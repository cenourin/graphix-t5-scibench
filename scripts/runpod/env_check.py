#!/usr/bin/env python
"""Environment checks for the phase-B pod sessions (docs/RUNPOD_PHASE_B.md).

  python scripts/runpod/env_check.py fingerprint          # print the Python-env fingerprint
  python scripts/runpod/env_check.py check OUT.json       # preflight checks; exit 1 on any failure

The container cannot read its own image digest, so the image is checked through what it
contains: a fingerprint of every installed Python distribution (name, version and the
sha256 of its RECORD file, which lists the sha256 of each installed file), compared with
the one recorded from silveirabruno/graphix-modern@sha256:371f61af... (docker/modern/
a5_env_fingerprint.json). The digest itself is pinned in the pod template.

Checks (all must pass): expected GPU (EXPECT_GPU, default "RTX 4090"), compute capability,
VRAM, driver, exact library versions, env fingerprint, CUDA usable by torch and by DGL (a
small message-passing op on the GPU), and fp32 strictness: TF32 off in the flags the
entrypoint sets, no environment override that forces TF32 (NVIDIA_TF32_OVERRIDE,
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE), no compile/autocast switches, and an EMPIRICAL test --
a 2048x2048 fp32 matmul on the GPU compared with a float64 reference: fp32 gives ~1e-6
relative error, TF32 ~1e-3, so a limit of 1e-4 tells them apart.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

EXPECTED = {"python": "3.11", "torch": "2.4.0", "torch_cuda": "12.1", "dgl": "2.4.0",
            "transformers": "4.57.6", "datasets": "2.21.0", "numpy": "1.26.4"}
FINGERPRINT_FILE = "docker/modern/a5_env_fingerprint.json"


def fingerprint():
    from importlib import metadata
    rows = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        record = dist.read_text("RECORD") or ""
        rows.append((name, dist.version, hashlib.sha256(record.encode()).hexdigest()))
    rows.sort()
    h = hashlib.sha256("\n".join("%s==%s %s" % r for r in rows).encode()).hexdigest()
    return {"python": platform.python_version(), "distributions": len(rows), "sha256": h,
            "packages": {n: v for n, v, _ in rows}}


def check(out_path):
    res, failures = {}, []

    def need(name, ok, value):
        res[name] = {"ok": bool(ok), "value": value}
        if not ok:
            failures.append(name)

    import numpy
    import torch
    need("python", platform.python_version().startswith(EXPECTED["python"] + "."), platform.python_version())
    need("torch", torch.__version__.split("+")[0] == EXPECTED["torch"], torch.__version__)
    need("torch_cuda_runtime", torch.version.cuda == EXPECTED["torch_cuda"], torch.version.cuda)
    need("numpy", numpy.__version__ == EXPECTED["numpy"], numpy.__version__)
    import transformers
    need("transformers", transformers.__version__ == EXPECTED["transformers"], transformers.__version__)
    import datasets
    need("datasets", datasets.__version__ == EXPECTED["datasets"], datasets.__version__)
    import dgl
    need("dgl", dgl.__version__.split("+")[0] == EXPECTED["dgl"], dgl.__version__)

    fp = fingerprint()
    ref = json.load(open(FINGERPRINT_FILE))
    need("env_fingerprint", fp["sha256"] == ref["sha256"],
         {"got": fp["sha256"], "expected": ref["sha256"], "image": ref.get("image")})
    res["image_declared"] = os.environ.get("GRAPHIX_IMAGE_DIGEST")  # declaration; the fingerprint is the check
    res["image_expected"] = ref["image"]
    try:  # the CPU share the container actually gets (cgroup v2), not the host's core count
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        res["cgroup_cpus"] = None if q == "max" else round(int(q) / int(per), 2)
    except Exception:
        res["cgroup_cpus"] = None
    res["os_cpu_count"] = os.cpu_count()

    need("cuda_available", torch.cuda.is_available(), torch.cuda.is_available())
    if not torch.cuda.is_available():
        return _finish(out_path, res, failures)
    name = torch.cuda.get_device_name(0)
    expect_gpu = os.environ.get("EXPECT_GPU", "RTX 4090")
    need("gpu_model", expect_gpu in name, name)
    need("gpu_count", torch.cuda.device_count() >= 1, torch.cuda.device_count())
    cap = torch.cuda.get_device_capability(0)
    expect_cap = os.environ.get("EXPECT_CAPABILITY", "8.9")
    need("compute_capability", "%d.%d" % cap == expect_cap, "%d.%d" % cap)
    arch = torch.cuda.get_arch_list()
    need("torch_has_kernels_for_gpu", any(a.startswith("sm_%d" % cap[0]) for a in arch), arch)
    vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    need("vram_gb", vram >= float(os.environ.get("EXPECT_MIN_VRAM_GB", "23")), round(vram, 2))
    try:
        drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                      text=True).strip()
    except Exception as e:  # nvidia-smi is injected by the NVIDIA runtime
        drv = None
        res["nvidia_smi_error"] = repr(e)
    need("driver", drv is not None, drv)
    res["cudnn"] = torch.backends.cudnn.version()

    # fp32 strictness -------------------------------------------------------------------
    need("env_GRAPHIX_ALLOW_TF32_off", os.environ.get("GRAPHIX_ALLOW_TF32", "0") != "1",
         os.environ.get("GRAPHIX_ALLOW_TF32"))
    need("env_NVIDIA_TF32_OVERRIDE_not_forcing", os.environ.get("NVIDIA_TF32_OVERRIDE", "0") == "0",
         os.environ.get("NVIDIA_TF32_OVERRIDE"))
    need("env_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE_unset", os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") == "0",
         os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))
    compile_vars = {k: v for k, v in os.environ.items() if k.startswith(("TORCHDYNAMO", "TORCHINDUCTOR", "TORCH_COMPILE"))}
    need("env_no_compile_switches", not compile_vars, compile_vars)
    # the entrypoint sets both flags from GRAPHIX_ALLOW_TF32 (graphix_modern/train.py); do the same
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    need("float32_matmul_precision_highest", torch.get_float32_matmul_precision() == "highest",
         torch.get_float32_matmul_precision())
    need("autocast_off", not torch.is_autocast_enabled(), torch.is_autocast_enabled())
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(2048, 2048, generator=g)
    b = torch.randn(2048, 2048, generator=g)
    ref64 = a.double() @ b.double()
    got = (a.cuda() @ b.cuda()).double().cpu()
    rel = ((got - ref64).norm() / ref64.norm()).item()
    need("empirical_fp32_matmul_not_tf32", rel < 1e-4, rel)

    # DGL on the GPU: the RGAT path's message passing -----------------------------------
    try:
        import dgl.function as fn
        gr = dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])), num_nodes=4, idtype=torch.int32).to("cuda")
        gr.ndata["h"] = torch.ones(4, 8, device="cuda")
        gr.edata["w"] = torch.full((4, 1), 2.0, device="cuda")
        gr.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "o"))
        from dgl.nn.functional import edge_softmax
        s = edge_softmax(gr, torch.ones(4, 1, device="cuda"))
        ok = bool(torch.allclose(gr.ndata["o"], torch.full((4, 8), 2.0, device="cuda"))) and s.is_cuda
        need("dgl_cuda_message_passing", ok, str(gr.device))
    except Exception as e:
        need("dgl_cuda_message_passing", False, repr(e))
    return _finish(out_path, res, failures)


def _finish(out_path, res, failures):
    res["failures"] = failures
    res["passed"] = not failures
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    for k, v in res.items():
        if isinstance(v, dict) and "ok" in v:
            print("%-40s %s  %s" % (k, "ok  " if v["ok"] else "FAIL", v["value"]))
    print("env_check", "PASSED" if not failures else "FAILED: %s" % failures)
    return not failures


if __name__ == "__main__":
    if sys.argv[1] == "fingerprint":
        print(json.dumps(fingerprint(), indent=1))
    else:
        sys.exit(0 if check(sys.argv[2]) else 1)
