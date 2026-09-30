#!/usr/bin/env python
"""CPU/RAM sampler for the pod session monitors: one CSV row per second until killed.
  python scripts/runpod/sampler.py OUT.csv
Columns: epoch_s, cpu_percent (all cores, 0-100), cpu_count, ram_used_gb, ram_total_gb,
top_python_rss_gb (largest python process), load1, gpu_util, gpu_mem_used_mib, gpu_power_w
(GPU columns from nvidia-smi, sampled on the same clock so they align with stages.jsonl)."""
import subprocess
import os
import sys
import time

import psutil

new = not os.path.exists(sys.argv[1])
with open(sys.argv[1], "a", buffering=1) as f:
    if new:
        f.write("epoch_s,cpu_percent,cpu_count,ram_used_gb,ram_total_gb,top_python_rss_gb,load1,gpu_util,gpu_mem_used_mib,gpu_power_w\n")
    psutil.cpu_percent(None)
    while True:
        time.sleep(1.0)
        vm = psutil.virtual_memory()
        rss = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                if "python" in (p.info["name"] or "") and p.pid != os.getpid():
                    rss = max(rss, p.info["memory_info"].rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            g = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                         "--format=csv,noheader,nounits", "-i", "0"], text=True, timeout=5)
            gu, gm, gp = [x.strip() for x in g.strip().split(",")]
        except Exception:
            gu = gm = gp = ""
        f.write("%.1f,%.1f,%d,%.2f,%.2f,%.2f,%.2f,%s,%s,%s\n" % (
            time.time(), psutil.cpu_percent(None), psutil.cpu_count(), vm.used / 2 ** 30, vm.total / 2 ** 30,
            rss / 2 ** 30, os.getloadavg()[0], gu, gm, gp))
