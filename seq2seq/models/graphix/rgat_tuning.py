import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils import FFN
from .functions import *


class RGAT_Layer(nn.Module):

    def __init__(self, ndim, edim, num_heads=1, feat_drop=0.2):
        super(RGAT_Layer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim), \
            nn.Linear(self.ndim, dim, bias=False), nn.Linear(self.ndim, dim, bias=False)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)

    def forward(self, x, lgx, graph):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
        # set the same device:
        # DGL's graph.to(device) allocates GPU memory via its own CUDA allocator
        # (dgl::runtime::CUDADeviceAPI), which is entirely separate from PyTorch's caching
        # allocator -- when the model weights alone already consume ~98% of VRAM (e.g.
        # Graphix-3B on an 8GB card: confirmed 8002/8192 MiB used right after
        # model.to(device), before any of this ever runs), whatever PyTorch is holding but
        # not actively using in its cache is invisible to DGL's allocator and this raises a
        # genuine CUDA OOM even though nothing is actually short on memory
        # account-for-account. Releasing PyTorch's idle cached blocks back to the driver
        # first gives DGL's separate allocator a chance to find the room it needs.
        #
        # torch.cuda.empty_cache() forces a CUDA sync and makes the *next* allocation go
        # through a real cudaMalloc instead of PyTorch's cache -- calling it unconditionally
        # here (once per encoder layer, i.e. up to 12x per example on t5-base) is fine when
        # VRAM is that tight, but is pure, severe overhead whenever it isn't: confirmed via a
        # real ScienceBenchmark fine-tuning run on t5-base (6-7 GiB/8 GiB used, real
        # headroom) where a single optimizer step (32 micro-batches) hadn't completed after
        # nearly an hour with the GPU pegged at ~90-98% "utilization" that was mostly sync
        # overhead, not useful compute. Try the allocation first; only pay for empty_cache()
        # (and retry) on the rare case it's actually needed, so the common case (headroom
        # available) pays nothing extra.
        if x.is_cuda:
            try:
                g = graph.to(x.device)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                g = graph.to(x.device)
        else:
            g = graph.to(x.device)

        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lgx.view(-1, self.num_heads, self.d_k) if lgx.size(-1) == q.size(-1) else \
            lgx.unsqueeze(1).expand(-1, self.num_heads, -1)
        # DGL's CUDA sparse ops (the SpMM/SpMV kernels behind update_all/apply_edges in
        # propagate_attention) do not support any 16-bit dtype -- confirmed via a direct
        # repro on fp16: "DGLError: Data type not recognized with bits 16" out of
        # dgl/src/array/cuda/spmm.cu. This is independent of the VRAM-capacity issue above:
        # it would fail the same way even with unlimited memory, any time this layer's
        # input is 16-bit (fp16 or bf16) on a CUDA graph. The graph here is tiny
        # (num_nodes/num_edges are per-example question+schema token counts, not the
        # model's hidden size), so upcasting just this local, short-lived slice of data to
        # fp32 for the DGL calls -- and back down to the model's working dtype (bf16) for
        # the `affine_o`/`layernorm`/`ffn` that consume its output -- is cheap regardless
        # of device.
        try:
            with g.local_scope():
                g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k).float(), k.view(-1, self.num_heads, self.d_k).float()
                g.ndata['v'] = v.view(-1, self.num_heads, self.d_k).float()
                g.edata['e'] = e.float()
                out_x = self.propagate_attention(g)
            out_x = out_x.to(x.dtype)
        except dgl.DGLError as err:
            # This used to skip the RGAT for the example and carry on. On CPU, DGL 0.8.2's
            # libxsmm SpMM fails for every graph, so a run that fell back to CPU trained with
            # no RGAT at all while looking normal (incident of 2026-09-26, docs/incidentes.md).
            # A result labeled RGAT must have run the RGAT: fail the run instead.
            raise RuntimeError(
                "RGAT_Layer.forward: DGL failed in propagate_attention (n_nodes={}, n_edges={}, "
                "device={}); refusing to skip the RGAT silently".format(
                    graph.number_of_nodes(), graph.number_of_edges(), x.device)
            ) from err

        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)

        return out_x, lgx

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x