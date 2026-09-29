"""Port of seq2seq/models/graphix/rgat_tuning.py (RGAT_Layer) to DGL 2.4 (step A3).

Minimal-scope changes, nothing else (PORTABILIDADE.md §4, rows 2, 3 and 5):
  - fn.copy_edge -> fn.copy_e (copy_edge was removed from DGL);
  - graph.number_of_nodes()/number_of_edges() -> num_nodes()/num_edges() (deprecated);
  - DGLError is no longer caught: the legacy layer skipped the RGAT for an example when
    DGL failed, which an equivalence port must not hide;
  - the empty_cache() + retry around graph.to(device) is dropped (an 8 GB-VRAM
    workaround; the plain graph.to(device) is the same operation).
Kept exactly: parameters and their names, the three independent feat_dropout calls, the
fp32 upcast around the DGL calls (DGL's sparse kernels have no 16-bit support), the
attention math and edge order.
"""
import math

import dgl.function as fn
import torch.nn as nn

from .functions import div_by_z, scaled_exp, src_sum_edge_mul_dst, src_sum_edge_mul_edge
from .model_utils import FFN


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
                graph: dgl.graph
        """
        g = graph.to(x.device)

        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lgx.view(-1, self.num_heads, self.d_k) if lgx.size(-1) == q.size(-1) else \
            lgx.unsqueeze(1).expand(-1, self.num_heads, -1)
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k).float(), k.view(-1, self.num_heads, self.d_k).float()
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k).float()
            g.edata['e'] = e.float()
            out_x = self.propagate_attention(g)
        out_x = out_x.to(x.dtype)

        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)

        return out_x, lgx

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x
