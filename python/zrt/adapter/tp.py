from enum import Enum, auto
from typing import Dict, List

import networkx as nx

from zrt.adapter.parallel_group import CommPattern, make_comm_node
from zrt.common.tensor_base import TensorBase
from zrt.graph.graph import GlobalGraph
from zrt.graph.node import Node


class TPRule(Enum):
    COLUMN = auto()     # shard out-dim; output is a shard, no post-op comm
    ROW = auto()        # shard in-dim; partial output, post-op all-reduce
    REPLICATE = auto()  # unchanged on every rank


# DeepSeek-V3 component → TP rule. Extend via `register_tp_rule`.
# MLA: `*_a_proj` down-projects to a small compressed latent that stays
# replicated; only `*_b_proj` (up from latent into per-head space) is sharded.
_TP_RULES: Dict[str, TPRule] = {
    "attn.q_a_proj": TPRule.REPLICATE,
    "attn.kv_a_proj": TPRule.REPLICATE,
    "attn.q_b_proj": TPRule.COLUMN,
    "attn.kv_b_proj": TPRule.COLUMN,
    "attn.o_proj": TPRule.ROW,
    "ffn.up_proj": TPRule.COLUMN,
    "ffn.gate_proj": TPRule.COLUMN,
    "ffn.down_proj": TPRule.ROW,
    "moe.shared.up_proj": TPRule.COLUMN,
    "moe.shared.gate_proj": TPRule.COLUMN,
    "moe.shared.down_proj": TPRule.ROW,
    "lm_head": TPRule.COLUMN,
}


def tp_rule_for(component: str) -> TPRule:
    return _TP_RULES.get(component, TPRule.REPLICATE)


def register_tp_rule(component: str, rule: TPRule) -> None:
    _TP_RULES[component] = rule


def _shard_last(t: TensorBase, tp: int) -> TensorBase:
    if not t.shape or t.shape[-1] % tp != 0:
        return TensorBase(shape=t.shape, dtype=t.dtype)
    return TensorBase(shape=t.shape[:-1] + (t.shape[-1] // tp,), dtype=t.dtype)


def _shard_first(t: TensorBase, tp: int) -> TensorBase:
    if not t.shape or t.shape[0] % tp != 0:
        return TensorBase(shape=t.shape, dtype=t.dtype)
    return TensorBase(shape=(t.shape[0] // tp,) + t.shape[1:], dtype=t.dtype)


def _apply_shape_rule(node: Node, rule: TPRule, tp: int) -> Node:
    """Clone `node` and adjust input/output shapes per TP rule.

    Heuristic: a rank-2 input is treated as weight, everything else activation.
    COLUMN shards weight dim 0 + output last dim. ROW shards weight dim 1 +
    activation last dim; output unchanged (post-op all-reduce).
    """
    clone = node.clone()
    if rule is TPRule.REPLICATE:
        return clone

    new_inputs: List[TensorBase] = []
    for t in clone.inputs:
        if len(t.shape) == 2:
            new_inputs.append(
                _shard_first(t, tp) if rule is TPRule.COLUMN else _shard_last(t, tp)
            )
        else:
            new_inputs.append(
                TensorBase(shape=t.shape, dtype=t.dtype)
                if rule is TPRule.COLUMN
                else _shard_last(t, tp)
            )
    clone.inputs = new_inputs

    if rule is TPRule.COLUMN:
        clone.outputs = [_shard_last(t, tp) for t in clone.outputs]
    return clone


def apply_tp(src: GlobalGraph, tp_size: int) -> GlobalGraph:
    """Rewrite `src` (single rank-0 graph) into a `tp_size`-rank TP graph.

    Per-rank subgraphs mirror the source topology; ROW ops get a post-op
    all-reduce node on `COMM_STREAM`, cross-rank-wired so every rank's comm
    waits on every rank's compute.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if tp_size == 1:
        return src
    if 0 not in src.ranks or len(src.ranks) != 1:
        raise ValueError(
            "apply_tp expects a single-rank source graph seeded at rank 0; "
            f"got ranks={list(src.ranks)}"
        )

    src_rank = src.get_rank(0)
    order = list(nx.topological_sort(src_rank))

    dst = GlobalGraph()
    ranks = [dst.create_rank(r) for r in range(tp_size)]

    clones: Dict[Node, List[Node]] = {}
    comms: Dict[Node, List[Node]] = {}

    for n in order:
        rule = tp_rule_for(n.component)
        per_rank = [_apply_shape_rule(n, rule, tp_size) for _ in range(tp_size)]
        for r, clone in enumerate(per_rank):
            ranks[r].add_op_node(clone)
        clones[n] = per_rank

        if rule is TPRule.ROW and per_rank[0].outputs:
            comm_nodes = []
            for r in range(tp_size):
                c = make_comm_node(
                    pattern=CommPattern.ALL_REDUCE,
                    tensor=per_rank[r].outputs[0],
                    group_size=tp_size,
                    index=n.index,
                    layer=n.layer,
                    module_path=n.module_path,
                )
                ranks[r].add_op_node(c)
                ranks[r].add_op_edge(per_rank[r], c)
                comm_nodes.append(c)
            # All-reduce sync: each rank's comm waits on every other rank's compute.
            for i in range(tp_size):
                for j in range(tp_size):
                    if i != j:
                        dst.add_cross_rank_edge(per_rank[i], comm_nodes[j])
            comms[n] = comm_nodes

    # Preserve intra-rank edges. If upstream was ROW, read from its comm node.
    for u, v in src_rank.edges():
        ups = comms.get(u, clones[u])
        dns = clones[v]
        for r in range(tp_size):
            ranks[r].add_op_edge(ups[r], dns[r])

    return dst
