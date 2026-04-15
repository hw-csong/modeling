import networkx as nx
import pytest

from zrt.adapter.parallel_group import CommPattern
from zrt.adapter.tp import TPRule, apply_tp, register_tp_rule, tp_rule_for
from zrt.common.tensor_base import DType, TensorBase
from zrt.graph.graph import GlobalGraph
from zrt.graph.node import Node


def _node(index: int, component: str, inputs=None, outputs=None) -> Node:
    return Node(
        index=index,
        aten_op="aten::mm",
        layer=0,
        module_path="m",
        component=component,
        inputs=inputs or [],
        outputs=outputs or [],
    )


def _act(last: int = 512) -> TensorBase:
    return TensorBase(shape=(1, 128, last), dtype=DType.FLOAT16)


def _weight(out: int, in_: int) -> TensorBase:
    return TensorBase(shape=(out, in_), dtype=DType.FLOAT16)


# -------------------------------------------------------- rule lookup


def test_default_rule_is_replicate():
    assert tp_rule_for("unknown_component") is TPRule.REPLICATE


def test_known_rules():
    assert tp_rule_for("attn.o_proj") is TPRule.ROW
    assert tp_rule_for("ffn.down_proj") is TPRule.ROW
    assert tp_rule_for("attn.q_a_proj") is TPRule.REPLICATE
    assert tp_rule_for("attn.q_b_proj") is TPRule.COLUMN


def test_register_override():
    register_tp_rule("custom_linear", TPRule.COLUMN)
    assert tp_rule_for("custom_linear") is TPRule.COLUMN


# -------------------------------------------------------- passthrough


def test_tp1_returns_source_unchanged():
    g = GlobalGraph()
    r = g.create_rank(0)
    r.add_op_node(_node(0, "ffn.up_proj"))
    assert apply_tp(g, 1) is g


def test_rejects_multi_rank_source():
    g = GlobalGraph()
    g.create_rank(0)
    g.create_rank(1)
    with pytest.raises(ValueError):
        apply_tp(g, 2)


# -------------------------------------------------------- structural rewrite


def test_column_shards_output_last_dim():
    g = GlobalGraph()
    r = g.create_rank(0)
    n = _node(
        0, "ffn.up_proj",
        inputs=[_act(512), _weight(1024, 512)],
        outputs=[_act(1024)],
    )
    r.add_op_node(n)

    out = apply_tp(g, 4)

    assert len(out.ranks) == 4
    for rid in range(4):
        clones = list(out.get_rank(rid).op_nodes())
        assert len(clones) == 1
        c = clones[0]
        # output last dim sharded
        assert c.outputs[0].shape == (1, 128, 1024 // 4)
        # weight dim-0 sharded
        assert c.inputs[1].shape == (1024 // 4, 512)
        # activation input unchanged (column rule)
        assert c.inputs[0].shape == (1, 128, 512)


def test_row_inserts_allreduce_per_rank_with_cross_rank_sync():
    g = GlobalGraph()
    r = g.create_rank(0)
    n = _node(
        0, "ffn.down_proj",
        inputs=[_act(1024), _weight(512, 1024)],
        outputs=[_act(512)],
    )
    r.add_op_node(n)

    tp = 2
    out = apply_tp(g, tp)

    # each rank now has: 1 compute + 1 comm
    computes, comm_nodes = [], []
    for rid in range(tp):
        nodes = list(out.get_rank(rid).op_nodes())
        assert len(nodes) == 2
        compute = next(x for x in nodes if not x.op_name.startswith("comm::"))
        comm = next(x for x in nodes if x.op_name.startswith("comm::"))
        computes.append(compute)
        comm_nodes.append(comm)

        # row-parallel shape: activation last dim / tp, weight col dim / tp
        assert compute.inputs[0].shape == (1, 128, 1024 // tp)
        assert compute.inputs[1].shape == (512, 1024 // tp)
        # output unchanged (all-reduce restores it)
        assert compute.outputs[0].shape == (1, 128, 512)

        # comm metadata
        assert comm.comm_pattern is CommPattern.ALL_REDUCE
        assert comm.group_size == tp
        assert comm.stream == 0

        # intra-rank: compute → comm
        assert out.get_rank(rid).has_edge(compute, comm)

    # cross-rank: every rank's compute feeds every other rank's comm
    for i in range(tp):
        for j in range(tp):
            if i == j:
                continue
            assert out.has_edge(computes[i], comm_nodes[j])


def test_downstream_reads_from_allreduce_after_row_op():
    g = GlobalGraph()
    r = g.create_rank(0)
    row = _node(
        0, "ffn.down_proj",
        inputs=[_act(1024), _weight(512, 1024)],
        outputs=[_act(512)],
    )
    downstream = _node(1, "ffn_norm", inputs=[_act(512)], outputs=[_act(512)])
    r.add_op_node(row)
    r.add_op_node(downstream)
    r.add_op_edge(row, downstream)

    out = apply_tp(g, 2)

    for rid in range(2):
        sub = out.get_rank(rid)
        comm = next(x for x in sub.op_nodes() if x.op_name.startswith("comm::"))
        down = next(x for x in sub.op_nodes() if x.component == "ffn_norm")
        assert sub.has_edge(comm, down)


def test_replicate_keeps_shapes_and_no_comm():
    g = GlobalGraph()
    r = g.create_rank(0)
    n = _node(
        0, "attn.q_a_proj",
        inputs=[_act(7168), _weight(1536, 7168)],
        outputs=[_act(1536)],
    )
    r.add_op_node(n)

    out = apply_tp(g, 4)
    for rid in range(4):
        nodes = list(out.get_rank(rid).op_nodes())
        assert len(nodes) == 1
        c = nodes[0]
        assert c.inputs[0].shape == (1, 128, 7168)
        assert c.inputs[1].shape == (1536, 7168)
        assert c.outputs[0].shape == (1, 128, 1536)


def test_chain_topology_preserved_per_rank():
    g = GlobalGraph()
    r = g.create_rank(0)
    a = _node(0, "attn_norm", outputs=[_act()])
    b = _node(1, "attn.q_b_proj", inputs=[_act(), _weight(2048, 512)], outputs=[_act(2048)])
    c = _node(2, "attn.o_proj", inputs=[_act(2048), _weight(512, 2048)], outputs=[_act(512)])
    d = _node(3, "ffn_norm", inputs=[_act(512)], outputs=[_act(512)])
    for n in (a, b, c, d):
        r.add_op_node(n)
    r.add_op_edge(a, b)
    r.add_op_edge(b, c)
    r.add_op_edge(c, d)

    out = apply_tp(g, 2)
    # topological order still valid globally
    assert list(nx.topological_sort(out))  # no cycle raised
    for rid in range(2):
        sub = out.get_rank(rid)
        # 4 compute + 1 all-reduce (for o_proj) = 5 nodes
        assert sub.number_of_nodes() == 5
