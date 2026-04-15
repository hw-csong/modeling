from zrt.adapter.tp import apply_tp
from zrt.config.runtime_config import RuntimeConfig
from zrt.graph.graph import GlobalGraph


class Adapter:
    """Rewrite the seeded single-rank GlobalGraph per RuntimeConfig.

    Current support: TP. DP/EP/MTP/prefix-cache/chunked-prefill are TODO.
    """

    def __init__(self, global_graph: GlobalGraph, rt_config: RuntimeConfig):
        self.global_graph = global_graph
        self.rt_config = rt_config

    def apply_parallel(self) -> GlobalGraph:
        pc = self.rt_config.parallel_config
        g = apply_tp(self.global_graph, pc.tp_size)
        return g
