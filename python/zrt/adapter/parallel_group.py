from enum import Enum, auto
from typing import Optional

from zrt.common.tensor_base import TensorBase
from zrt.graph.node import Node


class CommPattern(Enum):
    ALL_REDUCE = auto()
    ALL_GATHER = auto()
    REDUCE_SCATTER = auto()
    ALL_TO_ALL = auto()


def make_comm_node(
    pattern: CommPattern,
    tensor: TensorBase,
    group_size: int,
    index: int = -1,
    layer: Optional[int] = None,
    module_path: str = "",
) -> Node:
    """Build a communication Node.

    Kept on the default compute stream: downstream ops depend on the comm
    result so there's no overlap to gain. `group_size` is stashed on the node
    so cost models can estimate collective latency without walking cross-rank
    edges.
    """
    op_name = f"comm::{pattern.name.lower()}"
    node = Node(
        index=index,
        aten_op=op_name,
        layer=layer,
        module_path=module_path,
        component=op_name,
        inputs=[tensor],
        outputs=[tensor],
    )
    node.comm_pattern = pattern
    node.group_size = group_size
    return node
