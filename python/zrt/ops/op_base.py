from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Tuple

from zrt.common.chip_spec import ChipSpec
from zrt.common.tensor_base import DType, TensorBase


class OpType(Enum):
    VECTOR = auto()
    CUBE = auto()
    MIX = auto()
    COMMUNICATION = auto()

@dataclass
class OpResult():
    # static cost for launching kernel
    static_cost:float #in us
    # compute cost
    total_compute_flops:float
    total_compute_time:float # in us
    compute_formula:str
    # memory cost
    total_memory_bytes:float
    total_memory_time:float # in us
    memory_formula:str
    # TODO communication cost

    def duration(self) -> float:
        """Roofline duration: kernel launch + max(compute, memory), in us."""
        return self.static_cost + max(self.total_compute_time, self.total_memory_time)

    def peak_memory(self) -> float:
        """Peak live memory for this op. in bytes"""
        return self.total_memory_bytes


class OperatorBase(ABC):
    static_cost:float

    def __init__(self, op_type:OpType, op_name:str, chip_spec:ChipSpec):
        self.op_type = op_type
        self.name = op_name
        self.chip_spec = chip_spec

    def __call__(self, input_tensors:List[TensorBase], **kwds: Any) -> "OpResult":
        inputs, outputs = self._infer_shape(input_tensors)
        return self.get_cost(inputs, outputs)

    def get_cost(self, inputs:List[TensorBase], outputs:List[TensorBase])->"OpResult":
        total_compute_flops, total_compute_time, compute_formula = self._compute_cost(inputs, outputs)
        total_memory_bytes, total_memory_time, memory_formula = self._memory_cost(inputs, outputs)

        result = OpResult(
            static_cost=self.static_cost,
            total_compute_flops=total_compute_flops,
            total_compute_time=total_compute_time,
            compute_formula=compute_formula,
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula=memory_formula
        )
        return result

    @abstractmethod
    def _infer_shape(self, input_tensors:List[TensorBase]) -> Tuple[List[TensorBase], List[TensorBase]]:
        ...

    @abstractmethod
    def _infer_dtype(self, input_tensors:List[TensorBase]) -> DType:
        ...

    @abstractmethod
    def _compute_cost(self, inputs:List[TensorBase], outputs:List[TensorBase]) -> Tuple[float, float, str]:
        ...

    @abstractmethod
    def _memory_cost(self, inputs:List[TensorBase], outputs:List[TensorBase]) -> Tuple[float, float, str]:
        ...
   

class OpVectorBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.VECTOR, op_name, chip_spec)


class OpCubeBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.CUBE, op_name, chip_spec)


class OpMixBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.MIX, op_name, chip_spec)


class OpCommBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.COMMUNICATION, op_name, chip_spec)


OP_CLASS_REGISTRY = {}

def op_register(names: str | List[str]):
    def decorator(cls: type[OperatorBase]):
        if isinstance(names, str):
            op_names = [names]
        elif isinstance(names, list) and all(isinstance(name, str) for name in names):
            op_names = names
        else:
            raise ValueError("names must be a string or a list of strings")
        for name in op_names:
            OP_CLASS_REGISTRY[name] = cls
        return cls
    return decorator

def get_class_by_name(name: str) -> type[OperatorBase]:
    
    if name not in OP_CLASS_REGISTRY:
        raise ValueError(f"Operator class not found for name: {name}")
    return OP_CLASS_REGISTRY[name]