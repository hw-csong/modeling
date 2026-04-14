# Operators

Cost models that let the Runner convert a `Node` into a simulated `(duration, memory)` pair.

## Current State

Only the common scaffolding is implemented in [op_base.py](op_base.py); no concrete cost models yet.

## Scaffolding

### `OpType`
Enum classifying an op by the execution resource it primarily contends for:

| Member | Meaning |
|---|---|
| `VECTOR` | Element-wise / reduction ops running on vector units |
| `CUBE` | GEMM / conv-style ops running on tensor / matrix units |
| `MIX` | Ops that use both pipelines |
| `COMMUNICATION` | Cross-rank collectives (AllReduce, AllGather, Dispatch, Combine, …) |

### `OpResult`
Dataclass returned by a cost model for a single op:

| Field | Meaning |
|---|---|
| `static_cost` | Fixed kernel-launch overhead (µs) |
| `total_compute_flops` / `total_compute_time` / `compute_formula` | Compute cost + human-readable formula |
| `total_memory_bytes` / `total_memory_time` / `memory_formula` | Memory cost + formula |
| `duration()` | `static_cost + max(compute_time, memory_time)` — roofline duration consumed by the Runner |
| `peak_memory()` | Live memory footprint in bytes, consumed by the Runner's peak-memory sweep |

### `OperatorBase` and subclasses
Per-op-family base classes bound to an `OpType`:

- `OpVectorBase`, `OpCubeBase`, `OpMixBase`, `OpCommBase`

Each concrete op subclasses one of these and implements `get_memory_cost()` / `get_compute_cost()`.

### `op_register` / `OP_CLASS_REGISTRY`
Decorator-based registry mapping aten op names to `OperatorBase` subclasses:

```python
@op_register(["aten.add.Tensor", "aten.add.Scalar"])
class AddOp(OpVectorBase):
    ...
```

Look up a class by name via `get_class_by_name("aten.add.Tensor")`.

## Planned Cost Models

- **Theoretical** — roofline from shape / dtype / op type + `ChipSpec`.
- **DB** — measured lookup keyed by op signature (op name, input/output shapes, dtypes).

Both will return `OpResult`; the Runner is agnostic to which is used.
