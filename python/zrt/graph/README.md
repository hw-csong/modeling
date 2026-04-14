# Global Graph

The global graph is a multi_rank multi_stream directed graph.

## Rank

Each rank represents the computation graph on an actual device. `Rank` is a `networkx.DiGraph` subclass bound to a parent `GlobalGraph`; adding a node or edge to a `Rank` is mirrored onto the global view so both stay in sync.

### Keypoints

- Each rank is a subgraph.
- At most `MAX_STREAMS_PER_RANK = 2` streams per rank (stream ids `0` / `1`).
- Cross-rank dependencies (usually around communication ops) are added on the `GlobalGraph` via `add_cross_rank_edge(src, dst)`; both endpoints must already live on different ranks.

## Node

Each node represents an op, holding its tensor metadata and runtime placement. A Node carries the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_id` | `int` | Process-wide unique id auto-assigned at construction. Used as the graph key. |
| `index` | `int` | Row index in the source op-trace CSV. |
| `op_name` | `str` | Name of the op (e.g. `aten.add.Tensor`). |
| `layer` | `Optional[int]` | Layer index, if the op belongs to a transformer block. |
| `module_path` | `str` | Fully-qualified module path where the op was invoked. |
| `component` | `str` | High-level component label (e.g. `attn.q_proj`, `moe.gate`). |
| `stream` | `int` | Stream id (`0` or `1`) — used to model compute / communication overlap. |
| `inputs` | `List[TensorBase]` | Input tensor metadata (shape + dtype). |
| `outputs` | `List[TensorBase]` | Output tensor metadata (shape + dtype). |

Constructors: `Node.from_csv_row(row)` / `Node.from_csv(path)` build Nodes from the captured-ops CSV. `Node.clone()` returns a deep copy so Adapter passes can rewrite without aliasing the source graph.

## Builder

`GraphBuilder` turns the fused op sequence produced by Capturer into a `GlobalGraph`. It takes:

- `raw_graph: nx.DiGraph` — the linear fused op sequence from Capturer, with one node per fused op and edges representing dataflow.
- `rt_config: RuntimeConfig` — parallelism, disaggregation, and runtime-feature settings.

Flow:

1. **Seed the GlobalGraph** — create a single `Rank` for the driver and mirror every op from `raw_graph` onto it. At this stage the GlobalGraph is a one-rank, one-stream copy of the fused sequence.
2. **Hand off to the Adapter** — the Adapter consumes this seeded GlobalGraph and rewrites it per `rt_config`: splitting it across ranks for TP/DP/EP, inserting communication ops, adding cross-rank edges, and assigning stream ids so compute and communication can overlap.

The Builder itself stays minimal — it constructs the faithful initial graph; all feature- and parallelism-aware rewrites live in the Adapter.