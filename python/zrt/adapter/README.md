# Adapter

Adapter consumes the one-rank `GlobalGraph` seeded by `GraphBuilder` and rewrites it per `RuntimeConfig`: splitting nodes across ranks, inserting communication ops, adding cross-rank edges, and assigning stream ids so compute and communication can overlap. The result is the multi-rank multi-stream graph that the Runner executes.

## Status

Not yet implemented — [adapter.py](adapter.py) currently holds only an empty `Adapter` class. All features below are design targets.

## Features

### Parallel

#### TP

Tensor Parallel. We have to identify ops that support Tensor Parallel, changing their dim and adding communication op into right place.

#### DP

Data Parallel.

#### EP

Expert Parallel for MoE models. The raw op sequences usually represented FusedEP impl in modeling.py, we hope a DeepEP impl instead. So, we have to identify the MoE parts in raw op sequences, adding `Dispatch` and `Combine` op, changing Experts computation in `Gemm`. Additionally, placing shared experts in independent rank need to be considered.

### MTP

### Quant

### Prefix Cache

### Chunked Prefill
