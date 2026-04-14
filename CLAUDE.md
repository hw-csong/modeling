# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lightweight graph-based execution model for simulating LLM inference performance. The pipeline traces ATen-level ops from HuggingFace models into a `networkx.DiGraph`, rewrites that graph to model parallelism and runtime features, then walks it in topological order against per-op cost models to estimate memory and latency.

## Environment & Commands

- **Python env**: `conda run -n py311` (Python 3.11)
- **Run tests**: `conda run -n py311 python -m pytest tests/ -v`
- **Run one test**: `conda run -n py311 python -m pytest tests/<file>::<Class>::<test> -v -s`
- **Log level**: `LOG_LEVEL` env var (DEBUG/INFO/WARNING/ERROR)

`python/` is not an installed package. Scripts and tests must prepend `sys.path.insert(0, "python")` before importing `zrt`.

## Architecture

Cooperating components under `python/zrt/`:

### Capturer (`capturer/`)
Traces ATen-level operator sequences from HuggingFace model definitions (loaded on `torch.device("meta")` via `TorchDispatchMode`), records shapes/dtypes/module paths per op, and exports to CSV. The CSV is the boundary between capture and the rest of the system.

### Graph (`graph/`)
Builds a `networkx.DiGraph` from the captured CSV. Each node is an operator (or a fused group of operators). Supports operator fusion to produce higher-level computation graphs.

### Adapter (`adapter/`)
Rewrites the captured graph to model runtime features and parallelism. Target features:
- TP / DP / EP parallel
- MTP (DeepSeek-series)
- Prefix Cache
- Chunked Prefill
- Context Parallel via RingAttention

Adapter output is a **multi-rank multi-stream directed graph**: a global graph whose subgraphs represent individual ranks. Per-rank subgraphs may differ depending on enabled features. Each rank subgraph can carry multiple streams to model CUDA stream overlap (e.g. compute vs. communication).

### Runner (`runner/`)
Lightweight graph executor. Walks the `networkx.DiGraph` in topological order and assigns each node a `(start, end, duration)`. A node's start time is the max end time across its predecessors, accounting for stream and rank boundaries.

### Ops (`ops/`)
Cost models for simulating operator execution (memory + time):
- **Theoretical**: Roofline-based analytical model.
- **DB**: Looks up measured performance by op signature (inputs, dtype, shape).

### Common (`common/`)
Shared primitives — `TensorBase` (shape + dtype metadata), `ChipSpec` (vendor-neutral compute/bandwidth/interconnect specs), and logging utilities — used by every component above.

### Config (`config/`)
`RuntimeConfig` + `ParallelConfig` dataclasses. Carries chip spec, parallelism sizes (TP/DP/EP, PD disaggregation), MTP settings, prefix-cache and chunked-prefill toggles. Consumed by the GraphBuilder and Adapter.

## Layout

```
docs/                  # design docs
python/zrt/            # core library ("Zhanlu Runtime")
  capturer/            # HF → op trace
  graph/               # Node, Rank, GlobalGraph, GraphBuilder
  adapter/             # feature/parallelism rewrites → multi-rank multi-stream graph
  runner/              # topo-order simulator
  ops/                 # cost models + OperatorBase registry
  common/              # TensorBase, ChipSpec, logging
  config/              # RuntimeConfig / ParallelConfig
scripts/               # build / env / setup scripts
tests/                 # pytest suite
  deepseek_v3_ops.csv  # reference op trace (DeepSeek-V3)
```

## Repo Status

Early but no longer a pure skeleton. Current state:

| Module | Status |
|---|---|
| `common/` | Implemented: `TensorBase`, `ChipSpec`, logging |
| `graph/` | Implemented: `Node`, `Rank`, `GlobalGraph`, `GraphBuilder` (seeds one-rank copy) |
| `runner/` | Implemented: 4-step simulator (op results → ideal timeline → contention correction → peak memory) |
| `ops/` | Base only: `OperatorBase`, `OpResult`, `OpType`, `op_register` decorator. Theoretical / DB cost models not yet written |
| `config/` | `RuntimeConfig`, `ParallelConfig` dataclasses |
| `capturer/` | Empty stub classes; capture logic not yet integrated into `zrt/` |
| `adapter/` | Empty stub; all parallelism/feature rewrites TODO |

Tests only cover the runner (`tests/test_runner.py`). Verify actual file layout with `ls` before assuming a module exists.
