# Flash-Next reduced Top-k: prototype for scope discussion

This branch preserves a locally tested prototype on `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`
(engine 0.1.55). It is code for a feature discussion, not a ready-to-merge implementation.
It has not been rebased or executed on current upstream main. No default change is proposed.

## User problem and behavior

On one Windows PC with a Ryzen 7 5700X, 64 GB DDR4 and RX 9070 XT 16 GB, CPU expert work limits
generation throughput. An explicit reduced-expert mode allows a user to trade output quality for speed
without rewriting or duplicating the IQ3_S checkpoint. The user can return to full Top-10 between
requests without loading all model weights again.

The reduced mode still selects the original Top-10 probabilities, but evaluates only the first six
routed experts. Each retained expert keeps its original Top-10-normalized coefficient. Omitted
contributions are dropped; the shared expert stays unchanged. This changes model arithmetic and output
distribution. Keeping the original denominator does not make it lossless.

The experiment reduces decode, verifier and MTP expert width. The batched prefill path stays Top-10.
It does not reduce the resident checkpoint size or promise lower host-memory use.

## Prototype code and protocol

The relevant diff is in `src/program/generate.cpp`, `src/core/verify.cpp`, `src/core/mtp.cpp`,
`src/kernels/cuda/router_top10.cu`, their headers, `src/prefill/prefill.cpp`, and
`tests/hip/topk_reference.cpp`.

For the hot-switch experiment, start a native IQ3_S single-slot HIP server with:

```text
STRATA_EXPERIMENT_TOPK=10
STRATA_TOPK_REFERENCE=10
STRATA_HOT_TOPK=1
```

These environment names are prototype interfaces, not proposed stable public names. Unsupported batch,
pipeline, peer-device and multi-GPU combinations are rejected. The optional native router is not supported.
The experiment uses the HIP fast generic router; no CUDA, Metal or SYCL support is claimed.

While the engine is idle, a controller serializes access to the existing engine protocol and writes:

```text
TOPK 6
TOPK 10
```

Each command returns `TOPKED <k> <milliseconds>` or an error. This is an engine stdin control command,
not a chat prompt or a per-request sampling argument. The local service controller used in the recorded
integration tests is external to this branch; the stock Python server does not expose this control yet.

All width-dependent buffers start at Top-10 capacity. Compact expert work uses the active width, while
plan-field offsets keep the allocation capacity. A switch drains work, destroys width-dependent graphs,
clears sequence state and changes the active width. Model weights and GPU/RAM expert caches remain loaded.
Graphs are captured again on first use. Same-mode calls retain their prefix state; changing mode forces
the next request to prefill. Snapshots from different modes must not be reused.

## Recorded validation

Evidence is in `bench/results/2026-11-05-rx9070xt-topk-prototype/`. The JSON files contain measurements
from the local experiment, not a benchmark run made on current upstream main or an approved standard benchmark.

Passed on the recorded local build:

- Windows HIP Release build, `gfx1201`, native IQ3_S, ROCm SDK clang 22, MSVC toolchain.
- Router oracle: seven K/reference combinations, 129 rows each, including ties, extreme finite logits
  and compact-stride guard checks. Default Top-10 also matched the portable router in the recorded parity test.
- Five inference rounds, `10 -> 6 -> 10 -> 6 -> 10`, one engine PID. Each round checked arithmetic,
  same-mode prefix reuse, a structured tool call and executed generated Python: 20 checks completed.
- Four engine switch times: 44.228, 40.854, 43.728 and 50.189 ms. These exclude disk saving,
  HTTP/controller overhead, graph recapture and subsequent prefill.
- Cross-mode snapshot restore returned HTTP 400. Same-mode restore recovered 446 tokens.
  An integration request to switch while generating returned HTTP 409.

On 2026-11-06, the existing measured binary was also rerun with `topk_reference 10 10` and
`topk_reference 6 10`, with `STRATA_EXPERIMENT_TOPK=10 STRATA_TOPK_REFERENCE=10`.
Both returned zero failures and zero maximum absolute error against the CPU oracle. This was a
standalone router recheck, not a new full model run or a rebuild from this branch.

An earlier cold-start width sweep on the same PC measured 24.37 tok/s for Top-10/ref10 and
36.71 tok/s for Top-6/ref10. Each mode generated 384 tokens four times (two code and two Chinese
outputs), with 32 GiB expert RAM, 2,975 GPU expert slots (5.70 GiB), PCIe fraction 0.40,
spec=4, spec-min-p=0.5, `STRATA_SPEC_COST=1`, and `STRATA_ROUTE_TAIL_SKIP=0`.
Each mode was separately loaded and warmed. Sampling was temperature 0, seed 8167, reasoning off.
The aggregate is total generated tokens divided by total decode time, excluding prefill and loading.
Outputs and MTP acceptance differ, so this is not a fixed-trajectory kernel benchmark.

The 12-item smoke set scored 9/12 for both modes. It is far too small to establish quality equivalence.
Informal longer agent use produced inconsistent behavior, without a controlled comparison. No claim of
unchanged coding, reasoning or agent ability is made.

Failed during development:

- The first hot-switch run failed after 10 -> 6 because a dispatch plan used active width for offsets
  belonging to the allocated capacity. The `sink_.cap` changes address this; the five-round run was
  recorded after the fix. The earlier failure remains summarized in the evidence.
- In the earlier sweep, Top-6/ref6 and Top-5/ref10 scored 8/12; Top-5/ref5 was cancelled.
  These are not presented as equivalent-quality modes.

Not run:

- CUDA, SYCL and Metal builds or hardware execution.
- Current-main build and real-model execution.
- A broad default-off before/after byte-identity gate for this entire patch. Router parity alone is insufficient.
- Teacher-forced KL/PPL, standard large quality evaluations or controlled long-context agent evaluations.

## Scope proposed for an upstream implementation

1. An explicitly experimental, default-off reduced expert width, initially limited to the verified HIP
   native Flash-Next path. Full Top-10 remains the default.
2. Fixed Top-10 reference normalization for the first version, with clear quality tradeoff reporting.
3. A startup-only option first if maintainers prefer a smaller patch. Idle switching can follow separately.
4. If switching is included: serialization, state invalidation, mode-aware snapshot identity, capability
   reporting and failure handling belong in the native server, independent of model aliases and local paths.

Before a PR, the implementation needs scope agreement, a rebase, removal or separation of local build
plumbing, explicit backend guards, general configuration, full snapshot-identity coverage and targeted
backend/default-path checks. The Windows optional HIP diagnostics lookup included here is a build workaround
and should be considered separately. The older cold-start reference=K sweep is not a supported snapshot
mode of this prototype: its fingerprint currently assumes reference=10.

This differs from per-workload expert exclusion (#4445): all experts remain available and the checkpoint's
expert universe is unchanged; the active compute width is reduced. Neither mask calibration nor a smaller
checkpoint is proposed.

The local prototype and this proposal were prepared with AI assistance. Measurements come from saved local
runs; no unperformed tests are claimed. Maintainer scope agreement and acceptance are still pending.
