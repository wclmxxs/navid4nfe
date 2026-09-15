# Source and local changes

Source: NVlabs/Sana, `sol-engine`, commit
`8e0db4fa562d727ea28b8d63015c196db7d97cae`, directory
`models/minimax_h3/Sol-H3/h3_runtime`.

`UPSTREAM.json` records the original files and their SHA256 values before local edits.
The upstream project page describes Sol-H3 code as Apache 2.0; this source snapshot
does not contain a top-level LICENSE file. Existing third-party notices and the
FlashAttention license are preserved verbatim. Model weights have their own terms.

Local edit in `h3_runtime/engine.py`: load components and move them to CUDA on one
rank at a time, with barriers around each rank. This reduces peak host RAM during
startup. The serialized load itself does not alter weights or scheduler arithmetic.
An optional `before_gpu_load` callback rechecks the current rank's free GPU memory
after CPU loading and immediately before the CUDA transfer. Loading logs include
each rank's PID so other GPU processes can be distinguished from this worker.
Progress logs also mark each completed rank, elapsed loading time and the start
of LoRA fusion so the launcher can show progress during a long startup.

The wrapper `navid/h200.py` controls VAE compilation. BF16 compute and BF16
Ulysses transport remain selected. `navid/runtime.py` installs request-scoped
Dense/Sol-Triton-TMA dispatch, 4096-row packing with explicit exclusion of the
extra rows from attention, optional DiT compilation, and DBCache-style residual
reuse (Fn=1/Bn=0). It does not select the Blackwell BSA/MXFP8 paths.

Additional local changes:
- `engine.py`: integer 4..15-second native schedules, variable internal canvas,
  request reference short-edge resolver and metrics. Scheduler shifts and LoRA
  coefficients remain unchanged.
- `output.py`: keep the requested number of video frames/audio samples and resize
  the 32-aligned canvas to requested even dimensions, with no retiming.
- `third_party/sol_attn/triton_ref/fwd.py`: TMA runtime logical length distinct
  from descriptor capacity, 4096-bucket autotuning; exact/approximate KV masks and
  partial-block masses use live length. CuTe SM90 code remains untouched.
- `third_party/sol_attn/preprocess.py`: tau is a runtime value so changing tuning
  strength does not specialize a new threshold kernel.

GPU validation is explicit: `navid/kernel_check.py` tests dense-limit and sparse
bucket/unbucketed parity on each H200 before model load. Full request A/B and
output checks are available through `./deploy.sh verify`. CPU tests alone do not
validate GPU kernel correctness or performance.
