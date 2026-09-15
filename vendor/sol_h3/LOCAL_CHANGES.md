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
startup. Sampling, LoRA fusion, scheduler shifts and model outputs are unchanged.
An optional `before_gpu_load` callback rechecks the current rank's free GPU memory
after CPU loading and immediately before the CUDA transfer. Loading logs include
each rank's PID so other GPU processes can be distinguished from this worker.

The wrapper `navid/h200.py` controls VAE compilation. This deployment selects Dense
attention, BF16 compute and BF16 Ulysses transport, and uses VAE batches per clip.
No Blackwell sparse-attention or MXFP8 kernels are selected. The unused upstream
sparse kernels are retained to keep the upstream source snapshot complete.
