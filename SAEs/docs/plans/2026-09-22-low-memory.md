# Layer 26 extraction with occupied shared GPUs

Observed free VRAM: physical GPUs4–7 approximately8GiB each. User asks to use this remaining capacity. Existing37GiB-free default was a full-model conservative configuration, not a minimum requirement.

Implement opt-in --load-through-target: read original ChemDFM-R config, instantiate embedding + first26 Qwen2 blocks + final norm from unchanged BF16 checkpoint. Capture block25 before final norm; later blocks cannot affect it. No quantization, truncation of document context, or changed weights. Record loaded depth in extraction config and runtime memory/device map. Keep full-model default available. CPU tiny Qwen2 comparison must match full-depth block25 before real smoke.

Use four authorized GPUs with5GiB weight placement cap each and existing1GiB reserve guard. Run two actual documents per split in a separate smoke output. Observe peak memory and successful cache sealing. Do not launch full10M corpus until throughput/long-context feasibility is assessed. Never stop other GPU processes.
