# Intel Arc Pro B70 (Xe2/Battlemage) XPU Support Plan

## Overview

Goal: Make musubi-tuner work on Intel Arc Pro B70 (32GB VRAM) for Flux2 LoRA training.

**Context:** Arc Pro B70 uses Xe2 architecture (PCI ID `e223`). PyTorch XPU support is available via `torch-xpu` wheels. This is a porting effort since musubi-tuner was designed for NVIDIA CUDA.

---

## Phases

### Phase 0: Environment Setup
- [x] Install XPU PyTorch in musubi-tuner venv
- [x] Verify `torch.xpu.is_available()` returns `True`
- [x] Verify `torch.xpu.get_device_name(0)` returns `"Intel(R) Graphics [0xe223]"`

**Commands:**
```bash
source /home/mo/wan/musubi-tuner/venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

---

### Phase 1: Device Detection XPU-Aware
**Files to modify:**

1. **`src/musubi_tuner/cache_latents.py:337`**
   - Change `torch.cuda.is_available()` to check XPU too
   - Before: `device = args.device if ... else "cuda" if torch.cuda.is_available() else "cpu"`
   - After: detect both CUDA and XPU

2. **`src/musubi_tuner/cache_latents.py:384`** - Update `--device` help text to include `xpu`

3. **`src/musubi_tuner/flux_2_cache_latents.py:92`** - Same pattern

**Test:**
```bash
python src/musubi_tuner/flux_2_cache_latents.py \
  --dataset_config /home/mo/flux2/datasets/testcat/dataset_config.toml \
  --vae /home/mo/flux2/models/FLUX.2-dev/ae.safetensors \
  --model_version dev --batch_size 1
```

---

### Phase 2: Fix `clean_memory_on_device` for XPU
**File:** `src/musubi_tuner/modules/custom_offloading_utils.py:19-29`

Add XPU equivalents for:
- `torch.xpu.empty_cache()`
- `torch.xpu.synchronize()`

**Test:** Re-run Phase 1 test

---

### Phase 3: Fix Offloader Fallback (Sync Path) for XPU
**File:** `src/musubi_tuner/modules/custom_offloading_utils.py`

Changes:
- [ ] Line 111: Change `cuda_available = device.type == "cuda"` to also detect XPU
- [ ] Lines 320-321: Fix `torch.cuda.set_device()` for XPU or skip
- [ ] Ensure `swap_weight_devices_no_cuda` path is used when `cuda_available=False`

**Note:** Using synchronous fallback path (slow but safe). Async XPU streams are Phase 6.

**Test:** Re-run Phase 1 test with `blocks_to_swap=0`

---

### Phase 4: Test Full Caching (Latents + Text Encoder)
**Commands:**
```bash
# Latents
python src/musubi_tuner/flux_2_cache_latents.py \
  --dataset_config /home/mo/flux2/datasets/testcat/dataset_config.toml \
  --vae /home/mo/flux2/models/FLUX.2-dev/ae.safetensors \
  --model_version dev --batch_size 1

# Text encoder
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
  --dataset_config /home/mo/flux2/datasets/testcat/dataset_config.toml \
  --text_encoder /home/mo/flux2/models/FLUX.2-dev/text_encoder/model-00001-of-00010.safetensors \
  --model_version dev --device xpu --batch_size 1
```

**Success:** Both complete without errors

---

### Phase 5: Test Full Training Loop with blocks_to_swap
**Script:** `./train_lora_bf16_r32.sh testcat --training-only`

**Note:** Flux2 is ~24GB+ in bf16, needs `blocks_to_swap` to fit in 32GB VRAM.

**Success:** Training starts and runs without crashes

**If crash:** Analyze error → fix → restart Phase 5

---

### Phase 6: Optimize - XPU Async Streams (Future)
Once synchronous path works, implement proper XPU async offloading.

**Goal:** Use `torch.xpu.Stream` for async CPU↔XPU memory transfers instead of blocking `.to()` calls.

---

## File Changes Summary

| Phase | File | Lines | Change |
|-------|------|-------|--------|
| 0 | venv | - | Install XPU PyTorch (recreated venv with Python 3.13, installed XPU torch, dependencies manually) |
| 1 | `cache_latents.py` | ~337 | XPU device detection |
| 1 | `cache_latents.py` | ~384 | Update `--device` help |
| 1 | `flux_2_cache_latents.py` | ~92 | XPU device detection |
| 1 | `flux_2_cache_text_encoder_outputs.py` | ~38 | XPU device detection |
| 2 | `custom_offloading_utils.py` | ~19-29 | XPU empty_cache/synchronize (already had XPU support) |
| 3 | `custom_offloading_utils.py` | ~111 | Added `xpu_available` detection |
| 3 | `custom_offloading_utils.py` | ~321-323 | Guard `torch.cuda.set_device()` for XPU |

---

## Key Learnings

- PyTorch XPU uses `torch.xpu` namespace (not `torch.cuda`)
- `torch.xpu.is_available()` returns True when compute runtime is properly installed
- Arc Pro B70 needs compute-runtime v26.14+ for PCI ID `e223` support
- `blocks_to_swap` uses CUDA-specific async streams → fallback to sync path for XPU
- 32GB VRAM, Flux2 ~24GB+ bf16 → need offloading even on B70

---

## Success Criteria

1. ✅ Phase 1-4: Caching scripts run without errors
2. ✅ Phase 5: Training loop starts and runs without crashes
3. ⬜ LoRA weights are generated and work in inference

---

## References

- PyTorch XPU install: `/tmp/pytorch_intel_b70_install.md`
- musubi-tuner: `/home/mo/wan/musubi-tuner/`
- Training script: `/home/mo/flux2/train_lora_bf16_r32.sh`