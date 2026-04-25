# Intel Arc Pro B70 XPU Support

This project is being ported to run on Intel Arc Pro B70 (Xe2/Battlemage, 32GB VRAM).

## Current Status

We are actively working on XPU support. See `b70_support_plan.md` for the detailed implementation plan and progress tracking.

## Quick Reference

- **GPU:** Intel Arc Pro B70 (PCI ID `e223`, 32GB VRAM)
- **PyTorch:** XPU build from `https://download.pytorch.org/whl/xpu`
- **Target:** Flux2 LoRA training with musubi-tuner

## Key Files

- `b70_support_plan.md` - Detailed plan with phases and checkmarks for tracking progress
- `b70_agents.md` - This file

## Development Workflow

When implementing B70 support:
1. Check `b70_support_plan.md` for current phase and what's next
2. Make changes to the relevant files
3. Run the test command for the current phase
4. Update checkmarks in `b70_support_plan.md` when a phase passes
5. If crash, analyze error, fix, retry

## Important Notes

- PyTorch XPU uses `torch.xpu` namespace (not `torch.cuda`)
- `blocks_to_swap` uses CUDA-specific async streams → using synchronous fallback initially
- Async XPU optimization (proper `torch.xpu.Stream` usage) is Phase 6 (future)

## Contact

This is a work in progress by the flux2 project team.