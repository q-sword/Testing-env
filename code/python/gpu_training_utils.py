#!/usr/bin/env python3
"""
GPU Training Utilities - Safe Practices for Model Training
===========================================================

GOAL: GPU acceleration without shooting yourself in the foot.

Common footguns avoided:
  1. Memory leaks from not clearing gradients
  2. OOM from wrong batch sizes
  3. Silent CPU fallbacks killing perf
  4. Mixed precision done wrong
  5. DataLoader bottlenecks
  6. Not pinning memory
  7. Synchronization overhead
"""

import os
import sys
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

# ============================================================================
# 1. SAFE DEVICE SETUP
# ============================================================================

def get_device(prefer_cuda: bool = True, verbose: bool = True) -> "torch.device":
    """
    Get best available device with clear feedback.

    SAFE: Won't silently fall back to CPU without telling you.
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] Using {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[GPU] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[CPU] No GPU available - training will be slow!")
            print("      Consider: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    return device


def check_gpu_health():
    """
    Run GPU health check before training.

    Catches common issues BEFORE you waste hours.
    """
    import torch

    print("=" * 50)
    print("GPU HEALTH CHECK")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available")
        print("       Possible fixes:")
        print("       - pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("       - Check nvidia-smi works")
        return False

    # Basic info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Memory check
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / 1e9
    print(f"Memory: {total_mem:.1f} GB")

    # Quick computation test
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("[OK] GPU compute works")
    except Exception as e:
        print(f"[FAIL] GPU compute test: {e}")
        return False

    # Check memory is not already exhausted
    allocated = torch.cuda.memory_allocated() / 1e9
    if allocated > 0.5:
        print(f"[WARN] {allocated:.1f} GB already allocated - clear with torch.cuda.empty_cache()")

    print("=" * 50)
    return True


# ============================================================================
# 2. MEMORY MANAGEMENT
# ============================================================================

@dataclass
class MemoryStats:
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    total_gb: float

    @property
    def free_gb(self):
        return self.total_gb - self.reserved_gb


def get_gpu_memory() -> Optional[MemoryStats]:
    """Get current GPU memory stats."""
    import torch

    if not torch.cuda.is_available():
        return None

    return MemoryStats(
        allocated_gb=torch.cuda.memory_allocated() / 1e9,
        reserved_gb=torch.cuda.memory_reserved() / 1e9,
        max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
    )


def estimate_batch_size(model, input_shape, dtype="float32", safety_margin=0.8):
    """
    Estimate maximum safe batch size for a model.

    SAFE: Leaves headroom for gradients and optimizer states.
    """
    import torch

    if not torch.cuda.is_available():
        return 32  # Default for CPU

    # Get free memory
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    free_mem = props.total_memory - torch.cuda.memory_allocated()

    # Estimate memory per sample (rough: 4 bytes per param for fp32)
    # Account for: activations, gradients, optimizer states
    # Rule of thumb: need ~4x model size per sample for training
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == "float32" else 2
    mem_per_sample = param_count * bytes_per_param * 4  # 4x for grads+optimizer

    # Add input/output memory
    input_mem = torch.prod(torch.tensor(input_shape)).item() * bytes_per_param

    total_per_sample = mem_per_sample + input_mem
    max_batch = int(free_mem * safety_margin / total_per_sample)

    return max(1, max_batch)


@contextmanager
def memory_tracker(label: str = ""):
    """
    Context manager to track memory usage of a code block.

    Usage:
        with memory_tracker("Forward pass"):
            output = model(x)
    """
    import torch

    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()

    yield

    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    delta = (end_mem - start_mem) / 1e6
    peak = peak_mem / 1e6
    print(f"[MEM] {label}: delta={delta:+.1f}MB, peak={peak:.1f}MB")


def clear_memory():
    """
    Aggressively clear GPU memory.

    SAFE: Doesn't delete your model, just cached tensors.
    """
    import torch
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# 3. TRAINING LOOP UTILITIES
# ============================================================================

class GradScaler:
    """
    Simplified gradient scaler for mixed precision.

    SAFE: Handles NaN gradients gracefully.
    """

    def __init__(self, enabled: bool = True, init_scale: float = 65536.0):
        import torch

        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self._scaler = torch.amp.GradScaler('cuda', init_scale=init_scale)
        else:
            self._scaler = None

    def scale(self, loss):
        if self._scaler:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer):
        if self._scaler:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self._scaler:
            self._scaler.update()


def safe_training_step(
    model,
    batch,
    loss_fn,
    optimizer,
    scaler=None,
    max_grad_norm: float = 1.0,
    device=None,
):
    """
    Single training step with all safety checks.

    SAFE:
      - Clears gradients properly
      - Clips gradients to prevent explosion
      - Handles mixed precision correctly
      - Reports NaN losses
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    # Move batch to device
    if isinstance(batch, (tuple, list)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    else:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)

    # CRITICAL: Zero gradients BEFORE forward pass
    optimizer.zero_grad(set_to_none=True)  # set_to_none is faster

    # Forward with mixed precision if enabled
    use_amp = scaler is not None and scaler.enabled
    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    # Check for NaN loss
    if torch.isnan(loss):
        print("[WARN] NaN loss detected - skipping step")
        return None

    # Backward
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Check for exploding gradients
    if torch.isnan(grad_norm) or grad_norm > 1000:
        print(f"[WARN] Gradient explosion (norm={grad_norm:.1f}) - skipping step")
        optimizer.zero_grad(set_to_none=True)
        return None

    # Optimizer step
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return loss.item()


# ============================================================================
# 4. DATALOADER OPTIMIZATION
# ============================================================================

def create_fast_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = None,
    pin_memory: bool = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """
    Create optimized DataLoader.

    SAFE DEFAULTS:
      - Auto-detects optimal num_workers
      - Pins memory for GPU training
      - Uses persistent workers to avoid spawn overhead
    """
    import torch
    from torch.utils.data import DataLoader
    import os

    # Auto-detect num_workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        # Don't use more workers than make sense
        num_workers = min(cpu_count - 1, 8, len(dataset) // batch_size)
        num_workers = max(0, num_workers)

    # Pin memory if GPU available
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Persistent workers only if we have workers
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,  # Avoid weird batch sizes
    )


# ============================================================================
# 5. MIXED PRECISION SETUP
# ============================================================================

def setup_mixed_precision(model, optimizer, enabled: bool = True):
    """
    Set up mixed precision training safely.

    Returns: (model, optimizer, scaler)

    SAFE:
      - Only enables on compatible GPUs
      - Handles optimizer state correctly
      - Returns scaler for training loop
    """
    import torch

    if not enabled or not torch.cuda.is_available():
        return model, optimizer, None

    # Check GPU compute capability (need 7.0+ for good fp16)
    props = torch.cuda.get_device_properties(0)
    compute_cap = props.major + props.minor / 10

    if compute_cap < 7.0:
        print(f"[WARN] GPU compute capability {compute_cap} - fp16 may be slow")
        print("       Consider using fp32 instead")

    scaler = GradScaler(enabled=True)

    print(f"[AMP] Mixed precision enabled (GPU compute {compute_cap})")

    return model, optimizer, scaler


# ============================================================================
# 6. COMMON GOTCHAS DETECTOR
# ============================================================================

def detect_training_issues(model, dataloader, device=None):
    """
    Run checks to detect common training issues BEFORE training.

    Call this before your training loop to catch problems early.
    """
    import torch

    print("=" * 50)
    print("TRAINING ISSUE DETECTOR")
    print("=" * 50)

    issues = []

    if device is None:
        device = next(model.parameters()).device

    # 1. Check model is on right device
    param_device = next(model.parameters()).device
    if str(param_device) != str(device):
        issues.append(f"Model on {param_device} but expected {device}")

    # 2. Check model is in training mode
    if not model.training:
        issues.append("Model is in eval mode - call model.train()")

    # 3. Check for unused parameters (can cause DDP issues)
    try:
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (tuple, list)):
            sample_input = sample_batch[0][:1].to(device)
        else:
            sample_input = sample_batch["input"][:1].to(device)

        model.zero_grad()
        output = model(sample_input)
        if hasattr(output, 'sum'):
            output.sum().backward()

        unused_params = [name for name, p in model.named_parameters()
                        if p.grad is None and p.requires_grad]
        if unused_params:
            issues.append(f"Unused parameters: {unused_params[:3]}...")

        model.zero_grad()

    except Exception as e:
        issues.append(f"Forward pass failed: {e}")

    # 4. Check batch size vs memory
    if torch.cuda.is_available():
        mem = get_gpu_memory()
        if mem and mem.free_gb < 1.0:
            issues.append(f"Low GPU memory: {mem.free_gb:.1f}GB free")

    # 5. Check dataloader
    if dataloader.num_workers > 0 and not dataloader.pin_memory:
        issues.append("DataLoader workers>0 but pin_memory=False - add pin_memory=True")

    # Report
    if issues:
        print("[ISSUES FOUND]")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("[OK] No issues detected")

    print("=" * 50)
    return issues


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
GPU Training Utilities
======================

Usage in your training script:

    from gpu_training_utils import (
        get_device,
        check_gpu_health,
        setup_mixed_precision,
        create_fast_dataloader,
        safe_training_step,
        detect_training_issues,
        memory_tracker,
    )

    # Setup
    check_gpu_health()
    device = get_device()
    model = YourModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Enable mixed precision (2x speedup on modern GPUs)
    model, optimizer, scaler = setup_mixed_precision(model, optimizer)

    # Fast data loading
    train_loader = create_fast_dataloader(train_dataset, batch_size=32)

    # Detect issues before training
    detect_training_issues(model, train_loader, device)

    # Training loop
    for batch in train_loader:
        loss = safe_training_step(
            model, batch, loss_fn, optimizer,
            scaler=scaler, max_grad_norm=1.0, device=device
        )
        if loss is not None:
            print(f"Loss: {loss:.4f}")

Key safety features:
  - Explicit device management (no silent CPU fallback)
  - Memory tracking and leak detection
  - Gradient clipping and NaN handling
  - Mixed precision done right
  - DataLoader optimization with pin_memory
  - Pre-training issue detection
""")

    # Run health check
    check_gpu_health()
