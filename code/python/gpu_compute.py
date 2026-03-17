#!/usr/bin/env python3
"""
GPU Compute Utilities - Safe GPU Acceleration
==============================================

This module provides GPU acceleration with SAFE FALLBACKS.
It won't break if no GPU is available.

Supported backends (in priority order):
  1. CuPy - Drop-in numpy replacement for NVIDIA
  2. JAX - Functional numpy with auto-GPU
  3. PyTorch - Tensor operations
  4. NumPy CPU - Always works

Usage:
    from gpu_compute import xp, device_info, to_numpy

    # xp is numpy-like (cupy/jax/numpy depending on availability)
    x = xp.random.randn(1000, 1000)
    y = xp.dot(x, x.T)  # Runs on GPU if available

    # Convert back to numpy for I/O
    result = to_numpy(y)
"""

import os
import sys

# Configuration
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
PREFERRED_BACKEND = os.environ.get("GPU_BACKEND", "auto")  # auto, cupy, jax, torch, numpy

# Backend detection
_backend = None
_device = "cpu"
_backend_name = "numpy"

def _try_cupy():
    """Try to import CuPy (NVIDIA CUDA)."""
    global _backend, _device, _backend_name
    try:
        import cupy as cp
        # Test GPU is actually available
        cp.cuda.Device(0).compute_capability
        _backend = cp
        _device = "cuda"
        _backend_name = "cupy"
        return True
    except Exception:
        return False

def _try_jax():
    """Try to import JAX with GPU."""
    global _backend, _device, _backend_name
    try:
        # Suppress JAX warnings
        os.environ.setdefault("JAX_PLATFORMS", "gpu,cpu")
        import jax
        import jax.numpy as jnp

        # Check if GPU is available
        devices = jax.devices()
        if any(d.platform == "gpu" for d in devices):
            _backend = jnp
            _device = "gpu"
            _backend_name = "jax-gpu"
            return True
        else:
            _backend = jnp
            _device = "cpu"
            _backend_name = "jax-cpu"
            return True  # JAX CPU is still faster than numpy for large arrays
    except Exception:
        return False

def _try_torch():
    """Try to import PyTorch with GPU."""
    global _backend, _device, _backend_name
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            _backend_name = "torch-cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = "mps"  # Apple Silicon
            _backend_name = "torch-mps"
        else:
            _device = "cpu"
            _backend_name = "torch-cpu"

        # Torch doesn't have numpy API, but we can wrap it
        _backend = "torch"  # Special case
        return True
    except Exception:
        return False

def _use_numpy():
    """Fallback to NumPy (always works)."""
    global _backend, _device, _backend_name
    import numpy as np
    _backend = np
    _device = "cpu"
    _backend_name = "numpy"
    return True

def _init_backend():
    """Initialize the best available backend."""
    global _backend, _device, _backend_name

    if FORCE_CPU:
        _use_numpy()
        return

    backends = {
        "cupy": _try_cupy,
        "jax": _try_jax,
        "torch": _try_torch,
        "numpy": _use_numpy,
    }

    if PREFERRED_BACKEND != "auto" and PREFERRED_BACKEND in backends:
        if backends[PREFERRED_BACKEND]():
            return

    # Auto-detect best backend
    for name, init_fn in backends.items():
        if init_fn():
            return

# Initialize on import
_init_backend()

# Main export: numpy-compatible API
import numpy as np

if _backend_name.startswith("torch"):
    # PyTorch wrapper to provide numpy-like API
    import torch

    class TorchWrapper:
        """Wraps PyTorch to provide numpy-like API."""

        def __init__(self, device):
            self.device = device
            self._torch = torch

        def array(self, data, dtype=None):
            t = torch.tensor(data, device=self.device)
            if dtype is not None:
                t = t.to(getattr(torch, dtype) if isinstance(dtype, str) else dtype)
            return t

        def zeros(self, shape, dtype=None):
            return torch.zeros(shape, device=self.device)

        def ones(self, shape, dtype=None):
            return torch.ones(shape, device=self.device)

        def randn(self, *shape):
            return torch.randn(*shape, device=self.device)

        def rand(self, *shape):
            return torch.rand(*shape, device=self.device)

        def dot(self, a, b):
            return torch.matmul(a, b)

        def sum(self, x, axis=None):
            return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

        def mean(self, x, axis=None):
            return torch.mean(x, dim=axis) if axis is not None else torch.mean(x)

        def exp(self, x):
            return torch.exp(x)

        def sqrt(self, x):
            return torch.sqrt(x)

        def abs(self, x):
            return torch.abs(x)

        def sin(self, x):
            return torch.sin(x)

        def cos(self, x):
            return torch.cos(x)

        @property
        def random(self):
            return self  # randn/rand are top-level

        @property
        def linalg(self):
            return torch.linalg

        def __getattr__(self, name):
            # Fallback to torch
            return getattr(torch, name)

    xp = TorchWrapper(_device)
else:
    xp = _backend if _backend is not None else np


def to_numpy(x):
    """Convert any array type to numpy (for I/O, plotting, etc.)."""
    if isinstance(x, np.ndarray):
        return x

    if _backend_name.startswith("cupy"):
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)

    if _backend_name.startswith("jax"):
        import jax.numpy as jnp
        return np.asarray(x)

    if _backend_name.startswith("torch"):
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

    return np.asarray(x)


def to_device(x):
    """Convert numpy array to device array."""
    if isinstance(x, np.ndarray):
        return xp.array(x)
    return x


def device_info():
    """Return info about the current compute device."""
    info = {
        "backend": _backend_name,
        "device": _device,
        "gpu_available": _device != "cpu",
    }

    if _backend_name.startswith("cupy"):
        import cupy as cp
        dev = cp.cuda.Device(0)
        info["gpu_name"] = dev.pci_bus_id
        info["gpu_memory_gb"] = dev.mem_info[1] / 1e9

    if _backend_name.startswith("torch"):
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def benchmark_backend(size=2000, iterations=10):
    """
    Benchmark current backend with matrix operations.

    Returns dict with timing info.
    """
    import time

    print(f"Benchmarking {_backend_name} on {_device}...")
    print(f"Matrix size: {size}x{size}, iterations: {iterations}")

    # Generate test data
    a = xp.random.randn(size, size)
    if hasattr(a, 'astype'):
        a = a.astype('float32')  # GPU prefers float32

    # Warmup
    for _ in range(2):
        c = xp.dot(a, a.T) if hasattr(xp, 'dot') else a @ a.T

    # Synchronize if needed
    if _backend_name.startswith("cupy"):
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif _backend_name.startswith("torch") and _device != "cpu":
        import torch
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = xp.dot(a, a.T) if hasattr(xp, 'dot') else a @ a.T

        # Synchronize
        if _backend_name.startswith("cupy"):
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        elif _backend_name.startswith("torch") and _device != "cpu":
            import torch
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    flops = 2 * size**3 / avg_time / 1e9  # GFLOPS

    results = {
        "backend": _backend_name,
        "device": _device,
        "matrix_size": size,
        "avg_time_ms": avg_time * 1000,
        "gflops": flops,
    }

    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"Performance: {flops:.1f} GFLOPS")

    return results


# ============================================================================
# GPU-ACCELERATED PHYSICS FUNCTIONS
# ============================================================================

def pairwise_distances(pos, box_size=None):
    """
    Compute all pairwise distances efficiently.

    Args:
        pos: (N, 3) array of positions
        box_size: If given, apply periodic boundary conditions

    Returns:
        (N, N) distance matrix
    """
    pos = to_device(pos)
    n = pos.shape[0]

    # Vectorized distance calculation
    # diff[i,j] = pos[i] - pos[j]
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)

    if box_size is not None:
        # Minimum image convention
        diff = diff - box_size * xp.round(diff / box_size)

    # Euclidean distance
    dist = xp.sqrt(xp.sum(diff**2, axis=-1))

    return dist


def morse_potential_vectorized(r, D=4.8, alpha=2.0, r0=0.96):
    """
    Vectorized Morse potential.
    V(r) = D(1 - exp(-alpha*(r-r0)))^2 - D
    """
    r = to_device(r)
    return D * (1 - xp.exp(-alpha * (r - r0)))**2 - D


def lj_potential_vectorized(r, epsilon=0.01, sigma=3.0, cutoff=0.5):
    """
    Vectorized Lennard-Jones potential.
    V(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
    """
    r = to_device(r)

    # Avoid division by zero
    r_safe = xp.where(r < cutoff, cutoff, r)
    sr6 = (sigma / r_safe)**6
    energy = 4 * epsilon * (sr6**2 - sr6)

    # Hard core: very high energy for r < cutoff
    energy = xp.where(r < cutoff, 1e10, energy)

    return energy


def monte_carlo_energy_gpu(pos_H, pos_O, box_size,
                           D_OH=4.8, alpha_OH=2.0, r0_OH=0.96,
                           eps_HH=0.005, sig_HH=2.4,
                           eps_OO=0.010, sig_OO=3.0):
    """
    Compute total energy using GPU-accelerated pairwise calculations.

    This replaces the O(n^2) Python loops with vectorized operations.
    """
    pos_H = to_device(pos_H)
    pos_O = to_device(pos_O)

    # O-H distances (Morse)
    diff_OH = pos_H[None, :, :] - pos_O[:, None, :]  # (n_O, n_H, 3)
    diff_OH = diff_OH - box_size * xp.round(diff_OH / box_size)
    r_OH = xp.sqrt(xp.sum(diff_OH**2, axis=-1))  # (n_O, n_H)
    E_OH = xp.sum(morse_potential_vectorized(r_OH, D_OH, alpha_OH, r0_OH))

    # H-H distances (LJ)
    n_H = pos_H.shape[0]
    diff_HH = pos_H[:, None, :] - pos_H[None, :, :]
    diff_HH = diff_HH - box_size * xp.round(diff_HH / box_size)
    r_HH = xp.sqrt(xp.sum(diff_HH**2, axis=-1))
    # Upper triangle only (avoid double counting and self)
    mask_HH = xp.triu(xp.ones((n_H, n_H)), k=1)
    E_HH = xp.sum(lj_potential_vectorized(r_HH + 1e-10, eps_HH, sig_HH) * mask_HH)

    # O-O distances (LJ)
    n_O = pos_O.shape[0]
    diff_OO = pos_O[:, None, :] - pos_O[None, :, :]
    diff_OO = diff_OO - box_size * xp.round(diff_OO / box_size)
    r_OO = xp.sqrt(xp.sum(diff_OO**2, axis=-1))
    mask_OO = xp.triu(xp.ones((n_O, n_O)), k=1)
    E_OO = xp.sum(lj_potential_vectorized(r_OO + 1e-10, eps_OO, sig_OO) * mask_OO)

    total = E_OH + E_HH + E_OO
    return float(to_numpy(total))


# ============================================================================
# MAIN: Demo and benchmarks
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPU COMPUTE UTILITIES")
    print("=" * 60)

    info = device_info()
    print(f"\nBackend: {info['backend']}")
    print(f"Device: {info['device']}")
    print(f"GPU Available: {info['gpu_available']}")
    if "gpu_name" in info:
        print(f"GPU Name: {info['gpu_name']}")
    if "gpu_memory_gb" in info:
        print(f"GPU Memory: {info['gpu_memory_gb']:.1f} GB")

    print("\n" + "-" * 60)
    print("BENCHMARK")
    print("-" * 60)

    # Small benchmark
    results = benchmark_backend(size=1500, iterations=5)

    # Compare with numpy
    print("\n" + "-" * 60)
    print("NUMPY COMPARISON")
    print("-" * 60)

    import time
    size = 1500
    a_np = np.random.randn(size, size).astype(np.float32)

    times_np = []
    for _ in range(5):
        start = time.perf_counter()
        c = np.dot(a_np, a_np.T)
        times_np.append(time.perf_counter() - start)

    avg_np = sum(times_np) / len(times_np)
    gflops_np = 2 * size**3 / avg_np / 1e9

    print(f"NumPy avg time: {avg_np*1000:.2f} ms")
    print(f"NumPy performance: {gflops_np:.1f} GFLOPS")

    if info['gpu_available']:
        speedup = avg_np / (results['avg_time_ms'] / 1000)
        print(f"\nGPU Speedup: {speedup:.1f}x")

    # Test physics functions
    print("\n" + "-" * 60)
    print("PHYSICS FUNCTION TEST")
    print("-" * 60)

    np.random.seed(42)
    pos_H = np.random.uniform(0, 10, (20, 3))
    pos_O = np.random.uniform(0, 10, (10, 3))

    start = time.perf_counter()
    E = monte_carlo_energy_gpu(pos_H, pos_O, box_size=10.0)
    gpu_time = time.perf_counter() - start

    print(f"Monte Carlo energy: {E:.2f} eV")
    print(f"Compute time: {gpu_time*1000:.2f} ms")

    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
To use in your code:

    from gpu_compute import xp, to_numpy, device_info

    # Check what's running
    print(device_info())

    # Use xp like numpy (runs on GPU if available)
    x = xp.random.randn(1000, 1000)
    y = xp.dot(x, x.T)

    # For Monte Carlo simulations:
    from gpu_compute import monte_carlo_energy_gpu
    E = monte_carlo_energy_gpu(pos_H, pos_O, box_size)

Environment variables:
    FORCE_CPU=1          # Force CPU even if GPU available
    GPU_BACKEND=cupy     # Force specific backend (cupy/jax/torch/numpy)
""")
