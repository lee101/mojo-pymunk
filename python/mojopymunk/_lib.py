"""Shared-library loader and NumPy buffer helpers."""

from __future__ import annotations

import ctypes
import atexit
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "physics.mojo")
LIB = os.environ.get("MOJOPYMUNK_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pymunk.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mp_create_cpu_context": ([], I),
    "mp_destroy_cpu_context": ([I], None),
    "mp_moment_for_circle": ([F, F, F, F, F], F),
    "mp_area_for_circle": ([F, F], F),
    "mp_moment_for_segment": ([F, F, F, F, F, F], F),
    "mp_area_for_segment": ([F, F, F, F, F], F),
    "mp_moment_for_box": ([F, F, F], F),
    "mp_moment_for_poly": ([F, I, I, F, F, F], F),
    "mp_area_for_poly": ([I, I, F], F),
    "mp_centroid_for_poly": ([I, I, I], None),
    "mp_batch_moment_for_circle": ([I, I, I, I, I, I], None),
    "mp_point_query_nearest_circle": ([I, I, F, F, F, I, I, I, I], I),
    "mp_point_query_nearest_circle_bvh": (
        [I, I, I, I, F, F, F, I, I, I, I],
        I,
    ),
    "mp_integrate": ([I, I, F, F, F, F, I], None),
    "mp_collide": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPYMUNK_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo:
        command = [mojo]
    else:
        pixi = shutil.which("pixi")
        if not pixi:
            raise BuildError("neither mojo nor pixi is available")
        command = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        command + ["build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None
_parallel_context: int | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def parallel_context() -> int:
    global _parallel_context
    if _parallel_context is None:
        _parallel_context = int(lib().mp_create_cpu_context())
    return _parallel_context


def _close_parallel_context() -> None:
    global _parallel_context
    if _library is not None and _parallel_context:
        _library.mp_destroy_cpu_context(_parallel_context)
    _parallel_context = None


atexit.register(_close_parallel_context)


def f64(values, *, ndim: int | None = None) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind == "c":
        raise TypeError("complex values cannot cross the float64 FFI boundary")
    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("conversion to float64 would narrow the input dtype")
    if source.dtype.kind in "iu" and source.size:
        limit = 1 << 53
        if np.any(source > limit) or np.any(source < -limit):
            raise ValueError("integer values must be exactly representable as float64")
    array = np.ascontiguousarray(source, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}D array")
    return array


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous float64 arrays")
    address = int(array.ctypes.data)
    if array.size and not address:
        raise ValueError("non-empty FFI buffer has a null address")
    return address
