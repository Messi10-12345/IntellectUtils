# intellect_utils/jit.py

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def jit_compile(func=None, *, parallel=False, cache=True):
    """
    Decorator: JIT compile a function with numba.njit if available.

    Args:
        func (callable): Function to decorate.
        parallel (bool): Enable parallel loops inside function.
        cache (bool): Enable disk caching of compiled code.
    """
    if not NUMBA_AVAILABLE:
        # Dummy decorator - no effect if numba not installed
        def wrapper(f):
            return f
        return wrapper(func) if func else wrapper

    def decorator(f):
        return njit(f, parallel=parallel, cache=cache)

    return decorator(func) if func else decorator

# Optional: helper to create parallel loops inside jit functions
# Users can import prange directly from IntellectUtils.jit

if NUMBA_AVAILABLE:
    prange = prange
else:
    # fallback prange = range if numba not installed
    prange = range