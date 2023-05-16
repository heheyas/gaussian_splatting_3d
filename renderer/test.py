import torch
try:
    import _renderer as _backend
except ImportError:
    from .backend import _backend
    
    