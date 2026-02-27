"""Model management for MERIT."""

from .base import BaseModelAdapter


# Lazy imports for heavy modules
def __getattr__(name):
    _lazy_imports = {
        "ModelManager": ".manager",
        "EnhancedModelManager": ".manager",
        "create_model_manager": ".manager",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseModelAdapter", "ModelManager", "EnhancedModelManager", "create_model_manager"]
