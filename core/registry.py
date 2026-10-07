from typing import Dict, Type, Any, Optional
from .base_ocr import BaseOCREngine
from .base_transformer import BaseLayoutTransformer
from .easyocr_engine import EasyOCREngine
from .lilt_transformer import LiLTLayoutTransformer
from .alternative_transformers import HeuristicLayoutTransformer

class OCRRegistry:
    """Registry for plug-and-play OCR engines."""
    _registry: Dict[str, Type[BaseOCREngine]] = {}
    _instances: Dict[str, BaseOCREngine] = {}

    @classmethod
    def register(cls, name: str, engine_cls: Type[BaseOCREngine]):
        cls._registry[name.lower()] = engine_cls

    @classmethod
    def get(cls, name: str = "easyocr", **kwargs) -> BaseOCREngine:
        key = name.lower()
        if key not in cls._instances:
            if key not in cls._registry:
                raise ValueError(f"Unknown OCR engine '{name}'. Available: {list(cls._registry.keys())}")
            cls._instances[key] = cls._registry[key](**kwargs)
        return cls._instances[key]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())

class ModelRegistry:
    """Registry for plug-and-play Layout Transformer models."""
    _registry: Dict[str, Any] = {}
    _instances: Dict[str, BaseLayoutTransformer] = {}

    @classmethod
    def register(cls, name: str, factory_or_cls: Any):
        cls._registry[name.lower()] = factory_or_cls

    @classmethod
    def get(cls, name: str = "lilt_funsd", **kwargs) -> BaseLayoutTransformer:
        key = name.lower()
        if key not in cls._instances:
            if key not in cls._registry:
                raise ValueError(f"Unknown model '{name}'. Available: {list(cls._registry.keys())}")
            factory = cls._registry[key]
            cls._instances[key] = factory(**kwargs) if callable(factory) else factory
        return cls._instances[key]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())

# Register default engines
OCRRegistry.register("easyocr", EasyOCREngine)

# Register default models
ModelRegistry.register(
    "lilt_funsd",
    lambda **kw: LiLTLayoutTransformer(model_id="nielsr/lilt-roberta-en-base-finetuned-funsd", **kw)
)
ModelRegistry.register(
    "lilt_base",
    lambda **kw: LiLTLayoutTransformer(model_id="SCUT-DLVCLab/lilt-roberta-en-base", **kw)
)
ModelRegistry.register(
    "heuristic_baseline",
    lambda **kw: HeuristicLayoutTransformer(**kw)
)
