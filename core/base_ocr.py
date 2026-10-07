from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict, Any
from PIL import Image
import numpy as np
import io

@dataclass
class OCRWord:
    """Represents a single word/token detected by OCR."""
    id: int
    text: str
    bbox_pixels: List[int]  # [x0, y0, x1, y1] in absolute pixel coordinates
    confidence: float       # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox_pixels": self.bbox_pixels,
            "confidence": round(self.confidence, 4)
        }

@dataclass
class OCRResult:
    """Represents the complete result of OCR extraction on a document page."""
    image_width: int
    image_height: int
    words: List[OCRWord] = field(default_factory=list)
    engine_name: str = "BaseOCR"
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "image_size": [self.image_width, self.image_height],
            "word_count": len(self.words),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "words": [w.to_dict() for w in self.words],
            "full_text": self.full_text,
            "metadata": self.metadata
        }

class BaseOCREngine(ABC):
    """
    Abstract Base Class for OCR engines.
    Any OCR provider (EasyOCR, Tesseract, PaddleOCR, AWS Textract, Azure, etc.)
    can be plugged in by implementing `extract()`.
    """
    def __init__(self, name: str = "BaseOCREngine"):
        self.name = name

    @abstractmethod
    def extract(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> OCRResult:
        """
        Extract words, bounding boxes, and confidences from an image.
        
        Args:
            image_input: File path, image bytes, PIL Image, or numpy array.
            
        Returns:
            OCRResult object containing standardized word tokens and coordinates.
        """
        pass

    @staticmethod
    def load_pil_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """Helper to ensure input is a PIL RGB Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, (str, bytes, io.BytesIO)):
            if isinstance(image_input, bytes):
                image_input = io.BytesIO(image_input)
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
