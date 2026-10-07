from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class TokenPrediction:
    """Prediction for a single token in the layout transformer sequence."""
    token_index: int
    token: str
    bbox_norm: List[int]   # [0, 1000] scale
    label: str             # e.g. B-HEADER, B-QUESTION, B-ANSWER, etc.
    confidence: float
    word_index: int        # Index of original OCR word (-1 for special tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token": self.token,
            "bbox_norm": self.bbox_norm,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "word_index": self.word_index
        }

@dataclass
class ExtractedEntity:
    """Semantic Entity Recognition (SER) output."""
    id: int
    label_type: str        # e.g. "HEADER", "QUESTION", "ANSWER", "NAME", etc.
    text: str
    bbox_pixels: List[int] # [x0, y0, x1, y1] in pixel coords
    bbox_norm: List[int]   # [0, 1000] scale
    confidence: float      # SER confidence score (0.0 to 1.0)
    word_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label_type": self.label_type,
            "text": self.text,
            "bbox_pixels": self.bbox_pixels,
            "bbox_norm": self.bbox_norm,
            "confidence": round(self.confidence, 4),
            "ser_score": round(self.confidence * 100, 1),
            "word_indices": self.word_indices
        }

@dataclass
class ExtractedRelation:
    """Relation Extraction (RE) output linking two semantic entities (e.g. Question -> Answer)."""
    id: int
    relation_type: str       # e.g. "question_to_answer", "header_to_section"
    head_entity_id: int      # Question / Key entity ID
    tail_entity_id: int      # Answer / Value entity ID
    head_text: str           # Source text (Key)
    tail_text: str           # Target text (Value)
    head_bbox: List[int]     # [x0, y0, x1, y1]
    tail_bbox: List[int]     # [x0, y0, x1, y1]
    confidence: float        # RE relation score (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "relation_type": self.relation_type,
            "head_entity_id": self.head_entity_id,
            "tail_entity_id": self.tail_entity_id,
            "head_text": self.head_text,
            "tail_text": self.tail_text,
            "head_bbox": self.head_bbox,
            "tail_bbox": self.tail_bbox,
            "confidence": round(self.confidence, 4),
            "re_score": round(self.confidence * 100, 1)
        }

@dataclass
class WordPrediction:
    """Aggregated prediction at the OCR word level for visualization overlays."""
    word_id: int
    text: str
    bbox_pixels: List[int]
    bbox_norm: List[int]
    label: str             # Dominant BIO label or entity type
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "text": self.text,
            "bbox_pixels": self.bbox_pixels,
            "bbox_norm": self.bbox_norm,
            "label": self.label,
            "confidence": round(self.confidence, 4)
        }

@dataclass
class LayoutPredictionResult:
    """Standardized output returned by any Layout Transformer model."""
    model_name: str
    tokens: List[TokenPrediction] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    word_predictions: List[WordPrediction] = field(default_factory=list)
    processing_time_ms: float = 0.0
    label_map: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "token_count": len(self.tokens),
            "label_map": self.label_map,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "word_predictions": [w.to_dict() for w in self.word_predictions],
            "tokens": [t.to_dict() for t in self.tokens]
        }

class BaseLayoutTransformer(ABC):
    """
    Abstract Base Class for Layout Transformer models.
    Supports LiLT, LayoutLMv3, LayoutXLM, GeoLayoutLM, or any custom transformer.
    Any model conforming to this interface can be hot-swapped in code!
    """
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @abstractmethod
    def predict(
        self,
        words: List[str],
        boxes_pixels: List[List[int]],
        boxes_norm: List[List[int]],
        image_size: Tuple[int, int]
    ) -> LayoutPredictionResult:
        pass

    @abstractmethod
    def get_labels(self) -> Dict[int, str]:
        pass
