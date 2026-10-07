import time
from typing import List, Dict, Any, Tuple
import re

from .base_transformer import (
    BaseLayoutTransformer,
    LayoutPredictionResult,
    TokenPrediction,
    ExtractedEntity,
    WordPrediction
)
from .normalizer import BBoxNormalizer

class HeuristicLayoutTransformer(BaseLayoutTransformer):
    """
    Demonstration of an alternative layout transformer / layout model.
    Shows how a developer can plug in any custom model, heuristic model,
    LayoutLMv3, or GeoLayoutLM simply by implementing BaseLayoutTransformer!
    """
    def __init__(self, name: str = "Rule-Heuristic Baseline (Alternative Engine)"):
        super().__init__(name=name, model_id="local-heuristic-baseline")
        self._labels = {
            0: "O",
            1: "B-HEADER",
            2: "I-HEADER",
            3: "B-QUESTION",
            4: "I-QUESTION",
            5: "B-ANSWER",
            6: "I-ANSWER"
        }

    def get_labels(self) -> Dict[int, str]:
        return self._labels

    def predict(
        self,
        words: List[str],
        boxes_pixels: List[List[int]],
        boxes_norm: List[List[int]],
        image_size: Tuple[int, int]
    ) -> LayoutPredictionResult:
        start_time = time.perf_counter()
        img_w, img_h = image_size

        tokens: List[TokenPrediction] = []
        word_predictions: List[WordPrediction] = []
        entities: List[ExtractedEntity] = []

        # Simple demonstration logic based on layout geometry & text patterns
        for idx, (word, pix_box, norm_box) in enumerate(zip(words, boxes_pixels, boxes_norm)):
            text_upper = word.upper()
            y_pos = norm_box[1]

            # Header detection (top of document or keyword)
            if y_pos < 180 or any(kw in text_upper for kw in ["INCOME", "TAX", "DEPARTMENT", "GOVT", "INVOICE", "RECEIPT"]):
                tag = "B-HEADER" if idx == 0 or word_predictions[-1].label != "B-HEADER" else "I-HEADER"
                conf = 0.92
            # Key/Question detection (ends with colon or matches field label patterns)
            elif text_upper.endswith(":") or any(kw in text_upper for kw in ["NAME", "FATHER", "DATE", "DOB", "NUMBER", "NO", "PAN", "TOTAL", "AMOUNT"]):
                tag = "B-QUESTION"
                conf = 0.88
            # Answer/Value detection (dates, alphanumerics, values)
            elif re.search(r"\d", word) or (idx > 0 and "QUESTION" in word_predictions[-1].label):
                tag = "B-ANSWER"
                conf = 0.85
            else:
                tag = "O"
                conf = 0.75

            token = TokenPrediction(
                token_index=idx,
                token=word,
                bbox_norm=norm_box,
                label=tag,
                confidence=conf,
                word_index=idx
            )
            tokens.append(token)

            wp = WordPrediction(
                word_id=idx,
                text=word,
                bbox_pixels=pix_box,
                bbox_norm=norm_box,
                label=tag,
                confidence=conf
            )
            word_predictions.append(wp)

            if tag != "O":
                entity_type = tag.replace("B-", "").replace("I-", "")
                entities.append(
                    ExtractedEntity(
                        id=len(entities),
                        label_type=entity_type,
                        text=word,
                        bbox_pixels=pix_box,
                        bbox_norm=norm_box,
                        confidence=conf,
                        word_indices=[idx]
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return LayoutPredictionResult(
            model_name=self.name,
            tokens=tokens,
            entities=entities,
            word_predictions=word_predictions,
            processing_time_ms=elapsed_ms,
            label_map=self._labels
        )
