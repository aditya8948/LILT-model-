import time
import io
import base64
from typing import Union, Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .base_ocr import BaseOCREngine, OCRResult
from .base_transformer import BaseLayoutTransformer, LayoutPredictionResult
from .normalizer import BBoxNormalizer
from .registry import OCRRegistry, ModelRegistry

class DocumentPipeline:
    """
    Modular Document Understanding Pipeline.
    Orchestrates:
    Input Image -> OCR Engine -> [0, 1000] Normalization & Alignment -> Layout Transformer -> Entity Extraction.
    """
    def __init__(
        self,
        ocr_engine: Optional[Union[BaseOCREngine, str]] = None,
        layout_model: Optional[Union[BaseLayoutTransformer, str]] = None
    ):
        # Resolve OCR Engine
        if isinstance(ocr_engine, BaseOCREngine):
            self.ocr_engine = ocr_engine
        elif isinstance(ocr_engine, str):
            self.ocr_engine = OCRRegistry.get(ocr_engine)
        else:
            self.ocr_engine = OCRRegistry.get("easyocr")

        # Resolve Layout Transformer Model
        if isinstance(layout_model, BaseLayoutTransformer):
            self.layout_model = layout_model
        elif isinstance(layout_model, str):
            self.layout_model = ModelRegistry.get(layout_model)
        else:
            self.layout_model = ModelRegistry.get("lilt_funsd")

    def run(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        render_overlays: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline and records data flow at each stage.
        """
        total_start = time.perf_counter()

        # Step 1: Ingest Image
        pil_image = BaseOCREngine.load_pil_image(image_input)
        img_w, img_h = pil_image.size

        trace_steps = []
        trace_steps.append({
            "step": 1,
            "title": "Document Input",
            "description": f"Loaded image with resolution {img_w}x{img_h} px",
            "data": {"dimensions": [img_w, img_h], "channels": len(pil_image.getbands())}
        })

        # Step 2: Run OCR
        ocr_result: OCRResult = self.ocr_engine.extract(pil_image)
        words = [w.text for w in ocr_result.words]
        boxes_pixels = [w.bbox_pixels for w in ocr_result.words]
        confidences = [w.confidence for w in ocr_result.words]
        avg_ocr_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        trace_steps.append({
            "step": 2,
            "title": "OCR Extraction (Text & 2D Geometry)",
            "description": f"Extracted {len(words)} words using {self.ocr_engine.name} in {ocr_result.processing_time_ms:.1f}ms",
            "data": {
                "word_count": len(words),
                "avg_confidence": round(avg_ocr_conf, 4),
                "engine": self.ocr_engine.name,
                "elapsed_ms": ocr_result.processing_time_ms
            }
        })

        # Step 3: Coordinate Normalization [0, 1000]
        norm_start = time.perf_counter()
        boxes_norm = [BBoxNormalizer.normalize_box(box, img_w, img_h) for box in boxes_pixels]
        norm_elapsed_ms = (time.perf_counter() - norm_start) * 1000.0

        trace_steps.append({
            "step": 3,
            "title": "BBox Normalization & Subword Alignment",
            "description": "Normalized all pixel coordinates to LiLT standard [0, 1000] integer scale",
            "data": {
                "coordinate_space": "[0, 1000]",
                "total_normalized": len(boxes_norm),
                "elapsed_ms": round(norm_elapsed_ms, 2)
            }
        })

        # Step 4: Layout Transformer Forward Pass
        transformer_result: LayoutPredictionResult = self.layout_model.predict(
            words=words,
            boxes_pixels=boxes_pixels,
            boxes_norm=boxes_norm,
            image_size=(img_w, img_h)
        )

        trace_steps.append({
            "step": 4,
            "title": "Layout Transformer (LiLT Inference)",
            "description": f"Processed {len(transformer_result.tokens)} tokens and extracted {len(transformer_result.entities)} entities in {transformer_result.processing_time_ms:.1f}ms",
            "data": {
                "model_name": transformer_result.model_name,
                "token_count": len(transformer_result.tokens),
                "entity_count": len(transformer_result.entities),
                "elapsed_ms": transformer_result.processing_time_ms
            }
        })

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        # Step 5: Render Visualizations (OCR overlay vs LiLT overlay)
        ocr_image_b64 = None
        lilt_image_b64 = None
        original_image_b64 = None

        if render_overlays:
            original_image_b64 = self._pil_to_base64(pil_image)
            ocr_image_b64 = self._render_ocr_overlay(pil_image, ocr_result)
            lilt_image_b64 = self._render_lilt_overlay(pil_image, transformer_result)

        return {
            "success": True,
            "total_time_ms": round(total_elapsed_ms, 2),
            "image_size": [img_w, img_h],
            "ocr_result": ocr_result.to_dict(),
            "transformer_result": transformer_result.to_dict(),
            "pipeline_trace": trace_steps,
            "visuals": {
                "original": original_image_b64,
                "ocr_overlay": ocr_image_b64,
                "lilt_overlay": lilt_image_b64
            }
        }

    def _pil_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _render_ocr_overlay(self, base_img: Image.Image, ocr: OCRResult) -> str:
        """Draws OCR bounding boxes (crisp neon green) and confidence tags."""
        img = base_img.copy().convert("RGB")
        draw = ImageDraw.Draw(img)

        for w in ocr.words:
            x0, y0, x1, y1 = w.bbox_pixels
            # Green bounding box
            draw.rectangle([x0, y0, x1, y1], outline=(16, 185, 129), width=2)
            # Label badge
            conf_percent = int(w.confidence * 100)
            tag_text = f"{w.text} ({conf_percent}%)"
            badge_h = 14
            badge_w = min(x1 - x0 + 10, len(tag_text) * 7 + 4)
            draw.rectangle([x0, max(0, y0 - badge_h), x0 + badge_w, y0], fill=(16, 185, 129))
            draw.text((x0 + 2, max(0, y0 - badge_h + 1)), tag_text[:25], fill=(255, 255, 255))

        return self._pil_to_base64(img)

    def _render_lilt_overlay(self, base_img: Image.Image, pred: LayoutPredictionResult) -> str:
        """Draws LiLT semantic prediction (SER) bounding boxes and relation extraction (RE) linking arrows."""
        img = base_img.copy().convert("RGB")
        draw = ImageDraw.Draw(img)

        color_palette = {
            "HEADER": (59, 130, 246),      # Blue
            "QUESTION": (245, 158, 11),    # Amber
            "ANSWER": (16, 185, 129),      # Emerald Green
            "NAME": (139, 92, 246),        # Purple
            "PAN_NUMBER": (236, 72, 153),  # Pink
            "DOB": (20, 184, 166),         # Teal
            "FATHER_NAME": (99, 102, 241), # Indigo
            "O": (148, 163, 184)           # Slate Gray
        }

        # 1. Draw SER Entities
        for ent in pred.entities:
            if ent.label_type == "O":
                continue
            x0, y0, x1, y1 = ent.bbox_pixels
            color = color_palette.get(ent.label_type, (99, 102, 241))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

            # Badge with SER score
            ser_score_pct = int(ent.confidence * 100)
            tag_text = f"SER [{ent.label_type}]: {ser_score_pct}%"
            badge_h = 16
            badge_w = min(x1 - x0 + 10, len(tag_text) * 7 + 8)
            draw.rectangle([x0, max(0, y0 - badge_h), x0 + badge_w, y0], fill=color)
            draw.text((x0 + 3, max(0, y0 - badge_h + 2)), tag_text, fill=(255, 255, 255))

        # 2. Draw RE Relations (connecting Question -> Answer)
        for rel in pred.relations:
            hx0, hy0, hx1, hy1 = rel.head_bbox
            tx0, ty0, tx1, ty1 = rel.tail_bbox

            # Center coordinates
            hc_x = int((hx0 + hx1) / 2)
            hc_y = int((hy0 + hy1) / 2)
            tc_x = int((tx0 + tx1) / 2)
            tc_y = int((ty0 + ty1) / 2)

            # Draw connecting relation line
            rel_color = (236, 72, 153) # Vivid Pink/Magenta
            draw.line([(hc_x, hc_y), (tc_x, tc_y)], fill=rel_color, width=2)

            # Draw dot at endpoints
            draw.ellipse([hc_x - 3, hc_y - 3, hc_x + 3, hc_y + 3], fill=(245, 158, 11))
            draw.ellipse([tc_x - 3, tc_y - 3, tc_x + 3, tc_y + 3], fill=(16, 185, 129))

            # RE Score Badge at midpoint
            mid_x = int((hc_x + tc_x) / 2)
            mid_y = int((hc_y + tc_y) / 2)
            re_pct = int(rel.confidence * 100)
            re_label = f"RE: {re_pct}%"
            draw.rectangle([mid_x - 22, mid_y - 7, mid_x + 22, mid_y + 7], fill=rel_color)
            draw.text((mid_x - 19, mid_y - 6), re_label, fill=(255, 255, 255))

        return self._pil_to_base64(img)
