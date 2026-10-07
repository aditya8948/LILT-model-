import time
from typing import Union, List, Optional
from PIL import Image
import numpy as np
import io
import torch

from .base_ocr import BaseOCREngine, OCRResult, OCRWord

class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR implementation of BaseOCREngine.
    Uses CRAFT text detection + CRNN recognition.
    """
    def __init__(self, languages: Optional[List[str]] = None, gpu: Optional[bool] = None):
        super().__init__(name="EasyOCR")
        self.languages = languages or ['en']
        if gpu is None:
            self.gpu = torch.cuda.is_available()
        else:
            self.gpu = gpu
        self._reader = None

    @property
    def reader(self):
        """Lazy load reader to save memory until first use."""
        if self._reader is None:
            import easyocr
            print(f"[*] Initializing EasyOCR Reader (languages={self.languages}, gpu={self.gpu})...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> OCRResult:
        start_time = time.perf_counter()
        
        # Load PIL image to get width and height
        pil_img = self.load_pil_image(image_input)
        img_w, img_h = pil_img.size

        # Convert to numpy array for EasyOCR
        np_img = np.array(pil_img)

        # Run OCR
        raw_results = self.reader.readtext(np_img)

        words: List[OCRWord] = []
        for idx, (bbox_points, text, confidence) in enumerate(raw_results):
            text_str = str(text).strip()
            if not text_str:
                continue

            # Convert 4 corners to axis-aligned [x0, y0, x1, y1]
            xs = [pt[0] for pt in bbox_points]
            ys = [pt[1] for pt in bbox_points]
            x0 = max(0, int(min(xs)))
            y0 = max(0, int(min(ys)))
            x1 = min(img_w, int(max(xs)))
            y1 = min(img_h, int(max(ys)))

            # Prevent empty boxes
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1

            word = OCRWord(
                id=idx,
                text=text_str,
                bbox_pixels=[x0, y0, x1, y1],
                confidence=float(confidence)
            )
            words.append(word)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            image_width=img_w,
            image_height=img_h,
            words=words,
            engine_name="EasyOCR",
            processing_time_ms=elapsed_ms,
            metadata={
                "languages": self.languages,
                "gpu": self.gpu,
                "raw_detections": len(raw_results)
            }
        )
