# Language-Independent Layout Transformer (LiLT) & Modular OCR Pipeline

A clean, modular Document Intelligence framework pairing **OCR** with **Language-Independent Layout Transformers (LiLT)**, complete with an interactive web studio and pipeline data flow inspection.

---

## 🌟 Key Architecture & Highlights

1. **Decoupled, Modular Design**:
   - **Swappable OCR Engine**: Abstract `BaseOCREngine` allows dropping in EasyOCR, Tesseract, PaddleOCR, AWS Textract, or Google Cloud Vision without changing downstream logic.
   - **Swappable Layout Transformer**: Abstract `BaseLayoutTransformer` enables swapping LiLT with LayoutLMv3, LayoutXLM, GeoLayoutLM, or custom transformers seamlessly.
   - **Model & OCR Registries**: Clean factory pattern (`ModelRegistry.get()`, `OCRRegistry.get()`) to configure and switch engines dynamically at runtime.
2. **Language-Independent Layout Representation (LiLT)**:
   - LiLT separates the text and layout pathways:
     - **Text Branch**: Any pre-trained multilingual language model backbone (e.g. RoBERTa, XLM-RoBERTa, InfoXLM).
     - **Layout Branch**: Encodes 2D spatial coordinates normalized to standard `[0, 1000]`.
     - **Cross-Attention**: Bi-directional cross-attention bridges text and layout without coupling coordinates to language.
3. **Data Flow & Coordinate Bridge**:
   - Word-level pixel bounding boxes $\to$ Standardized $[0, 1000]$ integer coordinate space.
   - Subword tokenization alignment: Every subword token inherits its parent word's 2D bounding box.
4. **Interactive Studio UI**:
   - **Switch between OCR Results and final LiLT Results with a single click**.
   - Inspect data at every stage of the pipeline (Step 1 Image $\to$ Step 2 OCR $\to$ Step 3 Normalization $\to$ Step 4 LiLT).
   - Interactive bounding box hover & click linked directly to data tables and entity cards.

---

## 🏗️ Project Structure

```
/Users/aditya/lilt/
├── core/
│   ├── __init__.py                # Package exports
│   ├── base_ocr.py                # Abstract BaseOCREngine, OCRWord, OCRResult
│   ├── easyocr_engine.py          # EasyOCR implementation (CRAFT + CRNN)
│   ├── normalizer.py              # [0, 1000] normalizer & subword-bbox alignment
│   ├── base_transformer.py        # Abstract BaseLayoutTransformer & entity structures
│   ├── lilt_transformer.py        # LiLT implementation (LiltForTokenClassification)
│   ├── alternative_transformers.py# Example swappable baseline model
│   ├── registry.py                # ModelRegistry & OCRRegistry
│   └── pipeline.py                # End-to-end DocumentPipeline coordinator
├── samples/                       # Sample documents (PAN card, invoice, etc.)
│   ├── pan_card_varun.png
│   ├── sample_invoice.png
│   └── real_pan_card.jpg
├── static/                        # Frontend Web Studio
│   ├── index.html                 # Clean, modern UI with OCR vs LiLT switcher
│   ├── style.css                  # Sleek dark theme styling
│   └── app.js                     # Canvas overlays & data flow inspector
├── app.py                         # FastAPI backend server
├── run_pipeline.py                # Standalone CLI runner script
├── test_modular.py                # Test script validating swappability
└── README.md
```

---

## 🔄 How to Replace LiLT with Another Transformer

The system is designed so that you can replace LiLT with any other transformer (e.g., LayoutLMv3, GeoLayoutLM, or a custom neural network) **in just 3 simple steps**:

### Step 1: Subclass `BaseLayoutTransformer`
Create a new file `core/my_transformer.py`:

```python
from core.base_transformer import BaseLayoutTransformer, LayoutPredictionResult, TokenPrediction, ExtractedEntity

class MyCustomTransformer(BaseLayoutTransformer):
    def __init__(self, model_id="microsoft/layoutlmv3-base"):
        super().__init__(name="MyCustomTransformer", model_id=model_id)
        # Load your tokenizer / processor / model weights here
        ...

    def predict(self, words, boxes_pixels, boxes_norm, image_size) -> LayoutPredictionResult:
        # 1. Prepare inputs for your model
        # 2. Run model forward pass
        # 3. Return standardized LayoutPredictionResult
        return LayoutPredictionResult(
            model_name=self.name,
            tokens=[...],
            entities=[...],
            word_predictions=[...]
        )

    def get_labels(self):
        return {0: "O", 1: "B-HEADER", 2: "B-QUESTION", 3: "B-ANSWER"}
```

### Step 2: Register it in `ModelRegistry`
In `core/registry.py` (or directly in your script):

```python
from core.registry import ModelRegistry
from core.my_transformer import MyCustomTransformer

ModelRegistry.register("my_model", lambda **kw: MyCustomTransformer(**kw))
```

### Step 3: Run the Pipeline with your new model
```python
from core.pipeline import DocumentPipeline

pipeline = DocumentPipeline(ocr_engine="easyocr", layout_model="my_model")
result = pipeline.run("path/to/document.png")
```
*No modifications to OCR, image pre-processing, normalizers, or the UI are required!*

---

## 🔀 How to Replace the OCR Engine

Similarly, to plug in Tesseract, PaddleOCR, or AWS Textract:

```python
from core.base_ocr import BaseOCREngine, OCRResult, OCRWord
from core.registry import OCRRegistry

class TesseractOCREngine(BaseOCREngine):
    def extract(self, image_input) -> OCRResult:
        pil_img = self.load_pil_image(image_input)
        # Call pytesseract.image_to_data(...)
        words = [
            OCRWord(id=i, text="...", bbox_pixels=[x0, y0, x1, y1], confidence=0.95)
        ]
        return OCRResult(
            image_width=pil_img.width,
            image_height=pil_img.height,
            words=words,
            engine_name="Tesseract"
        )

OCRRegistry.register("tesseract", TesseractOCREngine)
```

---

## 🚀 Running the Studio UI

To launch the web interface:

```bash
/Users/aditya/ocr_learning/.venv/bin/python app.py
```

Then open your browser at:
👉 **`http://localhost:8001`**

### Features in the UI:
- **Switch Buttons**: Click **`[ 1. OCR Results ]`** to view raw OCR detections with confidence badges, or **`[ 2. Final LiLT Results ]`** to see color-coded semantic entity predictions (Header, Question/Key, Answer/Value).
- **Interactive Overlay**: Hover over any bounding box to view exact pixel coordinates and $[0, 1000]$ normalized coordinates.
- **Pipeline Data Flow Ribbon**: Track resolution, word count, token count, and latency across each stage.
- **Sample Switcher & File Upload**: Test preloaded sample ID cards, invoices, or upload your own files.

---

## 💻 Running via Command Line (CLI)

You can run end-to-end inference on any image directly:

```bash
/Users/aditya/ocr_learning/.venv/bin/python run_pipeline.py samples/pan_card_varun.png --ocr easyocr --model lilt_funsd
```

Output:
```
===========================================================================
DOCUMENT PIPELINE: EASYOCR  --->  LILT_FUNSD
Target Document: samples/pan_card_varun.png
===========================================================================

[+] PIPELINE EXECUTION SUMMARY:
    Total Duration:     1.8s
    Words Detected:     35
    Tokens Processed:   47
    Entities Extracted: 5

--- EXTRACTED ENTITIES (LiLT) ---
  [HEADER      ] INCOME TAX DEPARTMENT               (Confidence: 0.98, Pixels: [180, 42, 620, 78])
  [QUESTION    ] Permanent Account Number            (Confidence: 0.95, Pixels: [82, 190, 310, 215])
  [ANSWER      ] ABCDE1234F                          (Confidence: 0.99, Pixels: [82, 222, 290, 250])
...
```
