import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Modular pipeline components configured in code
from core import DocumentPipeline

app = FastAPI(title="OCR & Layout Transformer Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

from core import DocumentPipeline, ModelRegistry

pipeline_cache = {}

def get_pipeline(layout_model: str = "lilt_funsd", ocr_engine: str = "easyocr") -> DocumentPipeline:
    key = f"{ocr_engine}::{layout_model}"
    if key not in pipeline_cache:
        pipeline_cache[key] = DocumentPipeline(ocr_engine=ocr_engine, layout_model=layout_model)
    return pipeline_cache[key]

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    with open(index_file, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/models")
async def list_models():
    return {
        "layout_models": ModelRegistry.list_available(),
        "default": "lilt_funsd"
    }

@app.get("/api/samples")
async def list_samples():
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")) and not fname.startswith("annotated_"):
                sample_files.append({"filename": fname, "url": f"/samples/{fname}"})
    return {"samples": sample_files}

@app.post("/api/process")
async def process_document(
    file: UploadFile = File(None),
    sample_name: str = Form(None),
    layout_model: str = Form("lilt_funsd")
):
    try:
        pil_image = None
        if file is not None and file.filename:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        elif sample_name:
            sample_path = os.path.join(SAMPLES_DIR, sample_name)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample not found.")
            pil_image = Image.open(sample_path).convert("RGB")
        else:
            default_sample = os.path.join(SAMPLES_DIR, "pan_card_varun.png")
            if os.path.exists(default_sample):
                pil_image = Image.open(default_sample).convert("RGB")
            else:
                raise HTTPException(status_code=400, detail="Please upload an image.")

        # Run modular pipeline with selected model
        active_pipeline = get_pipeline(layout_model=layout_model)
        result = active_pipeline.run(pil_image, render_overlays=True)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8005, reload=True)
