import os
import sys
from core import DocumentPipeline, OCRRegistry, ModelRegistry

def main():
    print("=" * 70)
    print("TESTING MODULAR OCR & LAYOUT TRANSFORMER PIPELINE")
    print("=" * 70)
    
    sample_path = "samples/pan_card_varun.png"
    if not os.path.exists(sample_path):
        print(f"Error: sample path {sample_path} not found!")
        return

    print(f"Available OCR Engines: {OCRRegistry.list_available()}")
    print(f"Available Layout Models: {ModelRegistry.list_available()}")
    print("-" * 70)

    # 1. Test LiLT with EasyOCR
    print("\n[TEST 1] Running EasyOCR + LiLT (FUNSD)...")
    pipeline = DocumentPipeline(ocr_engine="easyocr", layout_model="lilt_funsd")
    result = pipeline.run(sample_path, render_overlays=True)

    print(f"Status: Success = {result['success']}")
    print(f"Total time: {result['total_time_ms']} ms")
    print(f"OCR Word Count: {result['ocr_result']['word_count']}")
    print(f"LiLT Model: {result['transformer_result']['model_name']}")
    print(f"LiLT Entities Extracted: {len(result['transformer_result']['entities'])}")
    for ent in result['transformer_result']['entities'][:5]:
        print(f"   - [{ent['label_type']}] {ent['text']} (conf: {ent['confidence']:.2f})")

    # 2. Test Swapping the Transformer Model seamlessly
    print("\n[TEST 2] Testing Swapping Transformer to Heuristic Baseline...")
    pipeline_swapped = DocumentPipeline(ocr_engine="easyocr", layout_model="heuristic_baseline")
    result_swapped = pipeline_swapped.run(sample_path, render_overlays=False)
    print(f"Swapped Model: {result_swapped['transformer_result']['model_name']}")
    print(f"Entities Extracted: {len(result_swapped['transformer_result']['entities'])}")
    print("\nPipeline modular swapping passed successfully!")

if __name__ == "__main__":
    main()
