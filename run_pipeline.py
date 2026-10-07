import argparse
import json
import os
import sys
from PIL import Image

from core import DocumentPipeline, OCRRegistry, ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Run modular OCR + Language-Independent Layout Transformer (LiLT)")
    parser.add_argument("image_path", nargs="?", default="samples/pan_card_varun.png", help="Path to input document image")
    parser.add_argument("--ocr", default="easyocr", choices=OCRRegistry.list_available(), help="OCR engine to use")
    parser.add_argument("--model", default="lilt_funsd", choices=ModelRegistry.list_available(), help="Layout Transformer model to use")
    parser.add_argument("--output_json", default="pipeline_output.json", help="Path to save output JSON")
    parser.add_argument("--save_overlays", action="store_true", help="Save OCR and LiLT visual overlay images")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[!] Error: File '{args.image_path}' does not exist.")
        sys.exit(1)

    print("=" * 75)
    print(f"DOCUMENT PIPELINE: {args.ocr.upper()}  --->  {args.model.upper()}")
    print(f"Target Document: {args.image_path}")
    print("=" * 75)

    pipeline = DocumentPipeline(ocr_engine=args.ocr, layout_model=args.model)
    result = pipeline.run(args.image_path, render_overlays=args.save_overlays)

    print("\n[+] PIPELINE EXECUTION SUMMARY:")
    print(f"    Total Duration:     {result['total_time_ms']} ms")
    print(f"    Words Detected:     {result['ocr_result']['word_count']}")
    print(f"    Tokens Processed:   {result['transformer_result']['token_count']}")
    print(f"    Entities Extracted: {len(result['transformer_result']['entities'])}")

    print("\n--- EXTRACTED ENTITIES (LiLT) ---")
    for entity in result['transformer_result']['entities']:
        print(f"  [{entity['label_type']:<12}] {entity['text']:<35} (Confidence: {entity['confidence']:.2f}, Pixels: {entity['bbox_pixels']})")

    # Save JSON output (strip base64 strings to keep JSON clean)
    clean_output = dict(result)
    if "visuals" in clean_output:
        clean_output["visuals"] = {k: ("data:image/...[omitted]" if v else None) for k, v in clean_output["visuals"].items()}

    with open(args.output_json, "w") as f:
        json.dump(clean_output, f, indent=2)
    print(f"\n[+] Saved detailed structured pipeline trace to: {args.output_json}")

    if args.save_overlays and "visuals" in result:
        print("[+] Visual overlays rendered.")

if __name__ == "__main__":
    main()
