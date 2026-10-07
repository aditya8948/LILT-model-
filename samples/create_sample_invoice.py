from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_invoice(out_path="samples/sample_invoice.png"):
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header section
    draw.rectangle([40, 40, 760, 110], fill=(245, 247, 250), outline=(220, 225, 230))
    draw.text((60, 55), "ACME GLOBAL LOGISTICS INC.", fill=(20, 30, 50))
    draw.text((60, 80), "OFFICIAL COMMERCIAL INVOICE", fill=(70, 80, 95))
    draw.text((580, 60), "INVOICE #: INV-2026-881", fill=(30, 40, 60))
    draw.text((580, 85), "DATE: 2026-10-07", fill=(30, 40, 60))

    # Bill To / Ship To
    draw.text((60, 140), "BILL TO:", fill=(15, 23, 42))
    draw.text((60, 165), "Nexus Corp Technologies", fill=(51, 65, 85))
    draw.text((60, 185), "452 Silicon Parkway, Suite 900", fill=(51, 65, 85))
    draw.text((60, 205), "San Francisco, CA 94107", fill=(51, 65, 85))

    draw.text((450, 140), "PAYMENT DETAILS:", fill=(15, 23, 42))
    draw.text((450, 165), "TERMS: Net 30 Days", fill=(51, 65, 85))
    draw.text((450, 185), "DUE DATE: 2026-11-06", fill=(51, 65, 85))
    draw.text((450, 205), "CURRENCY: USD ($)", fill=(51, 65, 85))

    # Table Header
    draw.rectangle([40, 260, 760, 295], fill=(230, 235, 245))
    draw.text((60, 270), "ITEM DESCRIPTION", fill=(30, 41, 59))
    draw.text((430, 270), "QTY", fill=(30, 41, 59))
    draw.text((530, 270), "UNIT PRICE", fill=(30, 41, 59))
    draw.text((660, 270), "TOTAL", fill=(30, 41, 59))

    # Items
    items = [
        ("Cloud Infrastructure Node v4", "4", "$450.00", "$1,800.00"),
        ("LiLT Model Inference Gateway", "1", "$750.00", "$750.00"),
        ("OCR Processing Pipeline License", "2", "$300.00", "$600.00"),
        ("Enterprise Support & SLA", "1", "$500.00", "$500.00")
    ]
    y = 315
    for desc, qty, rate, total in items:
        draw.text((60, y), desc, fill=(51, 65, 85))
        draw.text((440, y), qty, fill=(51, 65, 85))
        draw.text((540, y), rate, fill=(51, 65, 85))
        draw.text((660, y), total, fill=(51, 65, 85))
        draw.line([40, y + 25, 760, y + 25], fill=(240, 240, 240))
        y += 45

    # Totals Section
    y_totals = 540
    draw.text((500, y_totals), "SUBTOTAL:", fill=(71, 85, 105))
    draw.text((660, y_totals), "$3,650.00", fill=(15, 23, 42))

    draw.text((500, y_totals + 30), "TAX (8.5%):", fill=(71, 85, 105))
    draw.text((660, y_totals + 30), "$310.25", fill=(15, 23, 42))

    draw.rectangle([480, y_totals + 65, 760, y_totals + 105], fill=(241, 245, 249))
    draw.text((500, y_totals + 77), "TOTAL DUE:", fill=(15, 23, 42))
    draw.text((660, y_totals + 77), "$3,960.25", fill=(15, 23, 42))

    # Footer note
    draw.text((60, 750), "THANK YOU FOR YOUR BUSINESS!", fill=(100, 116, 139))
    draw.text((60, 775), "For wire transfers, refer to routing number 021000021.", fill=(148, 163, 184))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"Saved sample invoice to: {out_path}")

if __name__ == "__main__":
    create_sample_invoice()
