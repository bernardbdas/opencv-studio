import os
from PIL import Image, ImageDraw, ImageFont

def get_unicode_font(size: int):
    """Find a font that supports Unicode (CJK, accented Latin, etc.)."""
    # Prioritize fonts with full Unicode coverage
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size), path
            except Exception:
                continue
    return ImageFont.load_default(), None


def create_text_image(text: str, filename: str, title_text: str):
    """Create a high-contrast text card for OCR demo purposes."""
    width, height = 800, 500
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    font, font_path = get_unicode_font(36)
    title_font, _ = get_unicode_font(24)

    # Stylish border
    draw.rectangle([20, 20, width - 20, height - 20], outline=(56, 189, 248), width=3)

    # Title (top-left)
    draw.text((40, 40), title_text, fill=(148, 163, 184), font=title_font)

    # Word-wrap main text
    lines = [text]
    if len(text) > 35 and " " in text:
        words = text.split(" ")
        lines = []
        curr = ""
        for w in words:
            if len(curr + " " + w) < 30:
                curr = curr + " " + w if curr else w
            else:
                lines.append(curr)
                curr = w
        if curr:
            lines.append(curr)

    y_start = height // 2 - (len(lines) * 25)
    for idx, line in enumerate(lines):
        # Shadow + text for high contrast (OCR-friendly)
        draw.text((52, y_start + idx * 50 + 2), line, fill=(0, 0, 0), font=font)
        draw.text((50, y_start + idx * 50), line, fill=(248, 250, 252), font=font)

    # Save
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    assets_dir = os.path.join(root_dir, "assets", "demo")
    os.makedirs(assets_dir, exist_ok=True)

    dest = os.path.join(assets_dir, filename)
    img.save(dest, quality=95)
    print(f"  ✓ {filename} → {dest}  (font: {font_path or 'default'})")


if __name__ == "__main__":
    print("Generating OCR demo images…")
    create_text_image(
        "Hola, ¿cómo estás? Bienvenido a OpenCV Studio.",
        "ocr_spanish.jpg",
        "SPANISH OCR & TRANSLATION PRESET",
    )
    create_text_image(
        "La vision par ordinateur est incroyable.",
        "ocr_french.jpg",
        "FRENCH OCR & TRANSLATION PRESET",
    )
    create_text_image(
        "オープンCVスタジオへようこそ。",
        "ocr_japanese.jpg",
        "JAPANESE OCR & TRANSLATION PRESET",
    )
    print("Done.")
