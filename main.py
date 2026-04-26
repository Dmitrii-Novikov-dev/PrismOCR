# trains OCR model, evaluates accuracy, and runs inference on test image

from PIL import Image
import numpy as np

from src.preprocessing import to_grayscale_np, binarize
from src.components import connected_components
from src.model import TemplateOCR


def main():
    img = Image.open("data/test_images/test.png")

    gray = to_grayscale_np(img)
    mask = binarize(gray)

    comps = connected_components(mask)

    ocr = TemplateOCR()

    for c in comps:
        glyph = np.zeros_like(mask)
        glyph[c.pixels[:, 0], c.pixels[:, 1]] = True

        print(ocr.match_glyph(glyph))


if __name__ == "__main__":
    main()