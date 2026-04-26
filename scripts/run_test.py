# CLI script to run OCR inference on a test image
from PIL import Image
import numpy as np

from src.preprocessing import to_grayscale_np, binarize
from src.components import connected_components
from src.model import TemplateOCR


img = Image.open("data/test_images/test.png")

gray = to_grayscale_np(img)
mask = binarize(gray)

ocr = TemplateOCR()

comps = connected_components(mask)

for c in comps:
    glyph = np.zeros_like(mask)
    glyph[c.pixels[:, 0], c.pixels[:, 1]] = True
    print(ocr.match_glyph(glyph))