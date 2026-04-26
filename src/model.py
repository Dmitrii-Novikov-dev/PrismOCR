# stores glyph templates and performs matching using MSE

from typing import Dict, List
import numpy as np

from src.preprocessing import crop_to_foreground, resize_mask_nn


class TemplateOCR:
    def __init__(self, template_size=(32, 32)):
        self.template_size = template_size
        self.templates: Dict[str, List[np.ndarray]] = {}

    def add_template(self, label: str, mask: np.ndarray) -> None:
        label = label.replace("upper_", "").replace("lower_", "")
        mask = crop_to_foreground(mask.astype(bool))
        tmpl = resize_mask_nn(mask, self.template_size)
        self.templates.setdefault(label, []).append(tmpl)

    def mse(self, a, b):
        return float(np.mean((a - b) ** 2))

    def match_glyph(self, glyph_mask: np.ndarray) -> str:
        glyph = resize_mask_nn(
            crop_to_foreground(glyph_mask),
            self.template_size
        )

        best_label = "?"
        best_score = float("inf")

        for label, tmpls in self.templates.items():
            for t in tmpls:
                score = self.mse(glyph, t)
                if score < best_score:
                    best_score = score
                    best_label = label

        return best_label