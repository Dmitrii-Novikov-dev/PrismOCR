# Dataset loading, train/test split, training pipeline, and model evaluation

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.preprocessing import to_grayscale_np, binarize
from src.components import connected_components
from src.model import TemplateOCR


def load_dataset(root: str):
    dataset = []

    for label in sorted(os.listdir(root)):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue

        for f in os.listdir(folder):
            path = os.path.join(folder, f)

            try:
                img = Image.open(path)
            except:
                continue

            gray = to_grayscale_np(img)
            mask = binarize(gray)

            comps = connected_components(mask)

            for c in comps:
                glyph = np.zeros_like(mask)
                glyph[c.pixels[:, 0], c.pixels[:, 1]] = True
                dataset.append((glyph, label))

    return dataset


def prepare_and_split_data(root: str):
    data = load_dataset(root)
    X, y = zip(*data)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_ocr_model(ocr: TemplateOCR, X_train, y_train):
    for x, y in zip(X_train, y_train):
        ocr.add_template(y, x)


def evaluate_ocr_model(ocr: TemplateOCR, X_test, y_test):
    correct = 0

    for x, y in zip(X_test, y_test):
        if ocr.match_glyph(x) == y:
            correct += 1

    return correct / len(y_test)