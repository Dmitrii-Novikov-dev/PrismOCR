from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Utility: labels
# -----------------------------

def clean_label(label: str) -> str:
    label = str(label)

    if label.startswith("digit_"):
        return label.replace("digit_", "")

    if label.startswith("lower_"):
        return label.replace("lower_", "")

    if label.startswith("upper_"):
        return label.replace("upper_", "")

    return label


# -----------------------------
# Utility: image preprocessing
# -----------------------------

def to_grayscale_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 127

    for t in range(256):
        w_b += hist[t]

        if w_b == 0:
            continue

        w_f = total - w_b

        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        between = w_b * w_f * (m_b - m_f) ** 2

        if between > max_var:
            max_var = between
            threshold = t

    return threshold


def binarize(gray: np.ndarray, invert: bool = False, threshold: Optional[int] = None) -> np.ndarray:
    if threshold is None:
        threshold = otsu_threshold(gray)

    fg = gray <= threshold

    if invert:
        fg = ~fg

    return fg


def crop_to_foreground(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return mask.copy()

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)

    return mask[y0:y1 + 1, x0:x1 + 1]


def resize_mask_nn(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    H, W = size

    if h == 0 or w == 0:
        return np.zeros((H, W), dtype=np.float32)

    ys = (np.arange(H) * (h / H)).astype(int)
    xs = (np.arange(W) * (w / W)).astype(int)

    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    out = mask[ys[:, None], xs[None, :]]
    return out.astype(np.float32)


# -----------------------------
# Connected components
# -----------------------------

@dataclass
class Component:
    pixels: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int


def connected_components(mask: np.ndarray) -> List[Component]:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: List[Component] = []

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pix = []

            y0 = y1 = y
            x0 = x1 = x

            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))

                y0 = min(y0, cy)
                y1 = max(y1, cy)
                x0 = min(x0, cx)
                x1 = max(x1, cx)

                neighbors = [
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ]

                for ny, nx in neighbors:
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            pixels = np.array(pix, dtype=np.int32)

            comps.append(
                Component(
                    pixels=pixels,
                    bbox=(y0, x0, y1, x1),
                    area=len(pix),
                )
            )

    return comps


def remove_small_specks(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    comps = connected_components(mask)
    keep = np.zeros_like(mask, dtype=bool)

    for comp in comps:
        if comp.area >= min_area:
            keep[comp.pixels[:, 0], comp.pixels[:, 1]] = True

    return keep


# -----------------------------
# Template OCR model
# -----------------------------

class TemplateOCR:
    def __init__(self, template_size: Tuple[int, int] = (32, 32)):
        self.template_size = template_size
        self.templates: Dict[str, List[np.ndarray]] = {}

    def add_template(self, label: str, mask: np.ndarray) -> None:
        label = clean_label(label)

        mask = mask.astype(bool)
        mask = crop_to_foreground(mask, pad=1)
        tmpl = resize_mask_nn(mask, self.template_size)

        self.templates.setdefault(label, []).append(tmpl)

    def match_glyph(self, glyph_mask: np.ndarray) -> str:
        glyph_mask = crop_to_foreground(glyph_mask, pad=1)
        glyph = resize_mask_nn(glyph_mask, self.template_size)

        best_label = "?"
        best_score = float("inf")

        for label, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                score = self.mse(glyph, tmpl)

                if score < best_score:
                    best_score = score
                    best_label = label

        return best_label

    @staticmethod
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.mean(d * d))


# -----------------------------
# Typed dataset loader
# -----------------------------

def load_typed_dataset(
    root: str,
    invert: bool = False,
    min_area: int = 10
) -> Tuple[List[np.ndarray], List[str]]:
    masks = []
    labels = []

    for label in sorted(os.listdir(root)):

        # Skip multi-character sequence folders.
        # They break the one-glyph = one-label assumption.
        if label.startswith("seq_"):
            continue

        label_dir = os.path.join(root, label)

        if not os.path.isdir(label_dir):
            continue

        for fname in sorted(os.listdir(label_dir)):
            path = os.path.join(label_dir, fname)

            if not os.path.isfile(path):
                continue

            try:
                img = Image.open(path)
            except Exception:
                continue

            gray = to_grayscale_np(img)
            mask = binarize(gray, invert=invert)
            mask = remove_small_specks(mask, min_area=min_area)
            mask = crop_to_foreground(mask, pad=1)

            if mask.any():
                masks.append(mask)
                labels.append(clean_label(label))

    return masks, labels


# -----------------------------
# EMNIST handwritten loader
# -----------------------------

def load_emnist_openml(max_samples: int = 20000) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads EMNIST from OpenML.

    If one name fails, the code tries the next possible OpenML name.
    This is here because OpenML names can be picky.
    """

    possible_names = [
        "EMNIST_Balanced",
        "EMNIST-Balanced",
        "EMNIST Balanced",
        "emnist-balanced",
    ]

    last_error = None

    for dataset_name in possible_names:
        try:
            print(f"Trying to load EMNIST using OpenML name: {dataset_name}")
            data = fetch_openml(dataset_name, version="active", as_frame=False)
            X = data.data
            y = data.target.astype(str)
            break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(
            "Could not load EMNIST from OpenML. "
            "Download EMNIST Balanced CSV from Kaggle/NIST instead, "
            "or try changing the dataset name in possible_names."
        ) from last_error

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    masks = []

    for row in X:
        img = row.reshape(28, 28).astype(np.uint8)

        # EMNIST/MNIST-style images usually have bright ink on dark background.
        mask = img > 30

        # EMNIST is sometimes rotated/flipped depending on source.
        # For classification accuracy, this is usually okay as long as all images match.
        mask = crop_to_foreground(mask, pad=1)
        masks.append(mask)

    labels = list(y)

    return masks, labels


def load_emnist_from_csv(
    csv_path: str,
    max_samples: Optional[int] = 20000
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads EMNIST from a CSV file.

    Expected format:
    label,pixel1,pixel2,...,pixel784

    This works with Kaggle-style EMNIST CSV files.
    """

    print(f"Loading EMNIST CSV from: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if max_samples is not None:
        data = data[:max_samples]

    y = data[:, 0].astype(int).astype(str)
    X = data[:, 1:]

    masks = []

    for row in X:
        img = row.reshape(28, 28).astype(np.uint8)

        # Some EMNIST CSV versions are rotated.
        # This fixes visual orientation for many CSV versions.
        img = np.rot90(img, k=-1)
        img = np.fliplr(img)

        mask = img > 30
        mask = crop_to_foreground(mask, pad=1)
        masks.append(mask)

    return masks, list(y)


# -----------------------------
# Vector conversion for PCA + SVM
# -----------------------------

def masks_to_vectors(
    masks: List[np.ndarray],
    size: Tuple[int, int] = (32, 32)
) -> np.ndarray:
    vectors = []

    for mask in masks:
        resized = resize_mask_nn(mask.astype(bool), size)
        vectors.append(resized.flatten())

    return np.array(vectors, dtype=np.float32)


# -----------------------------
# Training / evaluation
# -----------------------------

def split_dataset(
    masks: List[np.ndarray],
    labels: List[str],
    test_size: float = 0.2
):
    return train_test_split(
        masks,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )


def train_ocr_model(
    ocr: TemplateOCR,
    X_train: List[np.ndarray],
    y_train: List[str]
) -> None:
    for mask, label in zip(X_train, y_train):
        ocr.add_template(label, mask)


def evaluate_ocr_model(
    ocr: TemplateOCR,
    X_test: List[np.ndarray],
    y_test: List[str]
) -> float:
    correct = 0

    for mask, label in zip(X_test, y_test):
        predicted = ocr.match_glyph(mask)
        actual = clean_label(label)

        if predicted == actual:
            correct += 1

    return correct / len(y_test)


def run_template_experiment(
    name: str,
    masks: List[np.ndarray],
    labels: List[str]
) -> TemplateOCR:
    print()
    print(f"===== {name} TEMPLATE OCR =====")
    print(f"Total samples: {len(labels)}")
    print(f"Classes: {sorted(set(labels))}")
    print(f"Number of classes: {len(set(labels))}")

    X_train, X_test, y_train, y_test = split_dataset(masks, labels)

    ocr = TemplateOCR(template_size=(32, 32))

    print("Training template OCR model...")
    train_ocr_model(ocr, X_train, y_train)

    print("Evaluating template OCR model...")
    accuracy = evaluate_ocr_model(ocr, X_test, y_test)

    print(f"{name} template OCR accuracy: {accuracy * 100:.2f}%")

    return ocr


def run_pca_svm_experiment(
    name: str,
    masks: List[np.ndarray],
    labels: List[str],
    n_components: int = 30
):
    print()
    print(f"===== {name} PCA + SVM =====")
    print(f"Using PCA components/features: {n_components}")

    labels = [clean_label(label) for label in labels]
    X = masks_to_vectors(masks, size=(32, 32))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    print("Training PCA + SVM model...")
    model.fit(X_train, y_train)

    print("Evaluating PCA + SVM model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} PCA + SVM accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(y_test, y_pred))

    return model


# -----------------------------
# Optional: OCR one typed image
# -----------------------------

def read_test_image(
    image_path: str,
    ocr: TemplateOCR,
    invert: bool = False,
    min_area: int = 10
) -> str:
    img = Image.open(image_path)
    gray = to_grayscale_np(img)

    mask = binarize(gray, invert=invert)
    mask = remove_small_specks(mask, min_area=min_area)

    components = connected_components(mask)

    components = [
        c for c in components
        if c.area >= min_area
    ]

    components = sorted(components, key=lambda c: c.bbox[1])

    result = ""

    for comp in components:
        glyph_mask = np.zeros_like(mask, dtype=bool)
        glyph_mask[comp.pixels[:, 0], comp.pixels[:, 1]] = True

        result += ocr.match_glyph(glyph_mask)

    return result


# -----------------------------
# Main
# -----------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Typed OCR dataset + EMNIST handwritten OCR experiment."
    )

    p.add_argument(
        "--typed_data",
        default=r"C:\Users\w33099\source\repos\CustomOCR\CustomOCR\training_data",
        help="Folder containing typed character image folders."
    )

    p.add_argument(
        "--emnist_csv",
        default=None,
        help="Optional path to EMNIST CSV file. If not given, code tries OpenML."
    )

    p.add_argument(
        "--emnist_samples",
        type=int,
        default=20000,
        help="Number of EMNIST samples to use."
    )

    p.add_argument(
        "--test_image",
        default=None,
        help="Optional image to OCR using the typed template model."
    )

    args = p.parse_args()

    # -----------------------------
    # Typed dataset
    # -----------------------------

    typed_masks, typed_labels = load_typed_dataset(args.typed_data)

    print()
    print("Typed labels loaded:")
    print(sorted(set(typed_labels)))

    typed_template_model = run_template_experiment(
        "Typed",
        typed_masks,
        typed_labels
    )

    typed_svm_model = run_pca_svm_experiment(
        "Typed",
        typed_masks,
        typed_labels,
        n_components=30
    )

    # -----------------------------
    # EMNIST handwritten dataset
    # -----------------------------

    if args.emnist_csv is not None:
        emnist_masks, emnist_labels = load_emnist_from_csv(
            args.emnist_csv,
            max_samples=args.emnist_samples
        )
    else:
        emnist_masks, emnist_labels = load_emnist_openml(
            max_samples=args.emnist_samples
        )

    handwritten_template_model = run_template_experiment(
        "EMNIST Handwritten",
        emnist_masks,
        emnist_labels
    )

    handwritten_svm_model = run_pca_svm_experiment(
        "EMNIST Handwritten",
        emnist_masks,
        emnist_labels,
        n_components=30
    )

    # -----------------------------
    # Optional external image test
    # -----------------------------

    if args.test_image is not None:
        print()
        print("===== TEST IMAGE OCR =====")
        predicted_text = read_test_image(args.test_image, typed_template_model)
        print(f"Predicted text: {predicted_text}")


if __name__ == "__main__":
    main()
