from __future__ import annotations

 

import os

import subprocess

import zipfile

from dataclasses import dataclass

from typing import Dict, List, Tuple, Optional

 

import numpy as np

import pandas as pd

from PIL import Image

 

from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

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

        self.pca = None

        self.pca_templates: Dict[str, List[np.ndarray]] = {}

 

    def add_template(self, label: str, mask: np.ndarray) -> None:

        label = clean_label(label)

 

        mask = mask.astype(bool)

        mask = crop_to_foreground(mask, pad=1)

        tmpl = resize_mask_nn(mask, self.template_size)

 

        self.templates.setdefault(label, []).append(tmpl)

 

    def fit_pca(self, n_components: int = 30) -> None:

        all_vectors = []

        all_labels = []

 

        for label, tmpl_list in self.templates.items():

            for tmpl in tmpl_list:

                all_vectors.append(tmpl.flatten())

                all_labels.append(label)

 

        all_vectors = np.array(all_vectors, dtype=np.float32)

 

        self.pca = PCA(n_components=n_components, random_state=42)

        transformed = self.pca.fit_transform(all_vectors)

 

        self.pca_templates = {}

 

        for label, vec in zip(all_labels, transformed):

            self.pca_templates.setdefault(label, []).append(vec)

 

    def match_glyph(self, glyph_mask: np.ndarray) -> str:

        glyph_mask = crop_to_foreground(glyph_mask, pad=1)

        glyph = resize_mask_nn(glyph_mask, self.template_size)

 

        scores = []

 

        if self.pca is not None:

            glyph_vec = glyph.flatten().reshape(1, -1)

            glyph_vec = self.pca.transform(glyph_vec)[0]

 

            for label, vec_list in self.pca_templates.items():

                for vec in vec_list:

                    score = self.mse(glyph_vec, vec)

                    scores.append((score, label))

        else:

            for label, tmpl_list in self.templates.items():

                for tmpl in tmpl_list:

                    score = self.mse(glyph, tmpl)

                    scores.append((score, label))

 

        scores.sort(key=lambda x: x[0])

 

        k = 6

        top_scores = scores[:k]

 

        votes = {}

 

        for score, label in top_scores:

            if label not in votes:

                votes[label] = 0.0

 

            votes[label] += 1.0 / (score + 1e-6)

 

        best_label = max(votes, key=votes.get)

 

        return best_label

 

    @staticmethod

    def mse(a: np.ndarray, b: np.ndarray) -> float:

        d = a - b

        return float(np.mean(d * d))

 

 

# -----------------------------

# Typed dataset loader

# -----------------------------

 

def is_number_or_letter(label: str) -> bool:

    label = str(label)

 

    if len(label) != 1:

        return False

 

    return label.isdigit() or label.isalpha()

 

 

def find_tmnist_alphabet_csv() -> Optional[str]:

    for fname in os.listdir("."):

        if not fname.lower().endswith(".csv"):

            continue

 

        try:

            df_check = pd.read_csv(fname, usecols=["labels"])

        except Exception:

            continue

 

        labels = df_check["labels"].astype(str)

        unique_labels = set(labels)

 

        has_letters = any(label.isalpha() for label in unique_labels)

        has_digits = any(label.isdigit() for label in unique_labels)

 

        if has_letters and has_digits and len(unique_labels) > 10:

            return fname

 

    return None

 

 

def download_tmnist_alphabet_if_missing() -> str:

    csv_path = find_tmnist_alphabet_csv()

 

    if csv_path is not None:

        return csv_path

 

    print("TMNIST Alphabet CSV not found.")

    print("Trying to download TMNIST Alphabet from Kaggle...")

 

    dataset_name = "nikbearbrown/tmnist-alphabet-94-characters"

    zip_path = "tmnist-alphabet-94-characters.zip"

 

    subprocess.run(

        [

            "kaggle",

            "datasets",

            "download",

            "-d",

            dataset_name,

            "-p",

            "."

        ],

        check=True

    )

 

    if os.path.exists(zip_path):

        with zipfile.ZipFile(zip_path, "r") as zip_ref:

            zip_ref.extractall(".")

 

    csv_path = find_tmnist_alphabet_csv()

 

    if csv_path is None:

        raise FileNotFoundError(

            "Could not find TMNIST Alphabet CSV after download. "

            "Make sure Kaggle downloaded nikbearbrown/tmnist-alphabet-94-characters."

        )

 

    return csv_path

 

 

def load_typed_dataset(

    root: str,

    invert: bool = False,

    min_area: int = 10

) -> Tuple[List[np.ndarray], List[str]]:

    csv_path = root

 

    if not os.path.exists(csv_path):

        csv_path = download_tmnist_alphabet_if_missing()

 

    print(f"Loading typed TMNIST CSV from: {csv_path}")

 

    df = pd.read_csv(csv_path)

 

    # keep TMNIST from being enormous

    df = df[:20000]

 

    if "labels" not in df.columns:

        raise ValueError("TMNIST CSV must have a 'labels' column.")

 

    labels = df["labels"].astype(str).tolist()

 

    drop_cols = []

 

    for col in ["names", "labels"]:

        if col in df.columns:

            drop_cols.append(col)

 

    X = df.drop(columns=drop_cols).values

 

    masks = []

    fixed_labels = []

 

    for row, label in zip(X, labels):

        label = clean_label(label)

 

        if not is_number_or_letter(label):

            continue

 

        img = row.reshape(28, 28).astype(np.uint8)

 

        mask = img > 30

        mask = crop_to_foreground(mask, pad=1)

 

        if mask.any():

            masks.append(mask)

            fixed_labels.append(label)

 

    print()

    print("Typed TMNIST number/letter labels loaded:")

    print(sorted(set(fixed_labels)))

    print(f"Number of typed TMNIST number/letter classes: {len(set(fixed_labels))}")

 

    return masks, fixed_labels

 

 

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

 

    print("Fitting PCA for template OCR model...")

    ocr.fit_pca(n_components=30)

 

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

    print(f"===== {name} PCA + KNN =====")

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

        ("knn", KNeighborsClassifier(n_neighbors=3))

    ])

 

    print("Training PCA + KNN model...")

    model.fit(X_train, y_train)

 

    print("Evaluating PCA + KNN model...")

    y_pred = model.predict(X_test)

 

    accuracy = accuracy_score(y_test, y_pred)

 

    print(f"{name} PCA + KNN accuracy: {accuracy * 100:.2f}%")

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

        default="94_character_TMNIST.csv",

        help="Path to TMNIST Alphabet CSV file."

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
