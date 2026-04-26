# CLI script to train OCR model from dataset
from src.model import TemplateOCR
from src.train_eval import prepare_and_split_data, train_ocr_model


X_train, X_test, y_train, y_test = prepare_and_split_data("data/training_data")

ocr = TemplateOCR()
train_ocr_model(ocr, X_train, y_train)

print("Training complete")