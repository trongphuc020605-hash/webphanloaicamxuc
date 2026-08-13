import re
import json
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except Exception:
    ViTokenizer = None
    HAS_PYVI = False

EXPORT_DIR = "phobert_pca_svm_export"

pipeline = joblib.load(f"{EXPORT_DIR}/pca_svm_pipeline.joblib")
label_encoder = joblib.load(f"{EXPORT_DIR}/label_encoder.joblib")

with open(f"{EXPORT_DIR}/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
MAX_LEN = config["max_len"]
USE_WORD_SEGMENTATION = config.get("use_word_segmentation", False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def basic_clean(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    text = basic_clean(text)
    if USE_WORD_SEGMENTATION and HAS_PYVI:
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            return text
    return text

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

@torch.no_grad()
def encode_texts(texts, batch_size=32):
    texts = [preprocess_text(t) for t in texts]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        emb = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        all_embs.append(emb.detach().cpu().numpy().astype("float32"))
    return np.vstack(all_embs)

def predict(texts):
    X = encode_texts(texts)
    y_pred = pipeline.predict(X)
    return label_encoder.inverse_transform(y_pred)

if __name__ == "__main__":
    samples = [
        "Sản phẩm rất tốt, giao hàng nhanh, mình rất hài lòng",
        "Quá tệ, dùng được vài ngày đã hỏng",
        "Bình thường, không quá nổi bật"
    ]
    preds = predict(samples)
    for text, pred in zip(samples, preds):
        print(f"{pred}: {text}")
