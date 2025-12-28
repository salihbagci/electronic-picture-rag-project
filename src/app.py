
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import faiss
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from config import INDEX_DIR, MODEL_ID, SCORE_THRESHOLD, VOTE_MIN, TOPK

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)

index = faiss.read_index(str(INDEX_DIR / "images.faiss"))
items = json.load(open(INDEX_DIR / "items.json", "r", encoding="utf-8"))

model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

@torch.inference_mode()
def embed_query(img: Image.Image) -> np.ndarray:
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].detach().float().cpu().numpy().astype("float32")

def predict_label(image: Image.Image, k=TOPK, score_threshold=SCORE_THRESHOLD, vote_min=VOTE_MIN):
    q = embed_query(image)[None, :]
    scores, idxs = index.search(q, k)

    results = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        results.append({"score": float(s), "label": items[i]["label"], "path": items[i]["path"]})

    if not results:
        return "unknown", results, {"top": None, "vote": None, "vote_count": 0}

    top_score = results[0]["score"]
    labels = [r["label"] for r in results]
    vote_label, vote_count = Counter(labels).most_common(1)[0]

    if top_score < score_threshold:
        return "unknown", results, {"top": top_score, "vote": vote_label, "vote_count": vote_count}

    if vote_count < vote_min:
        return "unknown", results, {"top": top_score, "vote": vote_label, "vote_count": vote_count}

    return vote_label, results, {"top": top_score, "vote": vote_label, "vote_count": vote_count}

def gradio_fn(image):
    if image is None:
        return "No image", ""

    label, results, info = predict_label(image)

    lines = []
    lines.append(f"Decision: {label}")
    lines.append(f"top_score={info['top']:.3f} | vote={info['vote']} | vote_count={info['vote_count']}/{TOPK}")
    lines.append("---- Top-k ----")
    for r in results[:TOPK]:
        lines.append(f"{r['label']} | score={r['score']:.3f}")

    return label, "\n".join(lines)

demo = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Image(type="pil", label="Upload component image"),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Debug")],
    title="Electronic Picture RAG Project",
    description="CLIP embeddings + FAISS nearest neighbor. Returns UNKNOWN when confidence is low."
)

if __name__ == "__main__":
    demo.launch(share=True)
