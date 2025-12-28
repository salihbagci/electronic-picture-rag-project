
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import faiss
from transformers import CLIPProcessor, CLIPModel

from config import DATA_DIR, INDEX_DIR, MODEL_ID

exts = {".jpg", ".jpeg", ".png", ".webp"}

def iter_images(data_dir: Path):
    for label_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for img_path in label_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                yield img_path, label

@torch.inference_mode()
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    def embed_image(img: Image.Image) -> np.ndarray:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().float().cpu().numpy().astype("float32")

    all_imgs = list(iter_images(DATA_DIR))
    print("Total images:", len(all_imgs))
    if len(all_imgs) == 0:
        raise RuntimeError(f"DATA_DIR empty: {DATA_DIR}")

    items = []
    embs = []

    for img_path, label in tqdm(all_imgs, desc="Embedding"):
        try:
            img = Image.open(img_path).convert("RGB")
            vec = embed_image(img)
            items.append({"path": str(img_path), "label": label})
            embs.append(vec)
        except Exception as e:
            print("Skip:", img_path, "err:", e)

    embs = np.stack(embs, axis=0)
    dim = embs.shape[1]
    print("Embeddings:", embs.shape)

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(INDEX_DIR / "images.faiss"))
    with open(INDEX_DIR / "items.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print("Saved:", INDEX_DIR / "images.faiss")
    print("Saved:", INDEX_DIR / "items.json")

if __name__ == "__main__":
    main()
