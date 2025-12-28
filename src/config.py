
from pathlib import Path

PROJECT_NAME = "electronic-picture-rag-project"
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "images"
INDEX_DIR = PROJECT_DIR / "index"

MODEL_ID = "openai/clip-vit-base-patch32"

# Unknown karar eşikleri
SCORE_THRESHOLD = 0.28
VOTE_MIN = 3
TOPK = 5
