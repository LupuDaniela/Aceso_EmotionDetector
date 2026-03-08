# ── model_logic.py ────────────────────────────────────────────────
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from preprocess import preproceseaza_model

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTII = ['Tristețe', 'Surpriză', 'Frică', 'Furie',
          'Neutru', 'Încredere', 'Bucurie']

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'best_model_3.pt'


# ──────────────────────────────────────────────────────────────────
#  Arhitectura modelului
# ──────────────────────────────────────────────────────────────────

class EmotionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout   = nn.Dropout(0.1)
        self.regressor = nn.Linear(768, len(EMOTII))
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.sigmoid(self.regressor(cls_output))


# ──────────────────────────────────────────────────────────────────
#  Încărcare model și tokenizer
# ──────────────────────────────────────────────────────────────────

def incarca_model():
    """Încarcă modelul antrenat și tokenizer-ul."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = EmotionRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model incarcat din {MODEL_PATH}")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────
#  Inferență brută (folosit în main.py și la calibrarea alpha)
# ──────────────────────────────────────────────────────────────────

def scoruri_model(text: str, model, tokenizer) -> dict:
    """Returnează scorurile brute ale modelului ca dict {emotie: scor}."""
    model.eval()
    encoding = tokenizer(
        preproceseaza_model(text),
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        scoruri = model(
            encoding['input_ids'].to(DEVICE),
            encoding['attention_mask'].to(DEVICE)
        ).cpu().numpy()[0]
    return {EMOTII[i]: float(scoruri[i]) for i in range(len(EMOTII))}