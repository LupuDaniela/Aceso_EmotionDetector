import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from preprocess import preproceseaza_model

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128 # textele mai lungi de 128 tokeni sunt trunchiate
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTII = ['Tristețe', 'Surpriză', 'Frică', 'Furie',
          'Neutru', 'Încredere', 'Bucurie']

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'best_model_3.pt'


class EmotionRegressor(nn.Module):
    """
    Implementeaza fine-tuning pentru modele Transformer:
    un encoder XLM-RoBERTa-base pre-antrenat extrage reprezentari contextuale din text,
    iar un strat de clasificare liniar proiecteaza reprezentarea
    token-ului [CLS] in spatiul celor 7 emotii din datasetul REDv2.

    Functia Sigmoid aplicata la iesire permite interpretarea fiecarei valori ca
    intensitatea independenta a emotiei corespunzatoare, modelul functionand in regim 
    de regresie multi-output, spre deosebire de clasificarea multi-clasa standard cu Softmax.
    """
    def __init__(self):
        """
        Initializeaza arhitectura retelei neuronale.
        """
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        self.dropout   = nn.Dropout(0.1) #la antrenare, 10% din neuronii stratului sunt dezactivati aleatoriu la fiecare pas, acest lucru previne overfitting-ul
        self.regressor = nn.Linear(768, len(EMOTII)) #strat liniar -> comprima reprezentarea de 768 dimensiuni in 7 scoruri, unul per emoție
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """
        Executa forward pass a datelor prin retea.

        Primeste secventa de tokeni si masca de atentie, le trece prin
        encoder-ul XLM-RoBERTa, extrage reprezentarea tokenului special
        [CLS] care agrega informatia intregii propozitii, aplica Dropout
        si stratul liniar, si returneaza sapte scoruri de intensitate
        emotionala normalizate cu Sigmoid.
        """
        output     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.sigmoid(self.regressor(cls_output))



def incarca_model():
    """
    Încarcă modelul antrenat și tokenizer-ul.

    Incaraca greutatile antrenate din best_model si seteaza modelul 
    in modul de evaluare.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model     = EmotionRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model incarcat din {MODEL_PATH}")
    return model, tokenizer



def scoruri_model(text: str, model, tokenizer) -> dict:
    """
    Calculeaza scorurile de intensitate emotionala pentru un text dat.

    Preprocesează textul, il tokenizeaza cu tokenizer-ul XLM-RoBERTa,
    executa inferenta prin model cu gradientii dezactivati pentru
    eficienta, si returneaza rezultatul ca dictionar {emotie: scor}.
    """
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