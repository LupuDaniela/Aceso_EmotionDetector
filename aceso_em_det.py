import datetime
import psycopg2
import torch
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import hamming_loss
from tqdm import tqdm
import json
from psycopg2.extras import Json
import sys

from preprocess    import preproceseaza_model
from model_logic    import (EmotionRegressor, incarca_model, scoruri_model,
                             EMOTII, MODEL_NAME, MAX_LENGTH, DEVICE, MODEL_PATH)
from lexical_module import RoEmoLexModule

BATCH_SIZE = 16
EPOCHS     = 20
LR         = 2e-5

BASE_DIR   = Path(__file__).parent
TRAIN_PATH = BASE_DIR / 'data_REDv2' / 'train.json'
VALID_PATH = BASE_DIR / 'data_REDv2' / 'valid.json'
TEST_PATH  = BASE_DIR / 'data_REDv2' / 'test.json'


DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'database': 'emotion_db',
    'user':     'postgres',
    'password': '1q2w3e'
}

def init_database():
    """
    Initializarea bazei de date.

    Schema tabelei:
        - id (SERIAL PRIMARY KEY)
        - message (TEXT): textul original al utilizatorului
        - emotie_dominanta (VARCHAR)
        - scor_dominant (FLOAT): scorul emotiei dominante in [0, 1]
        - toate_scorurile (JSONB): dictionarul complet {emotie: scor}
        - timestamp (TIMESTAMP): momentul inregistrarii
    """
    print("Conectare la PostgreSQL...")
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id               SERIAL PRIMARY KEY,
                message          TEXT NOT NULL,
                emotie_dominanta VARCHAR(50) NOT NULL,
                scor_dominant    FLOAT NOT NULL,
                toate_scorurile  JSONB,
                timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
            ON conversations (timestamp DESC)
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Baza de date conectata!\n")
    except psycopg2.OperationalError as e:
        print("Eroare PostgreSQL:", e)
        sys.exit(1)

def salveaza_in_db(message, emotie_dominanta, scor_dominant, toate_scorurile):
    """
    Salveaza rezultatul unei analize emotionale in baza de date.
    """
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations 
                (message, emotie_dominanta, scor_dominant, toate_scorurile, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            message,
            emotie_dominanta,
            float(scor_dominant),
            Json(toate_scorurile),
            datetime.datetime.now() 
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Eroare salvare:", e)

def afiseaza_statistici():
    """
    Interogheaza baza de date si afiseaza statistici despre conversatiile salvate.
    """
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT emotie_dominanta, COUNT(*) as count
            FROM conversations
            GROUP BY emotie_dominanta
            ORDER BY count DESC
        """)
        distributie = cursor.fetchall()

        cursor.execute("""
            SELECT message, emotie_dominanta, scor_dominant, timestamp
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recente = cursor.fetchall()
        cursor.close()
        conn.close()

        print(f"\n{'='*60}")
        print(f"STATISTICI - Total mesaje: {total}")
        print(f"{'='*60}")

        if distributie:
            print("\nDistributie emotii:")
            max_count = max(d[1] for d in distributie)
            for emotie, count in distributie:
                percentage = (count / total) * 100
                bar_length = int((count / max_count) * 30)
                bar        = '█' * bar_length
                print(f"  {emotie:12} [{count:3}] {bar} {percentage:5.1f}%")

        if recente:
            print(f"\nUltimele 10 mesaje:")
            for message, emotie, scor, timestamp in recente:
                time_str  = timestamp.strftime("%H:%M:%S")
                msg_short = (message[:40] + '...') if len(message) > 40 else message
                print(f"  {msg_short:43} -> {emotie:12} ({scor:.0%}) [{time_str}]")

        print('='*60 + '\n')

    except Exception as e:
        print("Eroare statistici:", e)



class REDv2Dataset(Dataset):
    """
    Dataset PyTorch pentru incarcarea si tokenizarea datelor REDv2.

    Implementeaza interfata torch.utils.data.Dataset necesara pentru
    utilizarea cu DataLoader in bucla de antrenare. Citeste fisierele
    JSON ale REDv2 si realizeaza tokenizarea la accesarea fiecarui
    element, returnand tensori gata de trimis modelului.
    """

    def __init__(self, path, tokenizer):
        """
        Incarca datele din fisierul JSON REDv2 si stocheaza tokenizatorul.
        """

        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returneaza numarul total de exemple din dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returneaza un exemplu tokenizat din dataset la indexul dat.

        Preproceseaza textul, il tokenizeaza la lungimea maxima cu
        padding si trunchiere, si converteste etichetele procentuale
        in tensor float. Etichetele procentuale ('procentual_labels')
        sunt folosite ca tinta pentru regresia MSE.
        """
        exemplu  = self.data[index]
        text     = preproceseaza_model(exemplu['text'])
        labels   = torch.tensor(exemplu['procentual_labels'], dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         labels
        }





def evalueaza(model, loader, loss_fn): 
    """
    Evalueaza modelul pe un set de date si calculeaza MSE si Hamming Loss.

    Ruleaza modelul in modul eval (fara gradient, dropout dezactivat)
    pe toate batch-urile din loader, acumuleaza predictiile si etichetele
    reale, si calculeaza cele 2 metrici standard din paper-ul REDv2:
    Mean Squared Error pentru setarea de regresie si Hamming Loss pentru
    comparabilitate cu setarea de clasificare.
    """ 
    model.eval()
    val_loss    = 0
    toate_pred  = []
    toate_label = []

    with torch.no_grad():
        for batch in loader: 
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            predictions    = model(input_ids, attention_mask)
            loss           = loss_fn(predictions, labels)
            val_loss      += loss.item()
            toate_pred.append((predictions.cpu().numpy() >= 0.5).astype(int))
            toate_label.append((labels.cpu().numpy() >= 0.5).astype(int))

    val_loss    /= len(loader)
    toate_pred   = np.vstack(toate_pred)
    toate_label  = np.vstack(toate_label)
    hl           = hamming_loss(toate_label, toate_pred)
    return val_loss, hl


def antreneaza():
    """
    Realizeaza fine-tuning-ul XLM-RoBERTa pe datasetul REDv2.

    Implementeaza bucla completa de antrenare cu early stopping:
    - Optimizer: AdamW cu learning rate 2e-5 si weight decay 0.01
    - Functie de pierdere: MSE (regresie pe procentual_labels)
    - Batch size: 16, maxim 20 de epoci
    - Early stopping cu patience=3: oprire daca val_loss nu scade
      timp de 3 epoci consecutive
    - Salvare automata a celui mai bun model in 'best_model_3.pt'

    La fiecare epoca afiseaza Train MSE, Val MSE si Hamming Loss
    pentru monitorizarea antrenarii. Modelul salvat corespunde
    epocii cu cel mai mic Val MSE.
    """
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = REDv2Dataset(TRAIN_PATH, tokenizer)
    val_dataset   = REDv2Dataset(VALID_PATH, tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} exemple")
    print(f"Validare: {len(val_dataset)} exemple\n")

    model     = EmotionRegressor().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn   = nn.MSELoss()

    best_val_loss   = float('inf')
    patience        = 3
    patience_contor = 0

    for epoca in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoca {epoca+1}/{EPOCHS}"):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            optimizer.zero_grad()
            predictions    = model(input_ids, attention_mask)
            loss           = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()

        train_loss /= len(train_loader)
        val_loss, hl = evalueaza(model, val_loader, loss_fn)

        print(f"Epoca {epoca+1:2d} | Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_loss:.4f} | Hamming Loss: {hl:.4f}")

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            patience_contor = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model salvat")
        else:
            patience_contor += 1
            if patience_contor >= patience:
                print(f"\nEarly stopping la epoca {epoca+1}")
                break

    print(f"\nAntrenare finalizata. Best val MSE: {best_val_loss:.4f}")
    return tokenizer

def evalueaza_test(tokenizer):
    """
    Evalueaza modelul salvat pe setul de test REDv2 si compara cu baseline-ul.

    Incarca greutatile din 'best_model_3.pt', ruleaza evaluarea pe test.json
    si afiseaza MSE si Hamming Loss alaturi de valorile baseline raportate(MSE: 10.06, Hamming Loss: 0.102).
    """
    test_dataset = REDv2Dataset(TEST_PATH, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model        = EmotionRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    loss_fn      = nn.MSELoss()
    test_loss, hl = evalueaza(model, test_loader, loss_fn)
    print(f"\n=== REZULTATE TEST ===")
    print(f"MSE:          {test_loss:.4f}  (baseline paper: 10.06)")
    print(f"Hamming Loss: {hl:.4f}   (baseline paper: 0.102)")


def detecteaza_emotii(text, model, tokenizer, salveaza=True):
    """
    Analizeaza un text si returneaza scorurile emotionale detectate.
    """  
    rezultat         = scoruri_model(text, model, tokenizer)  
    rezultat_sortat  = sorted(rezultat.items(), key=lambda x: x[1], reverse=True)
    emotie_dominanta = rezultat_sortat[0][0]
    scor_dominant    = rezultat_sortat[0][1]  

    print(f"\nText: {text}")
    print(f"{'─'*40}")
    for emotie, scor in rezultat_sortat:
        bara = '█' * int(scor * 30)
        print(f"  {emotie:12}: {scor:.3f}  {bara}")
    print(f"\nEmotie dominanta: {emotie_dominanta} ({scor_dominant:.3f})")

    if salveaza:
        salveaza_in_db(text, emotie_dominanta, scor_dominant, rezultat)
        print(f" Salvat in baza de date")

    return rezultat 



def mod_interactiv(model, tokenizer):
    """
    Interfata interactiva in linie de comanda.
    """
    print("\n" + "="*50)
    print("ACESO - Detectare Emotii")
    print("Comenzi: 'stats' - statistici | 'exit' - iesire")
    print("="*50)

    while True:
        text = input("\nMesaj: ").strip()
        if text.lower() in ['exit', 'quit', 'q']:
            break
        elif text.lower() == 'stats':
            afiseaza_statistici()
            continue
        elif not text:
            continue
        detecteaza_emotii(text, model, tokenizer)



if __name__ == '__main__':

    init_database()

    print("=== VERIFICARE DATE ===")
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        date = json.load(f)
    print(f"Train: {len(date)} exemple")
    print(f"Labels:  {date[0]['procentual_labels']}")
    print(f"Emotii:  {EMOTII}")
    print(f"Device:  {DEVICE}\n")

    skip_antrenare = '--skip' in sys.argv

    if skip_antrenare:
        print("Sar peste antrenare, incarc best_model_3.pt...")
        model, tokenizer = incarca_model()
    else:
        print("=== ANTRENARE ===")
        tokenizer = antreneaza()
        print("\n=== EVALUARE TEST ===")
        evalueaza_test(tokenizer)
        model, _  = incarca_model()

    model = EmotionRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"\nModel incarcat din {MODEL_PATH}")

    print("\n=== TEST RAPID ===")
    exemple = [
        "Sunt atât de fericită, nu-mi vine să cred!",
        "Mi-e frică și nu știu ce să fac.",
        "Sunt furioasă pe ce s-a întâmplat azi.",
    ]
    for ex in exemple:
        detecteaza_emotii(ex, model, tokenizer,salveaza=False)

    mod_interactiv(model, tokenizer)