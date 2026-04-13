"""
aceso_em_det.py — Pipeline principal Aceso.

Integreaza:
    1. Modul hibrid   (XLM-RoBERTa + RoEmoLex, alpha=0.9)
    2. Multi-Aspect Emotion Detection (MAED, clause-level via spaCy)
    3. Detectie diade Plutchik (toate 24, conditie dubla, prag=0.3)
"""

import datetime
import psycopg2
import torch
import os
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
from collections import Counter
from dotenv import load_dotenv

from preprocess     import preproceseaza_model
from model_logic    import (EmotionRegressor, incarca_model,
                             EMOTII, MODEL_NAME, MAX_LENGTH, DEVICE, MODEL_PATH)
from lexical_module import RoEmoLexModule
from hybrid_module  import analizeaza_text
from multi_aspect   import analizeaza_multi_aspect, afiseaza_rezultate as afiseaza_maed

# ─────────────────────────────────────────────
# Constante globale pipeline
# ─────────────────────────────────────────────
ALPHA       = 0.9   # calibrat prin ablation study pe REDv2 (MSE=0.0433)
PRAG_DIADE  = 0.25   # prag optim selectat dupa testarea la 0.2 / 0.3 / 0.4 / 0.5

BATCH_SIZE = 16
EPOCHS     = 20
LR         = 2e-5

BASE_DIR   = Path(__file__).parent
TRAIN_PATH = BASE_DIR / 'data_REDv2' / 'train.json'
VALID_PATH = BASE_DIR / 'data_REDv2' / 'valid.json'
TEST_PATH  = BASE_DIR / 'data_REDv2' / 'test.json'


load_dotenv()

DB_CONFIG = {
    'host':     os.getenv('DB_HOST', 'localhost'),
    'port':     int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'emotion_db'),
    'user':     os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

TOATE_DIADELE = [
    ('Iubire',         'Bucurie',    'Incredere',  'primara'),
    ('Supunere',       'Incredere',  'Frica',      'primara'),
    ('Teama',          'Frica',      'Surpriza',   'primara'),
    ('Dezamagire',     'Surpriza',   'Tristete',   'primara'),
    ('Remuscare',      'Tristete',   'Dezgust',    'primara'),
    ('Dispret',        'Dezgust',    'Furie',      'primara'),
    ('Agresivitate',   'Furie',      'Anticipare', 'primara'),
    ('Optimism',       'Anticipare', 'Bucurie',    'primara'),

    ('Vinovatie',      'Bucurie',    'Frica',      'secundara'),
    ('Curiozitate',    'Incredere',  'Surpriza',   'secundara'),
    ('Disperare',      'Frica',      'Tristete',   'secundara'),
    ('Rusine',         'Surpriza',   'Dezgust',    'secundara'),
    ('Invidie',        'Tristete',   'Furie',      'secundara'),
    ('Cinism',         'Dezgust',    'Anticipare', 'secundara'),
    ('Mandrie',        'Furie',      'Bucurie',    'secundara'),
    ('Speranta',       'Anticipare', 'Incredere',  'secundara'),

    ('Incantare',      'Bucurie',    'Surpriza',   'tertiara'),
    ('Sentimentalism', 'Incredere',  'Tristete',   'tertiara'),
    ('Pudoare',        'Frica',      'Dezgust',    'tertiara'),
    ('Indignare',      'Surpriza',   'Furie',      'tertiara'),
    ('Pesimism',       'Tristete',   'Anticipare', 'tertiara'),
    ('Morbiditate',    'Dezgust',    'Bucurie',    'tertiara'),
    ('Dominanta',      'Furie',      'Incredere',  'tertiara'),
    ('Anxietate',      'Anticipare', 'Frica',      'tertiara'),
]

EMOTII_NORM = {
    'Încredere': 'Incredere',
    'Frică':     'Frica',
    'Surpriză':  'Surpriza',
    'Tristețe':  'Tristete',
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
        - diade_detectate (JSONB): lista diadelor active la prag=0.3
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
                diade_detectate  JSONB,
                timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            ALTER TABLE conversations
            ADD COLUMN IF NOT EXISTS diade_detectate JSONB
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


def salveaza_in_db(message, emotie_dominanta, scor_dominant,
                   toate_scorurile, diade_detectate=None):
    """
    Salveaza rezultatul unei analize emotionale in baza de date.
    Campul diade_detectate stocheaza lista numelor diadelor active.
    """
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations
                (message, emotie_dominanta, scor_dominant,
                 toate_scorurile, diade_detectate, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            message,
            emotie_dominanta,
            float(scor_dominant),
            Json(toate_scorurile),
            Json(diade_detectate or []),
            datetime.datetime.now()
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Eroare salvare:", e)


def afiseaza_statistici():
    """
    Interogheaza baza de date si afiseaza:
        1. Distributia emotiilor dominante
        2. Distributia diadelor Plutchik detectate (din campul JSONB)
        3. Ultimele 10 mesaje
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

        # Extrage toate diadele din campul JSONB si numara frecventele.
        cursor.execute("""
            SELECT diada, COUNT(*) as count
            FROM conversations,
                 jsonb_array_elements_text(
                     COALESCE(diade_detectate, '[]'::jsonb)
                 ) AS diada
            GROUP BY diada
            ORDER BY count DESC
        """)
        distributie_diade = cursor.fetchall()

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
            print("\nDistributie emotii dominante:")
            max_count = max(d[1] for d in distributie)
            for emotie, count in distributie:
                percentage = (count / total) * 100
                bar_length = int((count / max_count) * 30)
                bar        = '█' * bar_length
                print(f"  {emotie:12} [{count:3}] {bar} {percentage:5.1f}%")

        if distributie_diade:
            print(f"\nDiade Plutchik detectate (prag={PRAG_DIADE}):")
            max_d = max(d[1] for d in distributie_diade)
            for diada, count in distributie_diade:
                percentage = (count / total) * 100
                bar_length = int((count / max_d) * 30)
                bar        = '░' * bar_length
                print(f"  {diada:18} [{count:3}] {bar} {percentage:5.1f}%")
        else:
            print("\n  (nicio diada detectata inca)")

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
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
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
    reale, si calculeaza cele 2 metrici standard din paper-ul REDv2.
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

    val_loss   /= len(loader)
    toate_pred  = np.vstack(toate_pred)
    toate_label = np.vstack(toate_label)
    hl          = hamming_loss(toate_label, toate_pred)
    return val_loss, hl


def antreneaza():
    """
    Realizeaza fine-tuning-ul XLM-RoBERTa pe datasetul REDv2.

    Bucla de antrenare cu early stopping (patience=3).
    Salveaza automat cel mai bun model in 'best_model_3.pt'.
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


def normalizeaza_scoruri(scoruri: dict) -> dict:
    """
    Normalizeaza cheile din scoruri pentru a elimina diacriticele
    variabile returnate de model (ex: 'Frică' → 'Frica').
    Necesar pentru a putea face lookup in TOATE_DIADELE.
    """
    return {EMOTII_NORM.get(k, k): v for k, v in scoruri.items()}


def detecteaza_diade(scoruri: dict, prag: float = PRAG_DIADE) -> list:
    """
    Detecteaza diadele Plutchik active pe baza scorurilor hibride.

    Conditie dubla: e1 > prag AND e2 > prag → diada prezenta.
    Aceasta conditie este corecta teoretic: o emotie complexa Plutchik
    apare doar cand ambele emotii componente sunt active simultan
    (Plutchik, 1980). Alternativa (media > prag) ar permite detectia
    chiar cand una dintre componente este absenta.

    Parametri:
        scoruri : dict {emotie: scor_hibrid}  — rezultat din analizeaza_text()
        prag    : float — prag minim pentru ambele componente (default: 0.3)

    Returneaza:
        lista de tuple (nume_diada, emotie1, emotie2, tip, scor_mediu)
        sortata descrescator dupa scor_mediu
    """
    scoruri_norm = normalizeaza_scoruri(scoruri)
    diade_active = []

    for ec, e1, e2, tip in TOATE_DIADELE:
        v1 = scoruri_norm.get(e1, 0.0)
        v2 = scoruri_norm.get(e2, 0.0)
        if v1 > prag and v2 > prag:
            scor_mediu = 0.5 * v1 + 0.5 * v2
            diade_active.append((ec, e1, e2, tip, scor_mediu))

    diade_active.sort(key=lambda x: x[4], reverse=True)
    return diade_active


def afiseaza_diade(diade_active: list):
    """
    Afiseaza diadele Plutchik detectate intr-un format tabelar.
    """
    if not diade_active:
        print("  (nicio diada detectata la pragul curent)")
        return

    print(f"  {'Emotie complexa':18} {'Tip':10} {'Componente':24} {'Scor'}")
    print(f"  {'─'*60}")
    for ec, e1, e2, tip, scor in diade_active:
        componente = f"{e1} + {e2}"
        bara       = '█' * int(scor * 20)
        print(f"  {ec:18} {tip:10} {componente:24} {scor:.3f}  {bara}")



def detecteaza_emotii(text, model, tokenizer, modul_lexical, salveaza=True):
    """
    Analizeaza un text prin intregul pipeline Aceso:

        1. MAED (Multi-Aspect Emotion Detection) — segmentare in clauze via spaCy
           si scor hibrid per clauza. Daca textul are o singura clauza, MAED
           produce acelasi scor ca modulul hibrid simplu.
        2. Scoruri hibride agregate (ponderate dupa lungimea clauzei)
        3. Diade Plutchik detectate pe scorurile agregate (prag=0.3)

    Afisare adaptiva:
        - text cu mai multe clauze → afiseaza breakdown per clauza + agregat
        - text cu o singura clauza → afiseaza doar scorurile (fara sectiune MAED redundanta)

    Parametri:
        text          : textul de analizat
        model         : EmotionRegressor incarcat
        tokenizer     : tokenizer XLM-RoBERTa
        modul_lexical : instanta RoEmoLexModule
        salveaza      : daca True, salveaza in PostgreSQL

    Returneaza:
        dict cu cheile 'scoruri', 'diade', 'maed'
    """
    print(f"\nText: {text}")
    print(f"{'═'*56}")

    rezultat_maed = analizeaza_multi_aspect(
        text, model, tokenizer, modul_lexical, ALPHA
    )

    nr_segmente = rezultat_maed['nr_segmente']
    scoruri     = rezultat_maed['agregat']  

    if nr_segmente > 1:
        print(f"  ANALIZA MULTI-ASPECT  ({nr_segmente} clauze)")
        print(f"  {'─'*50}")
        for i, seg in enumerate(rezultat_maed['segmente'], 1):
            scoruri_seg  = sorted(seg['scoruri'].items(),key=lambda x: x[1], reverse=True)
            dom_emotie   = scoruri_seg[0][0] if scoruri_seg else '—'
            dom_scor     = scoruri_seg[0][1] if scoruri_seg else 0.0
            print(f"\n  Clauza {i} [{dom_emotie} {dom_scor:.2f}]: \"{seg['text']}\"")
            print(f"  Aspect  : [{seg['aspect']}]")
            for emotie, scor in scoruri_seg:
                if scor >= 0.05:
                    bara = '█' * int(scor * 25)
                    print(f"    {emotie:12}: {scor:.3f}  {bara}")

        print(f"\n  {'─'*50}")
        print(f"  SCOR AGREGAT (ponderat pe lungimea clauzei):")
    else:
        print(f"  SCOR HIBRID (alpha={ALPHA}):")

    print(f"  {'─'*50}")
    rezultat_sortat  = sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
    emotie_dominanta = rezultat_sortat[0][0]
    scor_dominant    = rezultat_sortat[0][1]
    for emotie, scor in rezultat_sortat:
        bara = '█' * int(scor * 30)
        print(f"    {emotie:12}: {scor:.3f}  {bara}")
    print(f"\n  Emotie dominanta: {emotie_dominanta} ({scor_dominant:.3f})")

    diade_active = detecteaza_diade(scoruri, prag=PRAG_DIADE)
    print(f"\n  {'─'*50}")
    print(f"  DIADE PLUTCHIK (prag={PRAG_DIADE}):")
    afiseaza_diade(diade_active)
    print(f"{'═'*56}")

    if salveaza:
        nume_diade = [d[0] for d in diade_active]
        salveaza_in_db(text, emotie_dominanta, scor_dominant,
                       scoruri, diade_detectate=nume_diade)
        print(f"  ✓ Salvat in baza de date")

    return {
        'scoruri': scoruri,
        'diade':   diade_active,
        'maed':    rezultat_maed,
    }



def mod_interactiv(model, tokenizer, modul_lexical):
    """
    Interfata interactiva in linie de comanda.

    Comenzi:
        stats  - afiseaza statistici din baza de date
        exit   - iesire

    MAED ruleaza automat pe orice mesaj:
        - propozitie simpla → scor hibrid direct
        - text cu mai multe clauze → breakdown per clauza + scor agregat
    """
    print("\n" + "="*56)
    print("ACESO - Sistem Empatic de Detectare Emotii")
    print("  Hybrid: XLM-RoBERTa + RoEmoLex  (alpha=0.9)")
    print("  Diade : 24 diade Plutchik        (prag=0.3)")
    print("  MAED  : Multi-Aspect Detection   (auto, spaCy)")
    print("─"*56)
    print("Comenzi: 'stats' | 'exit'")
    print("="*56)

    while True:
        text = input("\nMesaj: ").strip()

        if text.lower() in ['exit', 'quit', 'q']:
            print("La revedere!")
            break

        elif text.lower() == 'stats':
            afiseaza_statistici()
            continue

        elif not text:
            continue

        detecteaza_emotii(text, model, tokenizer, modul_lexical, salveaza=True)



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
        model, _ = incarca_model()

    model = EmotionRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"\nModel incarcat din {MODEL_PATH}")

    modul_lexical = RoEmoLexModule()
    print("Modul lexical RoEmoLex incarcat\n")

    print("=== TEST RAPID ===")
    exemple = [
        "Sunt atât de fericită, nu-mi vine să cred!",
        "Mi-e frică și nu știu ce să fac.",
        "Sunt furioasă pe ce s-a întâmplat azi.",
        "Sunt îngrijorată de examen, dar sunt mândră că am învățat mult.",
    ]
    for ex in exemple:
        detecteaza_emotii(ex, model, tokenizer, modul_lexical, salveaza=False)

    mod_interactiv(model, tokenizer, modul_lexical)