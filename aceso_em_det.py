# from transformers import pipeline
# import psycopg2
# from psycopg2.extras import Json
# from datetime import datetime
# import matplotlib.pyplot as plt
# import sys
# from googletrans import Translator

# class EmotionDetector:
#     EMOTION_RO = {
#         'admiration': 'admirație', 'amusement': 'amuzament', 'anger': 'furie',
#         'annoyance': 'enervare', 'approval': 'aprobare', 'caring': 'grijă',
#         'confusion': 'confuzie', 'curiosity': 'curiozitate', 'desire': 'dorință',
#         'disappointment': 'dezamăgire', 'disapproval': 'dezaprobare',
#         'disgust': 'dezgust', 'embarrassment': 'jenă', 'excitement': 'entuziasm',
#         'fear': 'frică', 'gratitude': 'recunoștință', 'grief': 'durere',
#         'joy': 'bucurie', 'love': 'dragoste', 'nervousness': 'anxietate',
#         'optimism': 'optimism', 'pride': 'mândrie', 'realization': 'realizare',
#         'relief': 'ușurare', 'remorse': 'remușcare', 'sadness': 'tristețe',
#         'surprise': 'surpriză', 'neutral': 'neutru'
#     }

#     def __init__(self, db_config):
#         print("Initializing Emotion Detector...")
#         print("Loading RoBERTa GoEmotions...")
#         try:
#             self.classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
#             self.translator = Translator()
#             print("Model loaded! 28 emotions ready.")
#             print("Translation enabled (RO → EN)\n")
#         except Exception as e:
#             print("Error loading model:", e)
#             sys.exit(1)
#         self.db_config = db_config
#         self.init_database()

#     def init_database(self):
#         print("Connecting to PostgreSQL...")
#         try:
#             conn = psycopg2.connect(**self.db_config)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS conversations (
#                     id SERIAL PRIMARY KEY,
#                     message TEXT NOT NULL,
#                     message_en TEXT,
#                     primary_emotion VARCHAR(50) NOT NULL,
#                     primary_emotion_ro VARCHAR(50),
#                     primary_score FLOAT NOT NULL,
#                     all_scores JSONB,
#                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                 )
#             """)
#             cursor.execute("""
#                 CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
#                 ON conversations (timestamp DESC)
#             """)
#             conn.commit()
#             cursor.close()
#             conn.close()
#             print("Database connected!\n")
#         except psycopg2.OperationalError as e:
#             print("PostgreSQL connection error:", e)
#             sys.exit(1)

#     def detect_language(self, text):
#         try:
#             detection = self.translator.detect(text)
#             return detection.lang
#         except:
#             return 'en'

#     def translate_to_english(self, text):
#         try:
#             translation = self.translator.translate(text, src='auto', dest='en')
#             return translation.text
#         except Exception as e:
#             return text

#     def detect_emotions(self, message):
#         original_message = message
#         detected_lang = self.detect_language(message)
        
#         if detected_lang != 'en':
#             print(f" Language: {detected_lang.upper()}")
#             message_en = self.translate_to_english(message)
#             print(f"  🔄 Translation: {message_en}\n")
#         else:
#             message_en = message
        
#         word_count = len(message_en.split())
#         if word_count < 5:
#             print("Tip: Add more details for better accuracy!")
        
#         emotion_keywords = {
#             'anger': ['fight', 'angry', 'mad', 'furious', 'hate', 'pissed', 'annoyed', 'irritated', 'rage', 'argued', 'argument'],
#             'sadness': ['sad', 'depressed', 'crying', 'broke up', 'lost', 'died', 'death', 'lonely', 'heartbroken', 'miserable', 'passed away'],
#             'joy': ['happy', 'excited', 'great', 'amazing', 'wonderful', 'thrilled', 'delighted', 'pleased', 'fantastic'],
#             'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'panic', 'frightened'],
#             'love': ['love', 'adore', 'cherish', 'affection', 'romantic', 'crush', 'boyfriend', 'girlfriend'],
#             'disgust': ['disgusting', 'gross', 'revolting', 'nasty', 'sick', 'vomit']
#         }
        
#         message_lower = message_en.lower()
#         detected_keywords = []
        
#         for emotion, keywords in emotion_keywords.items():
#             if any(word in message_lower for word in keywords):
#                 detected_keywords.append(emotion)
        
#         results = self.classifier(message_en)[0]
#         sorted_result = sorted(results, key=lambda x: x['score'], reverse=True)
#         primary = sorted_result[0]
#         emotion_en_label = primary['label']
#         score = primary['score']
        
#         if detected_keywords and score < 0.50:
#             print(f"Keywords: {', '.join(detected_keywords)}")
            
#             for keyword_emotion in detected_keywords:
#                 matching = [r for r in sorted_result[:10] if keyword_emotion in r['label']]
#                 if matching and matching[0]['score'] > 0.05:
#                     print(f"  → Adjusted: {matching[0]['label']}\n")
#                     primary = matching[0]
#                     emotion_en_label = primary['label']
#                     score = primary['score']
#                     break
        
#         if score < 0.35:
#             print(f"Low confidence ({score:.1%})\n")
        
#         emotion_ro = self.EMOTION_RO.get(emotion_en_label, emotion_en_label)
        
#         top_5 = [
#             {
#                 'emotion_en': r['label'],
#                 'emotion_ro': self.EMOTION_RO.get(r['label'], r['label']),
#                 'score': r['score']
#             }
#             for r in sorted_result[:30]
#         ]
        
#         all_scores = {r['label']: r['score'] for r in results}
        
#         result = {
#             'text': original_message,
#             'text_en': message_en if detected_lang != 'en' else None,
#             'primary_emotion': emotion_en_label,
#             'primary_emotion_ro': emotion_ro,
#             'primary_score': float(score),
#             'top_5': top_5,
#             'all_scores': all_scores,
#             'timestamp': datetime.now()
#         }
        
#         self.save_to_database(result)
#         return result

#     def save_to_database(self, result):
#         try:
#             conn = psycopg2.connect(**self.db_config)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO conversations (message, message_en, primary_emotion, primary_emotion_ro, primary_score, all_scores, timestamp)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#             """, (
#                 result['text'],
#                 result.get('text_en'),
#                 result['primary_emotion'],
#                 result['primary_emotion_ro'],
#                 result['primary_score'],
#                 Json(result['all_scores']),
#                 result['timestamp']
#             ))
#             conn.commit()
#             cursor.close()
#             conn.close()
#         except Exception as e:
#             print("Save error:", e)

#     def print_result(self, result):
#         emotion = result['primary_emotion']
#         emotion_ro = result['primary_emotion_ro']
#         score = result['primary_score']
        
#         print(f"\n{'='*70}")
#         print(f"Emotion: {emotion} ({emotion_ro})")
#         print(f"Confidence: {score:.1%}")
        
#         bar_length = int(score * 50)
#         bar = '█' * bar_length + '░' * (50 - bar_length)
#         print(f"{bar}")
        
#         print(f"\nTop 5 Emotions:")
#         for idx, item in enumerate(result['top_5'], start=1):
#             emo_ro = item['emotion_ro']
#             emo_score = item['score']
#             print(f"  {idx}. {emo_ro:15} → {emo_score:.1%}")
#         print('='*70 + '\n')

#     def get_stats(self):
#         conn = psycopg2.connect(**self.db_config)
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM conversations")
#         total = cursor.fetchone()[0]
#         cursor.execute("""
#             SELECT primary_emotion_ro, COUNT(*) as count
#             FROM conversations
#             GROUP BY primary_emotion_ro
#             ORDER BY count DESC
#         """)
#         distribution = cursor.fetchall()
#         cursor.execute("""
#             SELECT message, primary_emotion_ro, primary_score, timestamp
#             FROM conversations
#             ORDER BY timestamp DESC
#             LIMIT 10
#         """)
#         recent = cursor.fetchall()
#         cursor.close()
#         conn.close()
#         return {'total': total, 'distribution': distribution, 'recent': recent}

#     def print_stats(self):
#         stats = self.get_stats()
#         print(f"\n{'='*70}")
#         print(f"STATISTICS")
#         print(f"{'='*70}")
#         print(f"\nTotal messages: {stats['total']}")
        
#         if stats['distribution']:
#             print(f"\nEmotion Distribution:")
#             max_count = max(d[1] for d in stats['distribution'])
#             for emotion_ro, count in stats['distribution']:
#                 percentage = (count / stats['total']) * 100
#                 bar_length = int((count / max_count) * 40)
#                 bar = '█' * bar_length
#                 print(f"  {emotion_ro:15} [{count:3}] {bar} {percentage:5.1f}%")
        
#         if stats['recent']:
#             print(f"\nLast 10 messages:")
#             for message, emotion_ro, score, timestamp in stats['recent']:
#                 time_str = timestamp.strftime("%H:%M:%S")
#                 msg_short = (message[:40] + '...') if len(message) > 40 else message
#                 print(f"  • {msg_short:43} → {emotion_ro:12} ({score:.0%}) [{time_str}]")
#         print('='*70 + '\n')

#     def plot_emotions(self):
#         stats = self.get_stats()
#         if not stats['distribution']:
#             print("No data to plot yet.\n")
#             return
        
#         emotions = [d[0] for d in stats['distribution'][:10]]
#         counts = [d[1] for d in stats['distribution'][:10]]
#         colors_list = ['#667eea','#764ba2','#f093fb','#4facfe','#43e97b',
#                       '#fa709a','#fee140','#30cfd0','#a8edea','#fed6e3']
        
#         plt.figure(figsize=(14, 7))
#         bars = plt.bar(emotions, counts, color=colors_list[:len(emotions)])
#         plt.title('Emotion Distribution', fontsize=18, fontweight='bold')
#         plt.xlabel('Emotions', fontsize=14)
#         plt.ylabel('Number of Messages', fontsize=14)
#         plt.xticks(rotation=45, ha='right', fontsize=12)
#         plt.grid(axis='y', alpha=0.3)
        
#         for bar in bars:
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
#                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        
#         plt.tight_layout()
#         filename = 'emotion_chart.png'
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Chart saved as {filename}\n")
#         try:
#             plt.show()
#         except:
#             pass


# def main():
#     db_config = {
#         'host': 'localhost',
#         'port': 5432,
#         'database': 'emotion_db',
#         'user': 'postgres',
#         'password': '1q2w3e'
#     }
    
#     try:
#         detector = EmotionDetector(db_config)
#         print("="*70)
#         print("EMOTION DETECTOR READY!")
#         print("="*70)
#         print("\nCommands:")
#         print("  [write message] - Analyze emotion (RO/EN)")
#         print("  stats           - Show statistics")
#         print("  graph           - Plot chart")
#         print("  quit            - Exit\n")
        
#         message_count = 0
        
#         while True:
#             try:
#                 text = input("Your message: ").strip()
                
#                 if text.lower() in ['exit', 'quit', 'q']:
#                     break
#                 elif text.lower() == 'stats':
#                     detector.print_stats()
#                     continue
#                 elif text.lower() == 'graph':
#                     detector.plot_emotions()
#                     continue
#                 elif not text:
#                     print("Please enter a message.\n")
#                     continue
                
#                 result = detector.detect_emotions(text)
#                 detector.print_result(result)
#                 message_count += 1
                
#                 if message_count == 5:
#                     print("💡 Type 'stats' to see statistics!\n")
                    
#             except KeyboardInterrupt:
#                 print("\n\nExiting...\n")
#                 break
#             except Exception as e:
#                 print("Error:", e, "\n")
        
#         if message_count > 0:
#             print("\nFinal Statistics:")
#             detector.print_stats()
        
#         print("Goodbye!\n")
        
#     except Exception as e:
#         print("Initialization error:", e)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

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
from model_logic   import EmotionRegressor, incarca_model, scoruri_model, \
                          EMOTII, MODEL_NAME, MAX_LENGTH, DEVICE
from lexical_module import RoEmoLexModule


#MODEL_NAME = "xlm-roberta-base"
#MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 2e-5
#DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#EMOTII = ['Tristețe', 'Surpriză', 'Frică', 'Furie',
 #         'Neutru', 'Încredere', 'Bucurie']


BASE_DIR   = Path(__file__).parent
TRAIN_PATH = BASE_DIR / 'data_REDv2' / 'train.json'
VALID_PATH = BASE_DIR / 'data_REDv2' / 'valid.json'
TEST_PATH  = BASE_DIR / 'data_REDv2' / 'test.json'
MODEL_PATH = BASE_DIR / 'best_model_3.pt'

DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'database': 'emotion_db',
    'user':     'postgres',
    'password': '1q2w3e'
}

def init_database():
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


#def preproceseaza(text):
 #   text = text.replace("<|PERSON|>", "persoana")
  #  return text.strip()

class REDv2Dataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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


class EmotionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        self.dropout   = nn.Dropout(0.1)
        self.regressor = nn.Linear(768, len(EMOTII))
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.sigmoid(self.regressor(cls_output))


def evalueaza(model, loader, loss_fn):  
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
    # model.eval()
    # encoding = tokenizer(
    #     preproceseaza(text),
    #     max_length=MAX_LENGTH,
    #     truncation=True,
    #     padding='max_length',
    #     return_tensors='pt'
    # )
    # with torch.no_grad():
    #     scoruri = model(
    #         encoding['input_ids'].to(DEVICE),
    #         encoding['attention_mask'].to(DEVICE)
    #     ).cpu().numpy()[0]

    rezultat         = scoruri_model(text, model, tokenizer)  #{EMOTII[i]: float(scoruri[i]) for i in range(len(EMOTII))}
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
        print(f"  ✓ Salvat in baza de date")

    return rezultat 



def mod_interactiv(model, tokenizer):
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
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model, _  = incarca_model()
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