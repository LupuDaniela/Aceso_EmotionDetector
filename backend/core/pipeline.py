import os, torch
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

from core.model_logic    import EmotionRegressor, incarca_model, DEVICE, MODEL_PATH
from core.lexical_module import RoEmoLexModule
from core.multi_aspect   import analizeaza_multi_aspect
from db.database         import get_conn

ALPHA      = 0.9
PRAG_DIADE = 0.25

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
    'Încredere': 'Incredere', 'Frică': 'Frica',
    'Surpriză':  'Surpriza',  'Tristețe': 'Tristete',
}


class AcesoPipeline:

    def __init__(self):
        self.model         = None
        self.tokenizer     = None
        self.modul_lexical = None

    def incarca(self):
        self.model, self.tokenizer = incarca_model()
        self.model = EmotionRegressor().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        self.modul_lexical = RoEmoLexModule()
        print("Pipeline Aceso incarcat.")

    def analizeaza(
        self,
        text: str,
        salveaza: bool = True,
        user_id: int = None,
        thread_id: int = None,
        genereaza_empatic: bool = False,
    ) -> dict:
        rezultat_maed    = analizeaza_multi_aspect(
            text, self.model, self.tokenizer, self.modul_lexical, ALPHA
        )
        scoruri          = rezultat_maed['agregat']
        sortat           = sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
        emotie_dominanta = sortat[0][0]
        scor_dominant    = sortat[0][1]
        diade            = self._detecteaza_diade(scoruri)

        raspuns_empatic = None
        if genereaza_empatic:
            try:
                from services.groq_integrare import genereaza_raspuns_empatic
                raspuns_empatic = genereaza_raspuns_empatic(text, scoruri)
            except Exception as e:
                print(f"[WARN] Eroare generare raspuns empatic: {e}")
                raspuns_empatic = "Îți mulțumesc că ai împărtășit asta cu mine."

        if salveaza:
            self._salveaza(
                text, emotie_dominanta, scor_dominant,
                scoruri, [d[0] for d in diade],
                user_id=user_id,
                thread_id=thread_id,
                raspuns_empatic=raspuns_empatic,
            )

        return {
            'scoruri':          {k: round(v, 4) for k, v in scoruri.items()},
            'emotie_dominanta': emotie_dominanta,
            'scor_dominant':    round(scor_dominant, 4),
            'raspuns_empatic':  raspuns_empatic,
            'diade': [
                {'nume': d[0], 'emotie1': d[1], 'emotie2': d[2],
                 'tip': d[3], 'scor': round(d[4], 4)}
                for d in diade
            ],
            'maed': {
                'nr_segmente': rezultat_maed['nr_segmente'],
                'segmente': [
                    {'text': s['text'], 'aspect': s.get('aspect', ''),
                     'scoruri': {k: round(v, 4) for k, v in s['scoruri'].items()}}
                    for s in rezultat_maed.get('segmente', [])
                ],
            },
        }

    def get_statistici(self, user_id: int = None) -> dict:
        try:
            conn = get_conn(); cursor = conn.cursor()

            if user_id:
                cursor.execute("SELECT COUNT(*) FROM conversations WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM conversations")
            total = cursor.fetchone()[0]

            if user_id:
                cursor.execute("""
                    SELECT emotie_dominanta, COUNT(*) cnt
                    FROM conversations WHERE user_id = %s
                    GROUP BY emotie_dominanta ORDER BY cnt DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT emotie_dominanta, COUNT(*) cnt
                    FROM conversations GROUP BY emotie_dominanta ORDER BY cnt DESC
                """)
            distributie = [{'emotie': r[0], 'count': r[1]} for r in cursor.fetchall()]

            if user_id:
                cursor.execute("""
                    SELECT jsonb_array_elements_text(diade_detectate) diada, COUNT(*) cnt
                    FROM conversations
                    WHERE diade_detectate IS NOT NULL AND user_id = %s
                    GROUP BY diada ORDER BY cnt DESC LIMIT 10
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT jsonb_array_elements_text(diade_detectate) diada, COUNT(*) cnt
                    FROM conversations WHERE diade_detectate IS NOT NULL
                    GROUP BY diada ORDER BY cnt DESC LIMIT 10
                """)
            diade = [{'diada': r[0], 'count': r[1]} for r in cursor.fetchall()]

            cursor.close(); conn.close()
            return {'total': total, 'distributie': distributie, 'diade': diade}
        except Exception as e:
            return {'error': str(e)}

    def get_istoric(self, limit: int = 20, user_id: int = None) -> list:
        try:
            conn = get_conn(); cursor = conn.cursor()
            cursor.execute("""
                SELECT id, message, emotie_dominanta, scor_dominant,
                       toate_scorurile, diade_detectate, raspuns_empatic, timestamp
                FROM conversations
                WHERE user_id = %s
                ORDER BY timestamp DESC LIMIT %s
            """, (user_id, limit))
            rows = cursor.fetchall()
            cursor.close(); conn.close()
            return [
                {'id': r[0], 'message': r[1], 'emotie_dominanta': r[2],
                 'scor_dominant': r[3], 'toate_scorurile': r[4],
                 'diade_detectate': r[5], 'raspuns_empatic': r[6],
                 'timestamp': r[7].isoformat() if r[7] else None}
                for r in rows
            ]
        except Exception as e:
            return [{'error': str(e)}]

    def _detecteaza_diade(self, scoruri: dict) -> list:
        norm = {EMOTII_NORM.get(k, k): v for k, v in scoruri.items()}
        activi = []
        for ec, e1, e2, tip in TOATE_DIADELE:
            v1, v2 = norm.get(e1, 0.0), norm.get(e2, 0.0)
            if v1 > PRAG_DIADE and v2 > PRAG_DIADE:
                activi.append((ec, e1, e2, tip, 0.5*v1 + 0.5*v2))
        return sorted(activi, key=lambda x: x[4], reverse=True)

    def _salveaza(
        self,
        message,
        emotie,
        scor,
        toate_scorurile,
        diade,
        user_id=None,
        thread_id=None,
        raspuns_empatic=None,
    ):
        try:
            conn = get_conn(); cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations
                    (user_id, thread_id, message, emotie_dominanta, scor_dominant,
                     toate_scorurile, diade_detectate, raspuns_empatic)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, thread_id, message, emotie, scor,
                Json(toate_scorurile), Json(diade), raspuns_empatic,
            ))
            if thread_id:
                cursor.execute(
                    "UPDATE chat_threads SET actualizat_la = NOW() WHERE id = %s",
                    (thread_id,)
                )
            conn.commit(); cursor.close(); conn.close()
        except Exception as e:
            print(f"Eroare DB: {e}")