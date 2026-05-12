"""
groq_integrare.py — Raspuns empatic Groq + citate din tabelul citate_emotii
"""

import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host':     os.getenv("DB_HOST",     "localhost"),
    'port':     int(os.getenv("DB_PORT", "5432")),
    'database': os.getenv("DB_NAME",     "emotion_db"),
    'user':     os.getenv("DB_USER",     "postgres"),
    'password': os.getenv("DB_PASSWORD", "1q2w3e"),
}

NORMALIZARE_EMOTIE = {
    'Încredere':  'Incredere',
    'Frică':      'Frica',
    'Bucurie':    'Bucurie',
    'Tristețe':   'Tristete',
    'Surpriză':   'Surpriza',
    'Anticipare': 'Anticipare',
    'Dezgust':    'Dezgust',
    'Furie':      'Furie',
    'Neutru':     'Neutru',
}


def fetch_citat(emotie: str) -> str:
    """
    Returneaza un citat aleator pentru emotia data din tabelul citate_emotii.
    Returneaza string gol daca nu exista citate sau conexiunea esueaza.
    """
    emotie_db = NORMALIZARE_EMOTIE.get(emotie, emotie)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT autor, citat FROM citate_emotii WHERE emotie = %s ORDER BY RANDOM() LIMIT 1",
                (emotie_db,)
            )
            row = cur.fetchone()
        conn.close()
        return f'„{row[1]}" — {row[0]}' if row else ""
    except Exception as e:
        print(f"  [WARN] Nu am putut accesa citatele: {e}")
        return ""


def construieste_prompt(text: str, scoruri: dict) -> tuple[str, str, str]:
    """
    Construieste prompt-ul sistem si utilizator pentru Groq.
    Identic cu construieste_prompt_cu_citat() din teste_llm_comparatie.py.

    Returns:
        (sistem, utilizator, citat)  — citat poate fi string gol
    """
    scoruri_filtrate = {
        e: round(s, 3)
        for e, s in sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
        if s > 0.05
    }
    emotie_dominanta = max(scoruri, key=scoruri.get)
    citat = fetch_citat(emotie_dominanta)

    citat_sectiune = (
        f"Sursă de inspirație (nu o cita direct, inspiră-te din tonul și ideea ei dacă "
        f"se potrivește natural cu mesajul persoanei):\n{citat}\n\n"
    ) if citat else ""

    sistem = (
        "Ești un asistent empatic care ajută oamenii să-și proceseze emoțiile. "
        "Răspunzi în limba română, cald și natural, în 2-3 propoziții. "
        "Nu menționezi că ești AI, nu folosești termeni tehnici despre analiză emoțională, "
        "nu dai sfaturi nesolicitate. Răspunsul trebuie să valideze emoția persoanei "
        "și să o facă să se simtă înțeleasă. "
        "Nu reproduce citate literare în răspuns — integrează doar esența lor dacă adaugă ceva valoros."
    )

    utilizator = (
        f"Mesajul utilizatorului: \"{text}\"\n\n"
        f"Emoție dominantă detectată: {emotie_dominanta}\n"
        f"Scoruri emoționale: {json.dumps(scoruri_filtrate, ensure_ascii=False)}\n\n"
        f"{citat_sectiune}"
        f"Răspunde empatic la mesajul utilizatorului."
    )

    return sistem, utilizator, citat


def genereaza_raspuns_empatic(
    text: str,
    scoruri: dict,
    afiseaza_citat: bool = False,
) -> str:
    """
    Punct de intrare principal — apelat din aceso_em_det.py.

    Args:
        text:           Textul original al utilizatorului.
        scoruri:        Dict {emotie: scor} de la analizeaza_text().
        afiseaza_citat: Daca True, printeaza citatul folosit.

    Returns:
        Raspunsul empatic generat de Groq.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "[EROARE] GROQ_API_KEY nu este setat în .env"

    sistem, utilizator, citat = construieste_prompt(text, scoruri)

    if afiseaza_citat:
        if citat:
            print(f"\n  Citat selectat: {citat}")
        else:
            print(f"\n  [INFO] Niciun citat disponibil în DB pentru această emoție.")

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": sistem},
                {"role": "user",   "content": utilizator},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[EROARE Groq] {e}"