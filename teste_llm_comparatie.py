import os
import sys
import json
import time
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))
load_dotenv()

from model_logic    import incarca_model, EmotionRegressor, MODEL_PATH, DEVICE
from lexical_module import RoEmoLexModule
from hybrid_module  import analizeaza_text
import torch

ALPHA = 0.9

DB_CONFIG = {
    'host':     os.getenv("DB_HOST",     "localhost"),
    'port':     int(os.getenv("DB_PORT", "5432")),
    'database': os.getenv("DB_NAME",     "emotion_db"),
    'user':     os.getenv("DB_USER",     "postgres"),
    'password': os.getenv("DB_PASSWORD", "1q2w3e"),
}

PROPOZITII_TEST = [
    "Nu sunt fericit de cadoul pe care l-am primit, dar apreciez gestul.",
    "A fost o zi buna ca m-am intalnit cu niste prietenii, chiar daca ne-am certat putin.",
    "Nu ma simt pregatit sa ies iar la intalniri.",
    "Niciodată nu am fost mai fericit.",
    "Nu sunt afectata de ce se intampla in jurul meu.",
    "Nu pot să cred că șeful meu și-a atribuit tot meritul pentru proiectul meu și a luat promovarea pe care o meritam eu. Sunt convins că voi face dreptate și îi voi arăta tuturor ce om fără caracter este.",
    "Sunt ingrijorat de examen, dar sunt mandru ca am invatat mult.",
    "M-am inscris la facultatea de medicina pentru ca am avut incredere in mama, si am ascultat-o, dar ma simt dezamagita de aceasta alegere",
]



def fetch_citat(emotie: str) -> str:
    """
    Returneaza un citat aleator pentru emotia data din tabelul citate_emotii.
    Returneaza string gol daca nu exista citate sau conexiunea esueaza.
    """
    normalizare = {
        'Încredere': 'Incredere',
        'Frică':     'Frica',
        'Bucurie':   'Bucurie',
        'Tristețe':  'Tristete',
        'Surpriză':  'Surpriza',
        'Anticipare':'Anticipare',
        'Dezgust':   'Dezgust',
        'Furie':     'Furie',
        'Neutru':    'Neutru',
    }
    emotie_db = normalizare.get(emotie, emotie)

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



def construieste_prompt_fara_citat(text: str, scoruri: dict) -> tuple:
    """
    Varianta A: prompt standard, fara citate.
    LLM-ul raspunde empatic bazat doar pe textul utilizatorului si emotia detectata.
    """
    scoruri_filtrate = {
        e: round(s, 3)
        for e, s in sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
        if s > 0.05
    }
    emotie_dominanta = max(scoruri, key=scoruri.get)

    sistem = (
        "Ești un asistent empatic care ajută oamenii să-și proceseze emoțiile. "
        "Răspunzi în limba română, cald și natural, în 2-3 propoziții. "
        "Nu menționezi că ești AI, nu folosești termeni tehnici despre analiză emoțională, "
        "nu dai sfaturi nesolicitate. Răspunsul trebuie să valideze emoția persoanei "
        "și să o facă să se simtă înțeleasă."
    )

    utilizator = (
        f"Mesajul utilizatorului: \"{text}\"\n\n"
        f"Emoție dominantă detectată: {emotie_dominanta}\n"
        f"Scoruri emoționale: {json.dumps(scoruri_filtrate, ensure_ascii=False)}\n\n"
        f"Răspunde empatic la mesajul utilizatorului."
    )

    return sistem, utilizator


def construieste_prompt_cu_citat(text: str, scoruri: dict) -> tuple:
    """
    Varianta B: prompt cu citat ca sursa de inspiratie.
    LLM-ul poate folosi tonul/ideea citatului, dar nu il reproduce direct.
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



def raspuns_groq(sistem: str, utilizator: str, api_key: str) -> str:
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


def raspunds_mistral(sistem: str, utilizator: str, api_key: str) -> str:
    try:
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        client = MistralClient(api_key=api_key)
        response = client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(role="system", content=sistem),
                ChatMessage(role="user",   content=utilizator),
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[EROARE Mistral] {e}"


def main():
    groq_key    = os.getenv("GROQ_API_KEY",    "")
    mistral_key = os.getenv("MISTRAL_API_KEY", "")

    if not any([groq_key, mistral_key]):
        print("EROARE: Nicio cheie API gasita.")
        print("Seteaza variabilele de mediu:")
        print("  set GROQ_API_KEY=...")
        print("  set MISTRAL_API_KEY=...")
        sys.exit(1)

    print("Incarc modelul Aceso...")
    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()
    print("Model incarcat.\n")

    rezultate = []

    for i, text in enumerate(PROPOZITII_TEST, 1):
        print(f"\n{'═'*65}")
        print(f"[{i}/{len(PROPOZITII_TEST)}] {text}")
        print(f"{'═'*65}")

        scoruri    = analizeaza_text(text, model, tokenizer, modul_lexical, ALPHA)
        emotie_dom = max(scoruri, key=scoruri.get)
        scor_dom   = scoruri[emotie_dom]

        print(f"  Emotie dominanta: {emotie_dom} ({scor_dom:.3f})")
        scoruri_top = {
            e: round(s, 3)
            for e, s in sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
            if s > 0.05
        }
        print(f"  Scoruri: {scoruri_top}\n")

        sistem_a, utilizator_a          = construieste_prompt_fara_citat(text, scoruri)
        sistem_b, utilizator_b, citat_b = construieste_prompt_cu_citat(text, scoruri)

        if citat_b:
            print(f"  Citat selectat pentru varianta B:\n  {citat_b}\n")
        else:
            print(f"  [INFO] Niciun citat disponibil pentru emotia '{emotie_dom}'.\n")

        raspunsuri_a = {}  # fara citat
        raspunsuri_b = {}  # cu citat

        if groq_key:
            print("  [Groq + Llama3] Varianta A (fara citat)...")
            ra = raspuns_groq(sistem_a, utilizator_a, groq_key)
            print(f"  → {ra}\n")
            raspunsuri_a['Groq+Llama3'] = ra
            time.sleep(30)

            print("  [Groq + Llama3] Varianta B (cu citat)...")
            rb = raspuns_groq(sistem_b, utilizator_b, groq_key)
            print(f"  → {rb}\n")
            raspunsuri_b['Groq+Llama3'] = rb
            time.sleep(30)

        if mistral_key:
            print("  [Mistral] Varianta A (fara citat)...")
            ra = raspunds_mistral(sistem_a, utilizator_a, mistral_key)
            print(f"  → {ra}\n")
            raspunsuri_a['Mistral'] = ra
            time.sleep(0.5)

            print("  [Mistral] Varianta B (cu citat)...")
            rb = raspunds_mistral(sistem_b, utilizator_b, mistral_key)
            print(f"  → {rb}\n")
            raspunsuri_b['Mistral'] = rb
            time.sleep(0.5)

        rezultate.append({
            'text':             text,
            'emotie_dominanta': emotie_dom,
            'scor_dominant':    round(scor_dom, 3),
            'scoruri':          scoruri_top,
            'citat_folosit':    citat_b,
            'raspunsuri': {
                'fara_citat': raspunsuri_a,
                'cu_citat':   raspunsuri_b,
            }
        })

    output_path = Path(__file__).parent / 'rezultate_llm_comparatie.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rezultate, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*65}")
    print(f"Rezultate salvate in: {output_path}")
    print(f"{'═'*65}")


if __name__ == '__main__':
    main()