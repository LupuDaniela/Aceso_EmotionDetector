"""
test_llm_comparatie.py — Comparatie raspunsuri empatice: Groq+Llama3 vs Gemini Flash vs Mistral

Testeaza cele 3 LLM-uri pe acelasi set de propozitii, folosind scorurile
emotionale produse de pipeline-ul Aceso (hybrid_module + model_logic).

Rulare:
    conda activate emotiondetector
    pip install groq google-generativeai mistralai
    python test_llm_comparatie.py

Necesita chei API in fisierul .env sau setate ca variabile de mediu:
    GROQ_API_KEY=...
    GEMINI_API_KEY=...
    MISTRAL_API_KEY=...
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# Incarca modelul Aceso
# ─────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent))
load_dotenv()

from model_logic    import incarca_model, EmotionRegressor, MODEL_PATH, DEVICE
from lexical_module import RoEmoLexModule
from hybrid_module  import analizeaza_text
import torch

ALPHA = 0.9

# ─────────────────────────────────────────────
# Propozitii de test
# ─────────────────────────────────────────────
PROPOZITII_TEST = [
    "Nu sunt fericit de cadoul pe care l-am primit, dar apreciez gestul.",
    "A fost o zi buna ca m-am intalnit cu niste prietenii, chiar daca ne-am certat putin.",
    "Nu ma simt pregatit sa ies iar la intalniri.",
    "Niciodată nu am fost mai fericit.",
    "Nu sunt afectata de ce se intampla in jurul meu.",
    "Nu pot să cred că șeful meu și-a atribuit tot meritul pentru proiectul meu și a luat promovarea pe care o meritam eu.Sunt convins că voi face dreptate și îi voi arăta tuturor ce om fără caracter este.",
    "Sunt ingrijorat de examen, dar sunt mandru ca am invatat mult.",
    "M-am inscris la facultatea de medicina pentru ca am avut incredere in mama, si am ascultat-o, dar ma simt dezamagita de aceasta alegere",
]

# ─────────────────────────────────────────────
# Prompt sistem — identic pentru toate LLM-urile
# ─────────────────────────────────────────────
def construieste_prompt(text: str, scoruri: dict) -> tuple:
    """
    Construieste prompt-ul sistem si cel de utilizator pentru LLM.
    Scorurile sunt formatate descrescator, doar cele > 0.05.
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


# ─────────────────────────────────────────────
# Apeluri API
# ─────────────────────────────────────────────
def raspuns_groq(sistem: str, utilizator: str, api_key: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",  "content": sistem},
                {"role": "user",    "content": utilizator},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[EROARE Groq] {e}"


def raspuns_gemini(sistem: str, utilizator: str, api_key: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=utilizator,
            config={"system_instruction": sistem, "max_output_tokens": 200},
        )
        return response.text.strip()
    except Exception as e:
        return f"[EROARE Gemini] {e}"


def raspuns_mistral(sistem: str, utilizator: str, api_key: str) -> str:
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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Chei API
    groq_key    = os.getenv("GROQ_API_KEY",    "")
    gemini_key  = os.getenv("GEMINI_API_KEY",  "")
    mistral_key = os.getenv("MISTRAL_API_KEY", "")

    if not any([groq_key, gemini_key, mistral_key]):
        print("EROARE: Nicio cheie API gasita.")
        print("Seteaza variabilele de mediu:")
        print("  set GROQ_API_KEY=...")
        print("  set GEMINI_API_KEY=...")
        print("  set MISTRAL_API_KEY=...")
        sys.exit(1)

    # Incarca modelul Aceso
    print("Incarc modelul Aceso...")
    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()
    print("Model incarcat.\n")

    rezultate = []

    for i, text in enumerate(PROPOZITII_TEST, 1):
        print(f"\n{'═'*65}")
        print(f"[{i}/{len(PROPOZITII_TEST)}] {text}")
        print(f"{'═'*65}")

        # Analiza emotionala Aceso
        scoruri = analizeaza_text(text, model, tokenizer, modul_lexical, ALPHA)
        emotie_dom = max(scoruri, key=scoruri.get)
        scor_dom   = scoruri[emotie_dom]

        print(f"  Emotie dominanta: {emotie_dom} ({scor_dom:.3f})")
        scoruri_top = {e: round(s, 3) for e, s in
                       sorted(scoruri.items(), key=lambda x: x[1], reverse=True)
                       if s > 0.05}
        print(f"  Scoruri: {scoruri_top}\n")

        sistem, utilizator = construieste_prompt(text, scoruri)

        raspunsuri = {}

        if groq_key:
            print("  [Groq + Llama3]")
            r = raspuns_groq(sistem, utilizator, groq_key)
            print(f"  → {r}\n")
            raspunsuri['Groq+Llama3'] = r
            time.sleep(30)

        if gemini_key:
            print("  [Gemini Flash]")
            r = raspuns_gemini(sistem, utilizator, gemini_key)
            print(f"  → {r}\n")
            raspunsuri['Gemini Flash'] = r
            time.sleep(0.5)

        if mistral_key:
            print("  [Mistral]")
            r = raspuns_mistral(sistem, utilizator, mistral_key)
            print(f"  → {r}\n")
            raspunsuri['Mistral'] = r
            time.sleep(0.5)

        rezultate.append({
            'text':             text,
            'emotie_dominanta': emotie_dom,
            'scor_dominant':    round(scor_dom, 3),
            'scoruri':          scoruri_top,
            'raspunsuri':       raspunsuri,
        })

    # Salveaza rezultatele in JSON
    output_path = Path(__file__).parent / 'rezultate_llm_comparatie.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rezultate, f, ensure_ascii=False, indent=2)
    print(f"\n{'═'*65}")
    print(f"Rezultate salvate in: {output_path}")
    print(f"{'═'*65}")


if __name__ == '__main__':
    main()