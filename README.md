# Aceso — Chatbot Empatic pentru Analiza Emoțională a Textului

Aceso este un chatbot empatic care detectează și analizează emoțiile din text în limba română, oferind răspunsuri personalizate bazate pe starea emoțională a utilizatorului.

## Arhitectură

- **`aceso-backend/`** — backend deployed pe [Hugging Face Spaces](https://huggingface.co/spaces), construit cu FastAPI
- **`frontend/`** — interfață React/TypeScript, deployed pe [Vercel](https://vercel.com)

## Tehnologii principale

- **Model NLP**: XLM-RoBERTa fine-tuned pe datasetul REDv2 pentru detectarea a 8 emoții primare (modelul Plutchik)
- **Modul lexical**: RoEmoLex v3 cu căutare pe 3 niveluri (formă exactă, fără diacritice, lemă)
- **Scor hibrid**: combinație ponderată între scorul neural și cel lexical (α=0.9)
- **Analiză multi-aspect (MAED)**: segmentare sintactică a propozițiilor cu spaCy
- **Generare răspunsuri**: Groq API cu modelul Llama 3.3 70B
- **Bază de date**: PostgreSQL via Supabase

## Demo

Aplicația este disponibilă la: [aceso-emotion.vercel.app](https://aceso-emotion.vercel.app)

## Lucrare de licență

Acest proiect reprezintă implementarea practică a lucrării de licență:
*„Chatbot empatic bazat pe detecția și analiza emoțională complexă a textului: Aceso"*
Universitatea Transilvania din Brașov, 2026
Absolvent: Lupu Daniela
