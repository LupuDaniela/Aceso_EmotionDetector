"""
test_diade.py
─────────────
Analizeaza frecventa de aparitie a tuturor celor 24 de diade Plutchik
pe setul de test REDv2, folosind scorurile finale hibride (alpha=0.9).

Conditie de detectie: AMBELE emotii componente trebuie sa depaseasca
pragul minim — conform teoriei lui Plutchik, o emotie complexa apare
doar cand ambele componente sunt active simultan.
"""

import json
from pathlib import Path
from collections import Counter

from model_logic    import incarca_model
from lexical_module import RoEmoLexModule
from hybrid_module  import analizeaza_text

BASE_DIR  = Path(__file__).parent
TEST_PATH = BASE_DIR / 'data_REDv2' / 'test.json'
ALPHA     = 0.9

TOATE_DIADELE = [
    # PRIMARE
    ('Iubire',         'Bucurie',    'Incredere',  'primara'),
    ('Supunere',       'Incredere',  'Frica',      'primara'),
    ('Teama',          'Frica',      'Surpriza',   'primara'),
    ('Dezamagire',     'Surpriza',   'Tristete',   'primara'),
    ('Remuscare',      'Tristete',   'Dezgust',    'primara'),
    ('Dispret',        'Dezgust',    'Furie',      'primara'),
    ('Agresivitate',   'Furie',      'Anticipare', 'primara'),
    ('Optimism',       'Anticipare', 'Bucurie',    'primara'),
    # SECUNDARE
    ('Vinovatie',      'Bucurie',    'Frica',      'secundara'),
    ('Curiozitate',    'Incredere',  'Surpriza',   'secundara'),
    ('Disperare',      'Frica',      'Tristete',   'secundara'),
    ('Rusine',         'Surpriza',   'Dezgust',    'secundara'),
    ('Invidie',        'Tristete',   'Furie',      'secundara'),
    ('Cinism',         'Dezgust',    'Anticipare', 'secundara'),
    ('Mandrie',        'Furie',      'Bucurie',    'secundara'),
    ('Speranta',       'Anticipare', 'Incredere',  'secundara'),
    # TERTIARE
    ('Incantare',      'Bucurie',    'Surpriza',   'tertiara'),
    ('Sentimentalism', 'Incredere',  'Tristete',   'tertiara'),
    ('Pudoare',        'Frica',      'Dezgust',    'tertiara'),
    ('Indignare',      'Surpriza',   'Furie',      'tertiara'),
    ('Pesimism',       'Tristete',   'Anticipare', 'tertiara'),
    ('Morbiditate',    'Dezgust',    'Bucurie',    'tertiara'),
    ('Dominanta',      'Furie',      'Incredere',  'tertiara'),
    ('Anxietate',      'Anticipare', 'Frica',      'tertiara'),
]

# mapare chei fara diacritice pentru potrivire cu scorurile din hybrid_module
EMOTII_NORM = {
    'Încredere': 'Incredere', 'Frică': 'Frica',
    'Surpriză': 'Surpriza',   'Tristețe': 'Tristete',
}


def normalizeaza_sf(sf: dict) -> dict:
    return {EMOTII_NORM.get(k, k): v for k, v in sf.items()}


def analizeaza_cu_prag(scoruri_test: list, prag: float) -> tuple:
    """
    Conditie dubla: e1 > prag AND e2 > prag → diada prezenta.
    """
    contor       = Counter()
    suma_scoruri = {d[0]: 0.0 for d in TOATE_DIADELE}

    for sf in scoruri_test:
        for ec, e1, e2, _ in TOATE_DIADELE:
            v1 = sf.get(e1, 0.0)
            v2 = sf.get(e2, 0.0)
            if v1 > prag and v2 > prag:
                contor[ec]       += 1
                suma_scoruri[ec] += 0.5 * v1 + 0.5 * v2

    total = len(scoruri_test)
    scoruri_medii = {
        ec: suma_scoruri[ec] / contor[ec] if contor[ec] > 0 else 0.0
        for ec in suma_scoruri
    }
    return contor, scoruri_medii


def afiseaza_tabel(contor, scoruri_medii, total, prag):
    print(f"\n{'='*60}")
    print(f"PRAG: {prag}  |  Total exemple: {total}")
    print(f"{'='*60}")
    tip_curent = None
    for ec, e1, e2, tip in TOATE_DIADELE:
        if tip != tip_curent:
            tip_curent = tip
            print(f"\n  -- {tip.upper()} --")
            print(f"  {'Emotie':16} {'Aparitii':8} {'%':8} {'Scor mediu'}")
            print(f"  {'-'*45}")
        aparitii  = contor.get(ec, 0)
        frecventa = aparitii / total * 100
        scor_med  = scoruri_medii.get(ec, 0.0)
        bara      = '#' * int(frecventa / 2)
        print(f"  {ec:16} {aparitii:6}    {frecventa:5.1f}%   {scor_med:.3f}  {bara}")
    print(f"\n  Total detectii: {sum(contor.values())}")
    print(f"{'='*60}")


if __name__ == '__main__':
    print("Incarc model si lexicon...")
    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()

    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        date_test = json.load(f)
    total = len(date_test)

    print(f"Calculez scoruri finale pentru {total} exemple...")
    scoruri_test = []
    for i, exemplu in enumerate(date_test):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total}...")
        sf = analizeaza_text(
            exemplu['text'], model, tokenizer, modul_lexical, ALPHA
        )
        scoruri_test.append(normalizeaza_sf(sf))

    print("\nAplic conditia dubla pentru praguri diferite...")
    rezultate_toate = {}
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor, scoruri_medii = analizeaza_cu_prag(scoruri_test, prag)
        rezultate_toate[prag] = (contor, scoruri_medii)
        afiseaza_tabel(contor, scoruri_medii, total, prag)

    print(f"\n{'='*60}")
    print(f"SUMAR COMPARATIV")
    print(f"{'='*60}")
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor, _ = rezultate_toate[prag]
        total_det    = sum(contor.values())
        diade_active = sum(1 for v in contor.values() if v > 0)
        print(f"  Prag {prag}: {total_det:5} detectii | {diade_active}/24 diade active")

    print("\nSalveaza rezultate_diade.json...")
    export = {}
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor, scoruri_medii = rezultate_toate[prag]
        export[str(prag)] = {
            'contor':        dict(contor),
            'scoruri_medii': scoruri_medii,
            'total':         total,
        }
    with open(BASE_DIR / 'rezultate_diade.json', 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print("Gata!")