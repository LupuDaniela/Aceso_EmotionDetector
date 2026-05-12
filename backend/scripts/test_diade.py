"""
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
from multi_aspect   import analizeaza_multi_aspect

BASE_DIR  = Path(__file__).parent
TEST_PATH = BASE_DIR / 'data_REDv2' / 'test.json'
MA_TEST_PATH  = BASE_DIR / 'set_date_multi_aspect.json'
ALPHA     = 0.9

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


def afiseaza_tabel(contor, scoruri_medii, total, prag, titlu):
    print(f"\n{'='*65}")
    print(f"{titlu}")
    print(f"PRAG: {prag}  |  Total exemple: {total}")
    print(f"{'='*65}")
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
    diade_active = sum(1 for v in contor.values() if v > 0)
    print(f"  Diade active:   {diade_active}/24")
    print(f"{'='*65}")

def afiseaza_comparatie(rezultate_redv2, rezultate_ma, total_redv2, total_ma):
    """Tabel comparativ REDv2 vs Multi-Aspect pentru toate pragurile."""
    print(f"\n\n{'='*75}")
    print(f"COMPARATIE REDv2 vs MULTI-ASPECT")
    print(f"{'='*75}")
 
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor_rv2, _ = rezultate_redv2[prag]
        contor_ma,  _ = rezultate_ma[prag]
 
        det_rv2    = sum(contor_rv2.values())
        det_ma     = sum(contor_ma.values())
        activ_rv2  = sum(1 for v in contor_rv2.values() if v > 0)
        activ_ma   = sum(1 for v in contor_ma.values() if v > 0)
 
        print(f"\n  Prag {prag}:")
        print(f"  {'':20} {'REDv2':>12} {'Multi-Aspect':>14}")
        print(f"  {'-'*48}")
        print(f"  {'Total exemple':20} {total_redv2:>12} {total_ma:>14}")
        print(f"  {'Total detectii':20} {det_rv2:>12} {det_ma:>14}")
        print(f"  {'Diade active':20} {activ_rv2:>11}/24 {activ_ma:>13}/24")
 
    prag = 0.3
    contor_rv2, _ = rezultate_redv2[prag]
    contor_ma,  _ = rezultate_ma[prag]
 
    print(f"\n\n  Detaliu per diada la prag={prag}:")
    print(f"  {'Emotie':16} {'Tip':10} {'REDv2 (%)':>12} {'MA (%)':>10}  {'Diferenta':>10}")
    print(f"  {'-'*62}")
 
    tip_curent = None
    for ec, e1, e2, tip in TOATE_DIADELE:
        if tip != tip_curent:
            tip_curent = tip
            print(f"  -- {tip.upper()} --")
        proc_rv2 = contor_rv2.get(ec, 0) / total_redv2 * 100
        proc_ma  = contor_ma.get(ec, 0)  / total_ma    * 100
        diff     = proc_ma - proc_rv2
        semn     = '+' if diff >= 0 else ''
        print(f"  {ec:16} {tip:10} {proc_rv2:>10.1f}%  {proc_ma:>8.1f}%  {semn}{diff:>+8.1f}%")
 
    print(f"\n{'='*75}")

if __name__ == '__main__':
    print("Incarc model si lexicon...")
    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()
 
    # ── 1. REDv2 ──────────────────────────────────────────────────
    print("\nIncarc setul de test REDv2...")
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        date_redv2 = json.load(f)
    total_redv2 = len(date_redv2)
 
    print(f"Calculez scoruri hibride pentru {total_redv2} exemple REDv2...")
    scoruri_redv2 = []
    for i, exemplu in enumerate(date_redv2):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total_redv2}...")
        sf = analizeaza_text(
            exemplu['text'], model, tokenizer, modul_lexical, ALPHA
        )
        scoruri_redv2.append(normalizeaza_sf(sf))
 
    # ── 2. Multi-Aspect ───────────────────────────────────────────
    print(f"\nIncarc setul multi-aspect ({MA_TEST_PATH.name})...")
    with open(MA_TEST_PATH, 'r', encoding='utf-8') as f:
        date_ma = json.load(f)
    total_ma = len(date_ma)
 
    print(f"Calculez scoruri MAED pentru {total_ma} exemple multi-aspect...")
    scoruri_ma = []
    for i, exemplu in enumerate(date_ma):
        print(f"  {i+1}/{total_ma}: {exemplu['text'][:50]}...")
        rezultat = analizeaza_multi_aspect(
            exemplu['text'], model, tokenizer, modul_lexical, ALPHA
        )
        scoruri_ma.append(normalizeaza_sf(rezultat['agregat']))
 
    # ── 3. Calcul diade pentru toate pragurile ─────────────────────
    print("\nAplic conditia dubla pentru praguri diferite...")
    rezultate_redv2 = {}
    rezultate_ma    = {}
 
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor_rv2, scoruri_medii_rv2 = analizeaza_cu_prag(scoruri_redv2, prag)
        contor_ma,  scoruri_medii_ma  = analizeaza_cu_prag(scoruri_ma,    prag)
        rezultate_redv2[prag] = (contor_rv2, scoruri_medii_rv2)
        rezultate_ma[prag]    = (contor_ma,  scoruri_medii_ma)
 
        afiseaza_tabel(contor_rv2, scoruri_medii_rv2, total_redv2, prag,
                       "REZULTATE REDv2")
        afiseaza_tabel(contor_ma,  scoruri_medii_ma,  total_ma,    prag,
                       "REZULTATE MULTI-ASPECT")
 
    # ── 4. Tabel comparativ ────────────────────────────────────────
    afiseaza_comparatie(rezultate_redv2, rezultate_ma, total_redv2, total_ma)
 
    # ── 5. Salvare JSON ───────────────────────────────────────────
    print("\nSalveaza rezultate...")
    export = {}
    for prag in [0.2, 0.3, 0.4, 0.5]:
        contor_rv2, scoruri_medii_rv2 = rezultate_redv2[prag]
        contor_ma,  scoruri_medii_ma  = rezultate_ma[prag]
        export[str(prag)] = {
            'redv2': {
                'contor':        dict(contor_rv2),
                'scoruri_medii': scoruri_medii_rv2,
                'total':         total_redv2,
            },
            'multi_aspect': {
                'contor':        dict(contor_ma),
                'scoruri_medii': scoruri_medii_ma,
                'total':         total_ma,
            },
        }
 
    out_path = BASE_DIR / 'rezultate_diade_comparatie.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"Salvat in: {out_path}")