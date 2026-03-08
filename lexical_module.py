import csv
from pathlib import Path

from preprocess import preproceseaza_lexical, elimina_diacritice

MAPARE_ROEMOLEX = {
    'Furie':      'Furie',
    'Anticipare': 'Anticipare',
    'Dezgust':    'Dezgust',
    'Frica':      'Frică',
    'Bucurie':    'Bucurie',
    'Tristete':   'Tristețe',
    'Surpriza':   'Surpriză',
    'Incredere':  'Încredere',
}

EMOTII_ROEMOLEX = list(MAPARE_ROEMOLEX.values())

DIADE_PLUTCHIK = [
    ('Iubire',       'Bucurie',    'Încredere'),
    ('Supunere',     'Încredere',  'Frică'),
    ('Teamă',        'Frică',      'Surpriză'),
    ('Dezamăgire',   'Surpriză',   'Tristețe'),
    ('Optimism',     'Anticipare', 'Bucurie'),
    ('Agresivitate', 'Furie',      'Anticipare'),
    ('Curiozitate',  'Surpriză',   'Încredere'),
    ('Disperare',    'Frică',      'Tristețe'),
    ('Invidie',      'Tristețe',   'Furie'),
]

BASE_DIR  = Path(__file__).parent
POS_PATH  = BASE_DIR / 'lexicon' / 'RoEmoLex_V3_pos_sept2021.csv'
EXPR_PATH = BASE_DIR / 'lexicon' / 'RoEmoLex_V3_expr_sept2021.csv'


class RoEmoLexModule:
    def __init__(self, cale_pos: Path = POS_PATH, cale_expr: Path = EXPR_PATH):
        self.lexicon  = {}
        self.expresii = {}

        self._incarca_fisier(cale_pos,  tip='cuvant')
        self._incarca_fisier(cale_expr, tip='expresie')

        self.ponderi_diade = self._calculeaza_ponderi_diade()

        print(f"RoEmoLex încărcat: {len(self.lexicon)} cuvinte + {len(self.expresii)} expresii")

    def _incarca_fisier(self, cale: Path, tip: str):
        with open(cale, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                cuvant_norm, _ = preproceseaza_lexical(row['word'].strip())
                scoruri = {
                    col_intern: int(row.get(col_lex, 0) or 0)
                    for col_lex, col_intern in MAPARE_ROEMOLEX.items()
                }
                if sum(scoruri.values()) == 0:
                    continue
                if tip == 'expresie' or ' ' in cuvant_norm:
                    self.expresii[cuvant_norm] = scoruri
                else:
                    self.lexicon[cuvant_norm] = scoruri

    def _calculeaza_ponderi_diade(self) -> dict:
        ponderi = {}
        for emotie_complexa, emotie1, emotie2 in DIADE_PLUTCHIK:
            suma_e1 = 0
            suma_e2 = 0
            co_ocurente = 0
            for scoruri in self.lexicon.values():
                s1 = scoruri.get(emotie1, 0)
                s2 = scoruri.get(emotie2, 0)
                if s1 > 0 and s2 > 0:
                    suma_e1     += s1
                    suma_e2     += s2
                    co_ocurente += 1
            if co_ocurente == 0 or (suma_e1 + suma_e2) == 0:
                w1, w2 = 0.5, 0.5
            else:
                total = suma_e1 + suma_e2
                w1    = suma_e1 / total
                w2    = suma_e2 / total
            ponderi[emotie_complexa] = {
                'emotie1': emotie1, 'emotie2': emotie2,
                'w1': round(w1, 4), 'w2': round(w2, 4),
                'co_ocurente': co_ocurente,
            }
        return ponderi

    def afiseaza_ponderi(self):
        print(f"\n{'─'*65}")
        print(f"  {'Emoție complexă':15} {'Comp. 1':12} w1     {'Comp. 2':12} w2     Co-oc.")
        print(f"{'─'*65}")
        for ec, date in self.ponderi_diade.items():
            print(f"  {ec:15} {date['emotie1']:12} {date['w1']:.3f}  "
                  f"{date['emotie2']:12} {date['w2']:.3f}  {date['co_ocurente']}")
        print(f"{'─'*65}\n")

    def analizeaza(self, text: str) -> tuple:
        text_norm, leme = preproceseaza_lexical(text)
        acumulat = {e: 0 for e in EMOTII_ROEMOLEX}
        gasite   = 0

        for expresie, scoruri_expr in self.expresii.items():
            if expresie in text_norm:
                for emotie in EMOTII_ROEMOLEX:
                    acumulat[emotie] += scoruri_expr.get(emotie, 0)
                gasite += 1

        for lema in leme:
            intrare = self.lexicon.get(lema)
            if intrare is None:
                intrare = self.lexicon.get(elimina_diacritice(lema))
            if intrare:
                for emotie in EMOTII_ROEMOLEX:
                    acumulat[emotie] += intrare.get(emotie, 0)
                gasite += 1

        scoruri_finale = (
            {e: acumulat[e] / gasite for e in EMOTII_ROEMOLEX}
            if gasite > 0
            else {e: 0.0 for e in EMOTII_ROEMOLEX}
        )
        return scoruri_finale, gasite


if __name__ == '__main__':
    modul = RoEmoLexModule()
    modul.afiseaza_ponderi()

    teste = [
        "Sunt atât de fericită, nu-mi vine să cred!",
        "Mi-e frică și nu știu ce să fac.",
        "Sunt furioasă pe ce s-a întâmplat azi.",
        "Îmi amintesc cu drag de copilărie, dar nu mai pot reveni.",
    ]
    for text in teste:
        scoruri, gasite = modul.analizeaza(text)
        print(f"\nText: {text}")
        print(f"Cuvinte găsite în lexicon: {gasite}")
        activi = {e: s for e, s in scoruri.items() if s > 0}
        for emotie, scor in sorted(activi.items(), key=lambda x: -x[1]):
            print(f"  {emotie:12}: {scor:.3f}")
        if not activi:
            print("  (nicio emoție detectată lexical)")