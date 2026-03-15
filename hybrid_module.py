import json
import numpy as np
from pathlib import Path

from model_logic   import scoruri_model, incarca_model, EMOTII
from lexical_module import RoEmoLexModule, EMOTII_ROEMOLEX
from preprocess    import preproceseaza_model

BASE_DIR   = Path(__file__).parent
VALID_PATH = BASE_DIR / 'data_REDv2' / 'valid.json'

# ── Emotii comune intre model si RoEmoLex ────────────────────────────────────
# Modelul are 7 emotii, RoEmoLex are 8 — Anticipare si Dezgust lipsesc din model
EMOTII_COMUNE  = ['Tristețe', 'Surpriză', 'Frică', 'Furie', 'Încredere', 'Bucurie']
DOAR_LEXICAL   = ['Anticipare', 'Dezgust']   # absente din REDv2
DOAR_MODEL     = ['Neutru']                   # absent din RoEmoLex

# mapare nume emotii model → nume emotii RoEmoLex (identice, dar verificam)
MAPARE_COMUNE = {e: e for e in EMOTII_COMUNE}


def combina_scoruri(scor_model: dict, scor_lexical: dict, alpha: float) -> dict:
    """
    Combina scorurile modelului neural si ale modulului lexical
    folosind formula de medie ponderata:

        scor_final(e) = α × scor_model(e) + (1-α) × scor_lexical(e)

    Cazuri speciale:
        - Anticipare, Dezgust: absente din model → doar scor lexical
        - Neutru: absent din RoEmoLex → doar scor model

    Parametri:
        scor_model   : dict {emotie: scor} din XLM-RoBERTa (7 emotii)
        scor_lexical : dict {emotie: scor} din RoEmoLex (8 emotii)
        alpha        : ponderea modelului neural, in [0, 1]

    Returneaza:
        dict {emotie: scor_final} cu toate 9 emotii
    """
    scor_final = {}

    # emotii comune — formula completa
    for emotie in EMOTII_COMUNE:
        sm = scor_model.get(emotie, 0.0)
        sl = scor_lexical.get(emotie, 0.0)
        scor_final[emotie] = alpha * sm + (1 - alpha) * sl

    # emotii doar din lexicon
    for emotie in DOAR_LEXICAL:
        scor_final[emotie] = scor_lexical.get(emotie, 0.0)

    # emotii doar din model
    for emotie in DOAR_MODEL:
        scor_final[emotie] = scor_model.get(emotie, 0.0)

    return scor_final


def calculeaza_mse_validare(model, tokenizer, modul_lexical: RoEmoLexModule,
                             alpha: float) -> float:
    """
    Calculeaza MSE pe setul de validare REDv2 pentru un alpha dat.

    Compara scorurile finale hibride cu etichetele procentuale din REDv2
    doar pentru cele 7 emotii comune (cele prezente in ground truth).

    Parametri:
        model, tokenizer : modelul XLM-RoBERTa incarcat
        modul_lexical    : instanta RoEmoLexModule
        alpha            : valoarea curenta a parametrului de combinare

    Returneaza:
        MSE mediu pe setul de validare
    """
    with open(VALID_PATH, 'r', encoding='utf-8') as f:
        date_validare = json.load(f)

    erori = []
    for exemplu in date_validare:
        text   = exemplu['text']
        labels = exemplu['procentual_labels']   # [Tristețe, Surpriză, Frică, Furie, Neutru, Încredere, Bucurie]

        sm, _ = scoruri_model_dict(text, model, tokenizer)
        sl, _ = modul_lexical.analizeaza(text)
        sf    = combina_scoruri(sm, sl, alpha)

        # MSE doar pe emotiile din REDv2 (cele 7)
        for i, emotie in enumerate(EMOTII):
            pred  = sf.get(emotie, 0.0)
            label = labels[i]
            erori.append((pred - label) ** 2)

    return float(np.mean(erori))


def scoruri_model_dict(text: str, model, tokenizer) -> tuple:
    """
    Wrapper care returneaza scorurile modelului ca dict si lista.
    """
    from model_logic import scoruri_model
    rezultat = scoruri_model(text, model, tokenizer)
    return rezultat, list(rezultat.values())


def ablation_study(model, tokenizer, modul_lexical: RoEmoLexModule,
                   valori_alpha: list = None) -> tuple:
    """
    Testeaza sistematic valori ale lui alpha pe setul de validare
    si returneaza alpha-ul cu MSE minim.

    Parametri:
        model, tokenizer : modelul incarcat
        modul_lexical    : instanta RoEmoLexModule
        valori_alpha     : lista de valori de testat (implicit 0.3 → 0.9)

    Returneaza:
        (alpha_optim, dict {alpha: mse})
    """
    if valori_alpha is None:
        valori_alpha = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    rezultate = {}
    print(f"\n{'='*45}")
    print(f"ABLATION STUDY — calibrare alpha")
    print(f"{'='*45}")
    print(f"  {'Alpha':8} {'MSE':10}")
    print(f"  {'─'*20}")

    for alpha in valori_alpha:
        mse = calculeaza_mse_validare(model, tokenizer, modul_lexical, alpha)
        rezultate[alpha] = mse
        print(f"  α={alpha:.1f}     MSE={mse:.4f}")

    alpha_optim = min(rezultate, key=rezultate.get)
    print(f"\n  Alpha optim: {alpha_optim} (MSE={rezultate[alpha_optim]:.4f})")
    print(f"{'='*45}\n")

    return alpha_optim, rezultate


def analizeaza_text(text: str, model, tokenizer,
                    modul_lexical: RoEmoLexModule,
                    alpha: float) -> dict:
    """
    Pipeline complet Pasul 3 + Pasul 4 + Pasul 5 pentru un text.

    Returneaza scorurile finale combinate pentru toate 9 emotii.
    """
    sm      = scoruri_model(text, model, tokenizer)
    sl, _   = modul_lexical.analizeaza(text)
    sf      = combina_scoruri(sm, sl, alpha)
    return sf


if __name__ == '__main__':
    from model_logic import incarca_model

    print("Incarc model si lexicon...")
    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()

    # ── Ablation study ────────────────────────────────────────────────────────
    alpha_optim, rezultate_alpha = ablation_study(model, tokenizer, modul_lexical)

    # ── Test pe exemple ───────────────────────────────────────────────────────
    teste = [
        "Sunt atât de fericită, nu-mi vine să cred!",
        "Mi-e frică și nu știu ce să fac.",
        "Sunt furioasă pe ce s-a întâmplat azi.",
        "Abia aștept să mă duc la film.",
        "Mi-e scârbă de ce a făcut.",
        "Nu suport minciuna.",
    ]

    print(f"\nTEST RAPID cu alpha optim = {alpha_optim}")
    print(f"{'='*55}")
    for text in teste:
        sf = analizeaza_text(text, model, tokenizer, modul_lexical, alpha_optim)
        sf_sortat = sorted(sf.items(), key=lambda x: x[1], reverse=True)
        print(f"\nText: {text}")
        print(f"{'─'*40}")
        for emotie, scor in sf_sortat:
            if scor > 0.05:
                bara = '█' * int(scor * 20)
                print(f"  {emotie:12}: {scor:.3f}  {bara}")