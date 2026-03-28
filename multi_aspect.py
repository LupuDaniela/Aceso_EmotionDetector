"""
Modul Multi-Aspect Emotion Detection (MAED).

Implementeaza detectia emotionala la nivel de clauza, in conformitate
cu literatura ABSA (Xia & Ding, ACL 2019; Ahmed et al., 2023), care
demonstreaza ca unitatile semantice relevante pentru analiza emotionala
sunt clauzele, nu documentele sau propozitiile intregi.

Extragerea termenilor de aspect foloseste relatiile de dependenta
sintactica identificate de spaCy (nsubj, dobj), metoda validata empiric
in Anwar et al. (2023).

Segmentarea in clauze foloseste subtree-urile nodurilor cu relatii
clauzale din arborele de dependente spaCy (Universal Dependencies):
    - conj      : propozitii coordonate ("Sunt fericit, dar sunt obosit")
    - advcl     : clauze adverbiale    ("Sunt fericit desi am obosit")
    - parataxis : clauze loose         ("Am venit, am vazut, am plecat")

Alegerea ro_core_news_lg in locul ro_core_news_sm: modelul mare ofera
precizie mai buna la parsarea dependentelor sintactice (~91% vs ~85%
UAS pe UD Romanian RRT), reducand cazurile in care conjunctele si
clauzele adverbiale nu sunt detectate corect. Pentru un chatbot care
proceseaza mesaje scurte, costul de memorie (~560MB vs ~15MB) este
acceptabil in schimbul calitatii superioare a segmentarii.

Fallback-ul pe virgula a fost eliminat: cu ro_core_news_lg, parataxis
acopera majoritatea cazurilor unde virgula separa clauze, fara a
introduce erori de segmentare pe virgule non-clauzale (enumerari,
apoziții, intercalari).

Adaptarea originala fata de ABSA clasic: inlocuieste clasificarea binara
pozitiv/negativ cu scorurile continue pe 9 emotii Plutchik produse de
modulul hibrid (hybrid_module.py).

Pipeline:
    text
      → spaCy ro_core_news_lg (doc.sents)
      → subtree-based split pe conj + advcl + parataxis
      → curatare punctuatie/conjunctii de la capetele clauzei
      → per clauza: extrage aspect term via nsubj/dobj/ROOT
      → hybrid_module.analizeaza_text pe textul curat
      → agregat ponderat dupa numarul de tokeni
"""

from preprocess     import nlp, preproceseaza_model
from hybrid_module  import analizeaza_text
from lexical_module import RoEmoLexModule

PRAG_AFISARE = 0.05

# Etichete Universal Dependencies care marcheaza granite de clauza.
# Toate sunt etichete UD standard, valabile pentru orice limba suportata de spaCy — nu sunt reguli specifice limbii romane.

# conj     : propozitie coordonata  ("Sunt trist, dar sunt mandru")
# advcl    : clauza adverbiala      ("Sunt fericit desi am obosit")
# parataxis: clauza loose/juxtapusa ("Am castigat, am pierdut")
# ccomp (complement clausal) este exclus intentionat: "Stiu ca esti trist"

TIPURI_CLAUZE = {'conj', 'parataxis'} # advcl


def _curata_text_clauza(tokeni: list) -> tuple:
    """
    Elimina punctuatia si conjunctiile coordonatoare (dep_='cc')
    de la inceputul si sfarsitul unei clauze extrase din subtree.

    Necesar deoarece subtree-ul unui nod clauzal include si conjunctia
    si virgula care il preceda (ex: ', desi sunt obosit' → 'sunt obosit').
    Textul curat e trimis la model si la parserul de aspect pentru a
    evita zgomotul sintactic la capetele fragmentului.

    dep_='cc' (coordinating conjunction) este eticheta UD pentru
    conjunctii coordonatoare — eliminarea lor e independenta de limba,
    bazata exclusiv pe eticheta sintactica, nu pe forma cuvantului.

    Returneaza:
        text_curat   : str  — textul fara punctuatie/conjunctii la capete
        tokeni_valizi: list — tokenii fara spatii si punctuatie (pentru numarare)
    """
    while tokeni and (tokeni[0].is_punct or tokeni[0].dep_ == 'cc'):
        tokeni = tokeni[1:]

    while tokeni and tokeni[-1].is_punct:
        tokeni = tokeni[:-1]

    tokeni_valizi = [t for t in tokeni if not t.is_space and not t.is_punct]
    text_curat    = ' '.join(t.text for t in tokeni).strip()
    return text_curat, tokeni_valizi


def _extrage_aspect(sent) -> str:
    """
    Extrage termenul de aspect principal dintr-o clauza spaCy.

    Relatiile de dependenta sintactica (nsubj, dobj) identifica
    termenii de aspect mai precis decat simpla selectie a primului substantiv.

    Prioritati:
        1. Token cu dep_='nsubj' si POS NOUN/PROPN, ne-stopword
        2. Token cu dep_='dobj', ne-stopword
        3. Token ROOT care e substantiv sau verb cu continut
        4. Primul substantiv ne-stopword
        5. Fallback: primele 3 cuvinte ne-spatiu
    """
    for token in sent:
        if (token.dep_ == 'nsubj'
                and token.pos_ in ('NOUN', 'PROPN')
                and not token.is_stop):
            return token.lemma_.lower()

    for token in sent:
        if token.dep_ == 'dobj' and not token.is_stop:
            return token.lemma_.lower()

    for token in sent:
        if token.dep_ == 'ROOT' and token.pos_ in ('NOUN', 'PROPN', 'VERB'):
            return token.lemma_.lower()

    for token in sent:
        if (token.pos_ in ('NOUN', 'PROPN')
                and not token.is_stop
                and not token.is_punct):
            return token.lemma_.lower()

    cuvinte = [t.text for t in sent if not t.is_space and not t.is_punct][:3]
    return ' '.join(cuvinte) if cuvinte else '—'


def _segmenteaza_clauze(text: str) -> list:
    """
    Segmenteaza textul in clauze folosind subtree-urile nodurilor clauzale
    din arborele de dependente spaCy (Universal Dependencies).

    Strategia: subtree-based pe TIPURI_CLAUZE = {conj, advcl, parataxis}
        - ROOT + dependentii sai care nu sunt in TIPURI_CLAUZE = prima clauza
        - fiecare nod din TIPURI_CLAUZE + subtree-ul sau = clauza separata

    Fata de versiunea anterioara (doar 'conj'), extinderea la
    parataxis rezolva cazul neacoperite:

        parataxis: "Am castigat, sunt mandru, dar sunt si obosit"
               INAINTE → fallback pe virgula (imprecis)
               ACUM    → detectat structural prin arborele de dependente

    Fallback eliminat: ro_core_news_lg detecteaza corect parataxis in
    majoritatea cazurilor unde versiunea sm esua si necesita fallback
    pe virgula. Eliminarea fallback-ului evita segmentarile gresite pe
    virgule non-clauzale (enumerari, apozitii, intercalari).

    Filtru minim 3 tokeni valizi: segmentele prea scurte nu au context
    suficient pentru detectia emotionala si sunt ignorate.

    Returneaza lista de tuple:
        (text_curat: str, nr_tokeni_valizi: int, sent_obj: spaCy Span)
    """
    doc    = nlp(preproceseaza_model(text))
    clauze = []

    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ == 'ROOT'), None)
        if root is None:
            continue

        # Noduri clauzale care atarna DIRECT de ROOT.
        # Conditia t.head == root exclude clauzele imbricate
        noduri_clauzale = [
            t for t in sent
            if t.dep_ in TIPURI_CLAUZE and t.head == root
        ]

        if not noduri_clauzale:
            tokeni_sent  = list(sent)
            text_s, tv_s = _curata_text_clauza(tokeni_sent)
            if len(tv_s) >= 3:
                sub_doc = nlp(text_s)
                clauze.append((text_s, len(tv_s), sub_doc[:]))
            continue

        # Prima clauza: ROOT + dependentii sai care NU sunt in subtree-urile nodurilor clauzale.
        tokeni_sub = {t.i for nod in noduri_clauzale for t in nod.subtree}
        prima      = [t for t in sent if t.i not in tokeni_sub]
        text_p, tv = _curata_text_clauza(prima)
        if len(tv) >= 3:
            sub_doc = nlp(text_p)
            clauze.append((text_p, len(tv), sub_doc[:]))

        # Clauzele secundare: fiecare nod clauzal + subtree-ul sau curatat.
        # Sortarea dupa t.i pastreaza ordinea originala din fraza.
        for nod in noduri_clauzale:
            subtree_tok  = sorted(nod.subtree, key=lambda t: t.i)
            text_c, tv_c = _curata_text_clauza(subtree_tok)
            if len(tv_c) >= 3:
                sub_doc = nlp(text_c)
                clauze.append((text_c, len(tv_c), sub_doc[:]))

    return clauze


def analizeaza_multi_aspect(text: str, model, tokenizer,
                             modul_lexical: RoEmoLexModule,
                             alpha: float) -> dict:
    """
    Pentru fiecare clauza identificata:
        - extrage termenul de aspect principal (nsubj/dobj/ROOT)
        - ruleaza pipeline-ul hibrid (XLM-RoBERTa + RoEmoLex) pe textul curat
        - calculeaza scorul agregat ponderat dupa lungimea clauzei

    Parametri:
        text          : textul de analizat (fraza sau paragraf)
        model         : modelul XLM-RoBERTa incarcat
        tokenizer     : tokenizer-ul corespunzator
        modul_lexical : instanta RoEmoLexModule
        alpha         : parametrul de combinare hibrid (calibrat pe REDv2)

    Returneaza dict cu:
        'segmente'    : lista de dict per clauza:
                        {text, aspect, scoruri, nr_tokeni}
        'agregat'     : dict {emotie: scor} — medie ponderata pe lungime
        'nr_segmente' : numarul de clauze detectate
    """
    clauze = _segmenteaza_clauze(text)

    if not clauze:
        scoruri = analizeaza_text(text, model, tokenizer, modul_lexical, alpha)
        return {
            'segmente':    [{'text':      text,
                             'aspect':    '—',
                             'scoruri':   scoruri,
                             'nr_tokeni': 1}],
            'agregat':     scoruri,
            'nr_segmente': 1,
        }

    segmente = []
    for text_clauza, nr_tok, sent_obj in clauze:
        scoruri = analizeaza_text(
            text_clauza, model, tokenizer, modul_lexical, alpha
        )
        aspect = _extrage_aspect(sent_obj)
        segmente.append({
            'text':      text_clauza,
            'aspect':    aspect,
            'scoruri':   scoruri,
            'nr_tokeni': nr_tok,
        })

    # Agregare ponderata dupa numarul de tokeni valizi per clauza.
    # O clauza mai lunga contribuie proportional mai mult la scorul final
    total_tokeni = sum(s['nr_tokeni'] for s in segmente)
    emotii       = list(segmente[0]['scoruri'].keys())

    agregat = {
        emotie: sum(
            s['scoruri'][emotie] * s['nr_tokeni']
            for s in segmente
        ) / total_tokeni
        for emotie in emotii
    }

    return {
        'segmente':    segmente,
        'agregat':     agregat,
        'nr_segmente': len(segmente),
    }


def afiseaza_rezultate(rezultat: dict, text_original: str = ''):
    if text_original:
        print(f"\nText: {text_original}")

    print(f"\n{'='*58}")
    print(f"MULTI-ASPECT EMOTION DETECTION  "
          f"({rezultat['nr_segmente']} clauza/clauze)")
    print(f"{'='*58}")

    for i, seg in enumerate(rezultat['segmente'], 1):
        print(f"\n  Clauza {i}: \"{seg['text']}\"")
        print(f"  Aspect: [{seg['aspect']}]")
        print(f"  {'─'*44}")
        scoruri_sortate = sorted(
            seg['scoruri'].items(), key=lambda x: x[1], reverse=True
        )
        for emotie, scor in scoruri_sortate:
            if scor >= PRAG_AFISARE:
                bara = '█' * int(scor * 20)
                print(f"    {emotie:12}: {scor:.3f}  {bara}")

    print(f"\n  {'─'*44}")
    print(f"  SCOR AGREGAT (ponderat dupa lungime clauza):")
    print(f"  {'─'*44}")
    agregat_sortat = sorted(
        rezultat['agregat'].items(), key=lambda x: x[1], reverse=True
    )
    for emotie, scor in agregat_sortat:
        if scor >= PRAG_AFISARE:
            bara = '█' * int(scor * 20)
            print(f"    {emotie:12}: {scor:.3f}  {bara}")
    print(f"{'='*58}\n")


if __name__ == '__main__':
    from model_logic import incarca_model

    model, tokenizer = incarca_model()
    modul_lexical    = RoEmoLexModule()
    ALPHA            = 0.9

    teste = [
        "Am fost suparat ca m-am certat cu un prieten, dar mi s-a imbunat ziua pentru ca ma pregatesc sa plec in vacanta.",
        "Mi-e dor de copilarie cand eram fericit. Acum sunt obosit si nu mai am chef de nimic.",
        "Sunt ingrijorat de examen, dar sunt mandru ca am invatat mult. Sper sa reusesc.",
        "Ma bucur ca vine vara, desi imi pare rau ca se termina scoala si nu o sa mai vad colegii in fiecare zi.",
    ]

    for text in teste:
        rezultat = analizeaza_multi_aspect(
            text, model, tokenizer, modul_lexical, ALPHA
        )
        afiseaza_rezultate(rezultat, text_original=text)