import unicodedata
import spacy 

nlp = spacy.load("ro_core_news_lg")

def preproceseaza_model(text: str) -> str:
    """
    Inlocuieste token-ul de anonimizarea cu substantivul persoana
    si elimina spatiile de la capetele sirului.
    """
    text = text.replace("<|PERSON|>", "persoana")
    return text.strip()

def normalizeaza_diacritice(text: str) -> str:
    """
    Corecteaza diacriticele la forma standard Unicode.
    """
    text = unicodedata.normalize("NFC", text)
    return (text.replace('ş', 'ș').replace('Ş', 'Ș')
                .replace('ţ', 'ț').replace('Ţ', 'Ț'))

def elimina_diacritice(text: str) -> str:
    """
    Elimina toate diacriticile, si e utilizat de rezerva 
    in caz ca un cuvant nu e gasit in RoEmoLex cu diacritice.
    """
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

def lemmatizeaza(text: str) -> list:
    """
    Transforma cuvintele la forma lor de baza/canonica.  
    Exemple:
    -> plural
    -> conjugare
    Lexiconul RoEmoLex contine cuvintele la forma de baza.
    """
    doc = nlp(text)
    return [token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space]

def preproceseaza_lexical(text: str) -> tuple:
    """
    Aplica lantul complet de preprocesare necesar modlului
    lexical RoEmoLex
    """
    text = text.replace("<|PERSON|>", "persoana")
    text = normalizeaza_diacritice(text.lower())
    leme = lemmatizeaza(text)
    return text, leme

def este_marker_negatie(token):
    return (token.dep_ == 'neg' or
            (token.dep_ == 'advmod' and token.lemma_ in ('nu', 'nici')))

def detecteaza_leme_negate(text: str) -> set:
    """
    Returneaza multimea lemelor aflate in scope-ul unui marker de negatie.

    Detectia se bazeaza pe relatiile de dependenta UD 'neg' si 'advmod'
    din spaCy (Nivre et al., 2016). In romana, spaCy ro_core_news_lg
    eticheteaza negatia 'nu' ca advmod (nu dep=neg), deci verificam ambele.

    Un token este considerat negat daca:
      (a) are un copil direct marker de negatie (dep=neg sau advmod cu lemma=nu)
      (b) este subiect sau complement direct (nsubj, dobj, iobj) al unui verb
          care are la randul sau un copil marker de negatie — propagare scope
          limitata la relatii argumental, pentru a evita false pozitive.

    Exceptie — constructii superlative:
      "Niciodată nu am fost mai fericit" contine atat 'nu' cat si 'mai'
      ca advmod al aceluiasi token. In acest caz negatia nu reduce
      intensitatea emotiei, ci face parte din constructia superlativa —
      tokenul este exclus din scope (Wiegand et al., 2010).

    Negarea reduce intensitatea emotiei, nu o inverseaza:
    "nu sunt fericit" != Tristete, ci Bucurie atenuata (Hogenboom et al., 2011).

    Returneaza:
        set de leme (str, lowercase) aflate in scope-ul negatiei,
        inclusiv varianta fara diacritice pentru acoperire maxima.

    """
    text_proc = normalizeaza_diacritice(text.replace("<|PERSON|>", "persoana").lower())
    doc = nlp(text_proc)

    def este_negatie(t):
        return (t.dep_ == 'neg' or
                (t.dep_ == 'advmod' and t.lemma_ in ('nu', 'nici')))

    negate = set()
    for token in doc:
        if token.is_punct or token.is_space or este_negatie(token):
            continue

        copii_advmod = [c.lemma_ for c in token.children if c.dep_ == 'advmod']
        are_negatie_directa = any(este_negatie(c) for c in token.children)
        este_superlativ = are_negatie_directa and 'mai' in copii_advmod

        if este_superlativ:
            continue

        are_negatie_prin_head = (
            token.head != token
            and token.dep_ in ('nsubj', 'dobj', 'iobj', 'nsubj:pass')
            and any(este_negatie(c) for c in token.head.children)
        )

        if are_negatie_directa or are_negatie_prin_head:
            lema = token.lemma_.lower()
            negate.add(lema)
            negate.add(elimina_diacritice(lema))
    return negate