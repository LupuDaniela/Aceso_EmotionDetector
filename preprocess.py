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