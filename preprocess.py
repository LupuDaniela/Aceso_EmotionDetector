import unicodedata
import spacy 

nlp = spacy.load("ro_core_news_sm")

def preproceseaza_model(text: str) -> str:
    text = text.replace("<|PERSON|>", "persoana")
    return text.strip()

def normalizeaza_diacritice(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return (text.replace('ş', 'ș').replace('Ş', 'Ș')
                .replace('ţ', 'ț').replace('Ţ', 'Ț'))

def elimina_diacritice(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

def lemmatizeaza(text: str) -> list:
    doc = nlp(text)
    return [token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space]

def preproceseaza_lexical(text: str) -> tuple:
    text = text.replace("<|PERSON|>", "persoana")
    text = normalizeaza_diacritice(text.lower())
    leme = lemmatizeaza(text)
    return text, leme