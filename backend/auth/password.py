import secrets
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_parola(parola: str) -> str:
    return pwd_context.hash(parola)

def verifica_parola(parola: str, hash_stocat: str) -> bool:
    return pwd_context.verify(parola, hash_stocat)

def genereaza_reset_token() -> tuple[str, datetime]:
    token   = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    return token, expires