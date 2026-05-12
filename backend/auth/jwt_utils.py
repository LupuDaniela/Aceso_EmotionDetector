import os
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "schimba-asta-in-productie")
ALGORITHM  = "HS256"
EXPIRE_MIN = 60 * 24 * 7

security = HTTPBearer()

def creeaza_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MIN)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decodifica_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def verifica_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    payload = decodifica_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid sau expirat.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload