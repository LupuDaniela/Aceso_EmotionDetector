import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr

from db.database    import get_conn
from auth.jwt_utils import creeaza_token, verifica_token
from auth.password  import hash_parola, verifica_parola, genereaza_reset_token
from auth.email_sender import trimite_email_reset

router = APIRouter()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:5173")


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/register", status_code=201)
def register(req: RegisterRequest):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (req.email,))
    if cursor.fetchone():
        cursor.close(); conn.close()
        raise HTTPException(400, "Email deja înregistrat.")
    cursor.execute(
        "INSERT INTO users (email, name, password_hash) VALUES (%s,%s,%s) RETURNING id",
        (req.email, req.name, hash_parola(req.password))
    )
    user_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"access_token": creeaza_token({"sub": str(user_id), "email": req.email})}


@router.post("/login")
def login(req: LoginRequest):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE email = %s", (req.email,))
    row = cursor.fetchone()
    cursor.close(); conn.close()
    if not row or not row[1] or not verifica_parola(req.password, row[1]):
        raise HTTPException(401, "Email sau parolă incorectă.")
    return {"access_token": creeaza_token({"sub": str(row[0]), "email": req.email})}


@router.get("/google/login")
def google_login():
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid email profile"
        "&access_type=offline"
    )
    return RedirectResponse(url)


@router.get("/google/callback")
async def google_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/login")

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            }
        )
    token_data = token_resp.json()

    if "access_token" not in token_data:
        error = token_data.get("error", "unknown_error")
        if error == "invalid_grant":
            return RedirectResponse(f"{FRONTEND_URL}/login")
        raise HTTPException(400, error)

    async with httpx.AsyncClient() as client:
        user_info = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
    g = user_info.json()
    email, name, google_id = g.get("email"), g.get("name", ""), g.get("id")

    if not email:
        raise HTTPException(400, "Nu s-a putut obține emailul de la Google.")

    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if row:
        user_id = row[0]
        cursor.execute("UPDATE users SET google_id=%s WHERE id=%s", (google_id, user_id))
    else:
        cursor.execute(
            "INSERT INTO users (email, name, google_id) VALUES (%s,%s,%s) RETURNING id",
            (email, name, google_id)
        )
        user_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()

    token = creeaza_token({"sub": str(user_id), "email": email})
    return RedirectResponse(f"{FRONTEND_URL}/auth/callback?token={token}")


@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (req.email,))
    row = cursor.fetchone()
    if row:
        token, expires = genereaza_reset_token()
        cursor.execute(
            "UPDATE users SET reset_token=%s, reset_expires=%s WHERE id=%s",
            (token, expires, row[0])
        )
        conn.commit()
        try:
            trimite_email_reset(req.email, token)
        except Exception:
            pass
    cursor.close(); conn.close()
    return {"message": "Dacă emailul există, vei primi un link de resetare."}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, reset_expires FROM users WHERE reset_token=%s", (req.token,)
    )
    row = cursor.fetchone()
    if not row:
        cursor.close(); conn.close()
        raise HTTPException(400, "Token invalid.")
    expires = row[1]
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        cursor.close(); conn.close()
        raise HTTPException(400, "Token expirat. Solicită un nou link.")
    cursor.execute(
        "UPDATE users SET password_hash=%s, reset_token=NULL, reset_expires=NULL WHERE id=%s",
        (hash_parola(req.new_password), row[0])
    )
    conn.commit(); cursor.close(); conn.close()
    return {"message": "Parola a fost resetată cu succes."}


@router.get("/me")
def get_me(user=Depends(verifica_token)):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, name, created_at FROM users WHERE id=%s", (user["sub"],)
    )
    row = cursor.fetchone()
    cursor.close(); conn.close()
    if not row:
        raise HTTPException(404, "User negăsit.")
    return {"id": row[0], "email": row[1], "name": row[2],
            "created_at": row[3].isoformat() if row[3] else None}