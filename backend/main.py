import os
import importlib.util
import pathlib
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.database    import init_db, get_conn
from auth.routes    import router as auth_router
from auth.jwt_utils import verifica_token
from pipeline       import AcesoPipeline

pipeline: Optional[AcesoPipeline] = None

# Mapare emotii romanesti (pipeline) -> mood_key (frontend)
EMOTIE_TO_MOOD: dict[str, str] = {
    'Bucurie':    'joy',
    'Tristete':   'sadness',
    'Tristețe':   'sadness',
    'Frica':      'fear',
    'Frică':      'fear',
    'Furie':      'anger',
    'Surpriza':   'surprise',
    'Surpriză':   'surprise',
    'Incredere':  'trust',
    'Încredere':  'trust',
    'Anticipare': 'anticipation',
    'Dezgust':    'disgust',
    'Neutru':     'neutral',
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    init_db()
    pipeline = AcesoPipeline()
    pipeline.incarca()
    yield


app = FastAPI(title="Aceso API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["Auth"])


class TextRequest(BaseModel):
    text:     str
    salveaza: bool = True


class MoodRequest(BaseModel):
    date_key: str
    mood_key: str


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/api/analyze")
def analyze(req: TextRequest, user=Depends(verifica_token)):
    if not pipeline:
        raise HTTPException(503, "Pipeline se incarca.")
    if not req.text.strip():
        raise HTTPException(400, "Textul nu poate fi gol.")

    user_id  = int(user["sub"])
    rezultat = pipeline.analizeaza(req.text, salveaza=req.salveaza, user_id=user_id)

    # Salveaza automat emotia dominanta in calendar pentru ziua de azi
    mood_key = EMOTIE_TO_MOOD.get(rezultat.get("emotie_dominanta", ""), "neutral")
    date_key = date.today().isoformat()   # "2026-05-26"
    try:
        conn   = get_conn(); cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO mood_log (user_id, date_key, mood_key)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, date_key) DO NOTHING
        """, (user_id, date_key, mood_key))
        conn.commit(); cursor.close(); conn.close()
    except Exception as e:
        print(f"[mood_log auto-save] {e}")

    raspuns = ""
    try:
        spec = importlib.util.spec_from_file_location(
            "groq_integrare",
            pathlib.Path(__file__).parent / "services" / "groq_integrare.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raspuns = mod.genereaza_raspuns_empatic(req.text, rezultat["scoruri"])
    except Exception as e:
        print(f"[GROQ ERROR] {e}")

    return {**rezultat, "raspuns_empatic": raspuns}


@app.get("/api/stats")
def stats(user=Depends(verifica_token)):
    return pipeline.get_statistici(user_id=int(user["sub"]))


@app.get("/api/history")
def history(limit: int = 20, user=Depends(verifica_token)):
    return pipeline.get_istoric(limit=limit, user_id=int(user["sub"]))


@app.get("/api/moods")
def get_moods(user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "SELECT date_key, mood_key FROM mood_log WHERE user_id = %s",
        (int(user["sub"]),)
    )
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return {row[0]: row[1] for row in rows}


@app.post("/api/moods")
def log_mood(req: MoodRequest, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO mood_log (user_id, date_key, mood_key)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, date_key)
        DO UPDATE SET mood_key = EXCLUDED.mood_key
    """, (int(user["sub"]), req.date_key, req.mood_key))
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}


@app.delete("/api/moods/{date_key}")
def delete_mood(date_key: str, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM mood_log WHERE user_id = %s AND date_key = %s",
        (int(user["sub"]), date_key)
    )
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}