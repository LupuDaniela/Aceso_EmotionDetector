from contextlib import asynccontextmanager
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.database    import init_db
from auth.routes    import router as auth_router
from auth.jwt_utils import verifica_token
from pipeline       import AcesoPipeline

pipeline: Optional[AcesoPipeline] = None

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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["Auth"])

class TextRequest(BaseModel):
    text: str
    salveaza: bool = True

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

@app.post("/api/analyze")
def analyze(req: TextRequest, user=Depends(verifica_token)):
    if not pipeline:
        raise HTTPException(503, "Pipeline se incarca.")
    if not req.text.strip():
        raise HTTPException(400, "Textul nu poate fi gol.")
    return pipeline.analizeaza(
        req.text,
        salveaza=req.salveaza,
        user_id=int(user["sub"])
    )

@app.get("/api/stats")
def stats(user=Depends(verifica_token)):
    return pipeline.get_statistici(user_id=int(user["sub"]))

@app.get("/api/history")
def history(limit: int = 20, user=Depends(verifica_token)):
    return pipeline.get_istoric(limit=limit, user_id=int(user["sub"]))