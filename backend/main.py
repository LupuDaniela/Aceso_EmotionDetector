from contextlib import asynccontextmanager
from typing import Optional
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database    import init_db
from core.pipeline  import AcesoPipeline
from api.auth       import router as auth_router
from api.chat       import router as chat_router, set_pipeline as chat_set_pipeline
from api.stats      import router as stats_router, set_pipeline as stats_set_pipeline

pipeline: Optional[AcesoPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    init_db()
    pipeline = AcesoPipeline()
    pipeline.incarca()
    chat_set_pipeline(pipeline)
    stats_set_pipeline(pipeline)
    yield


app = FastAPI(title="Aceso API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,  prefix="/auth", tags=["Auth"])
app.include_router(chat_router,  prefix="/api",  tags=["Chat"])
app.include_router(stats_router, prefix="/api",  tags=["Stats"])


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}