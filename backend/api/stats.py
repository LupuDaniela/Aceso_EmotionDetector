from collections import defaultdict
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from db.database    import get_conn
from auth.jwt_utils import verifica_token

router = APIRouter()

_pipeline = None

def set_pipeline(p):
    global _pipeline
    _pipeline = p


class MoodRequest(BaseModel):
    date_key: str
    mood_key: str


@router.get("/stats")
def stats(user=Depends(verifica_token)):
    return _pipeline.get_statistici(user_id=int(user["sub"]))


@router.get("/stats/timeline")
def stats_timeline(days: int = 30, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp) AS day, emotie_dominanta, COUNT(*) AS cnt
        FROM conversations
        WHERE user_id = %s AND timestamp >= NOW() - INTERVAL '1 day' * %s
        GROUP BY DATE(timestamp), emotie_dominanta
        ORDER BY day ASC
    """, (int(user["sub"]), days))
    rows = cursor.fetchall()
    cursor.close(); conn.close()

    daily: dict = defaultdict(list)
    for day, emotie, cnt in rows:
        daily[str(day)].append((emotie, int(cnt)))

    result = []
    for day in sorted(daily.keys()):
        emotions = daily[day]
        dominant = max(emotions, key=lambda x: x[1])
        total    = sum(e[1] for e in emotions)
        result.append({
            "date":             day,
            "total":            total,
            "emotie_dominanta": dominant[0],
        })
    return result


@router.get("/history")
def history(limit: int = 20, user=Depends(verifica_token)):
    return _pipeline.get_istoric(limit=limit, user_id=int(user["sub"]))


@router.get("/moods")
def get_moods(user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "SELECT date_key, mood_key FROM mood_log WHERE user_id = %s",
        (int(user["sub"]),)
    )
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return {r[0]: r[1] for r in rows}


@router.post("/moods", status_code=201)
def log_mood(req: MoodRequest, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO mood_log (user_id, date_key, mood_key)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, date_key) DO UPDATE SET mood_key = EXCLUDED.mood_key
    """, (int(user["sub"]), req.date_key, req.mood_key))
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}


@router.delete("/moods/{date_key}")
def delete_mood(date_key: str, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM mood_log WHERE user_id = %s AND date_key = %s",
        (int(user["sub"]), date_key)
    )
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}