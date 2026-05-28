from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db.database    import get_conn
from auth.jwt_utils import verifica_token

router = APIRouter()

_pipeline = None

def set_pipeline(p):
    global _pipeline
    _pipeline = p


class ThreadCreateRequest(BaseModel):
    titlu: Optional[str] = None

class ChatMessageRequest(BaseModel):
    text: str
    thread_id: int


@router.post("/chat/thread", status_code=201)
def create_thread(req: ThreadCreateRequest, user=Depends(verifica_token)):
    titlu  = req.titlu or "Conversatie noua"
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_threads (user_id, titlu) VALUES (%s, %s) RETURNING id, creat_la, actualizat_la",
        (int(user["sub"]), titlu)
    )
    row = cursor.fetchone()
    conn.commit(); cursor.close(); conn.close()
    return {
        "id":            row[0],
        "titlu":         titlu,
        "creat_la":      row[1].isoformat(),
        "actualizat_la": row[2].isoformat(),
        "last_message":  None,
        "last_emotie":   None,
        "last_diade":    [],
    }


@router.get("/chat/threads")
def get_threads(user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute("""
        SELECT
            t.id, t.titlu, t.creat_la, t.actualizat_la,
            (SELECT message FROM conversations WHERE thread_id = t.id ORDER BY timestamp ASC LIMIT 1),
            (SELECT emotie_dominanta FROM conversations WHERE thread_id = t.id ORDER BY timestamp DESC LIMIT 1),
            (SELECT diade_detectate FROM conversations WHERE thread_id = t.id ORDER BY timestamp DESC LIMIT 1)
        FROM chat_threads t
        WHERE t.user_id = %s
        ORDER BY t.actualizat_la DESC
    """, (int(user["sub"]),))
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return [
        {
            "id":            r[0],
            "titlu":         r[1],
            "creat_la":      r[2].isoformat(),
            "actualizat_la": r[3].isoformat(),
            "last_message":  r[4],
            "last_emotie":   r[5],
            "last_diade":    r[6] if r[6] else [],
        }
        for r in rows
    ]


@router.delete("/chat/thread/{thread_id}")
def delete_thread(thread_id: int, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM chat_threads WHERE id = %s AND user_id = %s",
        (thread_id, int(user["sub"]))
    )
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}


@router.patch("/chat/thread/{thread_id}")
def rename_thread(thread_id: int, req: ThreadCreateRequest, user=Depends(verifica_token)):
    if not req.titlu:
        raise HTTPException(400, "Titlul nu poate fi gol.")
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_threads SET titlu = %s WHERE id = %s AND user_id = %s",
        (req.titlu, thread_id, int(user["sub"]))
    )
    conn.commit(); cursor.close(); conn.close()
    return {"ok": True}


@router.get("/chat/thread/{thread_id}/messages")
def get_thread_messages(thread_id: int, user=Depends(verifica_token)):
    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM chat_threads WHERE id = %s AND user_id = %s",
        (thread_id, int(user["sub"]))
    )
    if not cursor.fetchone():
        cursor.close(); conn.close()
        raise HTTPException(403, "Acces interzis.")
    cursor.execute("""
        SELECT id, message, emotie_dominanta, scor_dominant,
               toate_scorurile, diade_detectate, raspuns_empatic, timestamp
        FROM conversations
        WHERE thread_id = %s
        ORDER BY timestamp ASC
    """, (thread_id,))
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return [
        {
            "id":               r[0],
            "message":          r[1],
            "emotie_dominanta": r[2],
            "scor_dominant":    r[3],
            "toate_scorurile":  r[4],
            "diade_detectate":  r[5],
            "raspuns_empatic":  r[6],
            "timestamp":        r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


@router.post("/chat/message")
def chat_message(req: ChatMessageRequest, user=Depends(verifica_token)):
    if not _pipeline:
        raise HTTPException(503, "Pipeline se incarca.")
    if not req.text.strip():
        raise HTTPException(400, "Textul nu poate fi gol.")

    conn   = get_conn(); cursor = conn.cursor()
    cursor.execute(
        "SELECT id, titlu FROM chat_threads WHERE id = %s AND user_id = %s",
        (req.thread_id, int(user["sub"]))
    )
    thread = cursor.fetchone()
    cursor.close(); conn.close()

    if not thread:
        raise HTTPException(403, "Thread invalid sau acces interzis.")

    if thread[1] == "Conversatie noua":
        titlu_nou = req.text.strip()[:50]
        conn   = get_conn(); cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_threads SET titlu = %s WHERE id = %s",
            (titlu_nou, req.thread_id)
        )
        conn.commit(); cursor.close(); conn.close()

    return _pipeline.analizeaza(
        req.text,
        salveaza=True,
        user_id=int(user["sub"]),
        thread_id=req.thread_id,
        genereaza_empatic=True,
    )