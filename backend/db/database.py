import os, sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host':     os.getenv('DB_HOST', 'localhost'),
    'port':     int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'emotion_db'),
    'user':     os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    print("Conectare PostgreSQL...")
    try:
        conn   = get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id             SERIAL PRIMARY KEY,
                email          VARCHAR(255) UNIQUE NOT NULL,
                name           VARCHAR(255),
                google_id      VARCHAR(255) UNIQUE,
                password_hash  VARCHAR(255),
                reset_token    VARCHAR(255),
                reset_expires  TIMESTAMP,
                created_at     TIMESTAMP DEFAULT NOW()
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mood_log (
                user_id  INTEGER REFERENCES users(id) ON DELETE CASCADE,
                date_key VARCHAR(10) NOT NULL,
                mood_key VARCHAR(20) NOT NULL,
                PRIMARY KEY (user_id, date_key)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mood_log_user_id
            ON mood_log(user_id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id            SERIAL PRIMARY KEY,
                user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
                titlu         TEXT NOT NULL DEFAULT 'Conversatie noua',
                creat_la      TIMESTAMP DEFAULT NOW(),
                actualizat_la TIMESTAMP DEFAULT NOW()
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_threads_user_id
            ON chat_threads(user_id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id               SERIAL PRIMARY KEY,
                user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
                thread_id        INTEGER REFERENCES chat_threads(id) ON DELETE CASCADE,
                message          TEXT NOT NULL,
                emotie_dominanta VARCHAR(50) NOT NULL,
                scor_dominant    FLOAT NOT NULL,
                toate_scorurile  JSONB,
                diade_detectate  JSONB,
                raspuns_empatic  TEXT,
                timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS
            thread_id INTEGER REFERENCES chat_threads(id) ON DELETE CASCADE
        """)
        cursor.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS raspuns_empatic TEXT
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_thread_id
            ON conversations(thread_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
            ON conversations(timestamp DESC)
        """)

        cursor.execute("""
            SELECT id, user_id, message, timestamp
            FROM conversations
            WHERE thread_id IS NULL AND user_id IS NOT NULL
            ORDER BY user_id, timestamp ASC
        """)
        orfane = cursor.fetchall()

        thread_cache = {}

        for (conv_id, uid, message, timestamp) in orfane:
            day_key = timestamp.strftime('%Y-%m-%d')
            cache_key = (uid, day_key)

            if cache_key not in thread_cache:
                titlu = message.strip()[:50] if message else day_key
                cursor.execute("""
                    INSERT INTO chat_threads (user_id, titlu, creat_la, actualizat_la)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (uid, titlu, timestamp, timestamp))
                thread_id = cursor.fetchone()[0]
                thread_cache[cache_key] = thread_id
            else:
                thread_id = thread_cache[cache_key]
                cursor.execute("""
                    UPDATE chat_threads SET actualizat_la = %s WHERE id = %s
                """, (timestamp, thread_id))

            cursor.execute("""
                UPDATE conversations SET thread_id = %s WHERE id = %s
            """, (thread_id, conv_id))

        conn.commit()
        cursor.close()
        conn.close()
        print("DB initializat.\n")
    except psycopg2.OperationalError as e:
        print("Eroare PostgreSQL:", e)
        sys.exit(1)