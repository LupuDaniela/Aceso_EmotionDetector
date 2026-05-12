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
            CREATE TABLE IF NOT EXISTS conversations (
                id               SERIAL PRIMARY KEY,
                user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
                message          TEXT NOT NULL,
                emotie_dominanta VARCHAR(50) NOT NULL,
                scor_dominant    FLOAT NOT NULL,
                toate_scorurile  JSONB,
                diade_detectate  JSONB,
                timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            ALTER TABLE conversations
            ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
        """)
        cursor.execute("""
            ALTER TABLE conversations
            ADD COLUMN IF NOT EXISTS diade_detectate JSONB
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
            ON conversations(timestamp DESC)
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("DB initializat.\n")
    except psycopg2.OperationalError as e:
        print("Eroare PostgreSQL:", e)
        sys.exit(1)