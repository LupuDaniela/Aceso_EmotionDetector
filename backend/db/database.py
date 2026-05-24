import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host':     os.getenv("DB_HOST",     "localhost"),
    'port':     int(os.getenv("DB_PORT", "5432")),
    'database': os.getenv("DB_NAME",     "emotion_db"),
    'user':     os.getenv("DB_USER",     "postgres"),
    'password': os.getenv("DB_PASSWORD", ""),
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            email         VARCHAR(255) UNIQUE NOT NULL,
            name          VARCHAR(255),
            password_hash VARCHAR(255),
            google_id     VARCHAR(255),
            reset_token   VARCHAR(255),
            reset_expires TIMESTAMP WITH TIME ZONE,
            created_at    TIMESTAMP DEFAULT NOW()
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id                SERIAL PRIMARY KEY,
            user_id           INTEGER REFERENCES users(id),
            message           TEXT,
            emotie_dominanta  VARCHAR(100),
            scor_dominant     FLOAT,
            toate_scorurile   JSONB,
            diade_detectate   JSONB,
            timestamp         TIMESTAMP DEFAULT NOW()
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_log (
            id       SERIAL PRIMARY KEY,
            user_id  INTEGER REFERENCES users(id) ON DELETE CASCADE,
            date_key VARCHAR(10) NOT NULL,
            mood_key VARCHAR(50) NOT NULL,
            UNIQUE(user_id, date_key)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()