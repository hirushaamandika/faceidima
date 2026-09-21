import os
from contextlib import contextmanager
from datetime import date, datetime

import pymysql
import pymysql.cursors

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads a local .env file if present; no-op in production if absent
except ImportError:
    pass

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "face_attendance")


@contextmanager
def get_connection():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS users (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       name VARCHAR(255) UNIQUE NOT NULL
                   ) ENGINE=InnoDB"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS attendance (
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       name VARCHAR(255) NOT NULL,
                       timestamp DATETIME NOT NULL,
                       attendance_date DATE NOT NULL,
                       INDEX idx_name_date (name, attendance_date)
                   ) ENGINE=InnoDB"""
            )
        conn.commit()


def add_user_if_missing(name: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT IGNORE INTO users (name) VALUES (%s)", (name,))
        conn.commit()


def get_all_users():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM users ORDER BY name")
            rows = cur.fetchall()
    return [r["name"] for r in rows]


def already_marked_today(name: str) -> bool:
    today = date.today()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM attendance WHERE name = %s AND attendance_date = %s LIMIT 1",
                (name, today),
            )
            row = cur.fetchone()
    return row is not None


def mark_attendance(name: str) -> bool:
    """Insert an attendance row for today if not already marked. Returns True if newly marked."""
    if already_marked_today(name):
        return False
    now = datetime.now()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO attendance (name, timestamp, attendance_date) VALUES (%s, %s, %s)",
                (name, now, now.date()),
            )
        conn.commit()
    return True


def get_attendance_records(limit: int = 200):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, timestamp FROM attendance ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    return rows
