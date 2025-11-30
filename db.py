# db.py
import sqlite3
import pandas as pd
from datetime import datetime
DB_PATH = "bookings.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        phone TEXT,
        hotel TEXT,
        destination TEXT,
        checkin TEXT,
        checkout TEXT,
        guests INTEGER,
        notes TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_booking(booking: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO bookings (name,email,phone,hotel,destination,checkin,checkout,guests,notes,created_at)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        booking.get("name"),
        booking.get("email"),
        booking.get("phone"),
        booking.get("hotel"),
        booking.get("destination"),
        booking.get("checkin"),
        booking.get("checkout"),
        booking.get("guests"),
        booking.get("notes"),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def get_bookings():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM bookings ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_booking(booking_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
    conn.commit()
    conn.close()

def export_bookings_csv(path="bookings_export.csv"):
    df = get_bookings()
    df.to_csv(path, index=False)
    return path
