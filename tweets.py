import asyncio
import datetime
import sqlite3
import time
import zoneinfo
from twikit import Client  # pip install twikit
from datetime import date, timedelta

# Tus credenciales de X (crea una cuenta gratuita si no tienes)
USERNAME = 'tu_username'  # Ej: 'miusuario'
EMAIL = 'tu_email@ejemplo.com'
PASSWORD = 'tu_contrasena'

# ID de Elon Musk
ELON_SCREEN_NAME = 'elonmusk'
ELON_ID = '44196397'

# Timezones (usamos ET 12:00, equivalente a CET 18:00)
et_tz = zoneinfo.ZoneInfo("America/New_York")
utc = zoneinfo.ZoneInfo("UTC")

# Inicializa DB
conn = sqlite3.connect('elon_tweets.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS tweets 
                  (id TEXT PRIMARY KEY, created_at TEXT, author_id TEXT, in_reply_to_user_id TEXT)''')
conn.commit()

def get_start_dates(current_date, days_back=30, days_forward=30):
    starts = []
    start_check = current_date - timedelta(days=days_back)
    end_check = current_date + timedelta(days=days_forward)
    current = start_check
    while current <= end_check:
        if current.day % 3 == 0 and current.day != 0:
            starts.append(current)
        current += timedelta(days=1)
    return starts

def generate_events():
    today = date.today()
    start_days = get_start_dates(today)
    events = []
    for d in start_days:
        # Inicio: 12:00 ET = 18:00 CET
        start_dt = datetime.datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=et_tz)
        end_dt = start_dt + timedelta(days=7)
        name = f"{start_dt.strftime('%b %d')} al {end_dt.strftime('%b %d')}, {start_dt.year}"
        events.append({"name": name, "start": start_dt, "end": end_dt})
    return events

async def main():
    client = Client('en-US')
    
    # Login (con retry si falla)
    for _ in range(3):  # 3 intentos
        try:
            await client.login(
                auth_info_1=USERNAME,
                auth_info_2=EMAIL,
                password=PASSWORD,
                cookies_file='cookies.json'
            )
            print("Login exitoso.")
            break
        except Exception as e:
            print(f"Error en login: {e}. Reintentando...")
            time.sleep(5)
    else:
        print("Falló el login después de intentos.")
        return

    # Carga inicial: Fetch histórico (aumentado a 200 para más cobertura)
    user = await client.get_user_by_screen_name(ELON_SCREEN_NAME)
    tweets = await client.get_user_tweets(user.id, 'Tweets', count=200)
    while tweets:
        for tweet in tweets:
            if tweet.in_reply_to_user_id and tweet.in_reply_to_user_id != ELON_ID:
                continue
            try:
                cursor.execute("INSERT INTO tweets (id, created_at, author_id, in_reply_to_user_id) VALUES (?, ?, ?, ?)",
                               (tweet.id, tweet.created_at, tweet.user_id, tweet.in_reply_to_user_id))
                conn.commit()
            except sqlite3.IntegrityError:
                pass
        tweets = await tweets.next()

    # Actualiza conteos iniciales
    update_counts()

    # Polling (con retry)
    last_tweet_id = None
    while True:
        try:
            new_tweets = await client.get_user_tweets(user.id, 'Tweets', count=10)
            added = False
            for tweet in new_tweets:
                if tweet.id == last_tweet_id:
                    break
                if tweet.in_reply_to_user_id and tweet.in_reply_to_user_id != ELON_ID:
                    continue
                try:
                    cursor.execute("INSERT INTO tweets (id, created_at, author_id, in_reply_to_user_id) VALUES (?, ?, ?, ?)",
                                   (tweet.id, tweet.created_at, tweet.user_id, tweet.in_reply_to_user_id))
                    conn.commit()
                    print(f"Nuevo post: ID {tweet.id} at {tweet.created_at}")
                    added = True
                except sqlite3.IntegrityError:
                    pass
            if new_tweets:
                last_tweet_id = new_tweets[0].id
            if added:
                update_counts()
        except Exception as e:
            print(f"Error en polling: {e}. Reintentando en 60s...")
            time.sleep(60)
            continue
        time.sleep(30)  # Polling normal

def update_counts():
    now = datetime.datetime.now(tz=utc)
    seven_days_future = now + timedelta(days=7)
    fourteen_days_ago = now - timedelta(days=14)
    events = generate_events()  # Genera dinámicamente
    for event in events:
        end_utc = event['end'].astimezone(utc)
        if end_utc < fourteen_days_ago:
            print(f"Evento {event['name']}: Histórico viejo, scraping limitado.")
            continue
        if end_utc > seven_days_future or end_utc < now:
            continue  # Solo eventos que acaban en próximos 7 días
        start_utc_str = event['start'].astimezone(utc).isoformat() + 'Z'
        end_utc_str = end_utc.isoformat() + 'Z'
        count = cursor.execute(
            "SELECT COUNT(*) FROM tweets WHERE created_at >= ? AND created_at < ?",
            (start_utc_str, end_utc_str)
        ).fetchone()[0]
        status = "Activo" if event['start'].astimezone(utc) <= now else "Futuro"
        print(f"Evento: {event['name']} ({status}) | Conteo: {count}")

asyncio.run(main())