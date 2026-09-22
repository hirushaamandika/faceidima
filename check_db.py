

import pymysql

HOST = "serverless-europe-west2.sysp0000.db2.skysql.com"
PORT = 4000
USER = "dbpgf38994718"
PASSWORD = "0OkVj8D3i~EF4B97dfx3olJD"
SSL_CA_PATH = "globalsignrootca.pem"  

conn = pymysql.connect(
    host=HOST,
    port=PORT,
    user=USER,
    password=PASSWORD,
    ssl={"ca": SSL_CA_PATH},
)

try:
    with conn.cursor() as cur:
        cur.execute("SHOW DATABASES;")
        print("Existing databases:")
        for row in cur.fetchall():
            print(" -", row[0])

        cur.execute("CREATE DATABASE IF NOT EXISTS face_attendance;")
        conn.commit()
        print("\nDone. 'face_attendance' database is ready to use.")
finally:
    conn.close()
