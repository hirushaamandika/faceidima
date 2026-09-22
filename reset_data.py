

import pymysql
import boto3


MYSQL_HOST = "serverless-europe-west2.sysp0000.db2.skysql.com"
MYSQL_PORT = 4000
MYSQL_USER = "dbpgf38994718"
MYSQL_PASSWORD = "0OkVj8D3i~EF4B97dfx3olJD"
MYSQL_DATABASE = "face_attendance"
MYSQL_SSL_CA = "globalsignrootca.pem"  

-
S3_BUCKET = "faceidima"
S3_ENDPOINT_URL = "https://d16b463548d4194f648de87bf02f61af.r2.cloudflarestorage.com"
S3_REGION = "auto"
AWS_ACCESS_KEY_ID = "0ab766c98864d9b86c8d04b83d485aee"
AWS_SECRET_ACCESS_KEY = "4ee1cb82ded57cc462c419ec97d23040ac3128b8ac571c09d4b3392d0f01a043"


def wipe_mysql():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
        ssl={"ca": MYSQL_SSL_CA},
    )
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM attendance;")
            cur.execute("DELETE FROM users;")
        conn.commit()
        print("MySQL: cleared 'users' and 'attendance' tables.")
    finally:
        conn.close()


def wipe_s3():
    client = boto3.client(
        "s3",
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    deleted = 0
    for prefix in ("dataset/", "model/"):
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                client.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": keys})
                deleted += len(keys)
    print(f"S3: deleted {deleted} object(s) under dataset/ and model/.")


if __name__ == "__main__":
    confirm = input(
        "This will permanently delete ALL registered users, attendance records, "
        "face images, and the trained model. Type YES to continue: "
    )
    if confirm.strip() == "YES":
        wipe_mysql()
        wipe_s3()
        print("Done. Everything has been reset.")
    else:
        print("Cancelled. Nothing was deleted.")
