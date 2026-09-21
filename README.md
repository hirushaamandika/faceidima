# Face Attendance System

Register faces via webcam, train a CNN to recognize them, then scan faces to
mark attendance in a local SQLite database.

## IMPORTANT: use Python 3.10, 3.11, or 3.12

TensorFlow does **not** yet publish wheels for very new Python releases
(e.g. 3.13/3.14) — `pip install tensorflow` will fail or silently not
install on those versions. Use Python 3.11 or 3.12 for this project.

## Environment variables (required for the hosted/CI-CD version)

Set these in your platform's dashboard (Render/Railway env vars) or as
GitHub Actions secrets injected into your server's `.env` — never commit
real credentials to the repo.

```
# MySQL (e.g. MariaDB SkySQL free tier)
MYSQL_HOST=your-db-host
MYSQL_PORT=3306
MYSQL_USER=your-db-user
MYSQL_PASSWORD=your-db-password
MYSQL_DATABASE=face_attendance

# S3-compatible storage (e.g. Cloudflare R2 free tier)
S3_BUCKET_NAME=your-bucket-name
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com   # omit entirely for real AWS S3
S3_REGION=auto
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

Check your version:
```
python --version
```

If you're on 3.13+, install Python 3.11 from python.org, then create a
virtual environment with it specifically:
```
py -3.11 -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (macOS/Linux)
```

## Setup

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## How to use

1. **Register** — go to "Register New Face", type a name, click
   "Start Capturing". It automatically grabs ~20 face images over a few
   seconds. Click "Finish & Train Model" when done.
   - **Register at least 2 different people** before training — a
     classifier needs multiple classes to tell people apart.
   - Re-running registration + train for a new person retrains the model
     on everyone registered so far (previous people are kept, since their
     images are still in `dataset/`).

2. **Mark Attendance** — go to "Mark Attendance", click "Scan & Mark
   Attendance". It detects your face, runs it through the trained CNN,
   and if confident enough, logs your name + timestamp to the database
   (once per person per day).

3. **View Records** — see the full attendance log.

## How it works

- `haarcascade_frontalface_default.xml` — OpenCV's classical face
  *detector* (finds the location of a face in a frame). This runs before
  the CNN, just to crop out the face region.
- `model.py` — defines and trains a small CNN *classifier* (the neural
  network that learns to recognize *which* registered person a cropped
  face belongs to). Saved to `model/face_cnn.keras` after training.
- `database.py` — SQLite storage for registered user names and
  attendance timestamps (`attendance.db`, created automatically).
- `dataset/<name>/` — the captured training face images per person.

## Known limitations (by design, for simplicity)

- **Closed-set recognition**: the CNN's final layer is a softmax over
  exactly the people you've registered. If a completely new, unregistered
  face is scanned, the confidence threshold (0.75 in `model.py`) is your
  only defense against a false match — it isn't foolproof. For stricter
  security, consider a face-embedding/verification approach instead of a
  fixed-class classifier.
- **Retraining is required after every new registration** — the model
  doesn't incrementally learn; `train_and_save` rebuilds it from scratch
  using everyone in `dataset/`.
- Small per-person image counts (~20) with a from-scratch CNN can overfit;
  accuracy will improve with more images per person and varied
  lighting/angles during registration.
