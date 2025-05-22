# Privacy_Social_media
# Face Recognition–Based Consent Platform

A privacy-preserving media sharing system that uses real-time face detection and recognition to enforce bystander consent before publishing live or recorded video streams. Ideal for AR devices (e.g., HoloLens 2) or webcams. 

---

## Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Configuration](#configuration)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Running Tests](#running-tests)  
7. [License](#license)  

---

## Features

- **User signup & face enrollment**  
  Upload or capture portrait images; system extracts and stores face encodings. 
- **Login & session management** via Flask-Login
- **Live video feed** with blur/unblur based on known face encodings, switchable between webcam and HoloLens stream. 
- **Start/stop recording** threadsafe AVI capture, FFmpeg conversion to H.264 MP4.
- **Friend network & requests** for implicit live unblur and targeted consent workflows.  
- **Consent management**: detect friends in saved recordings, issue ConsentRequest entries, track approve/deny, auto-share when all approve.  
- **Social feed**: displays public recordings and private friend-only recordings.  
- **Asynchronous tasks** via Celery for email notifications and reprocessing.  

---

## Tech Stack

- **Backend:** Python 3.8+, Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate  
- **Database:** SQLite (default) or any SQLAlchemy-supported DB 
- **Async & Mail:** Celery with Redis broker, Flask-Mail
- **AI / CV:**  
  - `facenet-pytorch` MTCNN for face detection
  - `face_recognition` & dlib for embeddings & matching
  - OpenCV for video I/O and blurring
- **Frontend:** Jinja2 templates, HTML/CSS/JS  
- **Dependencies:** see `requirements.txt`  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- [ffmpeg](https://ffmpeg.org/) installed & in `$PATH`  
- Redis server (for Celery broker & backend)  
- Optional: GPU + CUDA for faster MTCNN inference  

### Installation

# Clone repo
- git clone https://github.com/your-org/face-consent-platform.git
cd face-consent-platform

# Create & activate venv
- python3 -m venv venv
- source venv/bin/activate

# Install Python deps
- pip install -r requirements.txt

### Configuration 
- Copy & adjust environment variables in a .env or your shell:

- export SECRET_KEY="your-secret-key"
- export DATABASE_URL="sqlite:///app.db"         # or your DB URL
- export MAIL_USERNAME="you@example.com"
- export MAIL_PASSWORD="your-mail-password"
- export CELERY_BROKER_URL="redis://localhost:6379/0"
- export CELERY_RESULT_BACKEND="redis://localhost:6379/0"

Initialize database (optional, app also calls db.create_all() on first request):

-flask db init
-flask db migrate
-flask db upgrade

### Usage
#Start Redis
- redis-server

#Run Celery worker
- celery -A tasks.celery worker --loglevel=info

#Run Flask server
- flask run --host=127.0.0.1 --port=8000

- Open browser at http://127.0.0.1:8000

- Signup → enroll face images

- Login → go to Live → select source (webcam/HoloLens) → Start Recording → Stop Recording

- Share Recording → issues consent requests to detected friends

- Friends respond under Consents → once all approve, recording appears in Feed

If you're using hololens, connect it to the same network and change the IP address in app.py.
