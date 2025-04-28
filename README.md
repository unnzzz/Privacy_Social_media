# Privacy_Social_media
# Face Recognition–Based Consent Platform

A privacy-preserving media sharing system that uses real-time face detection and recognition to enforce bystander consent before publishing live or recorded video streams. Ideal for AR devices (e.g., HoloLens 2) or webcams. :contentReference[oaicite:0]{index=0}

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
  Upload or capture portrait images; system extracts and stores face encodings :contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}.  
- **Login & session management** via Flask-Login :contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}.  
- **Live video feed** with blur/unblur based on known face encodings, switchable between webcam and HoloLens stream :contentReference[oaicite:5]{index=5}.  
- **Start/stop recording** threadsafe AVI capture, FFmpeg conversion to H.264 MP4 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}.  
- **Friend network & requests** for implicit live unblur and targeted consent workflows :contentReference[oaicite:8]{index=8}.  
- **Consent management**: detect friends in saved recordings, issue ConsentRequest entries, track approve/deny, auto-share when all approve :contentReference[oaicite:9]{index=9}.  
- **Social feed**: displays public recordings and private friend-only recordings :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}.  
- **Asynchronous tasks** via Celery for email notifications and reprocessing :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}.  

---

## Tech Stack

- **Backend:** Python 3.8+, Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate  
- **Database:** SQLite (default) or any SQLAlchemy-supported DB :contentReference[oaicite:14]{index=14}&#8203;:contentReference[oaicite:15]{index=15}  
- **Async & Mail:** Celery with Redis broker, Flask-Mail :contentReference[oaicite:16]{index=16}&#8203;:contentReference[oaicite:17]{index=17}  
- **AI / CV:**  
  - `facenet-pytorch` MTCNN for face detection :contentReference[oaicite:18]{index=18}&#8203;:contentReference[oaicite:19]{index=19}  
  - `face_recognition` & dlib for embeddings & matching :contentReference[oaicite:20]{index=20}&#8203;:contentReference[oaicite:21]{index=21}  
  - OpenCV for video I/O and blurring :contentReference[oaicite:22]{index=22}&#8203;:contentReference[oaicite:23]{index=23}  
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

```bash
# Clone repo
git clone https://github.com/your-org/face-consent-platform.git
cd face-consent-platform

# Create & activate venv
python3 -m venv venv
source venv/bin/activate

# Install Python deps
pip install -r requirements.txt
