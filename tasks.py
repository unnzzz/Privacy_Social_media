from celery import Celery
from config import Config

celery = Celery(__name__, broker=Config.CELERY_BROKER_URL)
celery.conf.update(Config.__dict__)

@celery.task
def send_consent_email(recording_id, friend_email):
    # Code to send an email (using Flask-Mail) asking for consent.
    # This is a placeholder.
    print(f"Sending consent email for recording {recording_id} to {friend_email}")
    return "Email sent"

@celery.task
def reprocess_recording(recording_id):
    # Code to re-run the face recognition processing on a recording.
    print(f"Reprocessing recording {recording_id}")
    return "Reprocessed"
