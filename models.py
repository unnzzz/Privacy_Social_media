from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Association table for friends.
friend_association = db.Table('friend_association',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('friend_id', db.Integer, db.ForeignKey('user.id'))
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)  # User's name
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    reference_media = db.Column(db.String(200))
    face_encoding = db.Column(db.PickleType)
    # New field: Public vs. Private account. True means public.
    is_public_account = db.Column(db.Boolean, default=True)
    friends = db.relationship('User', secondary=friend_association,
                              primaryjoin=id==friend_association.c.user_id,
                              secondaryjoin=id==friend_association.c.friend_id,
                              backref='friend_of')

    def get_friends(self):
        # all users where a FriendRequest exists with status='accepted'
        sent = FriendRequest.query.filter_by(sender_id=self.id, status='accepted').all()
        recv = FriendRequest.query.filter_by(receiver_id=self.id, status='accepted').all()
        # gather unique User objects
        friends = {fr.receiver for fr in sent} | {fr.sender for fr in recv}
        return list(friends)

    def get_all_known_encodings(self):
        """Return own encodings plus those of friends."""
        encodings = []
        if self.face_encoding:
            encodings.extend(self.face_encoding)
        for friend in self.friends:
            if friend.face_encoding:
                encodings.extend(friend.face_encoding)
        return encodings

class FriendRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String(20), default='pending')  # pending, accepted, rejected
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    sender = db.relationship('User', foreign_keys=[sender_id])
    receiver = db.relationship('User', foreign_keys=[receiver_id])


class Recording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    filename = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    consent_status = db.Column(db.String(20), default='pending')
    is_shared     = db.Column(db.Boolean, default=False)
    user          = db.relationship('User', backref='recordings')

class ConsentRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending / approved / denied
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    recording = db.relationship('Recording', backref=db.backref('consent_requests', cascade='all,delete-orphan'))
    requester = db.relationship('User', foreign_keys=[requester_id])
    recipient = db.relationship('User', foreign_keys=[recipient_id])


