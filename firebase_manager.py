"""
Firebase integration for AMATS state management and real-time data.
Handles Firestore operations with robust error handling.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import config

class FirebaseManager:
    """Manages all Firebase Firestore operations for AMATS"""
    
    def __init__(self):
        self._client: Optional[firestore.Client] = None
        self._initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with error handling"""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.db.firebase_credentials_path)
                firebase_admin.initialize_app(