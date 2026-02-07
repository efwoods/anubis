import os
import firebase_admin
from firebase_admin import firestore, auth, db, storage, get_app, credentials
from firebase_admin.exceptions import FirebaseError
from typing import Optional, Dict, Any, List

import logging
logger = logging.getLogger(__name__)

class FirebaseDB:
    """Wrapper around Google Firestore, Auth, and Storage operations with emulator support."""
    
    def __init__(self, project: Optional[str] = None, use_emulator: bool = None):
        self.project_id = project or os.getenv("FIREBASE_PROJECT_ID", "neuralnexus-467517")

        # Auto-detect environment if not explicitly set
        if use_emulator is None:
            use_emulator = os.getenv("USE_EMULATOR", "false").lower() == "true"
        self.use_emulator = use_emulator

        if use_emulator:
            self._init_emulator()
        else:
            self._init_production()
    
    def _init_emulator(self):
        """Initialize for local emulator."""
        print("🔧 Initializing Firebase with EMULATORS")

        # 1. Configuration
        # project_id = os.getenv("FIRESTORE_PROJECT_ID")
        # cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # 2. Set Environment Variables explicitly for the clients
        os.environ["FIRESTORE_EMULATOR_HOST"] = os.getenv("FIRESTORE_EMULATOR_HOST")
        os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = os.getenv("FIREBASE_AUTH_EMULATOR_HOST")
        os.environ["STORAGE_EMULATOR_HOST"] = os.getenv("FIREBASE_STORAGE_EMULATOR_HOST")
        

        # 3. Handle Firebase Admin SDK
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(options={"projectId": "neuralnexus-467517"})

        except ValueError:
            logger.error(ValueError)
            # if cred_path and os.path.exists(cred_path):
            #     # Use actual file if it exists
            #     cred = credentials.Certificate(cred_path)
            #     initialize_app(cred, {'projectId': project_id})
        
        # 4. Initialize clients with EXPLICIT project_id
        # This tells the library NOT to go looking for a service account to find the project
        self.client = firestore.client()
        self.db = self.client
        self.auth = auth
        self.bucket = storage.bucket(f"{self.project_id}.appspot.com")
        
        # For storage, using AnonymousCredentials prevents the 'File not found' error
        # from google.auth import credentials as google_creds
        # self.storage_client = cloud_storage.Client(
        #     project=project_id,
        #     credentials=google_creds.AnonymousCredentials() if not cred_path else None
        # )

        print(f"  - Project: {self.project_id}")
        print(f"  - Firestore: {os.getenv('FIRESTORE_EMULATOR_HOST')}")
        print(f"  - Auth: {os.getenv('FIREBASE_AUTH_EMULATOR_HOST')}")
        print(f"  - Storage: {os.getenv('FIREBASE_STORAGE_EMULATOR_HOST')}")
        print(f"  - UI: http://localhost:4000")
    
    def _init_production(self, project: Optional[str]):
        """Initialize for production."""
        print("🚀 Initializing Firebase for PRODUCTION")
        os.environ.pop("FIRESTORE_EMULATOR_HOST")
        os.environ.pop("FIREBASE_AUTH_EMULATOR_HOST")
        os.environ.pop("STORAGE_EMULATOR_HOST")
        
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            raise ValueError(
                "Production mode requires GOOGLE_APPLICATION_CREDENTIALS "
                "pointing to a valid service account JSON file"
            )

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred, {
                'projectId': self.project_id,
                'storageBucket': f"{self.project_id}.appspot.com"
            })
        
        self.db     = firestore.client()
        self.auth   = auth
        self.bucket = storage.bucket()

        print(f" → Project: {self.project_id}")
        print(f" → Credentials: {creds_path}")
        
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return user document or None."""
        doc = self.client.collection("users").document(user_id).get()
        return doc.to_dict() if doc.exists else None

    def create_user(self, user_id: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Create a user document (id = user_id)."""
        payload = data or {}
        payload.setdefault("avatars", [])
        self.client.collection("users").document(user_id).set(payload, merge=True)

    def add_avatar_to_user(self, user_id: str, avatar_id: str) -> None:
        """Ensure avatar_id is in users/{userId}.avatars array."""
        user_ref = self.client.collection("users").document(user_id)
        try:
            user_ref.update({"avatars": firestore.ArrayUnion([avatar_id])})
        except Exception:
            # If user doc missing, create it with avatars array
            self.create_user(user_id, {"avatars": [avatar_id]})

    # --- Avatar documents (top-level collection 'avatars') ---

    def get_avatar(self, user_id: str, avatar_id: str) -> Optional[Dict[str, Any]]:
        """Get avatar document from users/{userId}/avatars/{avatarId} subcollection."""
        logger.info(f'get_avatar - user_id: {user_id}, avatar_id: {avatar_id}')
        doc = (
            self.client.collection("users")
            .document(user_id)
            .collection("avatars")
            .document(avatar_id)
            .get()
        )
        logger.info(f'DOC exists: {doc.exists}')
        return doc.to_dict() if doc.exists else None

    def create_avatar(self, avatar_id: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Create avatars/{avatarId} document. Keeps conversations array and default_conversation fields."""
        payload = data or {}
        payload.setdefault("conversations", [])
        payload.setdefault("default_conversation", None)
        self.client.collection("avatars").document(avatar_id).set(payload, merge=True)

    def update_avatar_fields(self, user_id: str, avatar_id: str, fields: Dict[str, Any]) -> None:
        """Merge-update fields on avatars/{avatarId}."""
        (
            self.client.collection("users")
             .document(user_id)
             .collection("avatars")
             .document(avatar_id)
             .set(fields, merge=True)
            )
        

    # # --- Conversations under avatars/{avatarId}/conversations/{conversationId} ---

    # def list_conversations(self, avatar_id: str) -> List[str]:
    #     avatar = self.get_avatar(avatar_id)
    #     return avatar.get("conversations", []) if avatar else []

    # def add_conversation_to_avatar(self, avatar_id: str, conversation_id: str, data: Optional[Dict[str, Any]] = None) -> None:
    #     """
    #     Add conversation id to avatars/{avatarId}.conversations and create the subcollection doc
    #     avatars/{avatarId}/conversations/{conversationId}.
    #     """
    #     # ensure avatar doc exists
    #     try:
    #         self.client.collection("avatars").document(avatar_id).update({"conversations": firestore.ArrayUnion([conversation_id])})
    #     except Exception:
    #         # create avatar doc with conversations array
    #         self.create_avatar(avatar_id, {"conversations": [conversation_id]})

    #     conv_ref = self.client.collection("avatars").document(avatar_id).collection("conversations").document(conversation_id)
    #     conv_ref.set(data or {}, merge=True)

    # def set_default_conversation(self, avatar_id: str, conversation_id: Optional[str]) -> None:
    #     """Set avatars/{avatarId}.default_conversation"""
    #     self.update_avatar_fields(avatar_id, {"default_conversation": conversation_id})

    # # --- Messages under avatars/{avatarId}/conversations/{conversationId}/messages/{messageId} ---

    # def create_message(self, avatar_id: str, conversation_id: str, message_id: str, message_data: Dict[str, Any]) -> None:
    #     """
    #     Create or overwrite a message document under messages subcollection.
    #     message_data should include structure like: {"message": {"text": "...", "media": []}, "sender": "user", ...}
    #     """
    #     msg_ref = (
    #         self.client.collection("avatars")
    #         .document(avatar_id)
    #         .collection("conversations")
    #         .document(conversation_id)
    #         .collection("messages")
    #         .document(message_id)
    #     )
    #     msg_ref.set(message_data, merge=True)

    # def get_message(self, avatar_id: str, conversation_id: str, message_id: str) -> Optional[Dict[str, Any]]:
    #     msg_ref = (
    #         self.client.collection("avatars")
    #         .document(avatar_id)
    #         .collection("conversations")
    #         .document(conversation_id)
    #         .collection("messages")
    #         .document(message_id)
    #     )
    #     doc = msg_ref.get()
    #     return doc.to_dict() if doc.exists else None

    # def list_messages(self, avatar_id: str, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    #     msgs_ref = (
    #         self.client.collection("avatars")
    #         .document(avatar_id)
    #         .collection("conversations")
    #         .document(conversation_id)
    #         .collection("messages")
    #     )
    #     query = msgs_ref.order_by("created_at") if "created_at" in [f.name for f in msgs_ref.getAll()[0].reference.parent._client._firestore._metadata] else msgs_ref
    #     docs = query.limit(limit).stream() if limit else query.stream()
    #     return [d.to_dict() for d in docs]

    # def append_media_to_message(self, avatar_id: str, conversation_id: str, message_id: str, media_item: Dict[str, Any]) -> None:
    #     """Append a media item to message.message.media list. If absent, creates media list."""
    #     msg_ref = (
    #         self.client.collection("avatars")
    #         .document(avatar_id)
    #         .collection("conversations")
    #         .document(conversation_id)
    #         .collection("messages")
    #         .document(message_id)
    #     )
    #     doc = msg_ref.get()
    #     if not doc.exists:
    #         # create message with media list
    #         msg_ref.set({"message": {"text": "", "media": [media_item]}}, merge=True)
    #         return

    #     current = doc.to_dict() or {}
    #     msg = current.get("message", {})
    #     media = msg.get("media", [])
    #     media.append(media_item)
    #     msg_ref.set({"message": {"media": media}}, merge=True)


