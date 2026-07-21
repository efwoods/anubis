# example.py
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
# from io import BytesIO

from typing import Optional, List

from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

async def create_instant_voice_clone_voice(
    runtime: Runtime, 
    store: BaseStore,
    assistant_id: str, 
    user_id: str,
    audio_file_b_list: Optional[List[bytes]] = None, 
    ):
    """ 
    Create a voice for the avatar using the reference audio file from the store for an avatar.

    # note: audio_file_bytes
    
    """
    
    if audio_file_b_list == None:
        """ List of audio bytes is none, attempt to retrieve reference audio from the store """
        namespace = (user_id, assistant_id, "reference_audio",)
        await store.aget(namespace, key="reference_audio")



    NN_ELEVENLABS_API_KEY = runtime.context.get("NN_ELEVENLABS_API_KEY", "")
    elevenlabs_client = ElevenLabs(NN_ELEVENLABS_API_KEY)

    voice = elevenlabs_client.voices.ivc.create(
        name=f"{user_id}_{assistant_id}",
        # Replace with the paths to your audio files.
        # The more files you add, the better the clone will be.
        # files=[BytesIO(open("/path/to/your/audio/file.mp3", "rb").read())]
        files = audio_file_b_list
    )

    namespace = (user_id, assistant_id, "voice_id")
    key="voice_id"
    await store.aput(namespace, key=key, value={"value":voice.voice_id})

    print(voice.voice_id)

