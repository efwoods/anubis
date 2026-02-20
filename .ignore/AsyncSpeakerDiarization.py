import asyncio
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cdist
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

class AsyncSpeakerDiarization:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.pipeline = None
        self.embedding_model = None
        self.audio = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize models asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Load models in thread pool (they're CPU/GPU intensive)
        self.pipeline = await loop.run_in_executor(
            self.executor,
            self._load_pipeline
        )
        
        self.embedding_model = await loop.run_in_executor(
            self.executor,
            self._load_embedding_model
        )
        
        self.audio = Audio(sample_rate=16000, mono=True)
        
    def _load_pipeline(self):
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
    
    def _load_embedding_model(self):
        return PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device="cpu"  # Change to "cuda" if GPU available
        )
    
    async def diarize(self, audio_path, num_speakers=None):
        """Run diarization asynchronously"""
        loop = asyncio.get_event_loop()
        
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        
        diarization = await loop.run_in_executor(
            self.executor,
            lambda: self.pipeline(audio_path, **kwargs)
        )
        
        return diarization
    
    async def get_num_speakers(self, diarization):
        """Get number of unique speakers"""
        return len(diarization.labels())
    
    async def extract_embedding(self, audio_path, segment=None):
        """Extract speaker embedding from audio"""
        loop = asyncio.get_event_loop()
        
        def _extract():
            if segment:
                waveform, _ = self.audio.crop(audio_path, segment)
            else:
                # Use first 5 seconds if no segment specified
                from pyannote.core import Segment
                waveform, _ = self.audio.crop(audio_path, Segment(0, 5))
            
            return self.embedding_model(waveform[None])
        
        return await loop.run_in_executor(self.executor, _extract)
    
    async def find_target_speaker(self, full_audio_path, target_audio_path, diarization):
        """Find which speaker matches the target"""
        # Get target embedding
        target_embedding = await self.extract_embedding(target_audio_path)
        
        # Compare with each speaker
        results = {}
        tasks = []
        
        for speaker in diarization.labels():
            # Get first segment for this speaker
            speaker_segments = [
                s for s, _, spk in diarization.itertracks(yield_label=True) 
                if spk == speaker
            ]
            
            if speaker_segments:
                task = self._compare_speaker(
                    full_audio_path, 
                    speaker_segments[0], 
                    target_embedding,
                    speaker
                )
                tasks.append(task)
        
        # Run all comparisons concurrently
        comparisons = await asyncio.gather(*tasks)
        
        for speaker, similarity in comparisons:
            results[speaker] = similarity
        
        return results
    
    async def _compare_speaker(self, audio_path, segment, target_embedding, speaker_label):
        """Compare a speaker segment with target embedding"""
        speaker_embedding = await self.extract_embedding(audio_path, segment)
        
        loop = asyncio.get_event_loop()
        
        def _calc_similarity():
            distance = cdist(target_embedding, speaker_embedding, metric="cosine")[0][0]
            return 1 - distance
        
        similarity = await loop.run_in_executor(self.executor, _calc_similarity)
        
        return (speaker_label, similarity)
    
    async def get_target_speaker_segments(self, diarization, target_speaker):
        """Get all segments where target speaker is talking"""
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == target_speaker:
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.duration
                })
        return segments
    
    def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

"""
USAGE EXAMPLE:

# Usage example
async def main():
    # Initialize
    diarizer = AsyncSpeakerDiarization(hf_token="your_hf_token_here")
    await diarizer.initialize()
    
    try:
        # Run diarization
        print("Running diarization...")
        diarization = await diarizer.diarize("full_audio.wav")
        
        # Get number of speakers
        num_speakers = await diarizer.get_num_speakers(diarization)
        print(f"Number of speakers detected: {num_speakers}")
        
        # Print all segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
        
        # Find target speaker
        print("\nFinding target speaker...")
        similarities = await diarizer.find_target_speaker(
            "full_audio.wav",
            "target_speaker.wav",
            diarization
        )
        
        # Print similarities
        for speaker, similarity in similarities.items():
            print(f"Speaker {speaker}: similarity = {similarity:.3f}")
        
        # Get target speaker (highest similarity)
        target_speaker = max(similarities, key=similarities.get)
        print(f"\nTarget speaker identified as: {target_speaker}")
        
        # Get all segments for target speaker
        segments = await diarizer.get_target_speaker_segments(diarization, target_speaker)
        print(f"\nTarget speaker segments ({len(segments)} total):")
        for seg in segments:
            print(f"  {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)")
    
    finally:
        diarizer.close()


# Run it
if __name__ == "__main__":
    asyncio.run(main())
"""