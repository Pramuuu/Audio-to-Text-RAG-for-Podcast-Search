"""
Advanced WhisperX processor for high-accuracy speech recognition with speaker diarization.
Handles audio preprocessing, transcription, and temporal indexing for podcast episodes.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import whisperx
from pydub import AudioSegment
import librosa

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Represents a transcribed audio segment with metadata."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 0.0
    language: str = "en"
    words: Optional[List[Dict[str, Any]]] = None

@dataclass
class EpisodeMetadata:
    """Metadata for a podcast episode."""
    episode_id: str
    title: str
    duration: float
    language: str
    speakers: List[str]
    transcript_path: str
    processed_at: str

class WhisperXProcessor:
    """
    Advanced WhisperX processor for podcast audio transcription.
    
    Features:
    - High-accuracy speech recognition
    - Speaker diarization
    - Word-level timestamp alignment
    - Multi-language support
    - Audio preprocessing and enhancement
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize the WhisperX processor."""
        self.model_name = model_name or config.WHISPERX_MODEL
        self.device = device or config.WHISPERX_DEVICE
        self.model = None
        self.align_model = None
        self.diarize_model = None
        
        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            
        logger.info(f"Initializing WhisperX with model: {self.model_name} on device: {self.device}")
        
    def load_models(self):
        """Load WhisperX models for transcription and alignment."""
        try:
            # Load main transcription model
            self.model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=config.WHISPERX_COMPUTE_TYPE,
                language="en"
            )
            
            # Load alignment model for word-level timestamps
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device
            )
            
            logger.info("WhisperX models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX models: {e}")
            raise
            
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for optimal transcription.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path to the preprocessed audio file
        """
        try:
            # Load audio with librosa for analysis
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Apply noise reduction if needed
            if self._needs_noise_reduction(audio):
                audio = self._apply_noise_reduction(audio, sr)
            
            # Resample to target sample rate if needed
            if sr != config.AUDIO_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.AUDIO_SAMPLE_RATE)
                sr = config.AUDIO_SAMPLE_RATE
            
            # Save preprocessed audio (use soundfile as librosa.output.write_wav is deprecated)
            preprocessed_path = audio_path.replace(".", "_preprocessed.")
            try:
                import soundfile as sf
                sf.write(preprocessed_path, audio, sr)
            except Exception:
                # Fallback to pydub if soundfile is not available
                temp_wav = AudioSegment(
                    (audio * 32767).astype(np.int16).tobytes(),
                    frame_rate=sr,
                    sample_width=2,
                    channels=1
                )
                temp_wav.export(preprocessed_path, format="wav")
            
            logger.info(f"Audio preprocessed and saved to: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_path  # Return original if preprocessing fails
            
    def _needs_noise_reduction(self, audio: np.ndarray) -> bool:
        """Determine if audio needs noise reduction."""
        # Simple heuristic based on signal-to-noise ratio
        signal_power = np.mean(audio**2)
        noise_power = np.var(audio)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr < 15  # Threshold for noise reduction
        
    def _apply_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply basic noise reduction using spectral gating."""
        # Simple spectral gating noise reduction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise from first few frames
        noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Apply spectral gating
        gate_threshold = 2.0
        gain = np.maximum(1 - gate_threshold * noise_estimate / (magnitude + 1e-10), 0.1)
        magnitude_filtered = magnitude * gain
        
        # Reconstruct audio
        stft_filtered = magnitude_filtered * np.exp(1j * np.angle(stft))
        audio_filtered = librosa.istft(stft_filtered)
        
        return audio_filtered
        
    def transcribe_episode(self, audio_path: str, episode_id: str) -> Tuple[List[TranscriptionSegment], EpisodeMetadata]:
        """
        Transcribe a podcast episode with full metadata.
        
        Args:
            audio_path: Path to the audio file
            episode_id: Unique identifier for the episode
            
        Returns:
            Tuple of transcription segments and episode metadata
        """
        try:
            # Ensure models are loaded
            if self.model is None:
                self.load_models()
                
            # Preprocess audio
            preprocessed_path = self.preprocess_audio(audio_path)
            
            # Transcribe audio
            logger.info(f"Starting transcription of episode: {episode_id}")
            result = self.model.transcribe(
                preprocessed_path,
                batch_size=config.WHISPERX_BATCH_SIZE,
                language="en"
            )
            
            # Align timestamps to word level
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                preprocessed_path,
                self.device,
                return_char_alignments=False
            )
            
            # Extract segments with metadata
            segments = []
            for segment in result["segments"]:
                # Create transcription segment
                trans_segment = TranscriptionSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("avg_logprob", 0.0),
                    language=result.get("language", "en"),
                    words=segment.get("words", [])
                )
                segments.append(trans_segment)
            
            # Create episode metadata
            metadata = EpisodeMetadata(
                episode_id=episode_id,
                title=Path(audio_path).stem,
                duration=segments[-1].end if segments else 0.0,
                language=result.get("language", "en"),
                speakers=[],  # Will be populated by speaker diarization
                transcript_path=f"{episode_id}_transcript.json",
                processed_at=str(np.datetime64('now'))
            )
            
            # Save transcript
            self._save_transcript(segments, metadata, episode_id)
            
            logger.info(f"Transcription completed for episode: {episode_id}")
            return segments, metadata
            
        except Exception as e:
            logger.error(f"Transcription failed for episode {episode_id}: {e}")
            raise
            
    def _save_transcript(self, segments: List[TranscriptionSegment], metadata: EpisodeMetadata, episode_id: str):
        """Save transcript to JSON file."""
        try:
            transcript_data = {
                "metadata": {
                    "episode_id": metadata.episode_id,
                    "title": metadata.title,
                    "duration": metadata.duration,
                    "language": metadata.language,
                    "speakers": metadata.speakers,
                    "processed_at": metadata.processed_at
                },
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "speaker": seg.speaker,
                        "confidence": seg.confidence,
                        "language": seg.language,
                        "words": seg.words
                    }
                    for seg in segments
                ]
            }
            
            transcript_path = config.DATA_DIR / f"{episode_id}_transcript.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Transcript saved to: {transcript_path}")
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            
    def get_episode_summary(self, segments: List[TranscriptionSegment]) -> Dict[str, Any]:
        """Generate a summary of the episode from transcription segments."""
        try:
            total_duration = segments[-1].end if segments else 0.0
            total_words = sum(len(seg.text.split()) for seg in segments)
            avg_confidence = np.mean([seg.confidence for seg in segments])
            
            # Extract key topics (simple keyword extraction)
            all_text = " ".join([seg.text for seg in segments])
            words = all_text.lower().split()
            
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            content_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Simple frequency-based topic extraction
            from collections import Counter
            word_freq = Counter(content_words)
            top_topics = [word for word, freq in word_freq.most_common(10)]
            
            summary = {
                "total_duration": total_duration,
                "total_segments": len(segments),
                "total_words": total_words,
                "average_confidence": avg_confidence,
                "top_topics": top_topics,
                "speaking_rate": total_words / (total_duration / 60) if total_duration > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate episode summary: {e}")
            return {}
            
    def cleanup(self):
        """Clean up loaded models to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("WhisperX models cleaned up")

# Example usage
if __name__ == "__main__":
    processor = WhisperXProcessor()
    processor.load_models()
    
    # Example transcription
    # segments, metadata = processor.transcribe_episode("sample_podcast.mp3", "episode_001")
    # summary = processor.get_episode_summary(segments)
    # print(f"Episode Summary: {summary}")
    
    processor.cleanup() 