"""
Advanced speaker diarization for podcast episodes.
Identifies and separates different speakers with high accuracy using state-of-the-art models.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
try:
    # Lazy import to avoid hard dependency during initial import
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
except Exception:  # pragma: no cover
    Pipeline = None
    ProgressHook = None
import whisperx

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and confidence."""
    start: float
    end: float
    speaker: str
    confidence: float
    text: Optional[str] = None

@dataclass
class SpeakerInfo:
    """Information about a specific speaker."""
    speaker_id: str
    total_duration: float
    segments_count: int
    average_confidence: float
    dominant_language: str
    speaking_patterns: Dict[str, Any]

class SpeakerDiarizer:
    """
    Advanced speaker diarization system for podcast episodes.
    
    Features:
    - Multi-speaker identification
    - Speaker clustering and labeling
    - Confidence scoring
    - Speaker pattern analysis
    - Integration with transcription
    """
    
    def __init__(self, auth_token: str = None):
        """Initialize the speaker diarization system."""
        self.auth_token = auth_token or os.getenv("HUGGINGFACE_TOKEN")
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not self.auth_token:
            logger.warning("HuggingFace token not provided. Speaker diarization may not work.")
            
        logger.info(f"Initializing Speaker Diarizer on device: {self.device}")
        
    def load_pipeline(self):
        """Load the speaker diarization pipeline."""
        try:
            if not self.auth_token:
                raise ValueError("HuggingFace token required for speaker diarization")
            if Pipeline is None:
                raise ImportError(
                    "pyannote.audio is not installed. Install it to enable speaker diarization."
                )
                
            self.pipeline = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL,
                use_auth_token=self.auth_token
            )
            
            # Move to appropriate device
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                
            logger.info("Speaker diarization pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise
            
    def diarize_episode(self, audio_path: str, episode_id: str) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on a podcast episode.
        
        Args:
            audio_path: Path to the audio file
            episode_id: Unique identifier for the episode
            
        Returns:
            List of speaker segments with timing information
        """
        try:
            if self.pipeline is None:
                self.load_pipeline()
                
            logger.info(f"Starting speaker diarization for episode: {episode_id}")
            
            # Perform diarization
            if ProgressHook is not None:
                with ProgressHook() as hook:
                    diarization = self.pipeline(
                        audio_path,
                        hook=hook,
                        min_speakers=config.DIARIZATION_MIN_SPEAKERS,
                        max_speakers=config.DIARIZATION_MAX_SPEAKERS
                    )
            else:
                diarization = self.pipeline(
                    audio_path,
                    min_speakers=config.DIARIZATION_MIN_SPEAKERS,
                    max_speakers=config.DIARIZATION_MAX_SPEAKERS
                )
            
            # Extract speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # Pyannote doesn't provide confidence scores
                )
                segments.append(segment)
                
            # Sort segments by start time
            segments.sort(key=lambda x: x.start)
            
            # Merge overlapping segments
            segments = self._merge_overlapping_segments(segments)
            
            # Assign speaker labels
            segments = self._assign_speaker_labels(segments)
            
            # Save diarization results
            self._save_diarization_results(segments, episode_id)
            
            logger.info(f"Speaker diarization completed for episode: {episode_id}")
            logger.info(f"Identified {len(set(seg.speaker for seg in segments))} speakers")
            
            return segments
            
        except Exception as e:
            logger.error(f"Speaker diarization failed for episode {episode_id}: {e}")
            raise
            
    def _merge_overlapping_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge overlapping speaker segments to avoid conflicts."""
        if not segments:
            return segments
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if current.end >= next_seg.start:
                # Overlapping segments, merge them
                current.end = max(current.end, next_seg.end)
                # Keep the speaker with longer duration
                if (next_seg.end - next_seg.start) > (current.end - current.start):
                    current.speaker = next_seg.speaker
            else:
                merged.append(current)
                current = next_seg
                
        merged.append(current)
        return merged
        
    def _assign_speaker_labels(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Assign meaningful speaker labels (e.g., Speaker 1, Speaker 2)."""
        speaker_mapping = {}
        speaker_counter = 1
        
        for segment in segments:
            if segment.speaker not in speaker_mapping:
                speaker_mapping[segment.speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            segment.speaker = speaker_mapping[segment.speaker]
            
        return segments
        
    def _save_diarization_results(self, segments: List[SpeakerSegment], episode_id: str):
        """Save diarization results to JSON file."""
        try:
            diarization_data = {
                "episode_id": episode_id,
                "speakers": list(set(seg.speaker for seg in segments)),
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "speaker": seg.speaker,
                        "confidence": seg.confidence,
                        "text": seg.text
                    }
                    for seg in segments
                ]
            }
            
            diarization_path = config.DATA_DIR / f"{episode_id}_diarization.json"
            with open(diarization_path, 'w', encoding='utf-8') as f:
                json.dump(diarization_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Diarization results saved to: {diarization_path}")
            
        except Exception as e:
            logger.error(f"Failed to save diarization results: {e}")
            
    def integrate_with_transcription(self, 
                                   diarization_segments: List[SpeakerSegment],
                                   transcription_segments: List[Any]) -> List[Any]:
        """
        Integrate speaker diarization with transcription segments.
        
        Args:
            diarization_segments: Speaker diarization results
            transcription_segments: WhisperX transcription segments
            
        Returns:
            Transcription segments with speaker information
        """
        try:
            # Create a mapping from time to speaker
            speaker_map = {}
            for seg in diarization_segments:
                # Map each time point to a speaker
                for t in np.arange(seg.start, seg.end, 0.1):  # 100ms intervals
                    speaker_map[round(t, 1)] = seg.speaker
                    
            # Assign speakers to transcription segments
            for trans_seg in transcription_segments:
                # Find the most common speaker for this segment
                segment_speakers = []
                for t in np.arange(trans_seg.start, trans_seg.end, 0.1):
                    t_rounded = round(t, 1)
                    if t_rounded in speaker_map:
                        segment_speakers.append(speaker_map[t_rounded])
                        
                if segment_speakers:
                    # Assign the most common speaker
                    from collections import Counter
                    most_common_speaker = Counter(segment_speakers).most_common(1)[0][0]
                    trans_seg.speaker = most_common_speaker
                    
            logger.info("Speaker diarization integrated with transcription")
            return transcription_segments
            
        except Exception as e:
            logger.error(f"Failed to integrate diarization with transcription: {e}")
            return transcription_segments
            
    def analyze_speaker_patterns(self, segments: List[SpeakerSegment]) -> Dict[str, SpeakerInfo]:
        """Analyze speaking patterns for each speaker."""
        try:
            speaker_stats = {}
            
            for segment in segments:
                speaker_id = segment.speaker
                duration = segment.end - segment.start
                
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        "total_duration": 0.0,
                        "segments_count": 0,
                        "confidence_scores": [],
                        "speaking_times": []
                    }
                    
                speaker_stats[speaker_id]["total_duration"] += duration
                speaker_stats[speaker_id]["segments_count"] += 1
                speaker_stats[speaker_id]["confidence_scores"].append(segment.confidence)
                speaker_stats[speaker_id]["speaking_times"].append(segment.start)
                
            # Convert to SpeakerInfo objects
            speaker_info = {}
            for speaker_id, stats in speaker_stats.items():
                avg_confidence = np.mean(stats["confidence_scores"])
                
                # Analyze speaking patterns
                speaking_patterns = {
                    "average_segment_duration": stats["total_duration"] / stats["segments_count"],
                    "speaking_frequency": len(stats["speaking_times"]) / (stats["total_duration"] / 60),
                    "time_distribution": self._analyze_time_distribution(stats["speaking_times"])
                }
                
                speaker_info[speaker_id] = SpeakerInfo(
                    speaker_id=speaker_id,
                    total_duration=stats["total_duration"],
                    segments_count=stats["segments_count"],
                    average_confidence=avg_confidence,
                    dominant_language="en",  # Could be enhanced with language detection
                    speaking_patterns=speaking_patterns
                )
                
            return speaker_info
            
        except Exception as e:
            logger.error(f"Failed to analyze speaker patterns: {e}")
            return {}
            
    def _analyze_time_distribution(self, speaking_times: List[float]) -> Dict[str, Any]:
        """Analyze the temporal distribution of speaking times."""
        try:
            if not speaking_times:
                return {}
                
            # Calculate intervals between speaking times
            intervals = np.diff(sorted(speaking_times))
            
            # Analyze patterns
            analysis = {
                "total_speaking_instances": len(speaking_times),
                "average_interval": np.mean(intervals) if len(intervals) > 0 else 0,
                "median_interval": np.median(intervals) if len(intervals) > 0 else 0,
                "speaking_clusters": self._identify_speaking_clusters(speaking_times)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze time distribution: {e}")
            return {}
            
    def _identify_speaking_clusters(self, speaking_times: List[float], 
                                  cluster_threshold: float = 30.0) -> List[Dict[str, Any]]:
        """Identify clusters of speaking activity."""
        try:
            if not speaking_times:
                return []
                
            sorted_times = sorted(speaking_times)
            clusters = []
            current_cluster = [sorted_times[0]]
            
            for time in sorted_times[1:]:
                if time - current_cluster[-1] <= cluster_threshold:
                    current_cluster.append(time)
                else:
                    # End current cluster
                    if len(current_cluster) > 1:
                        clusters.append({
                            "start": current_cluster[0],
                            "end": current_cluster[-1],
                            "duration": current_cluster[-1] - current_cluster[0],
                            "speaking_instances": len(current_cluster)
                        })
                    current_cluster = [time]
                    
            # Add final cluster
            if len(current_cluster) > 1:
                clusters.append({
                    "start": current_cluster[0],
                    "end": current_cluster[-1],
                    "duration": current_cluster[-1] - current_cluster[0],
                    "speaking_instances": len(current_cluster)
                })
                
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to identify speaking clusters: {e}")
            return []
            
    def get_speaker_summary(self, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """Generate a comprehensive summary of speaker analysis."""
        try:
            speaker_info = self.analyze_speaker_patterns(segments)
            
            total_duration = sum(seg.end - seg.start for seg in segments)
            unique_speakers = len(set(seg.speaker for seg in segments))
            
            # Calculate overall statistics
            summary = {
                "total_duration": total_duration,
                "unique_speakers": unique_speakers,
                "total_segments": len(segments),
                "speaker_details": {},
                "conversation_flow": self._analyze_conversation_flow(segments)
            }
            
            # Add individual speaker details
            for speaker_id, info in speaker_info.items():
                summary["speaker_details"][speaker_id] = {
                    "total_duration": info.total_duration,
                    "percentage_of_total": (info.total_duration / total_duration) * 100,
                    "segments_count": info.segments_count,
                    "average_confidence": info.average_confidence,
                    "speaking_patterns": info.speaking_patterns
                }
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate speaker summary: {e}")
            return {}
            
    def _analyze_conversation_flow(self, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """Analyze the flow and dynamics of conversation."""
        try:
            if not segments:
                return {}
                
            # Analyze turn-taking patterns
            speaker_transitions = []
            for i in range(len(segments) - 1):
                current_speaker = segments[i].speaker
                next_speaker = segments[i + 1].speaker
                if current_speaker != next_speaker:
                    speaker_transitions.append((current_speaker, next_speaker))
                    
            # Calculate transition probabilities
            transition_counts = {}
            for from_speaker, to_speaker in speaker_transitions:
                key = f"{from_speaker}_to_{to_speaker}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
                
            # Analyze conversation dynamics
            flow_analysis = {
                "total_speaker_changes": len(speaker_transitions),
                "transition_patterns": transition_counts,
                "conversation_turns": len(segments),
                "average_segment_duration": np.mean([seg.end - seg.start for seg in segments])
            }
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation flow: {e}")
            return {}
            
    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Speaker diarization resources cleaned up")

# Example usage
if __name__ == "__main__":
    diarizer = SpeakerDiarizer()
    
    # Example diarization
    # segments = diarizer.diarize_episode("sample_podcast.mp3", "episode_001")
    # summary = diarizer.get_speaker_summary(segments)
    # print(f"Speaker Analysis: {summary}")
    
    diarizer.cleanup() 