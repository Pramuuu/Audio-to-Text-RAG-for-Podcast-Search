"""
Tests for the audio processor components.
Tests WhisperX processor, speaker diarization, and audio enhancement.
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.audio_processor.whisperx_processor import (
    WhisperXProcessor, 
    TranscriptionSegment, 
    EpisodeMetadata
)
from src.audio_processor.speaker_diarizer import (
    SpeakerDiarizer, 
    SpeakerSegment
)

class TestTranscriptionSegment:
    """Test TranscriptionSegment dataclass."""
    
    def test_transcription_segment_creation(self):
        """Test creating a TranscriptionSegment."""
        segment = TranscriptionSegment(
            start=0.0,
            end=10.0,
            text="Hello world",
            speaker="Speaker 1",
            confidence=0.95,
            language="en"
        )
        
        assert segment.start == 0.0
        assert segment.end == 10.0
        assert segment.text == "Hello world"
        assert segment.speaker == "Speaker 1"
        assert segment.confidence == 0.95
        assert segment.language == "en"
        
    def test_transcription_segment_defaults(self):
        """Test TranscriptionSegment with default values."""
        segment = TranscriptionSegment(
            start=5.0,
            end=15.0,
            text="Test text"
        )
        
        assert segment.start == 5.0
        assert segment.end == 15.0
        assert segment.text == "Test text"
        assert segment.speaker is None
        assert segment.confidence == 0.0
        assert segment.language == "en"

class TestEpisodeMetadata:
    """Test EpisodeMetadata dataclass."""
    
    def test_episode_metadata_creation(self):
        """Test creating an EpisodeMetadata."""
        metadata = EpisodeMetadata(
            episode_id="ep_001",
            title="Test Episode",
            duration=1800.0,
            language="en",
            speakers=["Speaker 1", "Speaker 2"],
            transcript_path="transcript.json",
            processed_at="2024-01-01"
        )
        
        assert metadata.episode_id == "ep_001"
        assert metadata.title == "Test Episode"
        assert metadata.duration == 1800.0
        assert metadata.language == "en"
        assert metadata.speakers == ["Speaker 1", "Speaker 2"]
        assert metadata.transcript_path == "transcript.json"
        assert metadata.processed_at == "2024-01-01"

class TestWhisperXProcessor:
    """Test WhisperX processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a WhisperX processor instance."""
        with patch('src.audio_processor.whisperx_processor.config') as mock_config:
            mock_config.WHISPERX_MODEL = "base"
            mock_config.WHISPERX_DEVICE = "cpu"
            mock_config.WHISPERX_COMPUTE_TYPE = "float32"
            mock_config.WHISPERX_BATCH_SIZE = 8
            mock_config.AUDIO_SAMPLE_RATE = 16000
            mock_config.DATA_DIR = Path("/tmp/test_data")
            
            # Create processor
            processor = WhisperXProcessor()
            return processor
            
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.model_name == "base"
        assert processor.device == "cpu"
        assert processor.model is None
        assert processor.align_model is None
        
    @patch('torch.cuda.is_available')
    def test_cuda_fallback(self, mock_cuda_available, processor):
        """Test CUDA fallback to CPU."""
        mock_cuda_available.return_value = False
        
        # Reinitialize with CUDA device
        processor.device = "cuda"
        processor._validate_device()
        
        assert processor.device == "cpu"
        
    @patch('whisperx.load_model')
    @patch('whisperx.load_align_model')
    def test_load_models(self, mock_load_align, mock_load_model, processor):
        """Test loading WhisperX models."""
        # Mock model loading
        mock_load_model.return_value = Mock()
        mock_load_align.return_value = (Mock(), Mock())
        
        processor.load_models()
        
        assert processor.model is not None
        assert processor.align_model is not None
        mock_load_model.assert_called_once()
        mock_load_align.assert_called_once()
        
    def test_needs_noise_reduction(self, processor):
        """Test noise reduction detection."""
        # Create test audio with low SNR
        low_snr_audio = np.random.normal(0, 0.1, 1000)  # Low signal power
        
        needs_reduction = processor._needs_noise_reduction(low_snr_audio)
        assert needs_reduction is True
        
        # Create test audio with high SNR
        high_snr_audio = np.random.normal(0, 1.0, 1000) + 2.0  # High signal power
        
        needs_reduction = processor._needs_noise_reduction(high_snr_audio)
        assert needs_reduction is False
        
    def test_apply_noise_reduction(self, processor):
        """Test noise reduction application."""
        # Create test audio
        test_audio = np.random.normal(0, 1.0, 1000)
        sr = 16000
        
        # Apply noise reduction
        filtered_audio = processor._apply_noise_reduction(test_audio, sr)
        
        # Check that output is numpy array
        assert isinstance(filtered_audio, np.ndarray)
        assert len(filtered_audio) == len(test_audio)
        
    @patch('src.audio_processor.whisperx_processor.WhisperXProcessor.load_models')
    @patch('src.audio_processor.whisperx_processor.WhisperXProcessor.preprocess_audio')
    @patch('whisperx.align')
    def test_transcribe_episode(self, mock_align, mock_preprocess, mock_load, processor):
        """Test episode transcription."""
        # Mock preprocessing
        mock_preprocess.return_value = "/tmp/test_audio.wav"
        
        # Mock transcription result
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "Hello world",
                    "avg_logprob": 0.9
                }
            ],
            "language": "en"
        }
        
        # Mock model
        processor.model = Mock()
        processor.model.transcribe.return_value = mock_result
        
        # Mock alignment
        mock_align.return_value = mock_result
        
        # Test transcription
        segments, metadata = processor.transcribe_episode("/tmp/test.mp3", "ep_001")
        
        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.0
        assert segments[0].end == 10.0
        assert metadata.episode_id == "ep_001"
        
    def test_get_episode_summary(self, processor):
        """Test episode summary generation."""
        # Create test segments
        segments = [
            TranscriptionSegment(0.0, 10.0, "Hello world", confidence=0.9),
            TranscriptionSegment(10.0, 20.0, "How are you", confidence=0.8),
            TranscriptionSegment(20.0, 30.0, "I am fine", confidence=0.95)
        ]
        
        summary = processor.get_episode_summary(segments)
        
        assert summary["total_duration"] == 30.0
        assert summary["total_segments"] == 3
        assert summary["total_words"] == 6
        assert summary["average_confidence"] == pytest.approx(0.883, rel=1e-2)
        
    def test_cleanup(self, processor):
        """Test processor cleanup."""
        # Mock models
        processor.model = Mock()
        processor.align_model = Mock()
        
        processor.cleanup()
        
        assert processor.model is None
        assert processor.align_model is None

class TestSpeakerDiarizer:
    """Test speaker diarization functionality."""
    
    @pytest.fixture
    def diarizer(self):
        """Create a speaker diarizer instance."""
        with patch('src.audio_processor.speaker_diarizer.os.getenv') as mock_getenv:
            mock_getenv.return_value = "test_token"
            
            diarizer = SpeakerDiarizer()
            return diarizer
            
    def test_diarizer_initialization(self, diarizer):
        """Test diarizer initialization."""
        assert diarizer.auth_token == "test_token"
        assert diarizer.pipeline is None
        
    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda_available, diarizer):
        """Test device selection logic."""
        mock_cuda_available.return_value = True
        assert diarizer.device == "cuda"
        
        mock_cuda_available.return_value = False
        diarizer.device = "cuda"
        diarizer._validate_device()
        assert diarizer.device == "cpu"
        
    def test_merge_overlapping_segments(self, diarizer):
        """Test merging of overlapping speaker segments."""
        segments = [
            SpeakerSegment(0.0, 10.0, "Speaker 1", 0.9),
            SpeakerSegment(8.0, 18.0, "Speaker 2", 0.8),  # Overlaps with first
            SpeakerSegment(20.0, 30.0, "Speaker 1", 0.95)
        ]
        
        merged = diarizer._merge_overlapping_segments(segments)
        
        # Should have 2 segments after merging
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 18.0  # Merged end time
        assert merged[1].start == 20.0
        assert merged[1].end == 30.0
        
    def test_assign_speaker_labels(self, diarizer):
        """Test speaker label assignment."""
        segments = [
            SpeakerSegment(0.0, 10.0, "SPEAKER_00", 0.9),
            SpeakerSegment(10.0, 20.0, "SPEAKER_01", 0.8),
            SpeakerSegment(20.0, 30.0, "SPEAKER_00", 0.95)
        ]
        
        labeled = diarizer._assign_speaker_labels(segments)
        
        # Check that labels are assigned consistently
        assert labeled[0].speaker == "Speaker 1"
        assert labeled[1].speaker == "Speaker 2"
        assert labeled[2].speaker == "Speaker 1"
        
    def test_analyze_speaker_patterns(self, diarizer):
        """Test speaker pattern analysis."""
        segments = [
            SpeakerSegment(0.0, 10.0, "Speaker 1", 0.9),
            SpeakerSegment(10.0, 20.0, "Speaker 2", 0.8),
            SpeakerSegment(20.0, 30.0, "Speaker 1", 0.95)
        ]
        
        patterns = diarizer.analyze_speaker_patterns(segments)
        
        assert "Speaker 1" in patterns
        assert "Speaker 2" in patterns
        assert patterns["Speaker 1"].total_duration == 20.0
        assert patterns["Speaker 2"].total_duration == 10.0
        
    def test_cleanup(self, diarizer):
        """Test diarizer cleanup."""
        # Mock pipeline
        diarizer.pipeline = Mock()
        
        diarizer.cleanup()
        
        assert diarizer.pipeline is None

# Integration tests
class TestAudioProcessingIntegration:
    """Integration tests for audio processing pipeline."""
    
    @pytest.fixture
    def setup_components(self):
        """Setup test components."""
        with patch('src.audio_processor.whisperx_processor.config') as mock_config, \
             patch('src.audio_processor.speaker_diarizer.os.getenv') as mock_getenv:
            
            mock_config.WHISPERX_MODEL = "base"
            mock_config.WHISPERX_DEVICE = "cpu"
            mock_config.DATA_DIR = Path("/tmp/test_data")
            mock_getenv.return_value = "test_token"
            
            processor = WhisperXProcessor()
            diarizer = SpeakerDiarizer()
            
            return processor, diarizer
            
    def test_full_processing_pipeline(self, setup_components):
        """Test the complete audio processing pipeline."""
        processor, diarizer = setup_components
        
        # This would test the full integration
        # For now, just verify components can be created together
        assert processor is not None
        assert diarizer is not None

if __name__ == "__main__":
    pytest.main([__file__]) 