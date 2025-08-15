"""
Configuration management for the Audio-to-Text RAG Podcast Search system.
Centralized configuration for all components including API keys, model parameters, and system settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management for the RAG system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "cache"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # OpenAI Configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # Cohere Configuration
    COHERE_MODEL = os.getenv("COHERE_MODEL", "command")
    COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    
    # Audio Processing Configuration
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHUNK_DURATION = float(os.getenv("AUDIO_CHUNK_DURATION", "30.0"))  # seconds
    AUDIO_OVERLAP = float(os.getenv("AUDIO_OVERLAP", "2.0"))  # seconds
    
    # WhisperX Configuration
    WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v2")
    WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu")
    WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
    WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
    
    # Speaker Diarization
    DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    DIARIZATION_MIN_SPEAKERS = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "1"))
    DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "10"))
    
    # Text Processing Configuration
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    
    # Embedding Configuration
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = str(CACHE_DIR / "chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "podcast_episodes")
    CHROMA_DISTANCE_FUNCTION = os.getenv("CHROMA_DISTANCE_FUNCTION", "cosine")
    
    # RAG Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
    
    # Evaluation Configuration
    RAGAS_METRICS = [
        "answer_relevancy",
        "context_relevancy", 
        "faithfulness",
        "answer_correctness"
    ]
    
    # Streamlit Configuration
    STREAMLIT_PAGE_TITLE = "🎙️ Advanced Podcast RAG Search"
    STREAMLIT_PAGE_ICON = "🎙️"
    STREAMLIT_LAYOUT = "wide"
    STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"
    
    # File Upload Configuration
    ALLOWED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    
    # Caching Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues."""
        issues = {}
        
        # Check required API keys
        if not cls.OPENAI_API_KEY:
            issues["OPENAI_API_KEY"] = "OpenAI API key is required"
        if not cls.COHERE_API_KEY:
            issues["COHERE_API_KEY"] = "Cohere API key is required"
            
        # Check CUDA availability
        if cls.WHISPERX_DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues["CUDA"] = "CUDA requested but not available"
            except ImportError:
                issues["CUDA"] = "PyTorch not available for CUDA check"
                
        return issues
    
    @classmethod
    def get_audio_config(cls) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return {
            "sample_rate": cls.AUDIO_SAMPLE_RATE,
            "chunk_duration": cls.AUDIO_CHUNK_DURATION,
            "overlap": cls.AUDIO_OVERLAP,
            "whisperx_model": cls.WHISPERX_MODEL,
            "device": cls.WHISPERX_DEVICE,
            "batch_size": cls.WHISPERX_BATCH_SIZE,
            "compute_type": cls.WHISPERX_COMPUTE_TYPE
        }
    
    @classmethod
    def get_rag_config(cls) -> Dict[str, Any]:
        """Get RAG system configuration."""
        return {
            "top_k": cls.TOP_K_RETRIEVAL,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
            "reranking_enabled": cls.RERANKING_ENABLED,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "embedding_dimension": cls.EMBEDDING_DIMENSION
        }
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "persist_directory": cls.CHROMA_PERSIST_DIRECTORY,
            "collection_name": cls.CHROMA_COLLECTION_NAME,
            "distance_function": cls.CHROMA_DISTANCE_FUNCTION
        }

# Global configuration instance
config = Config()

# Configuration validation on import
if __name__ == "__main__":
    issues = config.validate_config()
    if issues:
        print("Configuration issues found:")
        for key, issue in issues.items():
            print(f"  {key}: {issue}")
    else:
        print("Configuration validated successfully!") 