"""
Database management package for the RAG Podcast Search system.
Handles ChromaDB integration, data storage, and retrieval operations.
"""

from .chroma_manager import ChromaDBManager

__all__ = [
    "ChromaDBManager"
] 