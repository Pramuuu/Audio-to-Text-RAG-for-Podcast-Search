"""
Advanced retriever for intelligent podcast search and retrieval.
Implements sophisticated retrieval strategies with semantic understanding and metadata filtering.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    content: str
    episode_id: str
    episode_title: str
    start_time: float
    end_time: float
    speaker: Optional[str]
    confidence: float
    relevance_score: float
    metadata: Dict[str, Any]

@dataclass
class SearchQuery:
    """Represents a search query with context."""
    query_text: str
    query_type: str  # "topic", "speaker", "temporal", "semantic"
    filters: Dict[str, Any]
    max_results: int
    similarity_threshold: float

class AdvancedRetriever:
    """
    Advanced retriever for intelligent podcast search and retrieval.
    
    Features:
    - Multi-vector retrieval with semantic understanding
    - Metadata-aware filtering
    - Temporal and speaker-based search
    - Ensemble retrieval strategies
    - Context-aware reranking
    """
    
    def __init__(self, vector_store: Chroma = None):
        """Initialize the advanced retriever."""
        self.vector_store = vector_store
        self.embeddings = None
        self.retriever = None
        self.ensemble_retriever = None
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        logger.info("Advanced Retriever initialized")
        
    def _initialize_embeddings(self):
        """Initialize embedding models."""
        try:
            # Primary embeddings (OpenAI)
            if config.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    api_key=config.OPENAI_API_KEY
                )
                logger.info("OpenAI embeddings initialized")
                
            # Note: Cohere embeddings via LangChain require the langchain-cohere package.
            # For now, require OpenAI key for embeddings.
                
            else:
                raise ValueError("No embedding API keys available")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
            
    def setup_vector_store(self, collection_name: str = None):
        """Setup and configure the vector store."""
        try:
            collection_name = collection_name or config.CHROMA_COLLECTION_NAME
            
            # Create or load vector store
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )
            
            # Base retriever via vector store
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.TOP_K_RETRIEVAL}
            )
            
            # Setup basic ensemble behavior using two search strategies
            self._setup_ensemble_retriever()
            
            logger.info(f"Vector store setup complete: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            raise
            
    def _setup_ensemble_retriever(self):
        """Setup ensemble retrieval strategies."""
        try:
            # Prepare two retrievers with different strategies; we'll merge results manually
            self._base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.TOP_K_RETRIEVAL}
            )
            self._mmr_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": config.TOP_K_RETRIEVAL,
                    "fetch_k": config.TOP_K_RETRIEVAL * 2,
                    "lambda_mult": 0.7,
                }
            )
            self.ensemble_retriever = True
            logger.info("Basic ensemble retriever configured (similarity + MMR)")
        except Exception as e:
            logger.error(f"Failed to setup ensemble retriever: {e}")
            
    def search_episodes(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform intelligent search across podcast episodes.
        
        Args:
            query: Search query with context and filters
            
        Returns:
            List of relevant search results
        """
        try:
            logger.info(f"Processing search query: {query.query_text}")
            
            # Apply query-specific retrieval strategy
            if query.query_type == "topic":
                results = self._topic_based_search(query)
            elif query.query_type == "speaker":
                results = self._speaker_based_search(query)
            elif query.query_type == "temporal":
                results = self._temporal_search(query)
            elif query.query_type == "semantic":
                results = self._semantic_search(query)
            else:
                results = self._general_search(query)
                
            # Apply filters
            results = self._apply_filters(results, query.filters)
            
            # Rerank results if enabled
            if config.RERANKING_ENABLED:
                results = self._rerank_results(results, query)
                
            # Limit results
            results = results[:query.max_results]
            
            # Convert to SearchResult objects
            search_results = self._convert_to_search_results(results)
            
            logger.info(f"Search completed: {len(search_results)} results found")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    def _topic_based_search(self, query: SearchQuery) -> List[Document]:
        """Perform topic-based search with semantic understanding."""
        try:
            # Use semantic search with topic expansion
            expanded_query = self._expand_topic_query(query.query_text)
            
            # Search with topic context
            results = self.vector_store.similarity_search_with_relevance_scores(
                expanded_query,
                k=query.max_results * 2  # Get more results for reranking
            )
            
            return [doc for doc, score in results if score >= query.similarity_threshold]
            
        except Exception as e:
            logger.error(f"Topic-based search failed: {e}")
            return []
            
    def _speaker_based_search(self, query: SearchQuery) -> List[Document]:
        """Perform speaker-specific search."""
        try:
            # Filter by speaker if specified
            if "speaker" in query.filters:
                speaker_filter = {"speaker": query.filters["speaker"]}
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query.query_text,
                    k=query.max_results,
                    filter=speaker_filter
                )
                return [doc for doc, score in results if score >= query.similarity_threshold]
            else:
                return self._general_search(query)
                
        except Exception as e:
            logger.error(f"Speaker-based search failed: {e}")
            return []
            
    def _temporal_search(self, query: SearchQuery) -> List[Document]:
        """Perform temporal search with time-based filtering."""
        try:
            # Apply temporal filters
            temporal_filter = {}
            if "start_time" in query.filters:
                temporal_filter["start_time"] = {"$gte": query.filters["start_time"]}
            if "end_time" in query.filters:
                temporal_filter["end_time"] = {"$lte": query.filters["end_time"]}
            if "episode_date" in query.filters:
                temporal_filter["episode_date"] = query.filters["episode_date"]
                
            # Search with temporal context
            results = self.vector_store.similarity_search_with_relevance_scores(
                query.query_text,
                k=query.max_results,
                filter=temporal_filter if temporal_filter else None
            )
            
            return [doc for doc, score in results if score >= query.similarity_threshold]
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return []
            
    def _semantic_search(self, query: SearchQuery) -> List[Document]:
        """Perform advanced semantic search."""
        try:
            # Use both retrievers and merge results uniquely
            if self.ensemble_retriever:
                results_a = self._base_retriever.get_relevant_documents(query.query_text)
                results_b = self._mmr_retriever.get_relevant_documents(query.query_text)
                # Deduplicate by metadata id if present, else by content
                seen = set()
                merged = []
                for doc in results_a + results_b:
                    key = doc.metadata.get("id") or doc.page_content[:200]
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
                return merged[: query.max_results * 2]
            else:
                return self._general_search(query)
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
            
    def _general_search(self, query: SearchQuery) -> List[Document]:
        """Perform general similarity search."""
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query.query_text,
                k=query.max_results
            )
            
            return [doc for doc, score in results if score >= query.similarity_threshold]
            
        except Exception as e:
            logger.error(f"General search failed: {e}")
            return []
            
    def _expand_topic_query(self, query_text: str) -> str:
        """Expand topic query with related concepts."""
        try:
            # Simple topic expansion (could be enhanced with LLM)
            topic_keywords = {
                "ai": "artificial intelligence machine learning",
                "ml": "machine learning artificial intelligence",
                "blockchain": "cryptocurrency bitcoin ethereum",
                "climate": "climate change global warming environment",
                "health": "healthcare medicine wellness",
                "finance": "investment banking economics",
                "tech": "technology innovation startup"
            }
            
            expanded_query = query_text.lower()
            for topic, expansion in topic_keywords.items():
                if topic in expanded_query:
                    expanded_query += " " + expansion
                    break
                    
            return expanded_query
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query_text
            
    def _apply_filters(self, results: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """Apply metadata filters to search results."""
        try:
            if not filters:
                return results
                
            filtered_results = []
            for doc in results:
                if self._document_matches_filters(doc, filters):
                    filtered_results.append(doc)
                    
            return filtered_results
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return results
            
    def _document_matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        try:
            metadata = doc.metadata
            
            for filter_key, filter_value in filters.items():
                if filter_key not in metadata:
                    continue
                    
                doc_value = metadata[filter_key]
                
                # Handle different filter types
                if isinstance(filter_value, dict):
                    # Range filters
                    if "$gte" in filter_value and doc_value < filter_value["$gte"]:
                        return False
                    if "$lte" in filter_value and doc_value > filter_value["$lte"]:
                        return False
                elif isinstance(filter_value, list):
                    # List filters
                    if doc_value not in filter_value:
                        return False
                else:
                    # Exact match filters
                    if doc_value != filter_value:
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Filter matching failed: {e}")
            return True  # Include document if filter check fails
            
    def _rerank_results(self, results: List[Document], query: SearchQuery) -> List[Document]:
        """Rerank results using advanced strategies."""
        try:
            if not results:
                return results
                
            # Calculate relevance scores
            scored_results = []
            for doc in results:
                score = self._calculate_relevance_score(doc, query)
                scored_results.append((doc, score))
                
            # Sort by relevance score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_results]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
            
    def _calculate_relevance_score(self, doc: Document, query: SearchQuery) -> float:
        """Calculate relevance score for a document."""
        try:
            base_score = 0.5
            
            # Content relevance
            query_words = set(query.query_text.lower().split())
            doc_words = set(doc.page_content.lower().split())
            word_overlap = len(query_words.intersection(doc_words))
            content_score = min(word_overlap / len(query_words), 1.0) if query_words else 0
            
            # Metadata relevance
            metadata_score = 0.0
            metadata = doc.metadata
            
            # Speaker relevance
            if "speaker" in query.filters and "speaker" in metadata:
                if query.filters["speaker"] == metadata["speaker"]:
                    metadata_score += 0.3
                    
            # Temporal relevance
            if "episode_date" in query.filters and "episode_date" in metadata:
                # Could implement date-based scoring here
                metadata_score += 0.1
                
            # Confidence relevance
            if "confidence" in metadata:
                confidence_score = metadata["confidence"] * 0.2
                metadata_score += confidence_score
                
            # Combine scores
            final_score = base_score + (content_score * 0.4) + (metadata_score * 0.1)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Relevance score calculation failed: {e}")
            return 0.5
            
    def _convert_to_search_results(self, documents: List[Document]) -> List[SearchResult]:
        """Convert documents to SearchResult objects."""
        try:
            search_results = []
            
            for doc in documents:
                metadata = doc.metadata
                
                result = SearchResult(
                    content=doc.page_content,
                    episode_id=metadata.get("episode_id", "unknown"),
                    episode_title=metadata.get("episode_title", "Unknown Episode"),
                    start_time=metadata.get("start_time", 0.0),
                    end_time=metadata.get("end_time", 0.0),
                    speaker=metadata.get("speaker", "Unknown Speaker"),
                    confidence=metadata.get("confidence", 0.0),
                    relevance_score=metadata.get("relevance_score", 0.0),
                    metadata=metadata
                )
                
                search_results.append(result)
                
            return search_results
            
        except Exception as e:
            logger.error(f"Document conversion failed: {e}")
            return []
            
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed episodes."""
        try:
            if not self.vector_store:
                return {}
                
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get metadata statistics
            metadata_stats = {}
            if count > 0:
                # Sample documents to analyze metadata
                sample_docs = collection.get(limit=min(100, count))
                
                if "metadatas" in sample_docs:
                    for metadata in sample_docs["metadatas"]:
                        for key, value in metadata.items():
                            if key not in metadata_stats:
                                metadata_stats[key] = {"values": set(), "count": 0}
                            metadata_stats[key]["values"].add(str(value))
                            metadata_stats[key]["count"] += 1
                            
            return {
                "total_documents": count,
                "metadata_fields": metadata_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get episode statistics: {e}")
            return {}
            
    def cleanup(self):
        """Clean up resources."""
        if self.vector_store:
            self.vector_store = None
            
        if self.embeddings:
            self.embeddings = None
            
        logger.info("Advanced Retriever cleaned up")

# Example usage
if __name__ == "__main__":
    retriever = AdvancedRetriever()
    
    # Example search
    # query = SearchQuery(
    #     query_text="artificial intelligence ethics",
    #     query_type="topic",
    #     filters={},
    #     max_results=10,
    #     similarity_threshold=0.7
    # )
    # results = retriever.search_episodes(query)
    # print(f"Found {len(results)} results")
    
    retriever.cleanup() 
            return [doc for doc, score in results if score >= query.similarity_threshold]

            

        except Exception as e:

            logger.error(f"Temporal search failed: {e}")

            return []

            

    def _semantic_search(self, query: SearchQuery) -> List[Document]:

        """Perform advanced semantic search."""

        try:

            # Use ensemble retriever for semantic search

            if self.ensemble_retriever:

                results = self.ensemble_retriever.get_relevant_documents(query.query_text)

                return results[:query.max_results]

            else:

                return self._general_search(query)

                

        except Exception as e:

            logger.error(f"Semantic search failed: {e}")

            return []

            

    def _general_search(self, query: SearchQuery) -> List[Document]:

        """Perform general similarity search."""

        try:

            results = self.vector_store.similarity_search_with_relevance_scores(

                query.query_text,

                k=query.max_results

            )

            

            return [doc for doc, score in results if score >= query.similarity_threshold]

            

        except Exception as e:

            logger.error(f"General search failed: {e}")

            return []

            

    def _expand_topic_query(self, query_text: str) -> str:

        """Expand topic query with related concepts."""

        try:

            # Simple topic expansion (could be enhanced with LLM)

            topic_keywords = {

                "ai": "artificial intelligence machine learning",

                "ml": "machine learning artificial intelligence",

                "blockchain": "cryptocurrency bitcoin ethereum",

                "climate": "climate change global warming environment",

                "health": "healthcare medicine wellness",

                "finance": "investment banking economics",

                "tech": "technology innovation startup"

            }

            

            expanded_query = query_text.lower()

            for topic, expansion in topic_keywords.items():

                if topic in expanded_query:

                    expanded_query += " " + expansion

                    break

                    

            return expanded_query

            

        except Exception as e:

            logger.error(f"Query expansion failed: {e}")

            return query_text

            

    def _apply_filters(self, results: List[Document], filters: Dict[str, Any]) -> List[Document]:

        """Apply metadata filters to search results."""

        try:

            if not filters:

                return results

                

            filtered_results = []

            for doc in results:

                if self._document_matches_filters(doc, filters):

                    filtered_results.append(doc)

                    

            return filtered_results

            

        except Exception as e:

            logger.error(f"Filter application failed: {e}")

            return results

            

    def _document_matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:

        """Check if a document matches the given filters."""

        try:

            metadata = doc.metadata

            

            for filter_key, filter_value in filters.items():

                if filter_key not in metadata:

                    continue

                    

                doc_value = metadata[filter_key]

                

                # Handle different filter types

                if isinstance(filter_value, dict):

                    # Range filters

                    if "$gte" in filter_value and doc_value < filter_value["$gte"]:

                        return False

                    if "$lte" in filter_value and doc_value > filter_value["$lte"]:

                        return False

                elif isinstance(filter_value, list):

                    # List filters

                    if doc_value not in filter_value:

                        return False

                else:

                    # Exact match filters

                    if doc_value != filter_value:

                        return False

                        

            return True

            

        except Exception as e:

            logger.error(f"Filter matching failed: {e}")

            return True  # Include document if filter check fails

            

    def _rerank_results(self, results: List[Document], query: SearchQuery) -> List[Document]:

        """Rerank results using advanced strategies."""

        try:

            if not results:

                return results

                

            # Calculate relevance scores

            scored_results = []

            for doc in results:

                score = self._calculate_relevance_score(doc, query)

                scored_results.append((doc, score))

                

            # Sort by relevance score

            scored_results.sort(key=lambda x: x[1], reverse=True)

            

            return [doc for doc, score in scored_results]

            

        except Exception as e:

            logger.error(f"Reranking failed: {e}")

            return results

            

    def _calculate_relevance_score(self, doc: Document, query: SearchQuery) -> float:

        """Calculate relevance score for a document."""

        try:

            base_score = 0.5

            

            # Content relevance

            query_words = set(query.query_text.lower().split())

            doc_words = set(doc.page_content.lower().split())

            word_overlap = len(query_words.intersection(doc_words))

            content_score = min(word_overlap / len(query_words), 1.0) if query_words else 0

            

            # Metadata relevance

            metadata_score = 0.0

            metadata = doc.metadata

            

            # Speaker relevance

            if "speaker" in query.filters and "speaker" in metadata:

                if query.filters["speaker"] == metadata["speaker"]:

                    metadata_score += 0.3

                    

            # Temporal relevance

            if "episode_date" in query.filters and "episode_date" in metadata:

                # Could implement date-based scoring here

                metadata_score += 0.1

                

            # Confidence relevance

            if "confidence" in metadata:

                confidence_score = metadata["confidence"] * 0.2

                metadata_score += confidence_score

                

            # Combine scores

            final_score = base_score + (content_score * 0.4) + (metadata_score * 0.1)

            

            return min(final_score, 1.0)

            

        except Exception as e:

            logger.error(f"Relevance score calculation failed: {e}")

            return 0.5

            

    def _convert_to_search_results(self, documents: List[Document]) -> List[SearchResult]:

        """Convert documents to SearchResult objects."""

        try:

            search_results = []

            

            for doc in documents:

                metadata = doc.metadata

                

                result = SearchResult(

                    content=doc.page_content,

                    episode_id=metadata.get("episode_id", "unknown"),

                    episode_title=metadata.get("episode_title", "Unknown Episode"),

                    start_time=metadata.get("start_time", 0.0),

                    end_time=metadata.get("end_time", 0.0),

                    speaker=metadata.get("speaker", "Unknown Speaker"),

                    confidence=metadata.get("confidence", 0.0),

                    relevance_score=metadata.get("relevance_score", 0.0),

                    metadata=metadata

                )

                

                search_results.append(result)

                

            return search_results

            

        except Exception as e:

            logger.error(f"Document conversion failed: {e}")

            return []

            

    def get_episode_statistics(self) -> Dict[str, Any]:

        """Get statistics about indexed episodes."""

        try:

            if not self.vector_store:

                return {}

                

            collection = self.vector_store._collection

            count = collection.count()

            

            # Get metadata statistics

            metadata_stats = {}

            if count > 0:

                # Sample documents to analyze metadata

                sample_docs = collection.get(limit=min(100, count))

                

                if "metadatas" in sample_docs:

                    for metadata in sample_docs["metadatas"]:

                        for key, value in metadata.items():

                            if key not in metadata_stats:

                                metadata_stats[key] = {"values": set(), "count": 0}

                            metadata_stats[key]["values"].add(str(value))

                            metadata_stats[key]["count"] += 1

                            

            return {

                "total_documents": count,

                "metadata_fields": metadata_stats

            }

            

        except Exception as e:

            logger.error(f"Failed to get episode statistics: {e}")

            return {}

            

    def cleanup(self):

        """Clean up resources."""

        if self.vector_store:

            self.vector_store = None

            

        if self.embeddings:

            self.embeddings = None

            

        logger.info("Advanced Retriever cleaned up")



# Example usage

if __name__ == "__main__":

    retriever = AdvancedRetriever()

    

    # Example search

    # query = SearchQuery(

    #     query_text="artificial intelligence ethics",

    #     query_type="topic",

    #     filters={},

    #     max_results=10,

    #     similarity_threshold=0.7

    # )

    # results = retriever.search_episodes(query)

    # print(f"Found {len(results)} results")

    

    retriever.cleanup() 
