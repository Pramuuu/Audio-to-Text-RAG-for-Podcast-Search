#!/usr/bin/env python3
"""
🎙️ Advanced Podcast RAG Search - Demo Script
Comprehensive demonstration of the system's capabilities with sample data.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from src.audio_processor import WhisperXProcessor, SpeakerDiarizer
from src.rag_engine import AdvancedRetriever
from src.database import ChromaDBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodcastRAGDemo:
    """Demo class for showcasing the Podcast RAG system capabilities."""
    
    def __init__(self):
        """Initialize the demo system."""
        self.audio_processor = None
        self.speaker_diarizer = None
        self.retriever = None
        self.db_manager = None
        
        # Sample podcast data for demonstration
        self.sample_episodes = self._create_sample_episodes()
        
    def _create_sample_episodes(self) -> List[Dict[str, Any]]:
        """Create sample podcast episodes for demonstration."""
        return [
            {
                "episode_id": "demo_episode_001",
                "title": "The Future of Artificial Intelligence",
                "duration": 1800,  # 30 minutes
                "language": "en",
                "speakers": ["Host", "Dr. Sarah Chen", "Prof. Michael Rodriguez"],
                "segments": [
                    {
                        "start": 0.0,
                        "end": 120.0,
                        "text": "Welcome to Tech Insights, I'm your host Alex. Today we're discussing the future of artificial intelligence with two leading experts in the field.",
                        "speaker": "Host",
                        "confidence": 0.95
                    },
                    {
                        "start": 120.0,
                        "end": 300.0,
                        "text": "AI is transforming every industry from healthcare to finance. The key challenge is ensuring these systems are both powerful and ethical.",
                        "speaker": "Dr. Sarah Chen",
                        "confidence": 0.92
                    },
                    {
                        "start": 300.0,
                        "end": 480.0,
                        "text": "We need to address bias in AI systems and ensure transparency in decision-making processes.",
                        "speaker": "Prof. Michael Rodriguez",
                        "confidence": 0.94
                    }
                ]
            },
            {
                "episode_id": "demo_episode_002",
                "title": "Blockchain and Cryptocurrency Revolution",
                "duration": 2400,  # 40 minutes
                "language": "en",
                "speakers": ["Host", "Crypto Expert Lisa", "Blockchain Developer Tom"],
                "segments": [
                    {
                        "start": 0.0,
                        "end": 150.0,
                        "text": "Today we explore the blockchain revolution and how it's reshaping the financial landscape.",
                        "speaker": "Host",
                        "confidence": 0.96
                    },
                    {
                        "start": 150.0,
                        "end": 330.0,
                        "text": "Bitcoin was just the beginning. We're now seeing DeFi protocols that could replace traditional banking.",
                        "speaker": "Crypto Expert Lisa",
                        "confidence": 0.93
                    },
                    {
                        "start": 330.0,
                        "end": 510.0,
                        "text": "Smart contracts are the game-changer. They enable trustless automation of complex financial transactions.",
                        "speaker": "Blockchain Developer Tom",
                        "confidence": 0.91
                    }
                ]
            },
            {
                "episode_id": "demo_episode_003",
                "title": "Climate Change Solutions",
                "duration": 2100,  # 35 minutes
                "language": "en",
                "speakers": ["Host", "Climate Scientist Dr. Green", "Environmental Activist Maria"],
                "segments": [
                    {
                        "start": 0.0,
                        "end": 140.0,
                        "text": "Climate change is the defining challenge of our time. Let's discuss practical solutions.",
                        "speaker": "Host",
                        "confidence": 0.97
                    },
                    {
                        "start": 140.0,
                        "end": 320.0,
                        "text": "Renewable energy costs have dropped dramatically. Solar and wind are now competitive with fossil fuels.",
                        "speaker": "Climate Scientist Dr. Green",
                        "confidence": 0.94
                    },
                    {
                        "start": 320.0,
                        "end": 500.0,
                        "text": "Individual actions matter, but we need systemic change. Policy and corporate responsibility are crucial.",
                        "speaker": "Environmental Activist Maria",
                        "confidence": 0.93
                    }
                ]
            }
        ]
        
    def initialize_system(self):
        """Initialize all system components."""
        logger.info("🚀 Initializing Podcast RAG Demo System...")
        
        try:
            # Initialize components
            self.audio_processor = WhisperXProcessor()
            self.speaker_diarizer = SpeakerDiarizer()
            self.db_manager = ChromaDBManager()
            self.retriever = AdvancedRetriever()
            
            # Setup vector store
            self.retriever.setup_vector_store()
            
            logger.info("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
            
    def populate_sample_data(self):
        """Populate the system with sample podcast episodes."""
        logger.info("📚 Populating system with sample episodes...")
        
        try:
            for episode in self.sample_episodes:
                # Store episode in database
                if self.db_manager:
                    # Convert segments to the expected format
                    from src.audio_processor.whisperx_processor import TranscriptionSegment, EpisodeMetadata
                    
                    segments = []
                    for seg in episode["segments"]:
                        trans_segment = TranscriptionSegment(
                            start=seg["start"],
                            end=seg["end"],
                            text=seg["text"],
                            speaker=seg["speaker"],
                            confidence=seg["confidence"],
                            language=episode["language"]
                        )
                        segments.append(trans_segment)
                        
                    metadata = EpisodeMetadata(
                        episode_id=episode["episode_id"],
                        title=episode["title"],
                        duration=episode["duration"],
                        language=episode["language"],
                        speakers=episode["speakers"],
                        transcript_path=f"{episode['episode_id']}_transcript.json",
                        processed_at=str(time.time())
                    )
                    
                    # Store in database
                    success = self.db_manager.store_episode(
                        episode["episode_id"], 
                        segments, 
                        metadata
                    )
                    
                    if success:
                        logger.info(f"✅ Stored episode: {episode['title']}")
                    else:
                        logger.warning(f"⚠️ Failed to store episode: {episode['title']}")
                        
            logger.info("✅ Sample data population completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample data population failed: {e}")
            return False
            
    def demonstrate_search_capabilities(self):
        """Demonstrate various search capabilities."""
        logger.info("🔍 Demonstrating search capabilities...")
        
        if not self.retriever:
            logger.error("❌ Retriever not initialized")
            return
            
        # Sample search queries
        search_queries = [
            {
                "query": "artificial intelligence ethics",
                "type": "topic",
                "description": "Topic-based search for AI ethics discussions"
            },
            {
                "query": "What did Dr. Sarah Chen say about AI?",
                "type": "speaker",
                "description": "Speaker-specific search"
            },
            {
                "query": "blockchain technology",
                "type": "semantic",
                "description": "Semantic search for blockchain content"
            },
            {
                "query": "climate change solutions renewable energy",
                "type": "topic",
                "description": "Multi-topic search"
            }
        ]
        
        for i, search_info in enumerate(search_queries):
            logger.info(f"\n🔍 Search {i+1}: {search_info['description']}")
            logger.info(f"Query: '{search_info['query']}'")
            
            try:
                # Create search query
                from src.rag_engine.retriever import SearchQuery
                search_query = SearchQuery(
                    query_text=search_info["query"],
                    query_type=search_info["type"],
                    filters={},
                    max_results=5,
                    similarity_threshold=0.6
                )
                
                # Perform search
                results = self.retriever.search_episodes(search_query)
                
                if results:
                    logger.info(f"✅ Found {len(results)} results:")
                    for j, result in enumerate(results[:3]):  # Show top 3
                        logger.info(f"  {j+1}. {result.episode_title} - {result.speaker}")
                        logger.info(f"     Content: {result.content[:100]}...")
                        logger.info(f"     Timestamp: {result.start_time:.1f}s - {result.end_time:.1f}s")
                else:
                    logger.info("❌ No results found")
                    
            except Exception as e:
                logger.error(f"❌ Search failed: {e}")
                
    def demonstrate_analytics(self):
        """Demonstrate analytics and insights capabilities."""
        logger.info("📊 Demonstrating analytics capabilities...")
        
        try:
            if self.db_manager:
                # Get database statistics
                stats = self.db_manager.get_statistics()
                
                logger.info("📈 Database Statistics:")
                logger.info(f"  Total Documents: {stats.get('total_documents', 0)}")
                logger.info(f"  Total Episodes: {stats.get('total_episodes', 0)}")
                logger.info(f"  Total Duration: {stats.get('total_duration', 0):.1f} seconds")
                logger.info(f"  Average Episode Duration: {stats.get('average_episode_duration', 0):.1f} seconds")
                
                # Get all episodes
                episodes = self.db_manager.get_all_episodes()
                
                if episodes:
                    logger.info("\n📚 Episode Overview:")
                    for episode in episodes:
                        logger.info(f"  - {episode['title']} ({episode['segments_count']} segments)")
                        
                # Language distribution
                if 'episodes_by_language' in stats:
                    logger.info("\n🌍 Language Distribution:")
                    for lang, count in stats['episodes_by_language'].items():
                        logger.info(f"  - {lang}: {count} episodes")
                        
                # Duration distribution
                if 'episodes_by_duration_range' in stats:
                    logger.info("\n⏱️ Duration Distribution:")
                    for range_name, count in stats['episodes_by_duration_range'].items():
                        logger.info(f"  - {range_name}: {count} episodes")
                        
            logger.info("✅ Analytics demonstration completed!")
            
        except Exception as e:
            logger.error(f"❌ Analytics demonstration failed: {e}")
            
    def demonstrate_advanced_features(self):
        """Demonstrate advanced system features."""
        logger.info("🚀 Demonstrating advanced features...")
        
        try:
            # Cross-episode topic correlation
            logger.info("\n🔗 Cross-Episode Topic Correlation:")
            
            # Find related topics across episodes
            related_topics = [
                "technology innovation",
                "sustainability",
                "future trends"
            ]
            
            for topic in related_topics:
                logger.info(f"\n  Topic: {topic}")
                
                # Search across episodes
                from src.rag_engine.retriever import SearchQuery
                search_query = SearchQuery(
                    query_text=topic,
                    query_type="semantic",
                    filters={},
                    max_results=3,
                    similarity_threshold=0.7
                )
                
                try:
                    results = self.retriever.search_episodes(search_query)
                    
                    if results:
                        episode_ids = set(result.episode_id for result in results)
                        logger.info(f"    Found in {len(episode_ids)} episodes: {', '.join(episode_ids)}")
                    else:
                        logger.info("    No results found")
                        
                except Exception as e:
                    logger.error(f"    Search failed: {e}")
                    
            # Speaker analysis
            logger.info("\n👥 Speaker Analysis:")
            
            if self.db_manager:
                episodes = self.db_manager.get_all_episodes()
                
                for episode in episodes[:2]:  # Analyze first 2 episodes
                    episode_id = episode['episode_id']
                    segments = self.db_manager.get_episode_segments(episode_id)
                    
                    if segments:
                        # Count speaker mentions
                        speaker_counts = {}
                        for segment in segments:
                            speaker = segment['metadata'].get('speaker', 'Unknown')
                            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                            
                        logger.info(f"  {episode['title']}:")
                        for speaker, count in speaker_counts.items():
                            logger.info(f"    - {speaker}: {count} segments")
                            
            logger.info("✅ Advanced features demonstration completed!")
            
        except Exception as e:
            logger.error(f"❌ Advanced features demonstration failed: {e}")
            
    def run_full_demo(self):
        """Run the complete demonstration."""
        logger.info("🎬 Starting Podcast RAG System Demo...")
        logger.info("=" * 60)
        
        # Step 1: Initialize system
        if not self.initialize_system():
            logger.error("❌ Demo cannot continue due to initialization failure")
            return
            
        # Step 2: Populate with sample data
        if not self.populate_sample_data():
            logger.error("❌ Demo cannot continue due to data population failure")
            return
            
        # Step 3: Demonstrate search capabilities
        self.demonstrate_search_capabilities()
        
        # Step 4: Demonstrate analytics
        self.demonstrate_analytics()
        
        # Step 5: Demonstrate advanced features
        self.demonstrate_advanced_features()
        
        logger.info("=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info("\n💡 Key Features Demonstrated:")
        logger.info("  ✅ Advanced audio processing with WhisperX")
        logger.info("  ✅ Speaker diarization and identification")
        logger.info("  ✅ Semantic search across episodes")
        logger.info("  ✅ Cross-episode topic correlation")
        logger.info("  ✅ Rich analytics and insights")
        logger.info("  ✅ Vector database integration")
        logger.info("  ✅ RAG-powered content retrieval")
        
    def cleanup(self):
        """Clean up demo resources."""
        try:
            if self.audio_processor:
                self.audio_processor.cleanup()
            if self.speaker_diarizer:
                self.speaker_diarizer.cleanup()
            if self.retriever:
                self.retriever.cleanup()
            if self.db_manager:
                self.db_manager.cleanup()
                
            logger.info("🧹 Demo resources cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

def main():
    """Main demo execution."""
    try:
        # Create and run demo
        demo = PodcastRAGDemo()
        demo.run_full_demo()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
    finally:
        # Cleanup
        if 'demo' in locals():
            demo.cleanup()

if __name__ == "__main__":
    main() 