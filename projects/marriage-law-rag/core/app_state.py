#!/usr/bin/env python3
"""
Application State Management - Handles startup and shutdown logic
"""

import logging
from utils.postgres_vector_store import PostgresVectorStore
from flows import create_offline_indexing_flow, create_online_query_flow

logger = logging.getLogger(__name__)


async def startup_handler(app, settings):
    """
    Initialize the application on startup
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    logger.info("Starting Marriage Law RAG System...")
    
    try:
        # Initialize database connection
        app.state.vector_store = PostgresVectorStore(settings.database_url)
        await app.state.vector_store.connect()
        await app.state.vector_store.create_tables()
        
        # Initialize PocketFlow workflows
        app.state.offline_flow = create_offline_indexing_flow()
        app.state.online_flow = create_online_query_flow()
        
        logger.info("✅ Application initialized successfully")
        logger.info(f"Database URL: {settings.database_url[:20]}...")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise


async def shutdown_handler(app):
    """
    Cleanup on application shutdown
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Shutting down Marriage Law RAG System...")
    
    try:
        # Close database connections
        if hasattr(app.state, 'vector_store') and app.state.vector_store:
            await app.state.vector_store.close()
            logger.info("Database connections closed")
        
        # Clean up other resources if needed
        if hasattr(app.state, 'offline_flow'):
            app.state.offline_flow = None
        
        if hasattr(app.state, 'online_flow'):
            app.state.online_flow = None
        
        logger.info("✅ Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        # Don't re-raise during shutdown to avoid additional errors