"""
Azure OpenAI client management.

This module provides functionalities to create and manage clients for interacting with Azure OpenAI services.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from openai import AzureOpenAI, AsyncAzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Singleton class for managing Azure OpenAI clients.

    Provides methods to create and retrieve clients for interacting with Azure OpenAI services.
    """

    _instance = None
    _sync_client = None
    _async_client = None

    def __new__(cls):
        """Ensures that only one instance of the class is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(AzureOpenAIClient, cls).__new__(cls)
        return cls._instance

    def get_sync_client(self) -> AzureOpenAI:
        """
        Creates and returns a synchronous client for Azure OpenAI services.

        Returns:
            AzureOpenAI: An instance of the synchronous AzureOpenAI client.
        """
        if self._sync_client is None:
            try:
                if not all([settings.azure_openai.api_key, settings.azure_openai.endpoint]):
                    raise ValueError("Azure OpenAI API key and endpoint are required")

                self._sync_client = AzureOpenAI(
                    api_key=settings.azure_openai.api_key,
                    api_version=settings.azure_openai.api_version,
                    azure_endpoint=settings.azure_openai.endpoint
                )
                logger.info("Synchronous Azure OpenAI client created successfully")
            except Exception as e:
                logger.error(f"Failed to create synchronous Azure OpenAI client: {e}")
                raise

        return self._sync_client

    def get_async_client(self) -> AsyncAzureOpenAI:
        """
        Creates and returns an asynchronous client for Azure OpenAI services.

        Returns:
            AsyncAzureOpenAI: An instance of the asynchronous AzureOpenAI client.
        """
        if self._async_client is None:
            try:
                self._async_client = AsyncAzureOpenAI(
                    api_key=settings.azure_openai.api_key,
                    api_version=settings.azure_openai.api_version,
                    azure_endpoint=settings.azure_openai.endpoint,
                    timeout=30.0
                )
                logger.info("Asynchronous Azure OpenAI client created successfully")
            except Exception as e:
                logger.error(f"Failed to create asynchronous Azure OpenAI client: {e}")
                raise

        return self._async_client

    def get_chat_deployment(self) -> str:
        """Get the chat deployment name."""
        return settings.azure_openai.chat_deployment

    def get_embedding_deployment(self) -> str:
        """Get the embedding deployment name."""
        return settings.azure_openai.embedding_deployment


# Global client instance
azure_client = AzureOpenAIClient()