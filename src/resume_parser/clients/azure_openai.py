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

                # Try creating client with minimal parameters first
                self._sync_client = AzureOpenAI(
                    api_key=settings.azure_openai.api_key,
                    api_version=settings.azure_openai.api_version,
                    azure_endpoint=settings.azure_openai.endpoint
                )
                logger.info("Synchronous Azure OpenAI client created successfully")
            except TypeError as e:
                if "proxies" in str(e):
                    # Handle proxies parameter issue with older httpx versions
                    logger.warning(f"Proxies parameter issue: {e}. Trying alternative initialization...")
                    try:
                        import httpx
                        # Create a basic httpx client without proxy configuration
                        http_client = httpx.Client(timeout=30.0)
                        self._sync_client = AzureOpenAI(
                            api_key=settings.azure_openai.api_key,
                            api_version=settings.azure_openai.api_version,
                            azure_endpoint=settings.azure_openai.endpoint,
                            http_client=http_client
                        )
                        logger.info("Synchronous Azure OpenAI client created with custom http client")
                    except Exception as e2:
                        logger.error(f"Failed alternative Azure OpenAI client creation: {e2}")
                        self._sync_client = None  # Set to None so we can continue without LLM
                else:
                    logger.error(f"Failed to create synchronous Azure OpenAI client: {e}")
                    self._sync_client = None
            except Exception as e:
                logger.error(f"Failed to create synchronous Azure OpenAI client: {e}")
                self._sync_client = None

        return self._sync_client

    def get_async_client(self) -> AsyncAzureOpenAI:
        """
        Creates and returns an asynchronous client for Azure OpenAI services.

        Returns:
            AsyncAzureOpenAI: An instance of the asynchronous AzureOpenAI client.
        """
        if self._async_client is None:
            try:
                if not all([settings.azure_openai.api_key, settings.azure_openai.endpoint]):
                    raise ValueError("Azure OpenAI API key and endpoint are required")

                # Try creating client with minimal parameters first
                self._async_client = AsyncAzureOpenAI(
                    api_key=settings.azure_openai.api_key,
                    api_version=settings.azure_openai.api_version,
                    azure_endpoint=settings.azure_openai.endpoint,
                    timeout=30.0
                )
                logger.info("Asynchronous Azure OpenAI client created successfully")
            except TypeError as e:
                if "proxies" in str(e):
                    # Handle proxies parameter issue with older httpx versions
                    logger.warning(f"Proxies parameter issue: {e}. Trying alternative initialization...")
                    try:
                        import httpx
                        # Create a basic async httpx client without proxy configuration
                        http_client = httpx.AsyncClient(timeout=30.0)
                        self._async_client = AsyncAzureOpenAI(
                            api_key=settings.azure_openai.api_key,
                            api_version=settings.azure_openai.api_version,
                            azure_endpoint=settings.azure_openai.endpoint,
                            http_client=http_client
                        )
                        logger.info("Asynchronous Azure OpenAI client created with custom http client")
                    except Exception as e2:
                        logger.error(f"Failed alternative async Azure OpenAI client creation: {e2}")
                        self._async_client = None  # Set to None so we can continue without LLM
                else:
                    logger.error(f"Failed to create asynchronous Azure OpenAI client: {e}")
                    self._async_client = None
            except Exception as e:
                logger.error(f"Failed to create asynchronous Azure OpenAI client: {e}")
                self._async_client = None

        return self._async_client

    def get_chat_deployment(self) -> str:
        """Get the chat deployment name."""
        return settings.azure_openai.chat_deployment

    def get_embedding_deployment(self) -> str:
        """Get the embedding deployment name."""
        return settings.azure_openai.embedding_deployment


# Global client instance
azure_client = AzureOpenAIClient()