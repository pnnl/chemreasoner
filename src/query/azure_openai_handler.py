"""
This module contains the AzureOpenAIHandler class for interacting with the Azure OpenAI service.
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from logging_utils import setup_logging
from typing import List, Dict

from logging_utils import LogManager

# Load the logging configuration
if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_azure_handler.log", 
        log_config_path="src/query/logging_config.ini"
        )

logger = LogManager.get_logger("azure_handler")

class AzureOpenAIHandler:
    """
    A class for handling interactions with the Azure OpenAI service.

    This class manages the authentication and communication with Azure OpenAI,
    allowing for easy generation of responses using the specified model.

    Attributes:
        openai_api_endpoint (str): The Azure OpenAI API endpoint.
        openai_api_key (str): The API key for authentication.
        openai_api_version (str): The API version to use.
        openai_api_deployment_name (str): The deployment name of the model to use.
        client (AzureOpenAI): The Azure OpenAI client instance.
    """

    def __init__(self, env_path: str) -> None:
        """
        Initialize the Azure OpenAI client credentials.

        Args:
            env_path (str): Path to the .env file containing the Azure OpenAI credentials.

        Raises:
            ValueError: If required environment variables are not set.
        """
        # Load the environment variables from .env file
        load_dotenv(env_path)

        # Load the environment variables
        self.openai_api_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        self.openai_api_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([self.openai_api_endpoint, self.openai_api_key, self.openai_api_version, self.openai_api_deployment_name]):
            logger.error("Missing required environment variables for Azure OpenAI")
            raise ValueError(
                "Please ensure all required environment variables are set: "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME"
            )
        
        # Set up the OpenAI API client with Azure-specific settings
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.openai_api_endpoint,
                api_key=self.openai_api_key, 
                api_version=self.openai_api_version,
            )
            logger.info("AzureOpenAI client is created successfully!")
        except Exception as e:
            logger.error(f"Failed to create AzureOpenAI client: {str(e)}")
            raise

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the Azure OpenAI service.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.
                Each dictionary should have 'role' and 'content' keys.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If there's an error in generating the response.
        """
        try:
            logger.debug(f"Sending request to Azure OpenAI. Deployment: {self.openai_api_deployment_name}")
            response = self.client.chat.completions.create(
                model=self.openai_api_deployment_name,
                messages=messages
            )
            response_str = response.choices[0].message.content.strip() 
            logger.debug(f"Generated response: {response_str[:50]}...")  # Log first 50 characters
            return response_str
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"API Endpoint: {self.openai_api_endpoint}")
            logger.error(f"API Version: {self.openai_api_version}")
            logger.error(f"Deployment Name: {self.openai_api_deployment_name}")
            raise

# Example usage
if __name__ == "__main__":
    env_path = ".env"  # Replace with the path to your .env file
    
    try:
        azure_handler = AzureOpenAIHandler(env_path)
        
        # Example conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in catalysis."},
            {"role": "user", "content": "What are the advantages of using platinum as a catalyst?"}
        ]
        
        response = azure_handler.generate_response(messages)
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred. Please check the logs for more information.")