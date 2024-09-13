import os
from openai import AzureOpenAI
from dotenv import load_dotenv

import logging
console = logging.StreamHandler()
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)


class AzureOpenAIHandler():
    def __init__(self, env_path : str) -> None:
        """
        Initialize the AZURE client credentials.

        input: env_path - path to the .env variable
        """
        # Load the environment variables from .env file
        load_dotenv(env_path)

        # Load the environment variables
        self.openai_api_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        self.openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        self.openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", None)
        self.openai_api_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None)
        

        if self.openai_api_endpoint is None or self.openai_api_key is None:
            raise ValueError(
                "Please set the AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY Environment Variables"
                " or pass values to openai_api_endpoint and openai_api_key parameters"
            )
        
        # Set up the OpenAI API client with Azure-specific settings
        self.client = AzureOpenAI(
            azure_endpoint = self.openai_api_endpoint,
            api_key = self.openai_api_key, 
            api_version = self.openai_api_version,
        )
        
        logger.info("AzureOpenAI client is created successfully !!")

        return
    
    def generate_response(self, messages : str):
        response = self.client.chat.completions.create(
            model=self.openai_api_deployment_name,
            messages=messages
        )
        response_str = response.choices[0].message.content.strip() 
        return response_str