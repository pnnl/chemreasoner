"""Functions to handle the openai llm interface."""
import asyncio
import logging
import os
from typing import Union, Optional

import openai
from dotenv import load_dotenv

logging.getLogger().setLevel(logging.INFO)

global openai_client
openai_client = None


def init_openai(api_key_env="OPENAI_API_KEY_DEV", base_url=None, dotenv_path=None):
    """Initialize connection to OpenAI or compatible API.
    
    Args:
        api_key_env: Environment variable name containing the API key
        base_url: Optional base URL for custom endpoints
        dotenv_path: Optional path to .env file
    """
    global openai_client
    
    # Load environment variables if dotenv_path is provided
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
    
    # Initialize client with custom settings if provided
    if openai_client is None:
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        
        openai_client = openai.AsyncOpenAI(
            api_key=os.getenv(api_key_env),
            **kwargs
        )


async def parallel_openai_chat_completion(
    prompt, system_prompt=None, model="gpt-3.5-turbo", **kwargs
):
    """Run chat completion calls on openai, in parallel."""
    global openai_client
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return await openai_client.chat.completions.create(
            messages=messages, model=model, **kwargs
        )
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        raise e


async def openai_chat_async_evaluation(
    prompts, system_prompts, model="gpt-3.5-turbo", **kwargs
):
    completions = [
        parallel_openai_chat_completion(p, system_prompts[i], model=model, **kwargs)
        for i, p in enumerate(prompts)
    ]

    answers = await asyncio.gather(*completions)
    return answers


def run_openai_prompts(
    prompts: list[str],
    system_prompts: list[Union[str, None]] = None,
    model="gpt-3.5-turbo",
    api_key_env="OPENAI_API_KEY_DEV",
    base_url=None,
    dotenv_path=None,
    **kwargs
):
    """Run the given prompts with the openai interface.
    
    Args:
        prompts: List of prompts to send to the API
        system_prompts: Optional list of system prompts
        model: Model name to use
        api_key_env: Environment variable containing the API key
        base_url: Optional base URL for custom endpoints
        dotenv_path: Optional path to .env file
        **kwargs: Additional parameters for the API call
    """
    init_openai(api_key_env=api_key_env, base_url=base_url, dotenv_path=dotenv_path)
    
    # Apply defaults to kwargs
    kwargs["temperature"] = kwargs.get("temperature", 0.6)
    kwargs["top_p"] = kwargs.get("top_p", 0.3)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1300)

    if system_prompts is None:
        system_prompts = [None] * len(prompts)

    answer_objects = asyncio.run(
        openai_chat_async_evaluation(
            prompts,
            system_prompts=system_prompts,
            model=model,
            **kwargs,
        )
    )
    answer_strings = [a.choices[0].message.content for a in answer_objects]
    usages = [
        {
            "completion_tokens": a.usage.completion_tokens,
            "prompt_tokens": a.usage.prompt_tokens,
        }
        for a in answer_objects
    ]

    return [{"answer": a, "usage": u} for a, u in zip(answer_strings, usages)]


class OpenAIInterface:
    """A class to handle communicating with OpenAI or compatible APIs."""

    def __init__(
        self, 
        dotenv_path: str = ".env", 
        model: str = "gpt-3.5-turbo",
        api_key_env: str = "OPENAI_API_KEY_DEV",
        base_url: Optional[str] = None
    ):
        """Load the client for the given dotenv path.
        
        Args:
            dotenv_path: Path to .env file
            model: Model name to use
            api_key_env: Environment variable containing the API key
            base_url: Optional base URL for custom endpoints
        """
        self.dotenv_path = dotenv_path
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url

    def __call__(
        self,
        prompts: list[str],
        system_prompts: list[Union[str, None]] = None,
        **kwargs,
    ):
        """Run the given prompts with the OpenAI interface."""
        return run_openai_prompts(
            prompts=prompts,
            system_prompts=system_prompts,
            model=self.model,
            api_key_env=self.api_key_env,
            base_url=self.base_url,
            dotenv_path=self.dotenv_path,
            **kwargs
        )


# Default implementation for backward compatibility
class StandardOpenAIInterface(OpenAIInterface):
    """Standard OpenAI API implementation for backward compatibility."""
    
    def __init__(self, dotenv_path: str = ".env", model: str = "gpt-3.5-turbo"):
        super().__init__(dotenv_path=dotenv_path, model=model)


# For the PNNL custom endpoint
class PNNLOpenAIInterface(OpenAIInterface):
    """PNNL custom API implementation."""
    
    def __init__(self, dotenv_path: str = ".env", model: str = None):
        # Load environment variables first to get model and base_url
        load_dotenv(dotenv_path=dotenv_path, override=True)
        
        # Use environment variables with fallbacks
        model = model or os.getenv("CUSTOM_OPENAI_MODEL", "gpt-4o-birthright")
        base_url = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://ai-incubator-api.pnnl.gov")
        
        super().__init__(
            dotenv_path=dotenv_path, 
            model=model,
            api_key_env="CUSTOM_OPENAI_API_KEY",
            base_url=base_url
        )


_test_prompt = (
    "What are the top-3 catalysts that perform the hydrodeoxygenation reaction and demonstrate higher adsorption energy for acetate?. You should include candidate catalysts with the following properties: high conversion. Provide scientific explanations for each of the catalysts. Finally, return a python list named final_answer which contains the top-5 catalysts."
    "Take a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
)
_test_system_prompt = (
    "You are a helpful chemistry expert with extensive knowledge of "
    "catalysis. You will give recommendations for catalysts, including "
    "chemically accurate descriptions of the interaction between the catalysts "
    "and adsorbate(s). Make specific recommendations for catalysts, including "
    "their chemical composition. Make sure to follow the formatting "
    "instructions. Do not provide disclaimers or notes about your knowledge of "
    "catalysis."
)

if __name__ == "__main__":
    print(run_openai_prompts([_test_prompt] * 20, [_test_system_prompt] * 20))
