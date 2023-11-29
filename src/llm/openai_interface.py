"""Functions to handle the openai llm interface."""
import asyncio
import os

from typing import Union

import openai

global openai_client
openai_client = None


def init_openai():
    """Initialize connection to OpenAI."""
    global openai_client
    if openai_client is None:
        openai_client = openai.AsyncOpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY_DEV")


async def parallel_openai_text_completion(
    prompt, system_prompt=None, model="text-davinci-003", **kwargs
):
    """Run chat completion calls on openai, in parallel."""
    global openai_client
    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "Say this is a test"})
    print(kwargs["max_tokens"])
    return await openai_client.chat.completions.create(
        messages=messages, model=model, **kwargs
    )


async def openai_text_async_evaluation(
    prompts, system_prompts, model="text-davinci-003", **kwargs
):
    completions = [
        parallel_openai_text_completion(p, system_prompts[i])
        for i, p in enumerate(prompts)
    ]

    answers = await asyncio.gather(*completions)
    return answers


async def parallel_openai_chat_completion(
    prompt, system_prompt=None, model="gpt-3.5-turbo", **kwargs
):
    """Run chat completion calls on openai, in parallel."""
    global openai_client
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "Say this is a test"})
    return await openai_client.chat.completions.create(
        messages=messages, model=model, **kwargs
    )


async def openai_chat_async_evaluation(
    prompts, system_prompts, model="gpt-3.5-turbo", **kwargs
):
    completions = [
        parallel_openai_chat_completion(p, system_prompts[i], **kwargs)
        for i, p in enumerate(prompts)
    ]

    answers = await asyncio.gather(*completions)
    return answers


def run_openai_prompts(
    prompts: list[str],
    system_prompts: list[Union[str, None]] = None,
    model="gpt-3.5-turbo",
    **kwargs
):
    """Run the given prompts with the openai interface."""
    init_openai()
    # Apply defaults to kwargs
    kwargs["temperature"] = kwargs.get("temperature", 0.6)
    kwargs["top_p"] = kwargs.get("top_p", 0.3)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1300)

    if system_prompts is None:
        system_prompts = [None] * len(prompts)

    if model == "text-davinci-003":
        pass

    elif "gpt-3.5" in model or "gpt-4" in model:
        answer_objects = asyncio.run(
            openai_chat_async_evaluation(
                prompts,
                system_prompts=system_prompts,
                model=model,
                **kwargs,
            )
        )
        answer_strings = [a.choices[0].message.content for a in answer_objects]
        # usages = [[a.choices[0].message.content for a in answer_objects]]
        return answer_strings  # [{"answer": a, "usage": u} for a, u in zip(answer_strings, usages)]


if __name__ == "__main__":
    print(run_openai_prompts(["test1", "test2"]))
