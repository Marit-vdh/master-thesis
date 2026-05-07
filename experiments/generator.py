import json
import requests


def generate_response(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 1.0,
    max_completion_tokens: int = 100,
) -> str:
    """Generate a response using a model from the Nebul platform.

    Args:
        api_key (str): API key for access to the Nebul platform.
        model (str): model to use for text generation. Must be available through Nebul.
        system_prompt (str): system prompt for guidance in the text generation.
        user_prompt (str): user prompt to base text generation on.

    Returns:
        str: generated text.
    """

    url = "https://api.inference.nebul.io/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }

    print("Generating response")

    response = requests.post(url, headers=headers, json=payload).json()
    return response["choices"][0]["message"]["reasoning"]
