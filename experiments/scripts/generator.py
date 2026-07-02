import json
import requests
import os


def generate_response(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    content_access: str = "content",
    verbose=False,
    seed=42,
    n_completions=1,
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
        "n": n_completions,
        "seed": seed,
    }

    print("Generating response") if verbose else None

    # Try to generate a response 4 times if an error occurs, otherwise skip the prompt
    for _ in range(4):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=300
            ).json()

            return [
                response["choices"][i]["message"][content_access]
                for i in range(n_completions)
            ]

        except Exception as e:
            str_error = e

    return "Failed to generate a response"


if __name__ == "__main__":

    response = generate_response(
        api_key=os.environ["NEBUL_API_KEY"],
        model="mistralai/Mistral-Large-3-675B-Instruct-2512",
        system_prompt="Je bent een behulpzame chatbot voor journalisten. Je antwoordt altijd in het Nederlands. Geef antwoord op de volgende vraag:",
        content_access="content",
        user_prompt="Wat kun je me vertellen over stichting SOS Dolfijn?",
        verbose=True,
        n_completions=4,
    )

    print(response)
