import json
import requests
import os


def generate_response(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 1.0,
    content_access: str = "content",
    verbose=False,
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
    }

    print("Generating response") if verbose else None

    # Try to generate a response 4 times if an error occurs, otherwise skip the prompt
    for _ in range(4):
        try:
            response = requests.post(url, headers=headers, json=payload).json()
            return response["choices"][0]["message"][content_access]

        except Exception as e:
            str_error = e

    return "Failed to generate a response"


if __name__ == "__main__":
    from analyse_results import analyse_with_reference

    directory = "../results/generation_only"

    # Create directory to store the responses
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate responses based on prompts from the prompt file
    prompts = []
    with open("../sources/prompts.csv", "r") as file:
        for line in file:
            split_line = line.split(",")
            prompts.append((split_line[0], split_line[1].strip()))

    # Load the references from the reference file
    references = []
    with open("../sources/references.txt", "r") as file:
        for line in file:
            references.append(line.strip())

    # Get all generative models from the generative file
    generators = []
    with open("../sources/generators.csv", "r") as file:
        for line in file:
            split_line = line.split(",")
            generators.append((split_line[0], split_line[1].strip()))

    # Generate responses using each of the generators
    for generator in generators:

        generator_name = generator[0].split("/")[1]
        generator_directory = f"{directory}/{generator_name}"

        # Create subdirectory to store the responses per generator
        if not os.path.exists(generator_directory):
            os.makedirs(generator_directory)

        # Empty the directory for a generator if it already exists
        for filename in os.listdir(generator_directory):
            file_path = os.path.join(generator_directory, filename)

            # Check if it is a file (not a subdirectory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file

        # Generate responses for each of the prompts
        for i in range(len(prompts)):
            prompt = prompts[i]

            response = generate_response(
                api_key=os.environ["NEBUL_API_KEY"],
                model=generator[0],
                system_prompt="Je bent een behulpzame chatbot voor journalisten. Je antwoordt altijd in het Nederlands. Geef antwoord op de volgende vraag:",
                content_access=generator[1],
                user_prompt=prompt[1],
                verbose=True,
            )

            # Calculate the relevance score of the answer
            scores = analyse_with_reference(
                prompt=prompt[1], reference=references[i], response=response
            )

            output_filename = f"{generator_directory}/prompt_{i}_response.json"

            with open(output_filename, "w") as output_file:
                output = {}
                output["generator"] = generator_name
                output["scores"] = scores
                output["user_prompt"] = prompt[0]
                output["prompt_type"] = prompt[1]
                output["response"] = response
                json.dump(output, output_file)
