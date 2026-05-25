import os
import json
from tqdm import tqdm
from pathlib import Path

from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import cosine_similarity, retrieve
from generator import generate_response

full_directory_name = (
    "/Users/maritvandenhelder/information_studies/thesis/master-thesis/experiments"
)


class Experiment:

    def __init__(
        self,
        api_key: str,
        embedder: str,
        generator: str,
        test: bool = True,
        verbose=False,
        embed_documents: bool = True,
        embedding_file: str = "data.tsv",
        system_prompt_setup: str = "You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: ",
    ):
        # Set up global variables
        self.api_key = api_key
        self.embedder = embedder
        self.generator = generator
        self.ids, self.texts = import_articles(flatten=True, test=test)
        self.system_prompt_setup = system_prompt_setup

        print("Loading embeddings") if verbose else None

        if embed_documents:

            # Embed texts and load into a dictionary
            embed(
                model=self.embedder,
                input_articles=self.texts,
                api_key=self.api_key,
                write=True,
                output_file=embedding_file,
                max_articles=50,
                verbose=verbose,
            )

        self.embeddings = load_embeddings(
            embedding_file=embedding_file, indices=self.ids, verbose=verbose
        )

    def request_prompt(
        self, manual_prompt: bool = False, automatic: str = "Amsterdam", verbose=False
    ):
        """Ask a user for a prompt and embed the prompt.

        Args:
            manual_prompt (bool, optional): setting to ask for user input for the prompt through
                the terminal. If it is set to false, the 'automatic' argument is
                used as the prompt. Defaults to False.
            automatic (str, optional): Prompt added in the code itself. Defaults to "Amsterdam".
        """

        if manual_prompt:
            self.user_prompt = input("Please enter your prompt: ")
        else:
            self.user_prompt = automatic

        print("Embedding user prompt") if verbose else None
        self.embedded_user_prompt = embed(
            model=self.embedder,
            input_articles=self.user_prompt,
            api_key=self.api_key,
            write=False,
            verbose=False,
        )

    def rag(
        self,
        manual_prompt: bool = False,
        automatic="Amsterdam",
        n_articles: int = 3,
        report_to_terminal: bool = False,
        write: bool = True,
        output_filename: str = "rag_response.json",
        verbose=False,
    ):
        """RAG pipeline with parameters for asking for a user prompt. Has option
        to write the response to a file and/or the terminal.

        Args:
            manual_prompt (bool, optional): Setting to ask for user input for the prompt
                through the terminal. If it is set to false, the 'automatic' argument is
                used as the prompt. Defaults to False.
            automatic (str, optional): Prompt added in the code itself. Defaults to "Amsterdam".
            n_articles (int, optional): Number of articles to use for response generation.
                Defaults to 3.
            report_to_terminal (bool, optional): Parameter to print the prompt, used articles and
                generator response to the terminal. Defaults to False.
            write (bool, optional): Parameter to write the prompt, used articles and
                generator response to the output_file in json format. Defaults to True.
            output_filename (str, optional): Filename to write response to.
                Defaults to "rag_response.json".

        Returns:
            str: response from generator.
        """
        self.request_prompt(manual_prompt=manual_prompt, automatic=automatic)

        relevant_documents, relevant_document_ids = retrieve(
            retriever=cosine_similarity,
            query=self.embedded_user_prompt,
            embeddings=self.embeddings,
            texts=dict(zip(self.ids, self.texts)),
            n_articles=n_articles,
            verbose=verbose,
        )

        # Set up the system prompt using the relevant articles
        system_prompt = (
            self.system_prompt_setup
            + f"{'\n'.join([f'{chunk}' for chunk in relevant_documents])}"
        )

        # Generate a response
        response = generate_response(
            api_key=self.api_key,
            model=self.generator,
            system_prompt=system_prompt,
            user_prompt=self.user_prompt,
            verbose=verbose,
        )

        if write:
            with open(output_filename, "w") as output_file:
                output = {}

                output["user_prompt"] = self.user_prompt
                output["articles_used"] = {
                    article_id: article_text
                    for article_id, article_text in zip(
                        relevant_document_ids, relevant_documents
                    )
                }
                output["response"] = response
                json.dump(output, output_file)

        if report_to_terminal:
            print(f"User prompt:\n{self.user_prompt}\n")
            print(f"Articles used:\n{relevant_document_ids}\n")
            print(
                f"{'\n'.join([f'{docid} - {article}\n' for docid, article in zip(relevant_document_ids, relevant_documents)])}\n"
            )
            print(f"\nResponse:\n")
            print(response)

        return response

    def request_multiple_prompts(
        self,
        prompt_file: str,
        n_articles: int = 3,
        verbose=False,
        directory="responses",
    ):
        """Request multiple responses within the existing experiment from a
        file containing the prompts.

        Args:
            prompt_file (str): a file with plaintext formatting containing a new prompt
             on each line.
            n_articles (int, optional): Number of articles to use for response generation.
             Defaults to 3.
        """

        # Load the prompts
        prompts = open(prompt_file, "r")

        i = 0

        print("Generate responses based on prompts") if verbose else None

        for prompt in tqdm(prompts):

            prompt = prompt.strip()

            self.rag(
                manual_prompt=False,
                automatic=prompt,
                n_articles=n_articles,
                report_to_terminal=False,
                write=True,
                output_filename=f"{directory}/prompt_{i}_response.json",
                verbose=False,
            )

            i += 1


def get_model_list(filename):
    contents = []
    with open(filename) as file:
        for line in file:
            contents.append(line.strip())
    return contents


class Experiments:
    def __init__(
        self,
        embedders_file: str,
        retrieval_file: str,
        generator_file: str,
        prompt_file: str,
    ):

        self.embedders = get_model_list(embedders_file)
        self.retrievers = get_model_list(retrieval_file)
        self.generators = get_model_list(generator_file)
        self.prompt_file = prompt_file

    def run(self):

        for embedder in self.embedders:

            embedder_name = embedder.split("/")[1]

            print(f"Using embedder {embedder_name}")

            experiment = Experiment(
                api_key=os.environ["NEBUL_API_KEY"],
                embedder=embedder,
                generator="",
                test=False,
                embed_documents=True,
            )
            for generator in self.generators:

                generator_name = generator.split("/")[1]

                print(f"Using generator: {generator_name}")

                directory = (
                    f"{full_directory_name}/results/{embedder_name}/{generator_name}"
                )

                experiment.generator = generator

                # Create a directory to add the results of an individual experiment to or clear
                # directory if it already exists
                if not os.path.exists(directory):
                    os.makedirs(directory)

                else:
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)

                        # Check if it is a file (not a subdirectory)
                        if os.path.isfile(file_path):
                            os.remove(file_path)  # Remove the file

                experiment.request_multiple_prompts(
                    self.prompt_file, directory=directory
                )


if __name__ == "__main__":
    # experiment = Experiment(
    #     api_key=os.environ["NEBUL_API_KEY"],
    #     embedder="infly/inf-retriever-v1",
    #     generator="Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     test=False,
    #     embed_documents=False,
    #     verbose=True,
    # )

    # experiment.rag(manual_prompt=True, report_to_terminal=True)

    # experiment.request_multiple_prompts("prompts.csv", verbose=True)

    experiments = Experiments(
        "embedders.csv", "retrievers.csv", "generators.csv", "prompts.csv"
    )
    experiments.run()
