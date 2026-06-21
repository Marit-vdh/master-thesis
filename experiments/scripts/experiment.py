import os
import json
from tqdm import tqdm
from pathlib import Path

from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import (
    cosine_similarity,
    manhattan,
    retrieve_vector,
    create_index,
    bm25_retriever,
)
from generator import generate_response
from analyse_results import analyse_rag_response

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
        verbose=True,
        load_new_index: bool = True,
        embedding_file: str = "data.tsv",
        system_prompt_setup: str = "Je bent een behulpzame chatbot voor journalisten. Je antwoordt altijd in het Nederlands. Gebruik alleen de volgende stukken informatie bij het opstellen van een antwoord: ",
    ):
        # Set up global variables
        self.api_key = api_key
        self.embedder = embedder
        self.generator = generator
        self.ids, self.texts = import_articles(flatten=True, test=test)
        self.system_prompt_setup = system_prompt_setup
        self.embedding_file = embedding_file

        if load_new_index and embedder != "BM25":
            print("Loading embeddings") if verbose else None

            # Embed texts and load to file
            embed(
                model=self.embedder,
                input_articles=self.texts,
                api_key=self.api_key,
                write=True,
                output_file=embedding_file,
                max_articles=10,
                verbose=verbose,
            )

        elif embedder == "BM25":
            self.retriever, self.tokenizer = create_index(
                ids=self.ids,
                texts=self.texts,
                verbose=verbose,
                load_new_index=load_new_index,
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

        if self.embedder != "BM25":
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
        retriever_func: function = cosine_similarity,
        content_access: str = "content",
        n_articles: int = 3,
        report_to_terminal: bool = False,
        write: bool = True,
        prompt_type: str = "question",
        output_filename: str = "rag_response.json",
        verbose=False,
        evaluate_response=True,
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

        if self.embedder == "BM25":
            relevant_documents, relevant_document_ids = bm25_retriever(
                retriever=self.retriever,
                tokenizer=self.tokenizer,
                queries=[self.user_prompt],
                n_articles=n_articles,
            )

        else:
            relevant_documents, relevant_document_ids = retrieve_vector(
                retriever=retriever_func,
                query=self.embedded_user_prompt,
                embeddings_file=self.embedding_file,
                indices=self.ids,
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
            content_access=content_access,
            verbose=verbose,
            # max_completion_tokens=250,
        )

        if evaluate_response:
            scores = analyse_rag_response(
                self.user_prompt, relevant_documents, response
            )
        else:
            scores = {}

        if write:
            with open(output_filename, "w") as output_file:
                output = {}
                output["embedder"] = self.embedder_name
                output["generator"] = self.generator_name
                output["scores"] = scores
                output["distance_metric"] = (
                    None if self.embedder_name == "BM25" else retriever_func.__name__
                )

                output["user_prompt"] = self.user_prompt
                output["prompt_type"] = prompt_type
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
        prompts: list[str],
        content_access: str = "content",
        retriever_func: function = cosine_similarity,
        n_articles: int = 3,
        verbose=False,
        directory="../responses",
        evaluate_response=True,
    ):
        """Request multiple responses within the existing experiment from a
        file containing the prompts.

        Args:
            prompt_file (str): a file with plaintext formatting containing a new prompt
             on each line.
            n_articles (int, optional): Number of articles to use for response generation.
             Defaults to 3.
        """

        i = 0

        print("Generate responses based on prompts") if verbose else None

        for prompt in tqdm(prompts):

            prompt_text = prompt[1]

            self.rag(
                manual_prompt=False,
                automatic=prompt_text,
                retriever_func=retriever_func,
                content_access=content_access,
                n_articles=n_articles,
                report_to_terminal=False,
                write=True,
                prompt_type=prompt[0],
                output_filename=f"{directory}/prompt_{i}_response.json",
                verbose=False,
                evaluate_response=evaluate_response,
            )

            i += 1


def get_embedders(filename):
    contents = []
    with open(filename) as file:
        for line in file:
            split_line = line.split(",")
            load_embeddings = True if split_line[0] == "True" else False
            generate = True if split_line[1] == "True" else False
            model = split_line[2].strip()
            contents.append((load_embeddings, generate, model))
    return contents


def get_generators(filename):
    contents = []
    with open(filename) as file:
        for line in file:
            split_line = line.split(",")
            contents.append((split_line[0], split_line[1].strip()))
    return contents


def get_prompts(filename):
    contents = []
    with open(filename) as file:
        for line in file:
            split_line = line.split(",")
            contents.append((split_line[0], split_line[1].strip()))
    return contents


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
        distances: list[function],
    ):

        self.embedders = get_embedders(embedders_file)
        self.retrievers = get_model_list(retrieval_file)
        self.generators = get_generators(generator_file)
        self.prompts = get_prompts(prompt_file)
        self.distances = distances

        self.embedding_directory = f"{full_directory_name}/embeddings"

        # Create a directory for embeddings if it does not exist
        if not os.path.exists(self.embedding_directory):
            os.makedirs(self.embedding_directory)

    def run(self):

        for embedder_tuple in self.embedders:
            load_new_index = embedder_tuple[0]
            generate = embedder_tuple[1]
            embedder = embedder_tuple[2]

            embedder_name = embedder.split("/")[1]

            # Skip generators that have already finished the experiments
            if not generate:
                print(f"Skipping embedder {embedder_name}")
                continue

            print(f"Using embedder {embedder_name}")

            experiment = Experiment(
                api_key=os.environ["NEBUL_API_KEY"],
                embedder=embedder.strip("/"),
                generator="",
                test=False,
                load_new_index=load_new_index,
                embedding_file=f"{self.embedding_directory}/{embedder_name}.tsv",
                verbose=True,
            )

            experiment.embedder_name = embedder_name

            for generator_tuple in self.generators:

                generator_name = generator_tuple[0].split("/")[1]
                generator_access = generator_tuple[1]

                print(f"Using generator: {generator_name}")

                directory = (
                    f"{full_directory_name}/results/{embedder_name}/{generator_name}"
                )

                experiment.generator = generator_tuple[0]
                experiment.generator_name = generator_name

                # Use BM25 retrieval
                if embedder_name == "BM25":

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
                        self.prompts,
                        directory=directory,
                        content_access=generator_access,
                    )

                # If embedders are used, use embedding based retrieval
                else:
                    for distance in self.distances:

                        distance_name = distance.__name__

                        print(f"Using distance metric {distance_name}")

                        directory = f"{full_directory_name}/results/{embedder_name}/{generator_name}/{distance_name}"

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
                            self.prompts,
                            retriever_func=distance,
                            directory=directory,
                            content_access=generator_access,
                        )


if __name__ == "__main__":
    # experiment = Experiment(
    #     api_key=os.environ["NEBUL_API_KEY"],
    #     embedder="BM25",
    #     generator="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    #     test=False,
    #     load_new_index=False,
    #     verbose=True,
    #     embedding_file="../embeddings/bge-m3.tsv",
    # )
    # experiment.embedder_name = "BM25"
    # experiment.generator_name = "NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    # experiment.request_multiple_prompts(
    #     prompts=[
    #         ("question", "Amsterdam"),
    #         ("question", "Trekkertrek"),
    #         ("question", "Bruinvis"),
    #     ],
    #     content_access="content",
    #     retriever_func=None,
    #     verbose=True,
    #     evaluate_response=False,
    # )

    print(f"Running experiments..")
    experiments = Experiments(
        "../sources/embedders.csv",
        "../sources/retrievers.csv",
        "../sources/generators.csv",
        "../sources/prompts.csv",
        [cosine_similarity, manhattan],
    )
    experiments.run()
