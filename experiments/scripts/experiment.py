import os
import json
from tqdm import tqdm
from pathlib import Path

from import_text import import_articles
from embed import embed_articles, embed_prompt
from retrieve import (
    cosine_similarity,
    manhattan,
    retrieve_vector,
    create_index,
    bm25_retriever,
)
from generator import generate_response

full_directory_name = (
    "/Users/maritvandenhelder/information_studies/thesis/master-thesis/experiments"
)


class Experiment:

    def __init__(
        self,
        api_key: str,
        embedder: str,
        retriever: str,
        generator: str,
        ids: list[str],
        texts: list[str],
        test: bool = True,
        verbose=True,
        load_new_index: bool = True,
        embedding_file: str = "data.tsv",
    ):
        # Set up global variables
        self.api_key = api_key
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.ids = ids
        self.texts = texts
        self.embedding_file = embedding_file

        if load_new_index and embedder != "BM25":
            print("Loading embeddings") if verbose else None

            # Embed texts and load to file
            embed_articles(
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

    def generate_responses(
        self,
        prompts: list[tuple[str, str]],
        evaluate_response: bool = False,
        write: bool = True,
        output_directory: str = f"{full_directory_name}/responses",
        report_to_terminal: bool = False,
    ):
        """Given a dict of relevant documents and relevant document ids, generate responses to the list of prompts."""

        print("Generating responses")

        for i in tqdm(range(len(prompts))):

            system_prompt_setup = prompts[i][0]
            user_prompt = prompts[i][1]

            # Retrieve relevant documents from the dictionaries
            relevant_doc_ids = self.relevant_docs[user_prompt][1]
            relevant_docs = self.relevant_docs[user_prompt][0]

            if len(relevant_doc_ids) == 0:
                system_prompt = system_prompt = (
                    "Je bent een tool die gebruikt wordt voor journalistieke doeleinden. Hou je altijd aan de journalistieke code. Antwoord in het Nederlands."
                    + system_prompt_setup
                )

            else:
                # Set up the system prompt using the relevant articles
                system_prompt = (
                    "Je bent een tool die gebruikt wordt voor journalistieke doeleinden. Hou je altijd aan de journalistieke code. Antwoord in het Nederlands."
                    + system_prompt_setup
                    + f"Gebruik, indien relevant, de volgende artikelen als extra context: {'\n'.join([f'{chunk}' for chunk in relevant_docs])}"
                )

            # Generate a response
            response = generate_response(
                api_key=self.api_key,
                model=self.generator,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                content_access=self.content_access,
                n_completions=5,
            )

            if write:
                with open(
                    f"{output_directory}/prompt_{i}_response.json", "w"
                ) as output_file:
                    output = {}
                    output["embedder"] = self.embedder_name
                    output["generator"] = self.generator_name
                    output["distance_metric"] = (
                        None
                        if self.embedder_name in {"BM25", "no_retrieval"}
                        else self.distance_name
                    )
                    output["instruction"] = system_prompt_setup
                    output["user_prompt"] = user_prompt
                    output["articles_used"] = {
                        article_id: article_text
                        for article_id, article_text in zip(
                            relevant_doc_ids, relevant_docs
                        )
                    }
                    output["response"] = response
                    json.dump(output, output_file)

            if report_to_terminal:
                print(f"User prompt:\n{self.user_prompt}\n")
                print(f"Articles used:\n{relevant_doc_ids}\n")
                print(
                    f"{'\n'.join([f'{docid} - {article}\n' for docid, article in zip(relevant_doc_ids, relevant_docs)])}\n"
                )
                print(f"\nResponse:\n")
                print(response)


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


def load_benchmark(filename):
    with open("../sources/benchmark.json", "r") as benchmark_file:
        benchmark = json.load(benchmark_file)

    prompts = []
    # Retrieve the individual prompts from the benchmark
    for item in benchmark:
        system_prompt = item["instruction"]
        model_input = item["input"]
        choices = (
            f" Kies uit de volgende opties: {item["choices"]}. Geef alleen een enkele letter als resultaat."
            if item["choices"] != "-"
            else ""
        )
        instructions = system_prompt + choices
        prompts.append((instructions, model_input))

    return benchmark, prompts


def get_skip_file(filename):
    contents = set()
    with open(filename) as file:
        for line in file:
            contents.add(f"{full_directory_name}/results/{line.strip()}")
    return contents


class Experiments:
    def __init__(
        self,
        skip_file: str,
        embedders_file: str,
        generator_file: str,
        prompt_file: str,
        distances: list[function],
        n_articles: int = 3,
    ):
        self.skip = get_skip_file(skip_file)
        self.ids, self.texts = import_articles(flatten=True, test=False)
        self.embedders = get_embedders(embedders_file)
        self.generators = get_generators(generator_file)
        self.benchmark, self.prompts = load_benchmark(prompt_file)
        self.distances = distances
        self.n_articles = n_articles

        self.embedding_directory = f"{full_directory_name}/embeddings"

        # Create a directory for embeddings if it does not exist
        if not os.path.exists(self.embedding_directory):
            os.makedirs(self.embedding_directory)

    def generate(self, experiment, directory):

        # Do not generate new output if the directory is in the skip file
        if directory in self.skip:
            print(f"Skipping")
            return

        if not os.path.exists(directory):
            os.makedirs(directory)

        else:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)

                # Check if it is a file (not a subdirectory)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove the file

        experiment.generate_responses(
            self.prompts,
            evaluate_response=False,
            write=True,
            output_directory=directory,
        )

        with open("../sources/skip_file.txt", "a") as skip_file:
            skip_file.write(f"{directory}\n")

    def run(self):

        for embedder_tuple in self.embedders:

            load_new_index = embedder_tuple[0]
            generate = embedder_tuple[1]
            embedder = embedder_tuple[2]

            embedder_name = embedder.split("/")[1]

            if f"{full_directory_name}/results/{embedder_name}" in self.skip:
                print(f"Skipping retrieval using {embedder_name}")
                continue

            # Skip generators that have already finished the experiments
            if not generate:
                print(f"Skipping retrieval using {embedder_name}")
                continue

            experiment = Experiment(
                api_key=os.environ["NEBUL_API_KEY"],
                embedder=embedder.strip("/"),
                retriever="",
                generator="",
                ids=self.ids,
                texts=self.texts,
                test=False,
                load_new_index=load_new_index,
                embedding_file=f"{self.embedding_directory}/{embedder_name}.tsv",
                verbose=True,
            )

            experiment.embedder_name = embedder_name

            if experiment.embedder_name == "no_retrieval":

                print("Testing without RAG.")

                experiment.relevant_docs = {}

                for prompt in tqdm(self.prompts):

                    experiment.relevant_docs[prompt[1]] = (
                        [],
                        [],
                    )

                for generator_tuple in self.generators:
                    experiment.generator = generator_tuple[0]
                    experiment.content_access = generator_tuple[1]
                    experiment.generator_name = generator_tuple[0].split("/")[1]

                    print(f"Using generator {experiment.generator_name}")

                    directory = f"{full_directory_name}/results/no_retrieval/{experiment.generator_name}"

                    self.generate(experiment=experiment, directory=directory)

            elif experiment.embedder_name == "BM25":

                print("Using BM25 retrieval")

                experiment.relevant_docs = {}

                for prompt in tqdm(self.prompts):
                    relevant_documents, relevant_document_ids = bm25_retriever(
                        retriever=experiment.retriever,
                        tokenizer=experiment.tokenizer,
                        queries=[prompt[1]],
                        n_articles=self.n_articles,
                    )
                    experiment.relevant_docs[prompt[1]] = (
                        relevant_documents,
                        relevant_document_ids,
                    )

                for generator_tuple in self.generators:
                    experiment.generator = generator_tuple[0]
                    experiment.content_access = generator_tuple[1]
                    experiment.generator_name = generator_tuple[0].split("/")[1]

                    print(f"Using generator {experiment.generator_name}")

                    directory = f"{full_directory_name}/results/{experiment.embedder_name}/{experiment.generator_name}"

                    self.generate(experiment=experiment, directory=directory)

            else:
                print(f"Using embedder {embedder_name}")

                for distance in self.distances:

                    experiment.relevant_docs = {}

                    experiment.distance_name = distance.__name__

                    if (
                        f"{full_directory_name}/results/{experiment.embedder_name}/{experiment.distance_name}"
                        in self.skip
                    ):
                        print(
                            f"Skipping embedder {experiment.embedder_name} with distance metric {experiment.distance_name}"
                        )
                        continue

                    print(
                        f"Retrieving relevant documents with distance metric {experiment.distance_name}"
                    )

                    for prompt in tqdm(self.prompts):

                        embedded_user_prompt = embed_prompt(
                            model=experiment.embedder,
                            prompt=prompt[1],
                            api_key=experiment.api_key,
                        )

                        relevant_documents, relevant_document_ids = retrieve_vector(
                            retriever=distance,
                            query=embedded_user_prompt,
                            embeddings_file=experiment.embedding_file,
                            indices=experiment.ids,
                            texts=dict(zip(experiment.ids, experiment.texts)),
                            n_articles=self.n_articles,
                        )

                        experiment.relevant_docs[prompt[1]] = (
                            relevant_documents,
                            relevant_document_ids,
                        )

                    for generator_tuple in self.generators:
                        experiment.generator = generator_tuple[0]
                        experiment.content_access = generator_tuple[1]
                        experiment.generator_name = generator_tuple[0].split("/")[1]

                        print(f"Using generator {experiment.generator_name}")

                        directory = f"{full_directory_name}/results/{experiment.embedder_name}/{experiment.distance_name}/{experiment.generator_name}"

                        self.generate(experiment=experiment, directory=directory)


if __name__ == "__main__":

    print(f"Running experiments")
    experiments = Experiments(
        "../sources/skip_file.txt",
        "../sources/embedders.csv",
        "../sources/generators.csv",
        "../sources/benchmark.json",
        [cosine_similarity, manhattan],
    )
    experiments.run()
