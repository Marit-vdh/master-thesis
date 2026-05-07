import os
import openai

from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import cosine_similarity, retrieve
from generator import generate_response


class Experiment:

    def __init__(
        self,
        api_key: str,
        embedder: str,
        generator: str,
        test: bool = True,
        embedding_file: str = "data.tsv",
        system_prompt_setup: str = "You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: ",
    ):
        # Set up global variables
        self.api_key = api_key
        self.embedder = embedder
        self.generator = generator
        self.ids, self.texts = import_articles(flatten=True, test=test)
        self.system_prompt_setup = system_prompt_setup

        # Embed texts and load into a dictionary
        embed(
            model=self.embedder,
            input_articles=self.texts[:5],
            api_key=self.api_key,
            write=True,
            output_file=embedding_file,
        )

        self.embeddings = load_embeddings(
            embedding_file=embedding_file, indices=self.ids
        )

    def request_prompt(self, manual_prompt: bool = False, automatic: str = "Amsterdam"):
        """Ask a user for a prompt and embed the prompt."""

        if manual_prompt:
            self.user_prompt = input("Please enter your prompt: ")
        else:
            self.user_prompt = automatic

        self.embedded_user_prompt = embed(
            model=self.embedder,
            input_articles=self.user_prompt,
            api_key=self.api_key,
            write=False,
        )

    def rag(
        self,
        manual_prompt: bool = False,
        n_articles: int = 3,
        write: bool = True,
        output_filename: str = "rag_response.txt",
    ):
        self.request_prompt(manual_prompt=manual_prompt)

        relevant_documents, relevant_document_ids = retrieve(
            retriever=cosine_similarity,
            query=self.embedded_user_prompt,
            embeddings=self.embeddings,
            texts=dict(zip(self.ids, self.texts)),
            n_articles=n_articles,
        )

        # Set up the system prompt using the relevant articles
        system_prompt = (
            self.system_prompt_setup
            + f"{'\n'.join([f' - {chunk}' for chunk in relevant_documents])}"
        )

        # Generate a response
        response = generate_response(
            api_key=self.api_key,
            model=self.generator,
            system_prompt=system_prompt,
            user_prompt=self.user_prompt,
        )

        if write:
            with open(output_filename, "w") as output_file:
                output_file.write(f"User prompt:\n{self.user_prompt}\n")
                output_file.write(f"\nArticles used:\n{relevant_document_ids}\n")
                output_file.write(
                    f"{'\n'.join([f'{docid} - {article}' for docid, article in zip(relevant_document_ids, relevant_documents)])}\n"
                )
                output_file.write(f"\nResponse:\n")
                output_file.write(response)

        return response


if __name__ == "__main__":
    experiment = Experiment(
        api_key=os.environ["OPENAI_API_KEY"],
        embedder="infly/inf-retriever-v1",
        generator="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    experiment.rag(manual_prompt=True)
