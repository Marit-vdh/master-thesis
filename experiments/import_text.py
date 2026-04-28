import os
from tqdm import tqdm
import json
from bs4 import BeautifulSoup


def import_articles(flatten=True, test=False):
    """_summary_

    Args:
        flatten (bool, optional): _description_. Defaults to True.
        test (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    print("Start loading the articles")

    article_texts = {}

    if test:
        n_folders = 1
    else:
        n_folders = len(os.listdir("../../articles/"))

    for year in tqdm(os.listdir("../../articles/")[:n_folders]):

        article_texts[year] = {}

        if test:
            n_folders = 1
            print(f"Loading year {year}")
        else:
            n_folders = len(os.listdir(f"../../articles/{year}"))

        for month in os.listdir(f"../../articles/{year}")[:n_folders]:

            if test:
                print(f"Loading month {month}")

            article_texts[year][month] = {}

            for article_name in os.listdir(f"../../articles/{year}/{month}"):

                # Retrieve the flat text
                with open(
                    f"../../articles/{year}/{month}/{article_name}",
                    mode="r",
                    encoding="utf-8",
                ) as file:

                    article = json.load(file)

                    text = ""

                    for section in article["included"]:

                        # Text is stored in two different types of blocks
                        if section["type"] in {"block-intro", "block-text"}:

                            # Add if statement for outtro block (leads with 'closing')

                            # Clean the text with the BeautifulSoup module
                            parsed = BeautifulSoup(
                                section["attributes"]["text"], "html.parser"
                            )
                            text += parsed.get_text()

                    # Add text to dictionary, remove .json extension
                    article_texts[year][month][article_name[:-5]] = text

    print("Finished loading articles")

    if not flatten:
        return article_texts

    # The embedding functions require the strings to be presented in an array
    else:

        print("Start flattening the articles")

        strings = []
        for year, _ in tqdm(article_texts.items()):
            for month, _ in article_texts[year].items():
                for article_name, text in article_texts[year][month].items():
                    strings.append(text)

        print("Finished flattening the articles")

        return strings


if __name__ == "__main__":
    print(import_articles())
