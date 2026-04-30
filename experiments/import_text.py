import os
from tqdm import tqdm
import json
from bs4 import BeautifulSoup


def import_articles(flatten=True, test=False):
    """Load articles as strings from locally stored files. Return as a
    dictionary if flatten is set to false, else return as a list of string.s

    Args:
        flatten (bool, optional): flatten the dictionary to a list. Defaults to True.
        test (bool, optional): limits the number of articles loaded to one month
            worth of articles to allow for small batch testing. Defaults to False.

    Returns:
        loaded articles (dict or bool): the articles loaded as strings. If they are not
            flattened, the key is the article identifier (string), the value is the loaded
            article. If they are flattened, only the articles are returned.
    """

    print("Start loading the articles")

    article_texts = {}

    # Limit the number of articles loaded when test is set to True
    if test:
        n_folders = 1
    else:
        n_folders = len(os.listdir("../../articles/"))

    for year in tqdm(os.listdir("../../articles/")[:n_folders]):

        article_texts[year] = {}

        if test:
            n_folders = 1

            # Report the loaded year for reproducibility of the tests
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

                    text = []

                    for section in article["included"]:

                        # Text is stored in two different types of blocks
                        if section["type"] in {"block-intro", "block-text"}:

                            # Clean the text with the BeautifulSoup module
                            parsed = BeautifulSoup(
                                section["attributes"]["text"], "html.parser"
                            )
                            text += [parsed.get_text().replace('\n', ' ').replace('\xa0', '')]

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
    print(import_articles(flatten=False, test=True)['2024']['04']['334789'])
