import os
from tqdm import tqdm
import json
from bs4 import BeautifulSoup


def import_articles(flatten: bool = True, test: bool = False) -> dict[str, str] | bool:
    """Load articles as strings from locally stored files. Return as a
    dictionary if flatten is set to false, else return as a list of string.s

    Args:
        flatten (bool, optional): flatten the dictionary to a list. Defaults to True.
        test (bool, optional): limits the number of articles loaded to one month
            worth of articles to allow for small batch testing. Defaults to False.

    Returns:
        dict[str, str] | bool: the articles loaded as strings. If they are not
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

    for year in os.listdir("../../articles/")[:n_folders]:

        # Skip invisible folders
        if year.startswith("."):
            continue

        article_texts[year] = {}

        if test:
            n_folders = 1

            # Report the loaded year for reproducibility of the tests
            print(f"Loading year {year}")
        else:
            n_folders = len(os.listdir(f"../../articles/{year}"))

        for month in os.listdir(f"../../articles/{year}")[:n_folders]:

            # Skip invisible folders
            if month.startswith("."):
                continue

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

                        # Text is stored in three different types of blocks
                        if section["type"] in {
                            "block-intro",
                            "block-text",
                            "block-liveblog-text",
                        }:

                            # Clean the text with the BeautifulSoup module
                            parsed = BeautifulSoup(
                                section["attributes"]["text"], "html.parser"
                            )
                            text += (
                                parsed.get_text().replace("\n", " ").replace("\xa0", "")
                            )

                    # Add text to dictionary, remove .json extension
                    article_texts[year][month][article_name[:-5]] = text

    print("Finished loading articles")

    if not flatten:
        return article_texts

    # The embedding functions require the strings to be presented in an array. The
    # paragraphs of each article are used as separate 'tokens' to feed to the embedder.
    else:

        print("Start flattening the articles")

        strings = []
        ids = []

        for year, _ in tqdm(article_texts.items()):
            for month, _ in article_texts[year].items():
                for article_name, text in article_texts[year][month].items():

                    # Store the texts and article ids separately.
                    strings += [text]
                    ids += [article_name]

        print("Finished flattening the articles")

        return ids, strings


if __name__ == "__main__":
    # print(import_articles(flatten=False, test=True)['2024']['04']['334789'])

    ids, strings = import_articles(flatten=True, test=True)
    print(len(ids), len(strings))
