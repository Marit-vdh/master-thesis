import pandas as pd
import wikipedia as wp
from bs4 import BeautifulSoup
import requests

# wp.set_lang("nl")

# html = wp.page("Lijst van steden en dorpen in Noord-Holland").html().encode("UTF-8")

# soup = BeautifulSoup(html, "html.parser")

# soup.find("head")
# df = pd.read_html(html)[0]
# print(df.to_string())


wiki = "https://nl.wikipedia.org/wiki/Lijst_van_steden_en_dorpen_in_Noord-Holland"
website_url = requests.get(wiki).text
soup = BeautifulSoup(website_url, "lxml")

print(soup.contents)

my_table = soup.find("table", {"class": "wikitable plainrowheaders"})
print(my_table)
