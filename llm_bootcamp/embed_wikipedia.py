"""Embedding Wikipedia articles for search

This script shows how to prepare a dataset of Wikipedia articles for search
used in `question_answering.py`

See:
- https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search
"""

from loguru import logger
import mwclient
from openai import OpenAI


client = OpenAI(
    api_key="mabeleda",
    base_url="http://openai-api-proxy.discovery:8888/v1",
)

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"

CATEGORY_TITLE = "Category:2022 Winter Olympics"
WIKI_SITE = "en.wikipedia.org"


def titles_from_category(
    category: mwclient.listing.Category,
    max_depth: int,
) -> set[str]:
    """Return a set of page titles in a given Wiki category and its subcategories."""
    titles = set()
    for cm in category.members():
        if type(cm) == mwclient.page.Page:
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)
    return titles


def main():
    site = mwclient.Site(WIKI_SITE)
    category_page = site.pages[CATEGORY_TITLE]
    titles = titles_from_category(category_page, max_depth=1)
    logger.info(f"Found {len(titles)} article titles in {CATEGORY_TITLE}")
