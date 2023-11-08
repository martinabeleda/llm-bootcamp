"""Embedding Wikipedia articles for search

This script shows how to prepare a dataset of Wikipedia articles for search
used in `question_answering.py`

See:
- https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search
"""

import re

from loguru import logger
import mwclient
import mwparserfromhell
from openai import OpenAI


client = OpenAI(
    api_key="mabeleda",
    base_url="http://openai-api-proxy.discovery:8888/v1",
)

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"

CATEGORY_TITLE = "Category:2022 Winter Olympics"
WIKI_SITE = "en.wikipedia.org"

SECTIONS_TO_IGNORE = {
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
}


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


def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,
    parent_titles: list[str],
    sections_to_ignore: set[str],
) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia section, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []
    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(
                all_subsections_from_section(subsection, titles, sections_to_ignore)
            )
        return results


def all_subsections_from_title(
    title: str,
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
    site_name: str = WIKI_SITE,
) -> list[tuple[list[str], str]]:
    """From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """
    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)
    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(
            all_subsections_from_section(subsection, [title], sections_to_ignore)
        )
    return results


def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)


def keep_section(section: tuple[list[str], str]) -> bool:
    """Return True if the section should be kept, False otherwise."""
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True


def main():
    site = mwclient.Site(WIKI_SITE)
    category_page = site.pages[CATEGORY_TITLE]
    titles = titles_from_category(category_page, max_depth=1)
    logger.info(f"Found {len(titles)} article titles in {CATEGORY_TITLE}")

    wikipedia_sections = []
    for title in titles:
        wikipedia_sections.extend(all_subsections_from_title(title))
    logger.info(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")

    wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]
    original_num_sections = len(wikipedia_sections)
    wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
    logger.info(
        f"Filtered out {original_num_sections-len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)} sections."
    )

    for ws in wikipedia_sections[:5]:
        logger.info(f"{ws[0]} {ws[1][:77]}...")
