import os
from urllib.parse import urlparse

from loguru import logger
import pandas as pd


def extract_filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def construct_cache_path(filename: str, cache_dir: str | None = None):
    if cache_dir is None:
        # Default to ~/.cache if no cache directory is explicitly provided
        cache_dir = os.path.expanduser("~/.cache/llmbootcamp")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def read_csv_with_cache(url: str) -> pd.DataFrame:
    filename = extract_filename_from_url(url)
    cache_path = construct_cache_path(filename)
    if os.path.exists(cache_path):
        logger.info(f"Reading from {cache_path=}")
        df = pd.read_csv(cache_path)
    else:
        logger.info(f"Reading from {url=}")
        df = pd.read_csv(url)
        df.to_csv(cache_path, index=False)
    return df
