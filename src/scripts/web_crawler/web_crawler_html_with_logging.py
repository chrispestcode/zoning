import os
import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configure logging
LOG_FILE = "crawler.log"
LOG_LEVEL_FILE = logging.DEBUG  # Log everything to file
LOG_LEVEL_CONSOLE = logging.WARNING  # Only show warnings and above in console

# Create a logger
logger = logging.getLogger("WebCrawler")
logger.setLevel(logging.DEBUG)  # Capture all logs at logger level
logger.propagate = False  # Prevent duplicate logs from propagating to the root logger

# File Handler - Logs everything
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(LOG_LEVEL_FILE)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console Handler - Logs only warnings and above
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL_CONSOLE)
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Directory to store crawled pages
SAVE_DIR = "../../output/aca_pages"
os.makedirs(SAVE_DIR, exist_ok=True)

# Track visited URLs to avoid duplicate crawling
visited_urls = set()

def crawl(url: str, depth: int = 1, max_depth: int = 3) -> None:
    """
    Recursively crawls a website, follows links, and saves HTML pages.

    Args:
        url (str): The URL to crawl.
        depth (int): The current depth of recursion.
        max_depth (int): The maximum allowed depth.

    Returns:
        None
    """
    if depth > max_depth:
        logger.info(f"Max depth reached for {url}. Skipping further crawling.")
        return

    if url in visited_urls:
        logger.debug(f"Already visited: {url}")
        return

    visited_urls.add(url)
    logger.info(f"Crawling: {url}")

    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            save_page(url, response.text)

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                link_url = link['href']
                if not link_url or link_url.startswith('mailto:'):
                    continue

                absolute_url = urljoin(url, link_url)
                if absolute_url.startswith('http'):
                    crawl(absolute_url, depth + 1, max_depth)

            time.sleep(1)  # Avoid overloading the server
        else:
            logger.warning(f"Failed to retrieve {url} (Status Code: {response.status_code})")
    except requests.RequestException as e:
        logger.error(f"Error crawling {url}: {e}")


def save_page(url: str, html_content: str) -> None:
    """
    Saves the crawled page's HTML content to a file.

    Args:
        url (str): The URL of the page.
        html_content (str): The HTML content of the page.

    Returns:
        None
    """
    filename = url.replace("http://", "").replace("https://", "").replace("/", "_") + ".html"
    file_path = os.path.join(SAVE_DIR, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        logger.info(f"Page saved: {file_path}")
    except IOError as e:
        logger.error(f"Error saving file {file_path}: {e}")


if __name__ == "__main__":
    START_URL = "https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/affordable-care-act/for-workers-and-families"
    crawl(START_URL)

