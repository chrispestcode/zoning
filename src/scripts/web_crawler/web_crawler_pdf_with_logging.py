import os
import requests
import time
import logging
import pdfkit
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configure logging
LOG_FILE = "crawler.log"
LOG_LEVEL_FILE = logging.DEBUG  
LOG_LEVEL_CONSOLE = logging.INFO 

logger = logging.getLogger("WebCrawler")
logger.setLevel(logging.DEBUG)  
logger.propagate = False  

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(LOG_LEVEL_FILE)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL_CONSOLE)
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Directory to store PDFs
SAVE_DIR = "../../output/aca_pages_pdf"
os.makedirs(SAVE_DIR, exist_ok=True)

visited_urls = set()

def crawl(url: str, depth: int = 1, max_depth: int = 3) -> None:
    """Recursively crawls a website, follows links, and saves pages as PDFs."""
    
    if depth > max_depth:
        logger.debug(f"Max depth reached for {url}. Skipping further crawling.")
        return

    if url in visited_urls:
        logger.debug(f"Already visited: {url}")
        return

    visited_urls.add(url)
    logger.info(f"Crawling: {url}")

    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            save_page_as_pdf(url)

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                link_url = link['href']
                if not link_url or link_url.startswith('mailto:'):
                    continue

                absolute_url = urljoin(url, link_url)
                if absolute_url.startswith('http'):
                    crawl(absolute_url, depth + 1, max_depth)

            time.sleep(1)  
        else:
            logger.warning(f"Failed to retrieve {url} (Status Code: {response.status_code})")
    except requests.RequestException as e:
        logger.error(f"Error crawling {url}: {e}")

def save_page_as_pdf(url: str) -> None:
    """Saves the page as a PDF using pdfkit."""
    
    filename = url.replace("http://", "").replace("https://", "").replace("/", "_") + ".pdf"
    file_path = os.path.join(SAVE_DIR, filename)

    try:
        pdfkit.from_url(url, file_path)
        logger.info(f"PDF saved: {file_path}")
    except Exception as e:
        logger.error(f"Error saving PDF {file_path}: {e}")

if __name__ == "__main__":
    START_URL = "https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/affordable-care-act/for-workers-and-families"
    crawl(START_URL)

