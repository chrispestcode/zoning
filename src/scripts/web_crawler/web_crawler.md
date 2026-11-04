# Web Crawler

This project is a simple web crawler designed to recursively crawl a website, follow links, and save web pages in both HTML and PDF formats.
It is configured to crawl pages related to the Affordable Care Act (ACA) from a specified starting URL.

## Features

- **Recursive Crawling**: The crawler follows links from the starting page up to a specified maximum depth.
- **HTML Page Saving**: Crawled pages are saved as HTML files in a designated directory.
- **PDF Page Saving**: Crawled pages are also saved as PDFs for easy offline viewing.
- **Logging**: The crawler logs its activities, including successes and failures, to both a file and the console.
- **User-Agent Header**: The crawler includes a `User-Agent` header in its requests to identify itself.
- **Depth Limiting**: Allows you to control the depth of recursion to avoid infinite crawling.

## Requirements

- Python 3.x
- `requests` library
- `beautifulsoup4` library
- `pdfkit` library
- `wkhtmltopdf` (required for `pdfkit` to function correctly)
- `logging` library (built-in)

## Installation

1. Clone this repository.
2. Install required libraries using pip:
   ```sh
   pip install requests beautifulsoup4 pdfkit
   ```
3. Install `wkhtmltopdf`:
   - On macOS (with Homebrew):
     ```sh
     brew install wkhtmltopdf
     ```
   - On Ubuntu/Debian:
     ```sh
     sudo apt install wkhtmltopdf
     ```
   - On Windows, download and install from: https://wkhtmltopdf.org/downloads.html

## Configuration

- **Logging**: The crawler logs everything to a file named `crawler.log` and warnings or above to the console.
- **Crawling Depth**: The maximum crawling depth is set to 3 by default but can be adjusted in the `crawl` function.
- **Save Directory**:
  - HTML pages are saved in `aca_pages/`.
  - PDFs are saved in `aca_pages_pdf/`.

## Usage

1. Clone or download the repository to your local machine.
2. Ensure that all dependencies are installed.
3. Run the crawler using Python:
   ```sh
   python crawler.py
   ```
4. The crawler starts from the URL specified in the `START_URL` variable in the script.

## HTML and PDF Saving

The crawler saves web pages in both HTML and PDF formats:
- **HTML Files**: Crawled pages are saved in `aca_pages/` with filenames based on URLs, replacing special characters with underscores.
- **PDF Files**: Crawled pages are saved in `aca_pages_pdf/`, using `pdfkit` and `wkhtmltopdf` to generate PDFs.

## Example Output Structure

```
aca_pages/
  www.dol.gov_agencies_ebsa_laws-and-regulations_laws_affordable-care-act_for-workers-and-families.html
  www.dol.gov_agencies_ebsa_laws-and-regulations_laws_affordable-care-act_for-workers-and-families_page2.html

aca_pages_pdf/
  www.dol.gov_agencies_ebsa_laws-and-regulations_laws_affordable-care-act_for-workers-and-families.pdf
  www.dol.gov_agencies_ebsa_laws-and-regulations_laws_affordable-care-act_for-workers-and-families_page2.pdf

crawler.log
```

## Customization

- To change the starting URL, modify the `START_URL` variable in the script.
- To adjust the maximum crawling depth, modify the `max_depth` parameter in the `crawl` function call.

## Logging Levels

- **File Logging**: The script logs everything at DEBUG level to a file (`crawler.log`).
- **Console Logging**: Only warnings and above are logged to the console.

You can adjust the log levels in the script based on your needs.

## Limitations

- The script does not handle dynamic content that is loaded via JavaScript. It only processes static HTML content.
- Requires `wkhtmltopdf` for proper PDF conversion.
- Be mindful of websites' `robots.txt` policies before running the crawler to avoid violating their terms of service.

## Notes

- Ensure you have permission to crawl the target website and comply with its terms of service.
- Be respectful of server resources by not overwhelming the site with too many requests.

---
This crawler is designed to provide an easy way to archive ACA-related web pages in both HTML and PDF formats for offline reference. 🚀


