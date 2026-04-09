#!/usr/bin/env python3
"""
TVPL Playwright Crawler - Cloudflare Bypass

Uses Playwright to bypass Cloudflare protection and crawl thuvienphapluat.vn.

Features:
- Cloudflare bypass (headless browser)
- State persistence (SQLite)
- Parallel workers
- Resume capability
- Rich metadata extraction

Requirements:
    pip install playwright scrapy-playwright
    playwright install chromium

Usage:
    python crawler/playwright_crawler.py --workers 2
    python crawler/playwright_crawler.py --resume
"""

import argparse
import json
import logging
import sqlite3
import signal
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Dict
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy.http import Response, HtmlResponse
from scrapy.playwright.page import PageMethod
from scrapy.utils.project import get_project_settings
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class StateManager:
    """Manage crawler state in SQLite."""

    def __init__(self, db_path: str = "data/raw/.playwright_crawler_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # URLs table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_urls (
                url TEXT PRIMARY KEY,
                first_seen TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                data JSON,
                crawled_at TEXT,
                source TEXT
            )
        """)

        # Stats table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value JSON
            )
        """)

        # Initialize stats
        self.conn.execute(
            "INSERT OR IGNORE INTO stats (key, value) VALUES ('global', ?)",
            (json.dumps({"pages_crawled": 0, "documents_extracted": 0, "start_time": ""}),)
        )
        self.conn.commit()

    def is_url_seen(self, url: str) -> bool:
        """Check if URL has been seen."""
        row = self.conn.execute(
            "SELECT 1 FROM seen_urls WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def mark_url_seen(self, url: str):
        """Mark URL as seen."""
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, first_seen) VALUES (?, ?)",
            (url, datetime.now().isoformat())
        )
        self.conn.commit()

    def save_document(self, doc: dict) -> bool:
        """Save document."""
        doc_id = doc.get("doc_id")
        if not doc_id:
            return False

        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO documents (doc_id, url, data, crawled_at, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, doc.get("url"), json.dumps(doc), datetime.now().isoformat(), "playwright_crawler")
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save document {doc_id}: {e}")
            return False

    def get_stats(self) -> dict:
        """Get statistics."""
        row = self.conn.execute("SELECT value FROM stats WHERE key = 'global'").fetchone()
        if row:
            return json.loads(row[0])
        return {}

    def update_stats(self, **kwargs):
        """Update statistics."""
        stats = self.get_stats()
        stats.update(kwargs)
        stats["last_update"] = datetime.now().isoformat()

        self.conn.execute(
            "UPDATE stats SET value = ? WHERE key = 'global'",
            (json.dumps(stats),)
        )
        self.conn.commit()

    def get_last_page(self) -> int:
        """Get last crawled page number."""
        stats = self.get_stats()
        return stats.get("last_page", 0) + 1

    def set_last_page(self, page: int):
        """Set last crawled page."""
        self.update_stats(last_page=page)

    def get_all_documents(self) -> Generator[dict, None, None]:
        """Yield all documents."""
        for row in self.conn.execute("SELECT data FROM documents"):
            yield json.loads(row[0])


class TVPLPlaywrightSpider(scrapy.Spider):
    """TVPL Spider using Playwright for Cloudflare bypass."""

    name = "tvpl_playwright"
    allowed_domains = ["thuvienphapluat.vn"]

    custom_settings = {
        "DOWNLOAD_DELAY": 3,
        "CONCURRENT_REQUESTS": 2,
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {
            "headless": True,
        },
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 60000,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = StateManager()
        self.base_url = "https://thuvienphapluat.vn"
        self.start_page = self.state.get_last_page()

    def start_requests(self):
        """Generate initial requests."""
        url = f"{self.base_url}/page/tim-van-ban.aspx"

        for page in range(self.start_page, 100000):
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    "page": page,
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "div.search-result-item, div.news-item, table", timeout=15000),
                    ],
                },
                dont_filter=True,
            )

    def parse(self, response: Response):
        """Parse search results page."""
        if not isinstance(response, HtmlResponse):
            logger.warning(f"Non-HTML response: {type(response)}")
            return

        page = response.meta.get("page", 1)
        logger.info(f"Parsing page {page}")

        # Update state
        self.state.set_last_page(page)

        # Extract document URLs
        selectors = [
            "div.search-result-item a.title-link",
            "div.news-item ul li a",
            "table.doc-list a",
            "a[href*='/van-ban/']",
        ]

        doc_links = []
        for selector in selectors:
            links = response.css(selector)
            if links:
                doc_links = links
                break

        logger.info(f"Page {page}: Found {len(doc_links)} document links")

        for link in doc_links:
            href = link.css("::attr(href)").get("")
            if not href:
                continue

            url = urljoin(self.base_url, href)

            # Check if seen
            if self.state.is_url_seen(url):
                continue

            self.state.mark_url_seen(url)

            # Extract doc_id for deduplication
            doc_id = self.extract_doc_id(url)
            if not doc_id:
                continue

            # Yield document request
            yield scrapy.Request(
                url,
                callback=self.parse_document,
                meta={
                    "doc_id": doc_id,
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "div.doc-content, div.content, article", timeout=10000),
                    ],
                },
                dont_filter=False,
            )

    def parse_document(self, response: Response):
        """Parse individual document."""
        if not isinstance(response, HtmlResponse):
            return

        url = response.url
        doc_id = response.meta.get("doc_id", "")

        logger.info(f"Parsing document: {doc_id}")

        # Extract document data
        document = self.extract_document_data(response, url, doc_id)

        if document and self.state.save_document(document):
            stats = self.state.get_stats()
            self.state.update_stats(documents_extracted=stats.get("documents_extracted", 0) + 1)

            yield document

    def extract_doc_id(self, url: str) -> Optional[str]:
        """Extract document ID from URL."""
        import re
        match = re.search(r"-(\d+)\.aspx$", url)
        return match.group(1) if match else None

    def extract_document_data(self, response: Response, url: str, doc_id: str) -> Optional[dict]:
        """Extract all document data."""
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md

        soup = BeautifulSoup(response.body, "lxml")

        document = {
            "url": url,
            "doc_id": doc_id,

            # Categorization from URL
            "category": self._extract_category(url),
            "sub_category": "",

            # Basic info
            "title": self._extract_text(soup, ["h1.doc-title", "h1.title", "h1"]),
            "doc_number": self._extract_text(soup, ["span.doc-number", "li.number"]),
            "doc_type": self._extract_text(soup, ["span.doc-type", "li.type"]),

            # Authority
            "issuing_authority": self._extract_text(soup, ["span.authority", "li.organ"]),
            "signatory": self._extract_text(soup, ["span.signer", "li.nguoi-ky"]),
            "signatory_title": self._extract_text(soup, ["span.signer-title", "li.chuc-danh"]),

            # Dates
            "issue_date": self._extract_text(soup, ["span.issue-date", "li.ngay-bh"]),
            "effective_date": self._extract_text(soup, ["span.effective-date", "li.ngay-hl"]),
            "expiry_date": self._extract_text(soup, ["span.expiry-date", "li.ngay-hhl"]),
            "published_date": self._extract_text(soup, ["span.published-date", "li.ngay-dcb"]),

            # Status
            "status": self._extract_text(soup, ["span.status", "li.trang-thai"]),
            "effect_status": self._extract_text(soup, ["span.effect-status"]),

            # Classification
            "sector": self._extract_text(soup, ["span.sector", "li.nganh"]),
            "field": self._extract_text(soup, ["span.field", "li.linh-vuc"]),
            "scope": self._extract_text(soup, ["span.scope", "li.pham-vi"]),

            # Content
            "content_html": self._extract_content_html(soup),
            "content_text": "",
            "content_markdown": "",

            # Relationships
            "related_docs": [],
            "amends_docs": [],
            "repeals_docs": [],
            "cites_docs": [],

            # Additional
            "tags": [],
            "language": "vn",
            "has_english_version": bool(soup.select("a:contains('English')")),
            "original_doc_url": None,

            # Technical
            "crawled_at": datetime.now().isoformat(),
            "crawl_source": "playwright_crawler",
        }

        # Extract content
        content = document["content_html"]
        if content:
            document["content_text"] = self._html_to_text(content)
            document["content_markdown"] = self._html_to_markdown(content)

        return document

    def _extract_category(self, url: str) -> str:
        """Extract category from URL."""
        path = urlparse(url).path
        parts = [p for p in path.split("/") if p and not p.endswith(".aspx")]
        return parts[0] if parts else ""

    def _extract_text(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text using multiple selectors."""
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                return " ".join(text.split())
        return ""

    def _extract_content_html(self, soup: BeautifulSoup) -> str:
        """Extract main content HTML."""
        selectors = [
            "div.doc-content",
            "div.content",
            "div.document-content",
            "div.vb-content",
            "article",
        ]
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return str(elem)
        return ""

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown."""
        if not html:
            return ""
        try:
            return md(
                html,
                heading_style="ATX",
                bullets="*",
                strip=["script", "style", "nav", "footer", "header"],
            ).strip()
        except Exception:
            return self._html_to_text(html)


def run_crawler(workers: int = 1, resume: bool = False, max_pages: int = None):
    """Run the Playwright crawler."""
    from scrapy.crawler import CrawlerProcess

    settings = get_project_settings()
    settings.set("FEEDS", {
        "data/raw/playwright_documents.jsonl": {
            "format": "jsonlines",
            "encoding": "utf8",
            "overwrite": False,
        },
    })

    if max_pages:
        settings.set("CLOSESPIDER_PAGE_COUNT", max_pages)

    process = CrawlerProcess(settings)
    process.crawl(TVPLPlaywrightSpider)
    process.start()

    # Export stats
    state = StateManager()
    stats = state.get_stats()
    logger.info(f"\n=== Final Statistics ===")
    logger.info(f"Pages crawled: {stats.get('pages_crawled', 0)}")
    logger.info(f"Documents extracted: {stats.get('documents_extracted', 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVPL Playwright Crawler")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of workers")
    parser.add_argument("--resume", action="store_true", help="Resume from last state")
    parser.add_argument("--max-pages", "-m", type=int, help="Maximum pages to crawl")

    args = parser.parse_args()

    try:
        run_crawler(workers=args.workers, resume=args.resume, max_pages=args.max_pages)
    except KeyboardInterrupt:
        logger.info("\nCrawler stopped by user")
        sys.exit(0)
