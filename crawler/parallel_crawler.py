#!/usr/bin/env python3
"""
TVPL Parallel Crawler - Single File, Robust, Resume-capable

Features:
- Stop/resume capability
- Parallel workers (multi-process)
- State persistence (SQLite)
- Deduplication (URL + doc_id)
- Rate limiting (polite crawling)
- Progress tracking
- Error handling & retry
- Statistics & reporting

Usage:
    # Single worker
    python crawler/parallel_crawler.py

    # Multiple workers
    python crawler/parallel_crawler.py --workers 4

    # Resume from last state
    python crawler/parallel_crawler.py --resume

    # Stats only
    python crawler/parallel_crawler.py --stats
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

import click
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlStats:
    """Crawler statistics."""
    total_urls_seen: int = 0
    unique_urls: int = 0
    pages_crawled: int = 0
    documents_extracted: int = 0
    errors: int = 0
    start_time: str = ""
    last_update: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CrawlStats":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class StateManager:
    """Manage crawler state in SQLite database."""

    def __init__(self, db_path: str = "data/raw/.crawler_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # URLs table (seen URLs)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_urls (
                url TEXT PRIMARY KEY,
                first_seen TEXT,
                worker_id TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Documents table (extracted documents)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                data JSON,
                crawled_at TEXT,
                worker_id TEXT
            )
        """)

        # Workers table (parallel workers)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                current_page INTEGER DEFAULT 1,
                last_heartbeat TEXT,
                documents_fetched INTEGER DEFAULT 0
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
            (json.dumps(CrawlStats().to_dict()),)
        )
        self.conn.commit()

    def is_url_seen(self, url: str) -> bool:
        """Check if URL has been seen."""
        row = self.conn.execute(
            "SELECT 1 FROM seen_urls WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def mark_url_seen(self, url: str, worker_id: str = "main"):
        """Mark URL as seen."""
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, first_seen, worker_id) VALUES (?, ?, ?)",
            (url, datetime.now().isoformat(), worker_id)
        )
        self.conn.commit()

    def is_doc_extracted(self, doc_id: str) -> bool:
        """Check if document already extracted."""
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def save_document(self, doc: dict, worker_id: str = "main"):
        """Save extracted document."""
        doc_id = doc.get("doc_id")
        if not doc_id:
            return False

        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO documents (doc_id, url, data, crawled_at, worker_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, doc.get("url"), json.dumps(doc), datetime.now().isoformat(), worker_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save document {doc_id}: {e}")
            return False

    def get_stats(self) -> CrawlStats:
        """Get crawler statistics."""
        row = self.conn.execute("SELECT value FROM stats WHERE key = 'global'").fetchone()
        if row:
            return CrawlStats.from_dict(json.loads(row[0]))
        return CrawlStats()

    def update_stats(self, **kwargs):
        """Update statistics."""
        stats = self.get_stats()
        for key, value in kwargs.items():
            if hasattr(stats, key):
                setattr(stats, key, value)
        stats.last_update = datetime.now().isoformat()

        self.conn.execute(
            "UPDATE stats SET value = ? WHERE key = 'global'",
            (json.dumps(stats.to_dict()),)
        )
        self.conn.commit()

    def register_worker(self, worker_id: str):
        """Register a parallel worker."""
        self.conn.execute(
            """INSERT OR REPLACE INTO workers (worker_id, status, last_heartbeat)
               VALUES (?, 'active', ?)""",
            (worker_id, datetime.now().isoformat())
        )
        self.conn.commit()

    def update_worker(self, worker_id: str, **kwargs):
        """Update worker status."""
        valid_fields = ["status", "current_page", "documents_fetched"]
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [datetime.now().isoformat(), worker_id]

        self.conn.execute(
            f"UPDATE workers SET {set_clause}, last_heartbeat = ? WHERE worker_id = ?",
            values
        )
        self.conn.commit()

    def get_next_page(self) -> int:
        """Get next page to crawl (for parallel workers)."""
        # Get max page from all workers
        row = self.conn.execute(
            "SELECT MAX(current_page) FROM workers"
        ).fetchone()
        return (row[0] or 0) + 1

    def get_all_documents(self) -> Generator[dict, None, None]:
        """Yield all saved documents."""
        for row in self.conn.execute("SELECT data FROM documents"):
            yield json.loads(row[0])

    def export_to_jsonl(self, output_path: str):
        """Export all documents to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self.get_all_documents():
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"Exported {count} documents to {output_path}")
        return count


class TVPLCrawler:
    """TVPL Website Crawler."""

    BASE_URL = "https://thuvienphapluat.vn"
    SEARCH_URL = f"{BASE_URL}/page/tim-van-ban.aspx"

    def __init__(
        self,
        state: StateManager,
        worker_id: str = "main",
        delay: float = 2.5,
        max_pages: int = None,
    ):
        self.state = state
        self.worker_id = worker_id
        self.delay = delay
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })

        # Register worker
        self.state.register_worker(worker_id)

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._shutdown = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Worker {self.worker_id}: Received shutdown signal, saving state...")
        self._shutdown = True

    def _rate_limit(self):
        """Apply rate limiting."""
        time.sleep(self.delay)

    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic."""
        for attempt in range(retries):
            if self._shutdown:
                return None

            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

        return None

    def extract_doc_id(self, url: str) -> Optional[str]:
        """Extract document ID from URL."""
        match = re.search(r"-(\d+)\.aspx$", url)
        return match.group(1) if match else None

    def parse_search_page(self, page: int) -> List[str]:
        """Parse search results page and extract document URLs."""
        params = {
            "page": page,
            "keyword": "",
            "area": "0",
            "match": "False",
        }

        url = f"{self.SEARCH_URL}?{ '&'.join(f'{k}={v}' for k, v in params.items()) }"

        response = self._make_request(url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, "lxml")

        # Multiple selectors for robustness
        selectors = [
            "div.search-result-item a.title-link",
            "div.news-item ul li a",
            "a.doc-title",
            "table.doc-list a",
        ]

        links = []
        for selector in selectors:
            found = soup.select(selector)
            if found:
                links = [a.get("href") for a in found if a.get("href")]
                break

        if not links:
            return []

        # Convert to absolute URLs
        urls = [urljoin(self.BASE_URL, link) for link in links]

        # Update stats
        self.state.update_stats(total_urls_seen=self.state.get_stats().total_urls_seen + len(urls))

        return urls

    def parse_document(self, url: str) -> Optional[dict]:
        """Parse individual document page."""
        doc_id = self.extract_doc_id(url)
        if not doc_id:
            return None

        # Check if already extracted
        if self.state.is_doc_extracted(doc_id):
            return None

        response = self._make_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.content, "lxml")

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
            "has_english_version": False,
            "original_doc_url": None,

            # Technical
            "crawled_at": datetime.now().isoformat(),
            "worker_id": self.worker_id,
            "crawl_source": "thuvienphapluat.vn",
        }

        # Extract content
        content = document["content_html"]
        if content:
            document["content_text"] = self._html_to_text(content)
            document["content_markdown"] = self._html_to_markdown(content)

        # Extract relationships
        relationships = self._extract_relationships(soup)
        document.update(relationships)

        # Detect language
        document["language"] = self._detect_language(content)
        document["has_english_version"] = bool(soup.select("a:contains('English')"))

        return document

    def _extract_category(self, url: str) -> str:
        """Extract category from URL path."""
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

    def _extract_relationships(self, soup: BeautifulSoup) -> dict:
        """Extract document relationships."""
        relationships = {
            "related_docs": [],
            "amends_docs": [],
            "repeals_docs": [],
            "cites_docs": [],
        }

        # Find relationship sections
        rel_sections = soup.select("div.relationships, div.related-docs, div.vb-lien-quan")
        for section in rel_sections:
            links = section.select("a[href*='/van-ban/']")
            section_text = section.get_text().lower()

            for link in links:
                href = link.get("href", "")
                if "sửa đổi" in section_text or " amendments " in section_text:
                    relationships["amends_docs"].append(urljoin(self.BASE_URL, href))
                elif "bãi bỏ" in section_text or " repeals " in section_text:
                    relationships["repeals_docs"].append(urljoin(self.BASE_URL, href))
                else:
                    relationships["related_docs"].append(urljoin(self.BASE_URL, href))

        # Deduplicate
        for key in relationships:
            relationships[key] = list(set(relationships[key]))

        return relationships

    def _detect_language(self, content: str) -> str:
        """Detect if content has Vietnamese."""
        if not content:
            return "unknown"
        viet_chars = "àáảãạăằẳẵặâầẩẫậèéẻẽẹêềểễệìíỉĩịòóỏõọôồổỗộơờởỡợùúủũụưừửữựỳýỷỹỵđ"
        return "vn" if re.search(f"[{viet_chars}]", content) else "en"

    def run(self, start_page: int = 1):
        """Run crawler."""
        logger.info(f"Worker {self.worker_id}: Starting from page {start_page}")

        page = start_page
        consecutive_empty = 0
        max_empty = 3

        while not self._shutdown:
            # Check max pages
            if self.max_pages and page > self.max_pages:
                logger.info(f"Worker {self.worker_id}: Reached max pages ({self.max_pages})")
                break

            # Rate limit
            self._rate_limit()

            # Update worker status
            self.state.update_worker(self.worker_id, current_page=page)

            # Parse search page
            urls = self.parse_search_page(page)
            if not urls:
                consecutive_empty += 1
                if consecutive_empty >= max_empty:
                    logger.info(f"Worker {self.worker_id}: No results for {max_empty} pages, stopping")
                    break
                page += 1
                continue

            consecutive_empty = 0
            logger.info(f"Worker {self.worker_id}: Page {page} - Found {len(urls)} URLs")

            # Process documents
            docs_fetched = 0
            for url in tqdm(urls, desc=f"Page {page}", disable=self.worker_id != "main"):
                if self._shutdown:
                    break

                # Check if URL seen
                if self.state.is_url_seen(url):
                    continue

                self.state.mark_url_seen(url, self.worker_id)

                # Parse document
                doc = self.parse_document(url)
                if doc:
                    if self.state.save_document(doc, self.worker_id):
                        docs_fetched += 1

            # Update stats
            stats = self.state.get_stats()
            self.state.update_stats(
                pages_crawled=stats.pages_crawled + 1,
                documents_extracted=stats.documents_extracted + docs_fetched,
            )
            self.state.update_worker(self.worker_id, documents_fetched=docs_fetched)

            logger.info(
                f"Worker {self.worker_id}: Page {page} complete - "
                f"{docs_fetched} documents, "
                f"total: {stats.documents_extracted + docs_fetched} docs"
            )

            page += 1

        logger.info(f"Worker {self.worker_id}: Stopping")


@click.command()
@click.option("--workers", "-w", default=1, help="Number of parallel workers")
@click.option("--start-page", "-s", default=1, help="Starting page")
@click.option("--max-pages", "-m", help="Maximum pages to crawl")
@click.option("--delay", "-d", default=2.5, help="Delay between requests (seconds)")
@click.option("--resume", is_flag=True, help="Resume from last state")
@click.option("--stats", is_flag=True, help="Show statistics only")
@click.option("--export", "-e", help="Export to JSONL file")
def main(workers: int, start_page: int, max_pages: int, delay: float, resume: bool, stats: bool, export: str):
    """TVPL Parallel Crawler."""

    state = StateManager()

    # Show stats
    if stats:
        stats_obj = state.get_stats()
        print("\n=== Crawler Statistics ===")
        print(f"Total URLs seen: {stats_obj.total_urls_seen:,}")
        print(f"Unique URLs: {stats_obj.unique_urls:,}")
        print(f"Pages crawled: {stats_obj.pages_crawled:,}")
        print(f"Documents extracted: {stats_obj.documents_extracted:,}")
        print(f"Errors: {stats_obj.errors:,}")
        print(f"Started: {stats_obj.start_time}")
        print(f"Last update: {stats_obj.last_update}")

        # Count in DB
        doc_count = state.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        url_count = state.conn.execute("SELECT COUNT(*) FROM seen_urls").fetchone()[0]
        print(f"\nDatabase:")
        print(f"  URLs in database: {url_count:,}")
        print(f"  Documents in database: {doc_count:,}")
        return

    # Export
    if export:
        count = state.export_to_jsonl(export)
        print(f"Exported {count} documents to {export}")
        return

    # Initialize stats if new
    stats_obj = state.get_stats()
    if not stats_obj.start_time:
        state.update_stats(start_time=datetime.now().isoformat())

    # Determine start page
    if resume:
        start_page = state.get_next_page()
        logger.info(f"Resuming from page {start_page}")

    # Run crawler(s)
    if workers == 1:
        # Single worker
        crawler = TVPLCrawler(
            state=state,
            worker_id="main",
            delay=delay,
            max_pages=max_pages,
        )
        crawler.run(start_page=start_page)
    else:
        # Parallel workers
        import multiprocessing

        def worker_run(worker_num: int):
            crawler = TVPLCrawler(
                state=state,
                worker_id=f"worker-{worker_num}",
                delay=delay,
                max_pages=max_pages,
            )
            # Each worker gets different starting page
            worker_start = start_page + worker_num
            crawler.run(start_page=worker_start)

        processes = []
        try:
            for i in range(workers):
                p = multiprocessing.Process(target=worker_run, args=(i,))
                p.start()
                processes.append(p)
                logger.info(f"Started worker-{i}")

            # Wait for all
            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            for p in processes:
                p.terminate()
                p.join()

    # Final stats
    final_stats = state.get_stats()
    logger.info(f"\n=== Final Statistics ===")
    logger.info(f"Documents extracted: {final_stats.documents_extracted:,}")
    logger.info(f"Pages crawled: {final_stats.pages_crawled:,}")
    logger.info(f"Runtime: {final_stats.last_update}")

    # Auto-export
    output_file = f"data/raw/documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    state.export_to_jsonl(output_file)
    logger.info(f"Auto-exported to {output_file}")


if __name__ == "__main__":
    main()
