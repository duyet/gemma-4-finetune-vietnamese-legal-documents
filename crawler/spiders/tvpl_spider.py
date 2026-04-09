"""
Thư Viện Pháp Luạt (TVPL) Spider

Crawls thuvienphapluat.vn to extract Vietnamese legal documents with rich metadata.

Features:
- Discovers documents via search pagination
- Extracts full document metadata
- Captures document relationships (amends, repeals, cites)
- Resume capability via state file
- Polite rate limiting
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse, parse_qs

import scrapy
from scrapy.http import Response
from scrapy.utils.project import get_project_settings

from crawler.items import LegalDocumentItem


class TvplSpider(scrapy.Spider):
    """Spider for crawling thuvienphapluat.vn legal documents."""

    name = "tvpl"
    allowed_domains = ["thuvienphapluat.vn"]

    # Custom settings
    custom_settings = {
        "DOWNLOAD_DELAY": 2.5,
        "CONCURRENT_REQUESTS": 2,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_file = Path("../data/raw/.spider_state.json")
        self.state = self.load_state()

        # Base URL for search
        self.base_url = "https://thuvienphapluat.vn"
        self.search_url = f"{self.base_url}/page/tim-van-ban.aspx"

    def start_requests(self) -> Generator[scrapy.Request, None, None]:
        """Generate initial search requests."""
        # Start from page 1 or resume from saved state
        start_page = self.state.get("last_page", 1)

        # Search parameters to get all documents
        params = {
            "keyword": "",
            "area": "0",  # All fields
            "match": "False",  # Contains words
            "type": "",  # All types
            "status": "0",  # All statuses
            "lan": "1",  # Vietnamese
            "sort": "1",  # Newest first
        }

        for page in range(start_page, 100000):  # Large number, will break when no more results
            url = f"{self.search_url}?page={page}&" + "&".join(f"{k}={v}" for k, v in params.items())
            yield scrapy.Request(
                url,
                callback=self.parse_search_results,
                meta={"page": page},
                dont_filter=True,
            )

    def parse_search_results(self, response: Response) -> Generator[scrapy.Request, None, None]:
        """Parse search results page and extract document URLs."""
        current_page = response.meta.get("page", 1)

        self.logger.info(f"Parsing search results page {current_page}")

        # Check if no results (end of pagination)
        no_results = response.css("div.no-results::text").get()
        if no_results and "không tìm thấy" in no_results.lower():
            self.logger.info("No more results found. Ending crawl.")
            return

        # Extract document URLs from search results
        # The site uses different layouts; try multiple selectors
        doc_links = response.css(
            "div.search-result-item a.title-link::attr(href), "
            "div.news-item ul li a::attr(href), "
            "table.doc-list td.title a::attr(href), "
            "a.doc-title::attr(href)"
        ).getall()

        self.logger.info(f"Found {len(doc_links)} documents on page {current_page}")

        # Deduplicate and filter
        seen_urls = self.state.get("seen_urls", set())
        unique_links = [urljoin(self.base_url, link) for link in doc_links if link]

        for link in unique_links:
            if link in seen_urls:
                continue
            seen_urls.add(link)

            # Extract doc_id from URL for deduplication
            doc_id = self.extract_doc_id(link)
            if not doc_id:
                continue

            yield scrapy.Request(
                link,
                callback=self.parse_document,
                meta={"doc_id": doc_id, "source_url": link},
            )

        # Update state
        self.state["last_page"] = current_page
        self.state["seen_urls"] = list(seen_urls)
        self.save_state()

        # Follow next page
        next_page = current_page + 1
        next_url = f"{self.search_url}?page={next_page}"
        yield scrapy.Request(
            next_url,
            callback=self.parse_search_results,
            meta={"page": next_page},
            dont_filter=True,
        )

    def parse_document(self, response: Response) -> Generator[LegalDocumentItem, None, None]:
        """Parse individual legal document page."""
        url = response.url
        doc_id = response.meta.get("doc_id", self.extract_doc_id(url))

        self.logger.info(f"Parsing document: {doc_id}")

        item = LegalDocumentItem()

        # Basic identification
        item["url"] = url
        item["doc_id"] = doc_id
        item["title"] = self.clean_text(response.css("h1.doc-title::text, h1.title::text, h1::text").get())

        # Categorization from URL
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p and not p.endswith(".aspx")]
        item["category"] = path_parts[0] if len(path_parts) > 0 else ""
        item["sub_category"] = path_parts[1] if len(path_parts) > 1 else ""

        # Document type
        item["doc_type"] = self.clean_text(
            response.css("span.doc-type::text, li.type::text").get() or
            self.extract_doc_type_from_title(item.get("title", ""))
        )

        # Document number (số ký hiệu)
        item["doc_number"] = self.clean_text(
            response.css("span.doc-number::text, li.number::text").get()
        )

        # Issuing authority (cơ quan ban hành)
        item["issuing_authority"] = self.clean_text(
            response.css("span.authority::text, li.organ::text").get()
        )

        # Signatory info
        item["signatory"] = self.clean_text(
            response.css("span.signer::text, li.nguoi-ky::text").get()
        )
        item["signatory_title"] = self.clean_text(
            response.css("span.signer-title::text, li.chuc-danh::text").get()
        )

        # Dates
        item["issue_date"] = self.clean_text(
            response.css("span.issue-date::text, li.ngay-bh::text").get()
        )
        item["effective_date"] = self.clean_text(
            response.css("span.effective-date::text, li.ngay-hl::text").get()
        )
        item["expiry_date"] = self.clean_text(
            response.css("span.expiry-date::text, li.ngay-hhl::text").get()
        )
        item["published_date"] = self.clean_text(
            response.css("span.published-date::text, li.ngay-dcb::text").get()
        )

        # Status
        item["status"] = self.clean_text(
            response.css("span.status::text, li.trang-thai::text, span.effect-status::text").get()
        )
        item["effect_status"] = self.clean_text(
            response.css("span.effect-status::text").get()
        )

        # Classification
        item["sector"] = self.clean_text(
            response.css("span.sector::text, li.nganh::text").get()
        )
        item["field"] = self.clean_text(
            response.css("span.field::text, li.linh-vuc::text").get()
        )
        item["scope"] = self.clean_text(
            response.css("span.scope::text, li.pham-vi::text").get()
        )

        # Content
        content_selectors = [
            "div.doc-content",
            "div.content",
            "div.document-content",
            "div.vb-content",
            "article",
        ]
        content_html = ""
        for selector in content_selectors:
            content = response.css(selector).get()
            if content:
                content_html = content
                break

        item["content_html"] = content_html
        item["content_text"] = self.extract_text_from_html(content_html)
        item["content_markdown"] = self.convert_html_to_markdown(content_html)

        # Summary if available
        item["summary"] = self.clean_text(
            response.css("div.summary::text, div.tom-tat::text").get()
        )

        # Extract document relationships
        relationships = self.extract_relationships(response)
        item.update(relationships)

        # Additional metadata
        item["tags"] = self.extract_tags(response)
        item["language"] = self.detect_language(response)
        item["has_english_version"] = self.has_english_version(response)
        item["original_doc_url"] = self.get_original_doc_url(response)

        # Technical metadata
        now = datetime.utcnow().isoformat()
        item["created_at"] = now  # First time we see it
        item["updated_at"] = now
        item["crawled_at"] = now
        item["crawl_source"] = "thuvienphapluat.vn"

        yield item

    def extract_relationships(self, response: Response) -> dict:
        """Extract document relationships (amends, repeals, cites, etc.)."""
        relationships = {
            "related_docs": [],
            "amends_docs": [],
            "amended_by_docs": [],
            "repeals_docs": [],
            "repealed_by_docs": [],
            "cites_docs": [],
            "cited_by_docs": [],
        }

        # Look for relationship sections
        # Common selectors for Vietnamese legal document relationship sections
        rel_sections = response.css(
            "div.relationships, div.related-docs, "
            "div.vb-lien-quan, div.vb-sua-doi, "
            "div.vb-bai-bo, div.vb-trich-dan"
        )

        for section in rel_sections:
            section_text = section.get().lower()
            links = section.css("a::attr(href)").getall()

            # Categorize links based on section text
            if "sửa đổi" in section_text or " amendments " in section_text:
                relationships["amends_docs"].extend(links)
            elif "bãi bỏ" in section_text or " repeals " in section_text:
                relationships["repeals_docs"].extend(links)
            elif "trích dẫn" in section_text or " cites " in section_text:
                relationships["cites_docs"].extend(links)

            # All related docs
            relationships["related_docs"].extend(links)

        # Extract from inline text citations
        # Pattern: [số văn bản] or references in parentheses
        text = response.get()
        citation_pattern = r'(?:về|theo|theo\s+điều|căn\s+cứ)\s+(?:luật|nghị\s+định|thông\s+tư|quyết\s+định)\s+(?:số\s+)?[\d/]+'
        citations = re.findall(citation_pattern, text, re.IGNORECASE)
        relationships["cites_docs"].extend(citations)

        # Deduplicate
        for key in relationships:
            relationships[key] = list(set(relationships[key]))

        return relationships

    def extract_doc_id(self, url: str) -> str | None:
        """Extract document ID from URL."""
        # Pattern: /van-ban/Category/DocNumber-Title-ID.aspx
        match = re.search(r"-(\d+)\.aspx$", url)
        if match:
            return match.group(1)
        return None

    def extract_doc_type_from_title(self, title: str) -> str:
        """Extract document type from title."""
        if not title:
            return ""
        # Common Vietnamese document types
        doc_types = [
            "Luật", "Nghị quyết", "Nghị định", "Thông tư",
            "Quyết định", "Quyết định số", "Chỉ thị", "Thông báo",
            "Công văn", "Pháp lệnh", "Lệnh", "Chương trình",
            "Kế hoạch", "Đề án", "Quy chế", "Quy trình",
        ]
        for dt in doc_types:
            if dt.lower() in title.lower():
                return dt
        return "Văn bản"

    def convert_html_to_markdown(self, html: str) -> str:
        """Convert HTML content to Markdown format using markdownify library."""
        if not html:
            return ""
        try:
            from markdownify import markdownify as md

            # Configure markdownify for legal documents
            markdown = md(
                html,
                heading_style="ATX",  # Use # style headings
                bullets="*",
                strip=["script", "style", "nav", "footer", "header", "aside"],
                convert=["p", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "a", "ul", "ol", "li", "table"],
            )
            return markdown.strip()
        except ImportError:
            self.logger.warning("markdownify not installed, falling back to text extraction")
            return self.extract_text_from_html(html)
        except Exception as e:
            self.logger.error(f"Error converting to markdown: {e}")
            return self.extract_text_from_html(html)

    def extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML content."""
        if not html:
            return ""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def extract_tags(self, response: Response) -> list[str]:
        """Extract tags/keywords from document."""
        tags = response.css("div.tags a::text, meta[name='keywords']::attr(content)").getall()
        return [self.clean_text(tag) for tag in tags if tag]

    def detect_language(self, response: Response) -> str:
        """Detect document language."""
        has_vi = response.css("body::text").re_first(r"[àáảãạăằẳẵặâầẩẫậèéẻẽẹêềểễệìíỉĩịòóỏõọôồổỗộơờởỡợùúủũụưừửữựỳýỷỹỵđ]")
        has_en = response.css("a[href*='en']").get()

        if has_vi and has_en:
            return "both"
        elif has_vi:
            return "vn"
        else:
            return "en"

    def has_english_version(self, response: Response) -> bool:
        """Check if document has English translation."""
        return bool(response.css("a:contains('English'), a:contains('Tiếng Anh')").get())

    def get_original_doc_url(self, response: Response) -> str | None:
        """Get URL to original document scan."""
        link = response.css("a:contains('Văn bản gốc'), a:contains('Original')::attr(href)").get()
        return urljoin(self.base_url, link) if link else None

    def clean_text(self, text: str | None) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        return " ".join(text.strip().split())

    def load_state(self) -> dict:
        """Load spider state for resume capability."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"last_page": 1, "seen_urls": [], "seen_ids": set()}

    def save_state(self):
        """Save spider state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
