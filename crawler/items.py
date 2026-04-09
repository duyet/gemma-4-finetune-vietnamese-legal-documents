"""Scrapy items for legal document data structures."""

import scrapy
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class LegalDocument:
    """Vietnamese legal document schema with rich metadata."""

    # Basic identification
    url: str
    doc_id: str
    title: str

    # Document categorization (URL structure based)
    category: str  # Main category from URL path (e.g., "Giao-thong-Van-tai")
    sub_category: str  # Sub-category if available
    doc_type: str  # Loại văn bản (Luật, Nghị định, Thông tư, etc.)

    # Document metadata
    doc_number: str  # Số ký hiệu
    issuing_authority: str  # Cơ quan ban hành
    signatory: Optional[str]  # Người ký
    signatory_title: Optional[str]  # Chức danh người ký

    # Dates
    issue_date: str  # Ngày ban hành
    effective_date: str  # Ngày có hiệu lực
    expiry_date: Optional[str]  # Ngày hết hiệu lực
    published_date: Optional[str]  # Ngày đăng công báo

    # Status
    status: str  # Tình trạng hiệu lực (Còn hiệu lực, Hết hiệu lực, etc.)
    effect_status: Optional[str]  # Hiệu lực thi hành

    # Classification
    sector: str  # Ngành
    field: str  # Lĩnh vực
    scope: Optional[str]  # Phạm vi áp dụng

    # Content
    content_html: str  # Raw HTML
    content_text: str  # Cleaned plain text
    content_markdown: str  # Markdown format
    summary: Optional[str]  # Tóm tắt (if available)

    # References and relationships
    related_docs: list[str]  # Referenced document IDs
    amends_docs: list[str]  # Documents this amends
    amended_by_docs: list[str]  # Documents that amend this
    repeals_docs: list[str]  # Documents this repeals
    repealed_by_docs: list[str]  # Documents that repeal this
    cites_docs: list[str]  # Documents this cites
    cited_by_docs: list[str]  # Documents that cite this

    # Additional metadata
    tags: list[str]  # Keywords/tags
    language: str  # vn, en, or bilingual
    has_english_version: bool  # Whether English translation exists
    original_doc_url: Optional[str]  # Link to original scan

    # Technical metadata
    created_at: str  # ISO timestamp when crawled
    updated_at: str  # ISO timestamp when last updated
    crawled_at: str  # ISO timestamp of this crawl
    crawl_source: str  # thuvienphapluat.vn


class LegalDocumentItem(scrapy.Item):
    """Scrapy Item for legal documents with enhanced metadata."""

    # ===== Basic Identification =====
    url = scrapy.Field()
    doc_id = scrapy.Field()
    title = scrapy.Field()

    # ===== Categorization =====
    category = scrapy.Field()  # From URL path structure
    sub_category = scrapy.Field()  # Additional categorization
    doc_type = scrapy.Field()  # Loại văn bản

    # ===== Document Metadata =====
    doc_number = scrapy.Field()  # Số ký hiệu
    issuing_authority = scrapy.Field()  # Cơ quan ban hành
    signatory = scrapy.Field()  # Người ký
    signatory_title = scrapy.Field()  # Chức danh

    # ===== Dates =====
    issue_date = scrapy.Field()  # Ngày ban hành
    effective_date = scrapy.Field()  # Ngày có hiệu lực
    expiry_date = scrapy.Field()  # Ngày hết hiệu lực
    published_date = scrapy.Field()  # Ngày đăng công báo

    # ===== Status =====
    status = scrapy.Field()  # Tình trạng hiệu lực
    effect_status = scrapy.Field()  # Hiệu lực thi hành

    # ===== Classification =====
    sector = scrapy.Field()  # Ngành
    field = scrapy.Field()  # Lĩnh vực
    scope = scrapy.Field()  # Phạm vi áp dụng

    # ===== Content =====
    content_html = scrapy.Field()  # Raw HTML
    content_text = scrapy.Field()  # Cleaned plain text
    content_markdown = scrapy.Field()  # Markdown format
    summary = scrapy.Field()  # Tóm tắt

    # ===== References & Relationships =====
    related_docs = scrapy.Field()  # All related documents
    amends_docs = scrapy.Field()  # Sửa đổi
    amended_by_docs = scrapy.Field()  # Được sửa đổi bởi
    repeals_docs = scrapy.Field()  # Bãi bỏ
    repealed_by_docs = scrapy.Field()  # Được bãi bỏ bởi
    cites_docs = scrapy.Field()  # Trích dẫn
    cited_by_docs = scrapy.Field()  # Được trích dẫn bởi

    # ===== Additional Metadata =====
    tags = scrapy.Field()  # Tags/keywords
    language = scrapy.Field()  # vn, en, both
    has_english_version = scrapy.Field()  # Boolean
    original_doc_url = scrapy.Field()  # Original scan link

    # ===== Technical Metadata =====
    created_at = scrapy.Field()  # First crawled
    updated_at = scrapy.Field()  # Last updated
    crawled_at = scrapy.Field()  # This crawl timestamp
    crawl_source = scrapy.Field()  # Source website
