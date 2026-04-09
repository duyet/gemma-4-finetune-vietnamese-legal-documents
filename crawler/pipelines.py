"""Scrapy pipelines for processing legal documents."""

import json
import os
from datetime import datetime
from pathlib import Path
from scrapy.exceptions import DropItem


class DuplicatesPipeline:
    """Filter out duplicate documents based on doc_id."""

    def __init__(self):
        self.seen_ids = set()
        # Load existing IDs if resuming
        self.load_existing_ids()

    def load_existing_ids(self):
        """Load existing document IDs from previous runs."""
        output_path = Path("../data/raw/documents.jsonl")
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        if "doc_id" in doc:
                            self.seen_ids.add(doc["doc_id"])
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self.seen_ids)} existing document IDs")

    def process_item(self, item, spider):
        if item.get("doc_id") in self.seen_ids:
            raise DropItem(f"Duplicate document found: {item['doc_id']}")
        self.seen_ids.add(item.get("doc_id"))
        return item


class LegalDocumentPipeline:
    """Process and validate legal document items."""

    def process_item(self, item, spider):
        # Ensure required fields
        required_fields = ["doc_id", "title", "url", "content_text"]
        for field in required_fields:
            if not item.get(field):
                spider.logger.warning(f"Missing required field '{field}' in {item.get('url')}")
                # Still allow through, but log it

        # Add timestamp
        item["crawled_at"] = datetime.utcnow().isoformat()

        # Clean whitespace in text fields
        for field in ["title", "doc_number", "content_text"]:
            if field in item and item[field]:
                item[field] = " ".join(item[field].split())

        return item


class JsonWriterPipeline:
    """Write items to JSONL file with proper encoding."""

    def __init__(self):
        self.file = None
        self.output_dir = Path("../data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def open_spider(self, spider):
        # Use Scrapy's built-in feed exports instead
        pass

    def close_spider(self, spider):
        # Summary stats
        pass

    def process_item(self, item, spider):
        # Let Scrapy's feed exports handle writing
        return item
