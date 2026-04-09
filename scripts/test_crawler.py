#!/usr/bin/env python3
"""
Quick test script to verify crawler works before running full crawl.

Tests:
1. Check website accessibility
2. Parse a few search results
3. Extract a sample document
4. Validate data extraction
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from scrapy.utils.project import get_project_settings


def test_website_access():
    """Test if website is accessible."""
    import requests
    try:
        response = requests.get("https://thuvienphapluat.vn/", timeout=10)
        print(f"✅ Website accessible: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Website access failed: {e}")
        return False


def test_search_parsing():
    """Test parsing of search results page."""
    import requests
    from bs4 import BeautifulSoup

    url = "https://thuvienphapluat.vn/page/tim-van-ban.aspx?page=1"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")

        # Try different selectors
        selectors = [
            "div.search-result-item a.title-link",
            "div.news-item ul li a",
            "a.doc-title",
        ]

        found_any = False
        for sel in selectors:
            links = soup.select(sel)
            if links:
                print(f"✅ Found {len(links)} links with selector: {sel}")
                print(f"   Sample: {links[0].get('href', '')[:80]}...")
                found_any = True
                break

        if not found_any:
            print("⚠️  No document links found - selectors may need update")
            print("   HTML preview:")
            print(soup.prettify()[:500])

        return found_any
    except Exception as e:
        print(f"❌ Search parsing failed: {e}")
        return False


def test_document_extraction():
    """Test extraction from a sample document page."""
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    # Try to get a document URL
    url = "https://thuvienphapluat.vn/van-ban/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")

        # Find first document link
        doc_link = soup.select_one("a[href*='/van-ban/']")
        if not doc_link:
            print("⚠️  Could not find document link to test")
            return False

        doc_url = "https://thuvienphapluat.vn" + doc_link.get('href', '')
        print(f"Testing document: {doc_url}")

        response = requests.get(doc_url, timeout=10)
        soup = BeautifulSoup(response.content, "lxml")

        # Extract title
        title = soup.select_one("h1.doc-title, h1.title, h1")
        print(f"✅ Title: {title.get_text().strip() if title else 'N/A'}")

        # Extract content
        content = soup.select_one("div.doc-content, div.content, article")
        if content:
            html = str(content)
            text = content.get_text(separator="\n", strip=True)
            markdown = md(html)

            print(f"✅ Content length: {len(text)} chars")
            print(f"✅ Markdown length: {len(markdown)} chars")
            print(f"\nContent preview:\n{text[:200]}...")
            return True
        else:
            print("⚠️  No content found - may need selector update")
            return False

    except Exception as e:
        print(f"❌ Document extraction failed: {e}")
        return False


def test_scrapy_spider():
    """Test running spider for just 1 page."""
    print("\n=== Testing TVPL Spider (1 page) ===")

    output_file = Path("test_output.jsonl")
    if output_file.exists():
        output_file.unlink()

    settings = get_project_settings()
    settings.set("FEEDS", {str(output_file): {"format": "jsonlines"}})
    settings.set("CLOSESPIDER_PAGE_COUNT", 1)  # Stop after 1 page
    settings.set("LOG_LEVEL", "INFO")

    from crawler.spiders.tvpl_spider import TvplSpider

    process = CrawlerProcess(settings)
    process.crawl(TvplSpider)
    process.start()

    if output_file.exists():
        with open(output_file, "r") as f:
            lines = f.readlines()
        print(f"✅ Spider ran successfully, extracted {len(lines)} items")

        if lines:
            sample = json.loads(lines[0])
            print(f"\nSample item keys: {list(sample.keys())}")
            print(f"Title: {sample.get('title', 'N/A')[:60]}...")
            print(f"Doc ID: {sample.get('doc_id', 'N/A')}")
            print(f"Has markdown: {bool(sample.get('content_markdown'))}")

        output_file.unlink()
        return True
    else:
        print("❌ No output file created")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print("\n=== Testing Dependencies ===")

    deps = {
        "scrapy": "Scrapy framework",
        "bs4": "BeautifulSoup4",
        "markdownify": "HTML to Markdown",
        "underthesea": "Vietnamese NLP",
        "datasets": "HuggingFace datasets",
        "pandas": "Data processing",
    }

    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module)
            print(f"✅ {module:15} - {desc}")
        except ImportError:
            print(f"❌ {module:15} - NOT INSTALLED")
            all_ok = False

    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("TVPL Crawler Test Suite")
    print("=" * 60)

    results = {}

    print("\n[1/6] Testing dependencies...")
    results["dependencies"] = test_dependencies()

    print("\n[2/6] Testing website access...")
    results["website"] = test_website_access()

    if results["website"]:
        print("\n[3/6] Testing search results parsing...")
        results["search"] = test_search_parsing()

        print("\n[4/6] Testing document extraction...")
        results["document"] = test_document_extraction()

    print("\n[5/6] Testing Scrapy spider...")
    # Skip if dependencies failed
    if results["dependencies"]:
        results["spider"] = test_scrapy_spider()
    else:
        print("⚠️  Skipping spider test - dependencies missing")
        results["spider"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {test}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready to run full crawl.")
        print("\nTo start:")
        print("  uv run tvpl-crawl")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before full crawl.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
