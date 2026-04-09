---
description: Test crawler and validate data
---

Test crawler before full crawl and validate processed data.

**Test Crawler:**
```bash
# Run all tests
uv run tvpl-test

# Test components:
# 1. Dependencies check
# 2. Website accessibility
# 3. Search parsing
# 4. Document extraction
# 5. Scrapy spider (1 page)
```

**Validate Data:**
```bash
# Validate processed documents
uv run tvpl-validate

# Checks:
# - Required fields present
# - Vietnamese character content
# - Content length distribution
# - Document type distribution
# - Duplicate detection
```

**Quick Test Before Crawl:**
```bash
# 1. Test crawler
uv run tvpl-test

# 2. If all pass, start crawling
uv run python crawler/parallel_crawler.py
```

**Test Output:**
```
[1/6] Testing dependencies...
✅ scrapy - Scrapy framework
✅ bs4 - BeautifulSoup4
✅ markdownify - HTML to Markdown
...

[2/6] Testing website access...
✅ Website accessible: 200

[3/6] Testing search results parsing...
✅ Found 20 links with selector: div.news-item ul li a
...
```

**After Crawl:**
```bash
# Validate data quality
uv run tvpl-validate --detailed

# Shows:
# - Missing fields
# - Non-Vietnamese documents
# - Content statistics
# - Document type distribution
```
