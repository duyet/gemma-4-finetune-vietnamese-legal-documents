"""Scrapy settings for thuvienphapluat.vn crawler.

Designed to be respectful to the target server:
- 2-3 second delays between requests
- Respect robots.txt
- Auto-throttling enabled
- Retry failed requests
"""

import os

# Scrapy project settings
BOT_NAME = "tvpl"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

# Project name
PROJECT_NAME = "tvpl"

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy
CONCURRENT_REQUESTS = 4  # Conservative to avoid overwhelming server
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 2.5  # 2.5 second delay - respectful to server
RANDOMIZE_DOWNLOAD_DELAY = 0.5  # Add 0-0.5s randomness

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    "scrapy.spidermiddlewares.httperror.HttpErrorMiddleware": 50,
    "scrapy_splash.SplashMiddleware": 743,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "scrapy_user_agents.middlewares.RandomUserAgentMiddleware": 400,
    "scrapy.downloadermiddlewares.retry.RetryMiddleware": 90,
    "scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware": 110,
}

# Enable or disable extensions
EXTENSIONS = {
    "scrapy.extensions.telnet.TelnetConsole": None,
}

# Configure item pipelines
ITEM_PIPELINES = {
    "crawler.pipelines.LegalDocumentPipeline": 300,
    "crawler.pipelines.DuplicatesPipeline": 200,
    "crawler.pipelines.JsonWriterPipeline": 400,
}

# Enable and configure the AutoThrottle extension
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = os.path.join(os.path.dirname(__file__), ".scrapy", "httpcache")
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504, 408, 429]
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Log settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# Feed exports (for output files)
FEEDS = {
    "../data/raw/documents.jsonl": {
        "format": "jsonlines",
        "encoding": "utf8",
        "indent": 0,
        "overwrite": False,  # Append mode for resume capability
    },
}

# Stats collection
STATS_CLASS = "scrapy.statscollectors.StatsCollector"

# Close spider on no data
CLOSESPIDER_TIMEOUT = 3600  # 1 hour max per run
CLOSESPIDER_ITEMCOUNT = None  # No item limit
