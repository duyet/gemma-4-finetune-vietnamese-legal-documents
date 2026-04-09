"""Main entry point for running the crawler from command line."""

import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def main():
    """Run the TVPL spider."""
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    process.crawl("tvpl")
    process.start()


if __name__ == "__main__":
    main()
