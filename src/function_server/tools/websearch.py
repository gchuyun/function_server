from typing import Iterator
from ..function_calling import tool
from search_engines import Google, Bing, config
import os
import random
from ..settings import WEB_SEARCH_ENGINE
from loguru import logger


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Config/92.2.2788.20",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5; rv:123.0esr) Gecko/20100101 Firefox/123.0esr",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
]

class MyGoogle(Google):
    def _selectors(self, element):
        '''Returns the appropriate CSS selector.'''
        selectors = {
            'url': 'a[href]', 
            'title': 'h3', 
            'text': 'div[style="-webkit-line-clamp:2"]',
            'links': 'div#search div.g', 
            'next': 'a[href][aria-label="Page {page}"]'
        }
        return selectors[element]

@tool
def web_search(input: str) -> Iterator[dict]:
    '''a search engine. useful when you need to answer questions about current events or are unsure or uncertain about certain things. input should be a search query.'''
    if WEB_SEARCH_ENGINE.lower() == "google":
        engine = MyGoogle()
    else:
        engine = Bing()

    i = random.randint(0, len(USER_AGENTS)-1)
    logger.info("FAKE_USER_AGENT: %s" % i)
    config.FAKE_USER_AGENT = USER_AGENTS[i]
    results = engine.search(input, 1)

    return [{"url": r.get('link'), "text": r.get('text')} for r in results.results() if r.get('text')][:10]

# requirements = """
#     googlesearch-python>=1.2.3
# """

def main():
    print(web_search("张学友"))