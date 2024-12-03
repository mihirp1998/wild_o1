
from scraper import Scraper
from extract import Extractor
from filters import title_filter, keywords_filter, abstract_filter
from selector import Selector
from utils import save_papers, load_papers
import ipdb
st = ipdb.set_trace

years = [
    '2023',
    '2022',
    '2021',
    '2020',
    '2019',
    '2018'
]
conferences = [
    'NeurIPS',
    'ICLR',
    'ICML',
    'ECCV',
    'CVPR',
    'ICCV',
    # 'EMNLP',
]
keywords = [
    'attention', 'transformer'
]

def modify_paper(paper):
  paper.forum = f"https://openreview.net/forum?id={paper.forum}"
  paper.content['pdf'] = f"https://openreview.net{paper.content['pdf']}"
  return paper

# what fields to extract
extractor = Extractor(fields=['forum'], subfields={'content':['title', 'keywords', 'abstract', 'pdf', 'match']})

# if you want to select papers manually among the scraped papers

# select all scraped papers
selector = None

scraper = Scraper(conferences=conferences, years=years, keywords=keywords, extractor=extractor, fpath='attention.csv', fns=[modify_paper], selector=selector)

# adding filters to filter on
scraper.add_filter(title_filter)
scraper.add_filter(keywords_filter)
scraper.add_filter(abstract_filter)

st()
scraper()
