from bs4 import BeautifulSoup
from typing import Dict, Set
import requests
import re

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

# soup('a') is an array of dictionaries,
# where each a is a key-value pair with key 'href'
all_urls = [a['href']
            for a in soup('a')
            if a.has_attr('href')]

# using RegEx to refine this
regex = r"^https?://.*\.house\.gov/?$"

# applying specific RegEx
good_urls = [url for url in all_urls if re.match(regex,url)]

# removing duplciates
good_urls = list(set(good_urls))

press_releases: Dict[str, Set[str]] = {} # key is a string, value is potentially array of strings

for url in good_urls:
    html = requests.get(url).text # getting html text
    soup = BeautifulSoup(html, 'html5lib') 
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text
                .lower()} # getting press release link for each good url
                          # a is typically used for urls in html
    print('house url:', url, 'press link:', pr_links)
    print()
    press_releases[url] = pr_links # creating dictionary
                                   # key: rep url, value: corresponding pr link

# defining function for paragraph mentions
# will return true if keyword is found in a paragraph
def paragraph_mentions(text, keyword):
    
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower() 
               for paragraph in paragraphs)

# testing
for url, prls in press_releases.items():
    for prl in prls:
        suburl = f"{url}/{prl}"
        html = requests.get(suburl).text

        if paragraph_mentions(html, 'America'):
            print(f"{url}")

