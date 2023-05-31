from bs4 import BeautifulSoup
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
print(len(good_urls))

# removing duplciates
good_urls = list(set(good_urls))
print(len(good_urls))

