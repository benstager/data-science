from bs4 import BeautifulSoup
import requests 

# Gets html text and tags from certain url
# could write url as string then pass through

# RETURNS FULL HTML PAGE
html = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')

# Finding all links with 'press releases' in them
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

print(links)