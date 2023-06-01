from bs4 import BeautifulSoup
import requests

url = ("https://raw.githubusercontent.com/"
       "joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html,'html5lib')

first_par = soup.find('p')
first_par_text = soup.p.text
first_par_words = soup.p.text.split()

all_pars = soup('p')
print(len(all_pars))

