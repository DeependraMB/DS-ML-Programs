import requests
from bs4 import BeautifulSoup


def simple_scraper(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content,"html.parser")
        print("Title=",soup.title.string)
        print("Content:")
        print(soup.get_text())
    else:
        print("Failed to fetch the page",response.status_code)


url_to_scrape = "https://ajce.in"
simple_scraper(url_to_scrape)
