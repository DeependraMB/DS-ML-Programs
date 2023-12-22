import requests
from bs4 import BeautifulSoup

def simple_crawler(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content,"html.parser")
        print("Title:",soup.title.string)
        print("Content:\n")
        print(soup.get_text())
    else:
        print("Error in fetching web page,status code :",response.status_code)

url_scrape = "http://ajce.in"
simple_crawler(url_scrape)