import requests

def simple_crawler(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("Content : ")
        print(response.text)
    else:
        print("Failed to fetch code:",response.status_code)

url_scrape = 'http://ajce.in'
simple_crawler(url_scrape)