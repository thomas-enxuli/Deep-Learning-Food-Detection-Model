# coding: utf-8
from bs4 import BeautifulSoup
import urllib.request
url = 'https://unsplash.com/'
req = urllib.request.Request(url)
response = urllib.request.urlopen(req)
the_page = response.read()
the_page
soup = BeautifulSoup(the_page)
soup = BeautifulSoup(the_page, 'html-parser')
soup = BeautifulSoup(the_page, 'html.parser')
soup
soup.find_all('img')
img=_
img
img[0]
img[0]['src']

