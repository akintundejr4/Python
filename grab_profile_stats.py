import os 
import sys
import ssl
import time
import json
import requests 
import pymysql
import urllib.parse
import urllib.error
import urllib.request
import pymysql.cursors
from random import choice
from datetime import datetime
from bs4 import BeautifulSoup

# This will tell you where your script is actually being run from. 
# print(os.getcwd())

provided_json_file = sys.argv[1]

class InstagramScraper:
    """Helper class for scrapping Instagram profiles, part of a social media campaign effort project."""

    def getinfo(self, url):
        html = urllib.request.urlopen(url, context=self.ctx).read()
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.find_all('meta', attrs={'property': 'og:description'
                             })
        text = data[0].get('content').split()
        user = '%s %s %s' % (text[-3], text[-2], text[-1])
        followers = text[0]
        following = text[2]
        posts = text[4]
        info={}
        info["User"] = user
        info["Followers"] = followers
        info["Following"] = following
        info["Posts"] = posts
        self.info_arr.append(info)

    def main(self):
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.info_arr=[]
        self.content = [] 

        with open (provided_json_file) as json_data:
            data = json.load(json_data,)
            for item in data:
                url = f'https://www.instagram.com/{item}'
                self.content.append(url)  

        self.content = [x.strip() for x in self.content]
            
        for url in self.content:
            try:
                self.getinfo(url)
                time.sleep(1)
            except Exception as e:
                print("This URL was not processed succesfully " + url) 
                print(e) 
            
        with open('ProfileStats.json', 'w') as outfile:
            json.dump(self.info_arr, outfile, indent=4)
        
        print("A json file containing profile statistic information has been created")    

instagram = InstagramScraper()
instagram.main()
