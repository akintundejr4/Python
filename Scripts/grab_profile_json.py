import os
import re
import sys
import ssl
import time
import glob
import json
import errno
import random
import pymysql
import operator
import requests
import urllib.error
import urllib.parse
import urllib.request
import pymysql.cursors
from random import choice
from datetime import datetime
from bs4 import BeautifulSoup
from operator import itemgetter


# This will tell you where your script is actually being run from.
# print(os.getcwd())

provided_json_file = sys.argv[1]
lower_follower_count = int(sys.argv[2])
upper_follower_count = int(sys.argv[3])

class InstagramScraper:

    def getinfo(self, url):
        html = urllib.request.urlopen(url, context=self.ctx).read()
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all(
            'script', type="text/javascript", text=re.compile('window._sharedData'))
        stringified_json = scripts[0].get_text().replace(
            'window._sharedData = ', '')[:-1]
        json_data = json.loads(stringified_json)[
            'entry_data']['ProfilePage'][0]

        info = {}
        info['Username'] = json_data['graphql']['user']['username']
        info['FullName'] = json_data['graphql']['user']['full_name']
        info['Following'] = json_data['graphql']['user']['edge_follow']['count']
        info['Followers'] = json_data['graphql']['user']['edge_followed_by']['count']
        info['FollowsViewer'] = json_data['graphql']['user']['follows_viewer']
        info['BlockedByViewer'] = json_data['graphql']['user']['blocked_by_viewer']
        info['ConnectedFbPage'] = json_data['graphql']['user']['connected_fb_page']
        info['IsBusinessAccount'] = json_data['graphql']['user']['is_business_account']
        info['IsVerified'] = json_data['graphql']['user']['is_verified']

        self.info_arr.append(info)

    def parse_output(self): 
        trimmedList = []

        for item in self.info_arr:
            if item['Followers'] >= self.lower_bound and item['Followers'] <= self.upper_bound:
                trimmedList.append(item)

        self.sortedTrimmedList = sorted(trimmedList, key=itemgetter('Followers'))

                
    def main(self):
        self.content = []
        self.info_arr = []
        self.sortedTrimmedList = []
        self.ctx = ssl.create_default_context()

        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.lower_bound = lower_follower_count
        self.upper_bound = upper_follower_count

        with open(provided_json_file) as json_data:
            data = json.load(json_data,)
            for item in data:
                url = f'https://www.instagram.com/{item}'
                self.content.append(url)

        # Removing all leading and trailing spaces from the list
        self.content = [x.strip() for x in self.content]

        for url in self.content:
            try:
                self.getinfo(url)
                time.sleep(3)
            except Exception as e:
                print("This URL was not processed succesfully " + url)
                print(e)

        inputfilename = provided_json_file.split('.')
        datestring = datetime.strftime(datetime.now(), '%Y-%m-%d')
        
        self.parse_output()

        with open("./output/" + inputfilename[0] + "_parsed-output_" + datestring + ".txt", "w") as outfile:
            for user in self.sortedTrimmedList:
                outfile.write("----------------------------------\n")
                outfile.write("Username: " + user["Username"] + '\n')
                outfile.write("Following: " + "{:,}".format(user["Following"]) + '\n')
                outfile.write("Followers: " + "{:,}".format(user["Followers"]) + '\n')
                outfile.write("IsVerified: " + str(user["IsVerified"]) + '\n')
                outfile.write("BlockedByViewer: " + str(user["BlockedByViewer"]) + '\n')
                outfile.write("----------------------------------\n")
        

        # with open("./output/" + inputfilename[0] + '_jsondata-' + datestring + '.json', 'w') as outfile:
        #     json.dump(self.info_arr, outfile, indent=4)
       
        print("A text file containing information as requested has been created")


        
# The class has to be declared before you call the main scrapper method.
instagram = InstagramScraper()
instagram.main()
