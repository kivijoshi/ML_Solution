# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:19:47 2016

@author: Kaustubh
"""

import requests
from bs4 import BeautifulSoup
response = requests.get('http://www.brokersadda.com/index.php?do=/profile-104180/')
parsed_html = BeautifulSoup(response.content)
print parsed_html.body.find("div", {"id": "js_custom_content_9"})