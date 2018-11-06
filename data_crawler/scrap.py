#!/usr/bin/env python

from lxml import html
import requests
import time
import json
import os
from BeautifulSoup import BeautifulSoup
import urllib2
import re
import sys
import unicodedata
import json
#from time import sleep

# from http://stackoverflow.com/questions/1197981/convert-html-entities

baseurl = "https://us.openfoodfacts.org"
data = {}
#count = sys.argv[2]
count = 1
path = "output/"
n = 0

def asciify2(s):
  matches = re.findall("&#\d+;", s)
  if len(matches) > 0:
    hits = set(matches)
    for hit in hits:
      name = hit[2:-1]
      try:
        entnum = int(name)
        s = s.replace(hit, unichr(entnum))
      except ValueError:
        pass

  matches = re.findall("&\w+;", s)
  hits = set(matches)
  amp = "&amp;"
  if amp in hits:
    hits.remove(amp)
  for hit in hits:
    name = hit[1:-1]
    if htmlentitydefs.name2codepoint.has_key(name):
      #s = s.replace(hit, unichr(htmlentitydefs.name2codepoint[name]))
      s = s.replace(hit, "")
  s = s.replace(amp, "&")
  return s

def opensoup(url):
  request = urllib2.Request(url)
  request.add_header("User-Agent", "Mozilla/5.0")
  # To mimic a real browser's user-agent string more exactly, if necessary:
  #   Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.1.14)
  #   Gecko/20080418 Ubuntu/7.10 (gutsy) Firefox/2.0.0.14
  pagefile = urllib2.urlopen(request)
  soup = BeautifulSoup(pagefile)
  pagefile.close()
  return soup

def asciify(s):
  #print "DEBUG[", type(s), "]"
  return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')

# remove extra whitespace, including stripping leading and trailing whitespace.
def condense(s):
  s = re.sub(r"\s+", " ", s, re.DOTALL);  #$s =~ s/\s+/\ /gs;
  return s.strip()

# this gets rid of tags and condenses whitespace
def striptags(s):
  #print "DEBUG[", type(s), "]"
  s = re.sub(r"\<span\s+style\s*\=\s*\"display\:none[^\"]*\"[^\>]*\>[^\<]*\<\/span\>","", s)
  s = re.sub(r"\&\#160\;", " ", s)
  return condense(re.sub(r"\<[^\>]*\>", " ", s))

#Get the urls of products
print("Commencing product URL crawling")
start_time = time.time()
for i in range(1,8769):
  print ("Snooping into products in page "+str(i))
  page = requests.get(baseurl+"/"+str(i))
  tree = html.fromstring(page.content)
  for product in tree.xpath('//ul[@class="products"]')[0]:
    for li in product:
      prod_url = baseurl+li.attrib['href']
      newpage = requests.get(prod_url)
      newtree = html.fromstring(newpage.content)
      if len(newtree.xpath('//img[@class="show-for-xlarge-up"]')) > 5:
        name = newtree.xpath('//h1[@itemprop="name"]/text()')[0]
        label_src = newtree.xpath('//img[@class="show-for-xlarge-up"]')[5].attrib['src']
        if "nutrition_en" not in label_src:
            continue
        flag = 0
        flag2 = 0
        data[name] = {}
        data[name]['sno'] = count
        data[name]['file_name'] = str(count)+'.jpg'
        data[name]['url'] = prod_url
        data[name]['nutrition_label_src'] = newtree.xpath('//img[@class="show-for-xlarge-up"]')[5].attrib['src']
        count+=1
        #Opening soup for nutrition truth
        soup = opensoup(prod_url)
        tables = soup.findAll("table")
        table = tables[n-1]
        for body in table.findAll('tbody'):
          for r in body.findAll('tr'):
            rl = []
            a = 0
            key = ""
            value = ""
            for c in r.findAll(re.compile('td')):
              if a==0:
                k = striptags(c.renderContents())
                ke = list(k)
                if "-" in ke:
                  del(ke[ke.index("-")])
                if ke[0]==" ":
                  del(ke[0])
                key = "".join(ke)
              if a==1:
                v = striptags(c.renderContents())
                l = v.split(" ")
                if key!="NutriScore":
                  li = list(l[0])
                  if ',' in li:
                    li.remove(',')
                  li = "".join(li)
                  try:
                    value = float(li)
                  except:
                    value = li
                else:
                  value = l[0]
                break
              a+=1
            if value=="&lt;" or value=="&gt;" or value=="?":
              continue
            else:
              flag = 1
              if key=="NutriScore":
                  flag2 = 1
              data[name][key]=value
        if flag!=1 or flag2!=1:
            del data[name]
            count-=1
        else:
            print(".....product "+str(count-1))

if not os.path.exists(path):
    os.makedirs(path)
#with open(path+'data.json', 'a') as outfile:
#    json.dump(data, outfile, sort_keys = True, indent = 4)

#print("Commencing Image scraping for:")
#for product in data:
#  img_src = data[product]['nutrition_label_src']
#  r = requests.get(img_src)
#  with open(path+data[product]['file_name'], 'wb') as f:
#    f.write(r.content)
#    f.close()
#  print (product)
  # Retrieve HTTP meta-data

print("Finished in :"+str(time.time()-start_time)+" seconds")
print json.dumps(data, sort_keys = True, indent = 4)
