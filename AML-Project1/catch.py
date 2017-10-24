# Much inspiration for this scraper came from https://github.com/h2non/bbscraper
# -*- coding: utf-8 -*-

import timeit

import pickle

from bs4 import BeautifulSoup

# from urllib import request as req
from urllib2 import urlopen
import numpy as np

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from lxml import etree

# Will hold all conversations
master_list_of_conversations = []


def convertToXML(master_list, filename):
    dialog = etree.Element('dialog')

    for s_list in master_list:
        s = etree.SubElement(dialog, 's')
        for utt_list in s_list:
            utt = etree.SubElement(s, 'utt')
            utt.text = utt_list[1].decode('utf-8')
            utt.set('uid', str(utt_list[0]))

    tree = etree.ElementTree(dialog)
    tree.write(filename, pretty_print=True)
    return 0


# Used to get URLs of new threads from the main page
def get_thread_URLs(url):
    url_list = []

    # Get html page from chosen url
    # html_doc = req.urlopen(url).read()
    html_doc = urlopen(url).read()

    # Create BeautifulSoup Object from webpage
    parsed_url = BeautifulSoup(html_doc, 'html.parser')

    # Collect all thread titles from URL
    thread_list = parsed_url.find_all("h3", class_="threadtitle")

    # For each thread HTML object, extract only the html and add it to our master URL list so that individual posts can be extracted
    for thread in thread_list:
        new_url = thread.find("a").get("href")

        if new_url is not None:
            url_list.append("http://forums.futura-sciences.com" + new_url)

    return url_list


# Get the posts within each thread url and output them as a list
def get_posts(url):
    # List of posts
    post_list = []
    # list of usernames
    usernames = []

    # The actual HTML page
    # html_doc = req.urlopen(url).read()
    html_doc = urlopen(url).read()

    # The parsed html page
    soup = BeautifulSoup(html_doc, 'html.parser')

    # List of all posts as HTML BeautifulSoup objects
    list_of_posts = soup.find_all("div", class_="postdetails")

    for post in list_of_posts:

        # Get username and post
        user = post.find("span", class_="username guest")
        content = post.find("blockquote", class_="postcontent restore ")

        if user is None or content is None:
            continue
        else:
            # Get rid of quotes within posts
            [x.extract() for x in content.find_all("div", class_="bbcode_container")]
            # If we haven't seen the user before, add it to our list of usernames
            if user.text not in usernames:
                usernames.append(user.text)

            # print ('########################################')
            # print (user.text)
            # print ('#############')
            # print (content.text)
            # print ('########################################')
            # Add username index + 1 and their post to our post_list for the thread
            # post_list.append([usernames.index(user.text) + 1, content.text])
            content_with_strip = content.text.encode('utf-8')
            # print (content_with_strip)
            content_with_strip = content_with_strip.replace('\r', ' ')
            content_with_strip = content_with_strip.replace('\t', ' ')
            content_with_strip = content_with_strip.replace('\n', ' ')
            content_with_strip = ' '.join(content_with_strip.split())
            # content_with_strip = content_with_strip.replace('\r\n', ' ')
            post_list.append([usernames.index(user.text) + 1, content_with_strip])

    return post_list


def main():
    main_url = "http://forums.futura-sciences.com/sante-medecine-generale/"

    # f = open("output.txt", "w")

    # for page in range(1, 101):
    for page in range(1, 101):
        # print (page)
        urls = get_thread_URLs(main_url + str(page) + "/")
        # print (main_url + str(page) + "/")

        for url in urls:
            conversation = get_posts(url)
            master_list_of_conversations.append(conversation)
            print (len(master_list_of_conversations))
    print ("Number of conversations: " + str(len(master_list_of_conversations)))


    with open('storelist', 'wb') as fp:
        pickle.dump(master_list_of_conversations, fp)

    # with open('storelist', 'rb') as fp:
    #     master_list_of_conversations = pickle.load(fp)

	# print (master_list_of_conversations[0][0][1].decode('utf-8'))

    convertToXML(master_list_of_conversations, 'tree.xml')

    # print (master_list_of_conversations)

# print ("number of conversations: " + str(len(master_list_of_conversations)))

main()
