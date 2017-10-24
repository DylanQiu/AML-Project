# Much inspiration for this scraper came from https://github.com/h2non/bbscraper
# -*- coding: utf-8 -*-
import timeit

from bs4 import BeautifulSoup

import urllib as req

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import matplotlib.pyplot as plt
import numpy as np

import pickle


item = ["symptômes",\
"Douleur",\
"Fatigue",\
"Infection",\
"Gonflement",\
"Fièvre",\
"Perte de poids",\
"Mal de tête",\
"Diabète sucré",\
"La nausée",\
"Vomissement",\
"Stress",\
"Douleur abdominale",\
"Dépression",\
"La diarrhée",\
"Essoufflement",\
"La toux",\
"Anxiété",\
"Rhume",\
"Inflammation",\
"Démangeaison de la peau",\
"Vertiges",\
"Allergie",\
"Constipation",\
"Démanger",\
"Infarctus du myocarde",\
"Douleur de poitrine",\
"Saignement",\
"Hypertension",\
"Confusion",\
"Hypothésie",\
"Paresthésie",\
"Anorexie",\
"Migraine",\
"Irritabilité",\
"Accident vasculaire cérébral",\
"Des ganglions lymphatiques enflés",\
"Perte auditive",\
"Mal au dos",\
"Acouphènes",\
"Douleur au genou",\
"Désordre mental",\
"Asymptomatique",\
"Méningite virale"]
# Will hold all conversations
items = [i.lower().decode("utf-8") for i in item]
master_list_of_conversations = []

# global varible
from collections import defaultdict,Counter


countdict = defaultdict(int)

turn_list = []
people_list = []
total_word_list = []
average_word_length_list = []
distinct_word_length_list = []
statistic_dict = {}
statistic_dict["turn_list"] = turn_list
statistic_dict["people_list"] = people_list
statistic_dict["total_word_list"] = total_word_list
statistic_dict["average_word_length_list"] = average_word_length_list
statistic_dict["distinct_word_list"] = distinct_word_length_list

# Ignore for now, was just used for testing
def write_to_file(filename):
	f = open(filename, 'w')

	for conv in master_list_of_conversations:
		f.write("NEW CONVERSATION\n")

		for post in conv:
			f.write(str(post[0]))
			f.write(str(post[1]))

		f.write("END OF CONVERSATION\n")

# Works, but the text output is strange and needs to be edited
def create_XML_tree(list):
	dialogue = Element("Dialogue")

	for conv in list:
		conversation = SubElement(dialogue, "s")

		for post in conv:
			# utt_tag = 'uid="' + str(post[0]) + '"'
			utt = SubElement(conversation, "utt", {'uid': str(post[0])})
			# utt.text = post[1].decode('utf-8')
			utt.text = post[1]

	return dialogue

# Used to get URLs of new threads from the main page
def get_thread_URLs(url):
	url_list = []

	# Get html page from chosen url
	html_doc = req.urlopen(url).read()

	# Create BeautifulSoup Object from webpage
	parsed_url = BeautifulSoup(html_doc, 'html.parser')

	# Collect all thread titles from URL
	thread_list = parsed_url.find_all("h3", class_= "threadtitle")

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
	resource = req.urlopen(url)
	html_doc = resource.read().decode("ISO-8859-1")

	# The parsed html page
	soup = BeautifulSoup(html_doc, 'html.parser')

	# List of all posts as HTML BeautifulSoup objects
	list_of_posts = soup.find_all("div", class_="postdetails")

	for post in list_of_posts:

		# Get username and post
		user = post.find("span", class_= "username guest")

		content = post.find("blockquote", class_="postcontent restore ")

		if user is None or content is None:
			continue
		else:
			# Get rid of quotes within posts
			[x.extract() for x in content.find_all("div", class_="bbcode_container")]
			# If we haven't seen the user before, add it to our list of usernames
			# print user.text.encode("utf-8")
			if user.text not in usernames:
				usernames.append(user.text)

			# print ('########################################')
			# print (user.text)
			# print ('#############')
			# print (content.text)
			# print ('########################################')
			# Add username index + 1 and their post to our post_list for the thread
			# post_list.append([usernames.index(user.text) + 1, content.text])		
			post_list.append((usernames.index(user.text) + 1, content.text))	

	return post_list

def global_statistic(master_list_of_conversations):
	print "number of conversation",len(master_list_of_conversations)
	print "==========="
	for i in range(len(master_list_of_conversations)):
		statistic(master_list_of_conversations[i])


def statistic (dialog_list):
	
	if len(dialog_list) == 0:
		return
	
	print "--------------------------"
	master_list = [i for i in dialog_list]

	len_dialog = len(dialog_list)
	max_people = max([i[0] for i in master_list])
	total_word = sum(len(i[1].split()) for i in master_list)
	average_word_length = sum(len(i[1].split()) for i in master_list)/len(dialog_list)

	join_list = []
	string_set = []
	for i in master_list:
		join_list += i[1].split()
		string_set += i[1]

	distinct_word_length = dict(Counter(join_list)).keys()
	for item in items:
		if item in join_list:
			countdict[item] +=1


	turn_list.append(len_dialog)
	people_list.append(max_people)
	average_word_length_list.append(average_word_length)
	total_word_list.append(total_word)
	distinct_word_length_list.append(distinct_word_length)

	print "number of turn for this dialog",len_dialog
	print "number of people in each dialog",max_people
	print "total of word in each dialog",total_word
	print "average word length in each dialog",average_word_length
	print "number of distinct word in each dialog", distinct_word_length


	
def main():
	main_url = "http://forums.futura-sciences.com/sante-medecine-generale/"
	test_url = "http://forums.futura-sciences.com/sante-medecine-generale/621932-fatigue-soleil.html"


	# 	# print (page)
	# 	urls = get_thread_URLs(main_url + str(page) + "/")
	# 	print urls
	# 	print (main_url + str(page) + "/")
	with open('storelist', 'rb') as fp:
		master_list_of_conversations = pickle.load(fp)

	# for i,dialog in enumerate(master_list_of_conversations):
	# 	print "conversation",i ,"has turns", len(dialog)
	


	global_statistic(master_list_of_conversations)
	print countdict
	with open('statistic_dict_new', 'wb') as fp:
		pickle.dump(statistic_dict,fp)
		# statistic_dict = pickle.load(fp)
	
	
	# preprocess statistic_dict
	one_people_list = [i for i in range(len(statistic_dict["people_list"])) if statistic_dict["people_list"][i] ==1 ]

	for feature in statistic_dict.iteritems():
		# print feature
		for index in sorted(one_people_list,reverse = True):
			del feature[1][index] 

	newCounter = Counter()
	for i in range(len(statistic_dict["distinct_word_list"])):
		
		newCounter += Counter(statistic_dict["distinct_word_list"][i] )
		print i
	print newCounter


	# import numpy
	# for feature in statistic_dict.iteritems():
	# 	arr = numpy.array(feature[1])
	# 	print feature[0]
	# 	print "max,",arr.max()
	# 	print "min,",arr.min()
	# 	print "avg,",int(numpy.mean(arr))
	# 	print "std,",int(numpy.std(arr))
	

	# print "conversation total,",len(statistic_dict["turn_list"])
	plt.figure(1)
	# plt.subplot(321)
	# plt.plot(statistic_dict["turn_list"],"b,")
	# plt.axis([0, 100, 0, 20])

	# plt.title('turns per dialog')

	# plt.subplot(322)
	# plt.plot(statistic_dict["people_list"],"r,")
	# plt.axis([0, 500, 0, 20])
	# plt.title('number of people per dialog')

	plt.subplot(212)
	plt.plot(statistic_dict["total_word_list"],"g,")
	plt.title('total word per conversation')
	plt.ylabel('counts')

	# plt.subplot(324)
	# plt.plot(statistic_dict["average_word_length_list"],"y,")
	# plt.title('average word length per dialog')


	plt.subplot(211)
	plt.plot( statistic_dict["distinct_word_list"],"k,")
	plt.title('distinct word per conversation')
	plt.ylabel('counts')
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50,
                    wspace=0.30)
	
	plt.show()
	

main()

