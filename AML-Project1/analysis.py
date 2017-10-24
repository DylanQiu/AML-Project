import pickle
import catch

with open('storelist', 'rb') as fp:
    master_list_of_conversations = pickle.load(fp)

print 'conversations with only one speaker:', len(master_list_of_conversations)

two_more_person = []

for s in master_list_of_conversations:
    if len(s) > 1:
        two_more_person.append(s)

print 'conversations with two or more speakers:', len(two_more_person)

with open('storelist_withtwoormore.txt', 'wb') as fp:
    pickle.dump(two_more_person, fp)

catch.convertToXML(two_more_person, 'tree_withtwoormore.xml')