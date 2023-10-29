# import requests
# import json
# import re
# import random
# import pprint


# # putting everything together
# # Distractors from http://conceptnet.io/
# def get_distractors_conceptnet(word):
#     word = word.lower()
#     original_word = word
#     if (len(word.split()) > 0):
#         word = word.replace(" ", "_")
#     distractor_list = []
#     url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
#     obj = requests.get(url).json()

#     for edge in obj['edges']:
#         link = edge['end']['term']

#         url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
#         obj2 = requests.get(url2).json()
#         for edge in obj2['edges']:
#             word2 = edge['start']['label']
#             if word2 not in distractor_list and original_word.lower() not in word2.lower():
#                 distractor_list.append(word2)

#     return distractor_list


# original_word = "California"
# distractors = get_distractors_conceptnet(original_word)
