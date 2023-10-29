# # load sense2vec vectors
# from sense2vec import Sense2Vec
# s2v = Sense2Vec().from_disk('s2v_old')

# from collections import OrderedDict
# def sense2vec_get_words(word,s2v):
#     output = []
#     word = word.lower()
#     word = word.replace(" ", "_")

#     sense = s2v.get_best_sense(word)
#     most_similar = s2v.most_similar(sense, n=20)

#     # print ("most_similar ",most_similar)

#     for each_word in most_similar:
#         append_word = each_word[0].split("|")[0].replace("_", " ").lower()
#         if append_word.lower() != word:
#             output.append(append_word.title())

#     out = list(OrderedDict.fromkeys(output))
#     return out

# word = "Natural Language processing"
# distractors = sense2vec_get_words(word,s2v)

# print ("Distractors for ",word, " : ")
# print (distractors)
