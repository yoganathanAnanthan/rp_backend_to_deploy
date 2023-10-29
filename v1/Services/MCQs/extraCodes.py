# out = []
# try:
#     extractor = pke.unsupervised.MultipartiteRank()
#     extractor.load_document(input=content, language='en')
#     #    not contain punctuation marks or stopwords as candidates.
#     pos = {'PROPN', 'NOUN'}
#     # pos = {'PROPN','NOUN'}
#     stoplist = list(string.punctuation)
#     stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
#     stoplist += stopwords.words('english')
#     # extractor.candidate_selection(pos=pos, stoplist=stoplist)
#     extractor.candidate_selection(pos=pos)
#     # 4. build the Multipartite graph and rank candidates using random walk,
#     #    alpha controls the weight adjustment mechanism, see TopicRank for
#     #    threshold/method parameters.
#     extractor.candidate_weighting(alpha=1.1,
#                                   threshold=0.75,
#                                   method='average')
#     keyphrases = extractor.get_n_best(n=15)
#
#     for val in keyphrases:
#         out.append(val[0])
# except:
#     out = []
#     traceback.print_exc()
#
# return out