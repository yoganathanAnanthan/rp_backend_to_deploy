def filter_keywords(keywords, num_best_keywords):
    # Sort the keywords by length in descending order
    sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)

    # Select the top 'num_best_keywords' keywords
    best_keywords = sorted_keywords[:num_best_keywords]

    # Remove similar keywords
    filtered_keywords = []
    for keyword in best_keywords:
        is_similar = False
        for filtered_keyword in filtered_keywords:
            if keyword.lower() in filtered_keyword.lower() or filtered_keyword.lower() in keyword.lower():
                is_similar = True
                break
        if not is_similar:
            filtered_keywords.append(keyword)

    return filtered_keywords