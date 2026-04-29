from pygooglenews import GoogleNews

gn = GoogleNews()

# s = gn.search('intitle:boeing')

# print(s['feed'].values)
# # "intitle:boeing" - Google News




business = gn.topic_headlines('business')
headlines = [article['title'] for article in business['entries']]
print(headlines)