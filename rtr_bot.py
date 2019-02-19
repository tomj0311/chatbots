import bs4 as bs
import urllib.request
import re
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sent_tokens = None
word_tokens = None

def lemtokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

#pre-processing raw text
def lemnormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))    

# This will be used to find the similarity between words entered by the user and the words in the scraped corpus.
# Searches the user’s utterance for one or more known keywords and returns one of several possible responses. 
def response(user_response):
    bot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=lemnormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        bot_response = bot_response + "I am sorry! I don't understand you"
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response

def load_topic(topic):
    scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/' + topic)
    article = scrapped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    article_text = article_text.lower()

    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
    article_text = re.sub(r'\s+', ' ', article_text)  

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)  

    sent_tokens = nltk.sent_tokenize(article_text)
    word_tokens = nltk.word_tokenize(article_text) 

    return sent_tokens, word_tokens

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

flag=True
print("bot: My name is bot. I will answer your queries about a topic. If you want to exit, type Bye!")

topic_response = input('Topic: ')
sent_tokens, word_tokens = load_topic(topic_response)

while(flag==True):
    user_response = input(": ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("bot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("bot: "+greeting(user_response))
            else:
                print("bot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("bot: Bye! take care..")



