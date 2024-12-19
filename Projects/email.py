# TEXT DATA
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# nltk.download('stopwords')
#print stopwords
data = pd.read_csv("PATH")
data['label'] = np.random.random_integers(low=0,high=1,size=5572)
#seprating target
X = data.drop(columns='Category')
Y = data['Category']
#words reducing

port_stem = PorterStemmer()
def stemming(content):
    #remove all comma or something like that
    steamded_content = re.sub("[^a-zA-Z]"," ",content)
    #convert to lovercase
    steamded_content.lower()
    steamded_content =  steamded_content.split()
    steamded_content = [port_stem.stem(word) for word in steamded_content if not word in stopwords.words("english")]
    steamded_content = " ".join(steamded_content)
    return steamded_content




data['Message'] = data['Message'].apply(stemming)
X = data['Message'].values
Y = data['label'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
