#!/usr/bin/env python
# coding: utf-8

# # Keseluruhan Code

# # Import Library

# In[1]:


import pandas as pd
import numpy as np
#Import Library untuk Tokenisasi
import string 
import re #regex library

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# import the dataset from sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# viz libs
import matplotlib.pyplot as plt
import seaborn as sns


# # Read Data PTA

# In[2]:


dataPTA = pd.read_csv('ta-manajemen.csv')


# In[3]:


dataPTA.head(10)


# # Case Folding

# In[4]:


# gunakan fungsi Series.str.lower() pada Pandas
dataPTA['abstraksi'] = dataPTA['abstraksi'].str.lower()

print('Case Folding Result : \n')

#cek hasil case fold
print(dataPTA['abstraksi'].head(5))
print('\n\n\n')


# # Removal

# In[5]:


#Import Library untuk Tokenisasi
import string 
import re #regex library

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

def remove_PTA_special(text):
    # menghapus tab, new line, dan back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # menghapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # menghapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # menghapus incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_PTA_special)

#menghapus nomor
def remove_number(text):
    return  re.sub(r"\d+", "", text)

dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_number)

#menghapus punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_punctuation)

#menghapus spasi leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_whitespace_LT)

#menghapus spasi tunggal dan ganda
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_whitespace_multiple)

# menghapus kata 1 abjad
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

dataPTA['abstraksi'] = dataPTA['abstraksi'].apply(remove_singl_char)

# Tokenisasi
def word_tokenize_wrapper(text):
    return word_tokenize(text)

dataPTA['abstrak_token'] = dataPTA['abstraksi'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(dataPTA['abstrak_token'].head())
print('\n\n\n')


# # STOPWORD 

# In[6]:


from nltk.corpus import stopwords

list_stopwords = stopwords.words('indonesian')

# Mengubah List ke dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

#aStopwording
dataPTA['abstrak_stop'] = dataPTA['abstrak_token'].apply(stopwords_removal) 


print(dataPTA['abstrak_stop'].head(20))


# In[7]:


dataPTA.head()


# # stemming

# In[8]:


# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# sfungsi stemmer
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in dataPTA['abstrak_stop']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# stemming pada dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

dataPTA['abstrak_stem'] = dataPTA['abstrak_stop'].swifter.apply(get_stemmed_term)
print(dataPTA['abstrak_stem'])


# In[9]:


dataPTA.head()


# # TF Process

# In[10]:


# initialize the vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(dataPTA['abstraksi'])
print (X)


# In[11]:


from sklearn.cluster import KMeans

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_


# In[12]:


from sklearn.decomposition import PCA

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[13]:


# assign clusters and pca vectors to our dataframe 
dataPTA['cluster'] = clusters
dataPTA['x0'] = x0
dataPTA['x1'] = x1


# In[14]:


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            
get_top_keywords(10)


# In[15]:


# map clusters to appropriate labels 
cluster_map = {0: "sport", 1: "tech", 2: "religion"}
# apply mapping
dataPTA['cluster'] = dataPTA['cluster'].map(cluster_map)


# In[16]:


# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("TF-IDF + KMeans 20newsgroup clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=dataPTA, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()


# In[ ]:




