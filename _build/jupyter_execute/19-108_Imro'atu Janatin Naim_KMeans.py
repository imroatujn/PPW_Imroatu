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
import matplotlib.pyplot as plt
import seaborn as sns

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


from sklearn.cluster import KMeans


# # Read Data PTA

# In[2]:


dataPTA = pd.read_csv('pta-manajemen.csv')


# In[3]:


dataPTA.head(10)


# # Case Folding

# In[4]:


# gunakan fungsi Series.str.lower() pada Pandas
dataPTA['Abstrak'] = dataPTA['Abstrak'].str.lower()

print('Case Folding Result : \n')

#cek hasil case fold
print(dataPTA['Abstrak'].head(5))
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
                
dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_PTA_special)

#menghapus nomor
def remove_number(text):
    return  re.sub(r"\d+", "", text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_number)

#menghapus punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_punctuation)

#menghapus spasi leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_whitespace_LT)

#menghapus spasi tunggal dan ganda
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_whitespace_multiple)

# menghapus kata 1 abjad
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_singl_char)

# Tokenisasi
def word_tokenize_wrapper(text):
    return word_tokenize(text)

dataPTA['abstrak_token'] = dataPTA['Abstrak'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(dataPTA['abstrak_token'].head())


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


vectorizer = TfidfVectorizer(stop_words='english')
berita = []
for data in dataPTA['abstrak_stem']:
    isi = ''
    for term in data:
        isi += term + ' '
    berita.append(isi)

vectorizer.fit(berita)

# Encode the Document
vector = vectorizer.transform(berita)

a= vectorizer.get_feature_names_out()

count = vector.toarray()
df = pd.DataFrame(data=count,columns = [a])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(berita)).toarray()

dfp =pd.DataFrame(data=tf, index=list(range(1,len(tf[:,1])+1, )),columns=[a])
dfp


# # Reduksi Dimensi Menggunakan PCA

# In[11]:


from sklearn.decomposition import PCA
df_pca = pd.DataFrame()
# initialize PCA with 2 components
pca = PCA(n_components=3, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(dfp)
# save our two dimensions into x0 and x1
df_pca['x0']=pca_vecs[:, 1]
df_pca['x1']=pca_vecs[:, 2]


# In[12]:


df_pca


# # Mencari nilai K terbaik menggunakan elbow

# In[13]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_pca)
    distortions.append(kmeanModel.inertia_)


# In[14]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Nilai k yang paling optimal adalah')
plt.show()


# di dapatkan nilai k terbaik = 4

# # Proses clusterisasi K-Means

# In[15]:


from sklearn.cluster import KMeans

# iinisisalisasi dengan 3 centroid
kmeans = KMeans(n_clusters=4, random_state=42)
# fit the model
kmeans.fit(dfp)
# memsukan klaster ke variable
clusters = kmeans.labels_
kmeans.labels_


# # #Menampilkan hasil cluster

# In[16]:


df_pca['cluster']=clusters
df_pca


# Menampilkan isi klaster

# In[17]:


X = vectorizer.fit_transform(dataPTA['Abstrak'])
def get_top_keywords(n_terms):
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # grup TF-IDF berdasarkan cluster
    terms = vectorizer.get_feature_names_out() # akses tf-idf
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
get_top_keywords(10)


# # # Visualisasi Cluster

# In[18]:


# m# map clusters to appropriate labels 
cluster_map = {0: 0, 1: 1, 2: 2,3:3}
# apply mapping
df_pca['cluster'] = df_pca['cluster'].map(cluster_map)


# In[19]:


# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("TF-IDF + KMeans clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df_pca, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()


# In[ ]:





# In[ ]:




