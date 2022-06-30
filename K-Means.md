# K-Means

K-Means Merupakan salah satu metode data mining untuk mengelompokkan suatu data ke dalam klaster. metode ini melakukan penghitungan jarak berdasarkan k tertangga terdekatnya

untuk menghiyung jarak, dapat emnggunakan euclidean distance dengan rumus :

$$
d=\sqrt{\left(x_{2}-x_{1}\right)^{2}+\left(y_{2}-y_{1}\right)^{2}}
$$

## Tahapan K-Means

Tahap K-Means meliputi :


### Menghitung TF-IDF

Langkah pertama pada klastering k-means pada data text adalah menghitung TF-IDF. TF-IDF dihitung menggunakan matrix mxn. Dimana m merupakan dokumen dan n merupakan fitur tiap kata pada dokumen. TF-IDF atau Term frequency-Inverse document frequency memberikan bobot untuk term j dalam dokumen i.

Perhitungan TF-IDF dapat dilakukan dengan :


$$
W_{i,j}=tfi,j\times\log \dfrac{N}{d_{fj}}
$$


**Keterangan :**

Wij = score TF-IDF

Tfi,j = term dari dokumen

N= Total Dokumen

Df j = dokumen

pada python dapat dilakukan program sebagai berikut :

```python
#IMPORT LIBRARY

import pandas as pd
import numpy as np
#Import Library untuk Tokenisasi
import string 
import re #regex library

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
```

```python
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
```

### Melakukan Reduksi Dimensi

reduksi dimensi merupakan langkah yang digunakan untuk mengurangi fitur, pada tahap ini dilakukan dengan metode PCA. metode PCA (*Principal component analysis*) merupakan suatu teknik yang digunakan untukmengurangi tabel dalam sklaa besar. pada python dapat digunakan dengan menggunakan kode berikut :

```
from sklearn.decomposition import PCA
df_pca = pd.DataFrame()
# initialize PCA with 2 components
pca = PCA(n_components=3, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(dfp)
# save our two dimensions into x0 and x1
df_pca['x0']=pca_vecs[:, 1]
df_pca['x1']=pca_vecs[:, 2]
```



### Menenentukan nilai K terbaik menggunakan elbow

untuk menentukan k terbaik digunakan metode elbow. metode elbow merupakan  salah satu metode untuk menentukan jumlah klaster  yang tepat melalui persentase hasil perbandingan antara jumlah klaster yang akan membentuk siku siku pada  suatu titik klaster  yang tepat melalui persentase hasil perbandingan antara jumlah klaster yang akan membentuk siku siku pada  suatu titik 

```python
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_pca)
    distortions.append(kmeanModel.inertia_)
```

pada kode di atas, dilakukan perulanngan sebanyak 10 dengan menggunakan data_pca yang telah direduksi. untuk melihat hasil elbow, dapat divisualisasikan melalui plot dengan kode sebagai berikut :

```python
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Nilai k yang paling optimal adalah')
plt.show()
```

### Proses Klasterisasi kmeans

K-Means Merupakan salah satu metode data mining untuk mengelompokkan suatu data ke dalam klaster. metode ini melakukan penghitungan jarak berdasarkan k tertangga terdekatnya

```python
from sklearn.cluster import KMeans

# iinisisalisasi dengan 3 centroid
kmeans = KMeans(n_clusters=4, random_state=42)
# fit the model
kmeans.fit(dfp)
# memsukan klaster ke variable
clusters = kmeans.labels_
kmeans.labels_
```

### Menampilkan Hasil Cluster

untuk menampilkan hasil klaster, hasil klaster dapat dimasukkan ke dalam dataframe dengan cara berikut :

```python
df_pca['cluster']=clusters
df_pca

```

### Menampilkan beberapa hasil berdasarkan masing-masing klaster

```python
X = vectorizer.fit_transform(dataPTA['Abstrak'])
def get_top_keywords(n_terms):
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # grup TF-IDF berdasarkan cluster
    terms = vectorizer.get_feature_names_out() # akses tf-idf
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
get_top_keywords(10)

```

### Visualisasi Cluster

visualisasi dilakukan dengan menggunakan scatter plot

```python
# m# map clusters to appropriate labels 
cluster_map = {0: 0, 1: 1, 2: 2,3:3}
# apply mapping
df_pca['cluster'] = df_pca['cluster'].map(cluster_map)

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

```
