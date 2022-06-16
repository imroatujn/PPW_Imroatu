# Topic Modelling - LSA (Latent Semantic Analysis)

Topic modelling merupakan Teknik unsupervised learning atau pembelajaran tidak terawasi untuk menemukan suatu tema pada dokumen. Topic modeling akan melakukan extract data berdasarkan keyword yang menunjukkan topik.

Algoritma Topic Modelling :

\-    Setiap dokumen berhubungan dengan topik topik

\-    Setiap topik berhubungan dengan kata-kata.

Terdapat beberapa macam topic modelling, yang sering digunakan adalah Latent Dirchlet Allocation (LDA) dan Latent Semantic Analysis (LSA)

## Mengapa Menggunakan LSA?

Latent Semantic Analysis (LSA) adalah salah satu pemodelan topik yang mengekstrak topik dari dokumen teks yang diberikan. pembelajaran LSA meliputi dekomposisi matriks pada matriks term dokumen menggunakan dekomposisi nilai Singular. LSA mengambil matriks dokumen dan istilah dan mencoba menguraikannya menjadi dua matriks yang terpisah yaitu :

\-    Matriks topik dokumen

\-    Matriks istilah topik.

## Tahapan LSA

tahap Ltent Semantic Analysis meliputi :


### Menghitung TF-IDF

Langkah pertama pada topik modelling menggunakan LSA adalah menghitung TF-IDF. TF-IDF dihitung menggunakan matrix mxn. Dimana m merupakan dokumen dan n merupakan fitur tiap kata pada dokumen. TF-IDF atau Term frequency-Inverse document frequency memberikan bobot untuk term j dalam dokumen i.

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
vect =TfidfVectorizer(stop_words=list_stopwords,max_features=1000) 
vect_text=vect.fit_transform(dataPTA['Abstrak'])
print(vect_text.shape)
print(vect_text)
```

### Menampilkan SVD

Setelah menghitung TF-IDF, maka dilakukan Tindakan terhadap latent atau topik yang tersembunyi yang merepresentasikan dokumen. Sehingga kita dapat mengetahui matriks yang jarang muncul, masih terdapat noisy dan redudndan. Sehingga dilakukan pengurangan dimensi pada matrix A menjadi k.

Untuk itu dilakukan Singular Value Decomposition (SVD).

SVD merupakan proses menemukan informasi yang paling penting dengan menggunakan dimensi t untuk merepresentasikan hal yang sama. Nilai singular memfaktorkan matriks dan menjadikannya 3 matrix yang berbeda meliputi :

\-    Matrix kolom ortogonal (V)

\-    Matrix baris ortogonal, (U)

\-    Singular matrix (S)
$$
M=U*S*V
$$
Sementara itu pada SVD dilakukan perhitungan terhadap matrix U dan juga Matrix V. dengan rumus :
$$
A=USV^{T}
$$
Keterangan :

Matriks U = pada matriks ini, baris mewakili vector dokumen pada topik

Matriks V = baris pada matriks ini mewakili vector istilah yang dinyatakan dalam topik.

S= matriks diagonal yang memiliki elemen-elemen diagonal sebagai nilai singular dari A.

Sehingga SVD menghasilkan representasi vektor untuk setiap dokumen dan istilah dalam data.

 Panjang setiap vektor adalah k. 

Salah satu kegunaan penting dari vektor ini adalah kita dapat menemukan kata dan dokumen serupa dengan bantuan metrik kesamaan kosinus.

implementasi SVD pada python

```python
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)
print(lsa_top)
print(lsa_top.shape)
```

```python
l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
```

```python
print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)
```



### Mengekstrak Topic dan Term

Setelah dilakukan perhitungan matriks terhadap dokumen, dapat dilakukan ekstrak topik dokumen. Pada percobaan kali ini, dilakukan extrak sebanyak 10 topik.

```python
#kata yang paling penting pada suatu topik
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
```

