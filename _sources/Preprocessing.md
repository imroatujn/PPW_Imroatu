# Text Preprocessing

Data yang telah didapatkan merupakan data mentah yang butuh dilakukan preprosesing sebelum masuk ke dalam tahap pemrosesan data.

*Text preprocessing* digunakna untuk menyeleksi data text agar menjadi terstruktur melalui tahapan-tahapan preprosessing data.

Preprosesing meliputi case folding, tokenisasi, stopword removal, dan Stemming.

## Import Library

sebelum melakukan semua tahapan preprocessing, sebaiknya menyiapkan library yang akan digunakan terlebih dahulu.

```python
import pandas as pd
import numpy as np
#Import Library untuk Tokenisasi
import string 
import re #regex library

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
```



## Read Data

sebelum melakukan preproses data, terlebih dahulu membaca data yang akan digunakan

```python
dataPTA = pd.read_excel('PTAscrawl.xlsx')
```

## Case Folding

*Case folding* merupakan suatu tahap preprosessing data yang digunakan untuk menyam rataakan semua case atau ukuran huruf. Biasanya semua kata diubah menjadi lowercase (huruf kecil) pada *case folding*.

Contoh : kata “LaPtop” jika masuk pada tahap *case folding* maka akan berubah menjadi “laptop”. Semua huruf rata menggunakan *lowercase*.

pada python dapat dilakukan percobaan seperti berikut :

```python
# gunakan fungsi Series.str.lower() pada Pandas
dataPTA['Abstrak'] = dataPTA['Abstrak'].str.lower()

print('Case Folding Result : \n')

#cek hasil case fold
print(dataPTA['Abstrak'].head(5))
print('\n\n\n')
```

## Tokenisasi

*Tokenizing* merupakan salah satu tahap preprocessing yang merupakan tahap untuk memotong atau membuat kalimat menjadi potongan kata.

Contoh : saya makan bakso

Maka ketika masuk pada tahap tokenizing maka akan menjadi saya, makan, bakso.


## Stopword Removal

Pada tahap ini, dilakukan pembuangan terhadap kata atau karakter yang tidak dibutuhkan. Misalnya penggunaan kata hubung, kata preposisi, karakter, nomor, punctuation, selain itu juga dilakukan penghapusan terhadap kata yang tidak penting.

kode di bawah ini akan menunjukkan bagaimana proses remove karakter, number, punctuation.

```python
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

print('Result : \n') 
print(dataPTA['Abstrak'].head())
print('\n\n\n')
```

setelah itu dapat dilakukan stopword bahasa indonesia menggunakan library NLTK dan juga extend kata yang tidak memiliki makna.

```python
list_stopwords = stopwords.words('indonesian')
list_stopwords.extend(["aam","absolute","abstract","abstrakxd","adm","ahp","ai","aid","akanxd","akhirxd","alert","algorithm","alpha","alternative","ambroxol","analysis","analytic","analytical","and","angkaangka","angular","anp","apl","aplikasixd","application","architecture","artifical","as","asesoris","attribute","automatic","average","babnyajika","background","bahanbahan","baikxd","balanced","base","based","basic","bc","beasiwa","benarbenar","benedict","beratxd","berbedabeda","berturutturut","bifurcation","binaryzation","bisoprolol","bkd","block","blue","bold","bolditalic","bpp","bps","browsing","bsc","business","by","canny","caps","cbir","cefixime","center","centroid","chain","chaining","chainning","character","cipherteks","class","classfier","classification","classifier","close","cluster","clustering","coding","combat","commerce","component","compute","computer","confix","content","contex","context","corepoint","corpus","cosine","criteria","criteriaxd","crm","crossing","customer","cut","cycle","dalamxd","darixd","database","datadata","dataxd","daviesbouldin","decision","decomposition","defuzzyfikasi","dekripsi","denga","denganxd","depanxd","depth","design","development","dibatasixd","dicarixd","difference","diprosesxd","direction","disegmentasi","disk","disperindag","distance","distemming","dkpp","dlda","dominant","download","dperoleh","dr","drop","dsebut","eclipse","ecommerce","ecommercexd","emulator","engghibunten","engine","engineering","enginexd","english","enhanced","enjeiyeh","enterprise","environment","eoq","epoch","epoh","error","eucledian","euclidean","exploiting","exponential","express","fahp","fanp","feature","fighting","filter","filtering","fine","fingerprint","fingerprintbitmaps","finite","firewall","first","fisik","fmeasure","fmop","font","foreign","forward","framework","free","frustasi","fsm","function","fuzzy","fvc","galis","game","games","garisgaris","gateway","gaussian","geometry","gizixd","gldm","glrm","gr","gradient","gradients","gray","grayscaling","grcitra","ground","growth","ha","haar","had","handwriting","harris","hash","hh","hidden","hierarchy","high","hijauxd","hl","hog","ht","idb","ii","ij","iksass","image","indahmulya","indicator","indicators","infoinfo","inginxd","inixd","inktech","interface","interprise","intervace","intervensi","interview","intraseluler","intrusion","invariant","inventori","ips","iptables","italic","jaringanjaringan","jarixd","java","jejaring","jiwaxd","jst","kabupatenkabupaten","kaganga","kallista","karakterkarakter","karapanxd","kec","kerapan","kerjasamaxd","kesejahteraanya","key","keypoint","keyword","keywords","kg","kit","kkm","kluster","kmeans","kohonen","kokop","komulatif","konang","konekasi","kpi","kriteriakriteria","kriteriaxd","ksom","kub","kuisioner","kuisoner","lainlain","lainxd","langkahlangkah","language","languange","latent","layer","lda","learning","least","length","lerning","leveling","lh","life","light","linier","link","listening","ll","load","log","logic","low","lsa","lsasom","lt","lunakxd","lunturnya","lvq","lyapunov","machine","madistrindo","maduraindonesia","maduraxd","mail","making","malan","mamdani","management","manager","mandiriauto","map","mape","maps","mapserver","martodirdjo","masingmasing","masking","matching","matrix","maze","mazexd","mcdm","mdf","mean","melakukanxd","membatu","memilikixd","mengenkripsi","menggunakanxd","message","metadata","method","mg","middleware","minimnya","minutea","minutiae","modung","momentum","monitoring","morfologi","mosaic","mosaikpanoramik","moving","mpc","mse","multiatribute","multimedia","multiobjective","multiple","naive","nave","nbc","negaranegara","network","neural","ngram","node","nomor","non","npc","number","numberxd","nya","obatobatan","objective","obyek","obyektif","of","offline","ofr","oldinary","ols","omax","ontologi","ontology","open","optical","optimized","optimiztion","optimum","ordered","organizing","oriented","orl","output","owl","panoramic","panoramik","panoteng","parameterparameter","parsing","part","particle","pasienxd","pattern","pca","pe","pejualan","pelajarsantri","pelevelan","pemvalidasian","penjadwaln","perankingan","perankingannya","percentage","perconbaan","performance","periodeperiode","permasalaha","perusahaanxd","pihakpihak","pihakxd","pixel","pixels","plainteks","plasmodium","plastec","platform","playable","player","pmg","podhek","point","pose","ppa","prakandidat","precision","preference","presentase","preshion","prevention","prim","principal","prinsipnya","print","prism","probabilitasmetode","process","processing","produksipada","produktivitas","profitabilitas","programing","programming","programprogram","project","prosentase","prosesnya","pso","pt","ptxd","quantity","quantization","query","rangkebbhan","rank","ranks","raskin","ratarata","rate","rater","rating","ratus","rbfn","rbfnn","rbfnnxd","rc","rdf","reading","real","realistisxd","realitas","reality","realtime","recall","recognition","rekomndasi","relative","release","resource","resources","responden","retrieval","reuse","ridge","riilxd","rill","riwayatxd","rehabilitasi","roughness","rts","run","saaty","salafiyah","roughness","sales","sasaranxd","satunya","satunya","scale","scm","scorecard","scoring","screen","sdk","sdkxd","sdlc","sdm","search","second","security","segmentasinya","seharihari","sekuensial","self","semantic","sencitivity","seolaholah","separation","seringkali","server","service","ses","seseorangxd","shop","shortest","sift","sikannya","similar","similaritas","similarity","simple","simtak","single","singular","sistemxd","skenarioskenario","sky","sma","smarter","smartphone","smoothing","smooting","smp","sms","snort","software","solusinya","solusinyaxd","solution","som","sort","source","spare","spasial","spci","speaking","specificity","speech","spk","square","stakeholder","state","statistik","statusxd","stemmer","stemming","stockpile","strategi","strategy","straw","stripping","style","sub","subkriteria","subset","subsistem","subtropics","subyektivitas","sumenep","suplier","supplier","supply","swarm","syafiiyah","system","tab","tamansepanjang","technique","telekomunikasi","terater","terhadapxd","termination","terpisahpisah","tersebutxd","tertentuumumnya","test","testes","testing","thinning","thomas","threshold","tiaptiap","time","tinggixd","titiktitik","tnpk","to","toba","toefl","toeflxd","togaf","tool","tools","tooltool","topsis","traffic","tragah","training","transform","treshold","truth","tsai","tujuansetelah","tulangan","tuneup","two","ujicoba","userxd","utnuk","validitas","value","vector","velocity","vii","virtual","vision","vr","wachid","waktuxd","waterfall","watershed","wavelet","web","website","webxd","wide","window","winnowing","world","www","xd","xna","yakersuda","yangxd",'baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama','tidaknya'])
# Mengubah List ke dictionary
list_stopwords = set(list_stopwords)
```

## Stemming

*Stemming* merupakan tahap preprocessing yang digunakan untuk mengubah semua kata ke dalam bentuk baku atau dasar. Pada python *stemming* dilakukan menggunakan algoritma nadzief andriani pada library sastrawi.

Namun, pada topic modelling tidak diperlukan proses stem, dikarenakan setiap kata memiliki arti yang berbeda dan tujuan topic modelling adalah mengetahui proporsi *term* terhadap topik.

berikut kode stemming :

```python
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
```

