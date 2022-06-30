# Crawling Data

Crawling data merupakan proses pengumpulan data yang dilakukan pada suatu web atau sumber lain dari internet seperti media sosial, berita, artikel, dll.

Salah satu framework pada python yang digunakan untuk crawling data adalah scrapy. Scrapy dapat digunakan untuk mengekstrak, memproses dna menyimpan data dari sebuah website dengan skala besar ke format yang diinginkan.

## Instalasi Scrapy

Untuk menginstall scrapy, dapat dilakukan pada command prompt dengan menuliskan code

```
pip install scrapy
```



## Membuat File Scrapy baru

Untuk memulai pembuatan file scrapy baru dapat dilakukan menggunakan kode di bawah dan kemudian diikuti dnegan namaproject yang diinginkan.

```
scrapy startproject namaproject
```




## Menjalankan Spider Baru

Sebelum membuat spider baru, maka terlebih dahulu masuk ke dalam projectscrapy yang telah dibuat terlebih dahulu

```
cd nama project
```

lalu, menuliskan command yang berisikan nama file yang akan dibuat dan juga alamat website yang akan diambil datanya sebagai berikut

```
scrapy genspider example example.com
```



## Menulis Program Scapper

Setelah membuat file spider, maka pada file tersebut telah tersedia file dengan nama file yang telah dibuat pada pembuatan spider.

Pada file tersebut telah tersedia code default untuk melakukan scraping yang nantinya akan diubah sesuai kebutuhan.

```python
import scrapy

class scrapPTA(scrapy.Spider):
    name = 'PTA'
    allowed_domains = ['pta.trunojoyo.ac.id']
    start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/7/'+str(x)+" " for x in range(2,20)]

    def parse(self, response):
        for link in response.css('a.gray.button::attr(href)') :
            yield response.follow(link.get(),callback=self.parse_categories)

    def parse_categories(self, response):
        products = response.css('div#content_journal ul li')
        for product in products:
            yield {
                'judul' : product.css('div a.title::text').get().strip(),
                'penulis' : product.css('div div:nth-child(2) span::text').get().strip(),
                'dosen 1' : product.css('div div:nth-child(3) span::text').get().strip(),
                'dosen 2' : product.css('div div:nth-child(4) span::text').get().strip(),
                'abstrak' : product.css('div div:nth-child(2) p::text').get().strip()
            }
```

Class scrapPTA() digunakan untuk melakukan perayapan (spidering) website sesuai dengan alamat yang telah ditentukan.

Class Parse digunakan untuk melakukan follow link pada button yang telah ditenukan

Calss parse_categories merupakan callback function yang akan dipanggil pada saat setelah mengikuti link yang ditentukan pada parse(). 

Pada class ini, ditentukan elemen elemen apa yang akan diambil dengan melakukan copy select elemen atau class pada website yang akan diambil datanya.

## Menjalankan File Spider

Sebelum menjalankan file spider, terlebih dahulu masuk ke dalam direktori spider.

Kemudian dapat dilakukan running spider menggunakan command 

```
scrapy runspider namafile.py
```

## Menyimpan data ke file excel

Untuk menyimpan data yang telah diabil dari suatu web dapat dilakukan dengan menggunakan perintah 

```
scrapy crawl namafile -O namafileyangdiinginkan.xlsx
```

data yang telah di crawl dapat disimpan dalam format lain.
