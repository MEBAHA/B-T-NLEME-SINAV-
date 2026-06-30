# Endüstriyel Arıza Kodları Eşleştirme Sistemi
## NLP Final Ödev-2

**Öğrenci:** Rodolfo Mba NDONG MEBAHA
**Öğrenci No:** 2311081639
**Ders:** Doğal Dil İşleme
**Danışman:** Dr. Rabia Yaşa Koştaş
**Üniversite:** Gümüşhane Üniversitesi


---

## Proje Hakkında

Bu proje, endüstriyel arıza kodlarının açıklamalarını doğal dil ifadesiyle eşleştiren bir sistemdir. Kullanıcı Türkçe bir arıza açıklaması girdiğinde sistem, Word2Vec tabanlı vektör benzerliği kullanarak veri setindeki en benzer 5 arıza kaydını bulur.

**Proje Konusu:** #33 — Endüstriyel Arıza Kodları Eşleştirme
**Veri Kaynağı:** Teknik belgelerden derlenen 80 Türkçe arıza kodu
**Yöntem:** Word2Vec (CBOW / SkipGram) + Kosinüs Benzerliği

---

## Model Dosyaları (Google Drive)

16 adet Word2Vec modeli boyut nedeniyle Google Drive'da paylaşılmıştır.
Linke sahip olan herkes erişebilir:

🔗 **https://drive.google.com/drive/folders/1pW9pJiBN9Wuxamajx6E9j84Bqti1ThEe?usp=sharing**

Modelleri indirip projedeki `models/` klasörüne koyun.

---

## Kurulum

Python 3.11 gereklidir.

```bash
# 1. Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. NLTK kaynaklarını indir
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## Çalıştırma

### Adım 1 — Modelleri Eğit
```bash
python notebooks/02_word2vec_egitim.py
```
16 Word2Vec modeli `models/` klasörüne kaydedilir (~1-2 dakika).

### Adım 2 — Benzerlik & Değerlendirme
```bash
python notebooks/03_benzerlik_hesaplama.py
```
3 değerlendirme tablosu + Jaccard heatmap (`jaccard_heatmap.png`) üretilir.

### Adım 3 — Web Arayüzünü Başlat
```bash
streamlit run app.py
```
Tarayıcıda `http://localhost:8501` adresinde açılır.

---

## Arayüz Sekmeleri

| Sekme | Açıklama |
|-------|----------|
| 🔍 Arıza Sorgula | Türkçe arıza açıklaması yaz, en benzer 5 belgeyi bul |
| 📖 Örnek Senaryolar | 8 gerçek arıza senaryosu — tek tıkla test et |
| 📊 Model Karşılaştırma | 16 modeli aynı anda karşılaştır |

---

## Proje Yapısı

```
arizakodlari/
│
├── data/
│   ├── fault_codes_raw.csv        ← Ham veri (80 arıza kodu, Türkçe)
│   ├── lemmatized.csv             ← Zeyrek ile lemmatize edilmiş corpus
│   └── stemmed.csv                ← Snowball Turkish ile stemlenmiş corpus
│
├── models/                        ← 16 Word2Vec modeli (Drive'dan indir)
│   ├── lemmatized_cbow_win2_dim100.model
│   ├── lemmatized_skipgram_win2_dim100.model
│   └── ... (toplam 16 model)
│
├── notebooks/
│   ├── 01_onisleme.ipynb          ← Veri ön işleme (Jupyter)
│   ├── 02_word2vec_egitim.py      ← 16 model eğitimi
│   └── 03_benzerlik_hesaplama.py  ← Benzerlik + 3 değerlendirme
│
├── app.py                         ← Streamlit web arayüzü (3 sekme)
├── requirements.txt               ← Gerekli Python kütüphaneleri
├── jaccard_heatmap.png            ← 16×16 Jaccard benzerlik heatmap
└── README.md
```

---

## Eğitilen Modeller (16 adet)

| # | Model Adı | Tür | Pencere | Boyut | Corpus |
|---|-----------|-----|---------|-------|--------|
| 1 | lemmatized_cbow_win2_dim100 | CBOW | 2 | 100 | Lemmatized |
| 2 | lemmatized_skipgram_win2_dim100 | SkipGram | 2 | 100 | Lemmatized |
| 3 | lemmatized_cbow_win4_dim100 | CBOW | 4 | 100 | Lemmatized |
| 4 | lemmatized_skipgram_win4_dim100 | SkipGram | 4 | 100 | Lemmatized |
| 5 | lemmatized_cbow_win2_dim300 | CBOW | 2 | 300 | Lemmatized |
| 6 | lemmatized_skipgram_win2_dim300 | SkipGram | 2 | 300 | Lemmatized |
| 7 | lemmatized_cbow_win4_dim300 | CBOW | 4 | 300 | Lemmatized |
| 8 | lemmatized_skipgram_win4_dim300 | SkipGram | 4 | 300 | Lemmatized |
| 9 | stemmed_cbow_win2_dim100 | CBOW | 2 | 100 | Stemmed |
| 10 | stemmed_skipgram_win2_dim100 | SkipGram | 2 | 100 | Stemmed |
| 11 | stemmed_cbow_win4_dim100 | CBOW | 4 | 100 | Stemmed |
| 12 | stemmed_skipgram_win4_dim100 | SkipGram | 4 | 100 | Stemmed |
| 13 | stemmed_cbow_win2_dim300 | CBOW | 2 | 300 | Stemmed |
| 14 | stemmed_skipgram_win2_dim300 | SkipGram | 2 | 300 | Stemmed |
| 15 | stemmed_cbow_win4_dim300 | CBOW | 4 | 300 | Stemmed |
| 16 | stemmed_skipgram_win4_dim300 | SkipGram | 4 | 300 | Stemmed |

---

## Sonuçlar Özeti

- **En iyi anlamsal model:** `lemmatized_skipgram_win2_dim300` (ort. 4.20/5.0)
- **En yüksek cosine skoru:** CBOW + dim=300 modelleri (≈0.9999)
- **En tutarlı model çifti (Jaccard=1.0):** lemmatized_cbow_win4 ↔ lemmatized_skipgram_win2
