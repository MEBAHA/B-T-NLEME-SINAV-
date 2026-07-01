# Final Ödev-1: Türkçe Haber Metinleri – NLP Ön İşleme Pipeline

**Öğrenci:** Rodolfo Mba NDONG MEBAHA
**Öğrenci No:** 2311081639  
**Ders:** Doğal Dil İşleme  
**Öğretim Üyesi:** Dr. Rabia Yaşa Koştaş  
**Üniversite:** Gümüşhane Üniversitesi 

---

##  Projenin Amacı

Bu proje, gerçek dünya Türkçe haber metinleri üzerinde temel NLP ön işleme adımlarını uygulamak, verinin istatistiksel yapısını (Zipf Yasası) analiz etmek ve veriyi model eğitimine hazır hale getirmek amacıyla hazırlanmıştır.

---

## Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | Milliyet, Hürriyet, Sabah gazeteleri |
| Toplam Makale | 40 |
| Kategori Sayısı | 8 (Spor, Ekonomi, Teknoloji, Sağlık, Eğitim, Siyaset, Kültür, Dünya) |
| Dosya Boyutu | ~14 KB (0.014 MB) |
| Dosya Formatı | CSV (UTF-8-BOM) |

### Sütun Yapısı

| Sütun | Açıklama |
|---|---|
| `Makale_ID` | Benzersiz makale kimliği (TR001–TR040) |
| `Baslik` | Haber başlığı |
| `Kategori` | Haber kategorisi |
| `Kaynak` | Haber kaynağı (Milliyet / Hürriyet / Sabah) |
| `Yil` | Yayın yılı |
| `Icerik` | Ham haber metni |

---

## Uygulanan Ön İşleme Adımları

| Adım | Kullanılan Kütüphane | Açıklama |
|---|---|---|
| Metin Temizleme | `re` | HTML etiketleri, sayılar, özel karakterler kaldırıldı |
| Küçük Harf | Python built-in | Türkçe İ→i, I→ı düzeltmesiyle normalize edildi |
| Tokenization | `nltk` | Kelime ve cümle düzeyinde tokenization |
| Stop Word Removal | `nltk` + özel liste | Türkçe'ye özgü anlamsız kelimeler çıkarıldı |
| Stemming | `snowballstemmer` | Snowball Turkish algoritması ile gövdeye indirgeme |
| Lemmatization | Kural tabanlı | Türkçe çekim ekleri çıkarılarak sözlük köküne indirgeme |

---

## Zipf Yasası Analizi

Ham veri ve temizlenmiş veri üzerinde ayrı ayrı Log-Log Zipf grafikleri oluşturulmuştur. Türkçe haber metinleri Zipf yasasına uygun dağılım sergilemiştir.

- Ham veride en sık kelimeler: `ve` (28), `türkiye` (20), `bu` (17)
- Temizlenmiş veride: `türkiye` (20), `türk` (13), `yıl` (10)

---

## Repo İçeriği

```
├── turk_haberler_nlp.ipynb        # Ana Jupyter Notebook (kod + rapor)
├── turk_haberler_ham.csv          # Ham veri seti
├── turk_haberler_stemmed.csv      # Stemming uygulanmış veri
├── turk_haberler_lemmatized.csv   # Lemmatization uygulanmış veri
└── README.md                      # Bu dosya
```

---

## Kurulum ve Çalıştırma

```bash
pip install nltk pandas matplotlib numpy snowballstemmer
```

Jupyter Notebook'u açın ve tüm hücreleri sırayla çalıştırın (`Run All`).

---

## Kullanılan Kütüphaneler

- `pandas` — veri yükleme ve işleme
- `re` — regex ile metin temizleme
- `nltk` — tokenization ve stop word
- `snowballstemmer` — Türkçe stemming
- `matplotlib` — Zipf grafikleri
- `numpy` — matematiksel işlemler
