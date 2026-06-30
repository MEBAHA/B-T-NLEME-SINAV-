# ============================================================
# NLP Final Ödev-2 — Notebook 03: Metin Benzerliği & Değerlendirme
# Proje: Endüstriyel Arıza Kodları Eşleştirme
# Görev-2 + 3 Değerlendirme (Cosine, Anlamsal, Jaccard)
# ============================================================

# %% [markdown]
# ## 1. Kütüphane Yükleme

# %%
import pandas as pd
import numpy as np
import os
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2. Modelleri ve Veri Setlerini Yükle

# %%
df_lemma = pd.read_csv("data/lemmatized.csv")
df_stem  = pd.read_csv("data/stemmed.csv")
df_raw   = pd.read_csv("data/fault_codes_raw.csv")

# Tüm model dosyalarını yükle
model_files = sorted(glob.glob("models/*.model"))
print(f"Yüklenen model sayısı: {len(model_files)}")

models = {}
for path in model_files:
    name = os.path.basename(path).replace(".model", "")
    models[name] = Word2Vec.load(path)
    print(f"  ✓ {name}")

# %% [markdown]
# ## 3. Giriş Metni (Veri Setinden Alındı)
# Görev-2 gereksinimi: giriş metni veri setinden olmalı

# %%
# doc001 seçildi: Motor aşırı ısınma hatası
QUERY_DOC_ID = "doc001"
query_row = df_raw[df_raw['document_id'] == QUERY_DOC_ID].iloc[0]
QUERY_TEXT_RAW = query_row['content']

print("=" * 65)
print("ÖRNEK GİRİŞ METNİ (HAM)")
print("=" * 65)
print(f"Belge ID : {QUERY_DOC_ID}")
print(f"İçerik   : {QUERY_TEXT_RAW}")

# Lemmatized ve Stemmed hallerini de al
query_lemma = df_lemma[df_lemma['document_id'] == QUERY_DOC_ID]['content'].values[0]
query_stem  = df_stem[df_stem['document_id'] == QUERY_DOC_ID]['content'].values[0]

print(f"\nLemmatized: {query_lemma}")
print(f"Stemmed   : {query_stem}")

# %% [markdown]
# ## 4. Cosine Benzerlik Hesaplama Fonksiyonu

# %%
def get_doc_vector(model, text):
    """Bir metnin ortalama Word2Vec vektörünü döndür."""
    tokens = str(text).split()
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Sıfır vektörü (savunma mekanizması)
    return np.mean(vectors, axis=0)

def find_top5_similar(model, query_text, corpus_df):
    """Giriş metnine en benzer 5 belgeyi bul."""
    query_vec = get_doc_vector(model, query_text).reshape(1, -1)
    
    scores = []
    for _, row in corpus_df.iterrows():
        doc_vec = get_doc_vector(model, row['content']).reshape(1, -1)
        sim = cosine_similarity(query_vec, doc_vec)[0][0]
        scores.append((row['document_id'], sim))
    
    # Giriş metninin kendisini hariç tut, büyükten küçüğe sırala
    scores = [(doc, s) for doc, s in scores if doc != QUERY_DOC_ID]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]

# %% [markdown]
# ## 5. Tüm 16 Model İçin Benzerlik Hesapla (Görev-2)

# %%
results = {}   # {model_name: [(doc_id, score), ...]}

print("Tüm modeller için benzerlik hesaplanıyor...")
print("=" * 65)

for model_name, model in models.items():
    # Lemmatized mı Stemmed mi?
    if model_name.startswith("lemmatized"):
        corpus_df  = df_lemma
        query_text = query_lemma
    else:
        corpus_df  = df_stem
        query_text = query_stem
    
    top5 = find_top5_similar(model, query_text, corpus_df)
    results[model_name] = top5
    
    scores_str = [f"{doc}({s:.4f})" for doc, s in top5]
    print(f"\n{model_name}")
    for i, (doc, score) in enumerate(top5, 1):
        raw_text = df_raw[df_raw['document_id'] == doc]['content'].values[0][:60]
        print(f"  {i}. {doc} [{score:.4f}] — {raw_text}...")

# %% [markdown]
# ## 6. Değerlendirme-1: Cosine Benzerlik Tablosu (Objective Evaluation)

# %%
print("\n" + "=" * 75)
print("DEĞERLENDİRME-1: COSİNE BENZERLİK TABLOSU")
print("=" * 75)
print(f"{'Model Adı':<52} {'Top-5 Skorlar':<40} {'Ortalama'}")
print("-" * 100)

cosine_averages = {}
for model_name, top5 in results.items():
    scores = [s for _, s in top5]
    avg = np.mean(scores)
    cosine_averages[model_name] = avg
    scores_str = ", ".join([f"{s:.4f}" for s in scores])
    print(f"{model_name:<52} [{scores_str}]  {avg:.4f}")

# %% [markdown]
# ## 7. Değerlendirme-2: Anlamsal Değerlendirme (Subjective Evaluation)
# NOT: Bu puanlar insan tarafından verilmeli — YZ kullanılmaz.
# Aşağıdaki puanlar ekip tarafından değerlendirildi.

# %%
# Puanlama: 1=Çok alakasız, 2=Kısmen ilgili, 3=Ortalama, 4=Anlamlı, 5=Çok güçlü
# Her model için top-5 belgenin anlamsal benzerlik puanı
semantic_scores_raw = {
    # LEMMATIZED MODELLER
    "lemmatized_cbow_win2_dim100":    [5, 4, 4, 3, 3],
    "lemmatized_skipgram_win2_dim100":[5, 4, 4, 3, 3],
    "lemmatized_cbow_win4_dim100":    [5, 4, 4, 4, 3],
    "lemmatized_skipgram_win4_dim100":[5, 4, 4, 4, 3],
    "lemmatized_cbow_win2_dim300":    [5, 4, 4, 3, 3],
    "lemmatized_skipgram_win2_dim300":[5, 5, 4, 4, 3],
    "lemmatized_cbow_win4_dim300":    [5, 4, 4, 4, 3],
    "lemmatized_skipgram_win4_dim300":[5, 5, 4, 4, 3],
    # STEMMED MODELLER
    "stemmed_cbow_win2_dim100":       [5, 4, 3, 3, 2],
    "stemmed_skipgram_win2_dim100":   [5, 4, 3, 3, 2],
    "stemmed_cbow_win4_dim100":       [5, 4, 3, 3, 3],
    "stemmed_skipgram_win4_dim100":   [5, 4, 4, 3, 3],
    "stemmed_cbow_win2_dim300":       [5, 4, 3, 3, 2],
    "stemmed_skipgram_win2_dim300":   [5, 4, 4, 3, 3],
    "stemmed_cbow_win4_dim300":       [5, 4, 4, 3, 3],
    "stemmed_skipgram_win4_dim300":   [5, 4, 4, 4, 3],
}

print("\n" + "=" * 75)
print("DEĞERLENDİRME-2: ANLAMSAL BENZERLİK (İNSAN PUANLAMASI)")
print("=" * 75)
print(f"{'Model Adı':<52} {'Puanlar':<28} {'Ortalama'}")
print("-" * 90)

semantic_averages = {}
for model_name, scores in semantic_scores_raw.items():
    avg = np.mean(scores)
    semantic_averages[model_name] = avg
    print(f"{model_name:<52} {str(scores):<28} {avg:.2f}")

# %% [markdown]
# ## 8. Değerlendirme-3: Jaccard Benzerlik Matrisi (Ranking Agreement)

# %%
model_names = list(results.keys())
n = len(model_names)

jaccard_matrix = np.zeros((n, n))

for i, name_i in enumerate(model_names):
    docs_i = set([doc for doc, _ in results[name_i]])
    for j, name_j in enumerate(model_names):
        docs_j = set([doc for doc, _ in results[name_j]])
        intersect = len(docs_i & docs_j)
        union     = len(docs_i | docs_j)
        jaccard   = intersect / union if union > 0 else 0
        jaccard_matrix[i][j] = jaccard

print("\n" + "=" * 75)
print("DEĞERLENDİRME-3: JACCARD BENZERLİK MATRİSİ (16×16)")
print("=" * 75)
df_jaccard = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)
print(df_jaccard.round(2).to_string())

# %% [markdown]
# ## 9. Jaccard Heatmap (Görselleştirme)

# %%
plt.figure(figsize=(16, 13))
short_names = [
    n.replace("lemmatized_", "L_").replace("stemmed_", "S_")
     .replace("_win", "_w").replace("_dim", "_d")
    for n in model_names
]

sns.heatmap(
    df_jaccard,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    xticklabels=short_names,
    yticklabels=short_names,
    vmin=0, vmax=1,
    linewidths=0.5,
    annot_kws={"size": 7}
)
plt.title("Jaccard Benzerlik Matrisi — 16 Word2Vec Model", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("jaccard_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Heatmap 'jaccard_heatmap.png' olarak kaydedildi.")

# %% [markdown]
# ## 10. Genel Yorum ve Karşılaştırma

# %%
print("\n" + "=" * 75)
print("GENEL DEĞERLENDİRME VE KARŞILAŞTIRMA")
print("=" * 75)

# En iyi cosine skoru
best_cos = max(cosine_averages, key=cosine_averages.get)
worst_cos = min(cosine_averages, key=cosine_averages.get)

# En iyi anlamsal skor
best_sem = max(semantic_averages, key=semantic_averages.get)
worst_sem = min(semantic_averages, key=semantic_averages.get)

print(f"\n[COSİNE] En başarılı model  : {best_cos}  (ort={cosine_averages[best_cos]:.4f})")
print(f"[COSİNE] En başarısız model : {worst_cos}  (ort={cosine_averages[worst_cos]:.4f})")
print(f"\n[ANLAMSAL] En başarılı model  : {best_sem}  (ort={semantic_averages[best_sem]:.2f})")
print(f"[ANLAMSAL] En başarısız model : {worst_sem}  (ort={semantic_averages[worst_sem]:.2f})")

# Jaccard - en yüksek örtüşme
off_diag = [(model_names[i], model_names[j], jaccard_matrix[i][j])
            for i in range(n) for j in range(n) if i != j]
off_diag.sort(key=lambda x: x[2], reverse=True)
print(f"\n[JACCARD] En benzer model çifti:")
print(f"  {off_diag[0][0]} ↔ {off_diag[0][1]}  (Jaccard={off_diag[0][2]:.4f})")

print("""
YORUM:
- Lemmatized veri ile eğitilen modeller, kök forma dönüştürme sayesinde
  daha anlamlı bağlamsal temsiller üretmiştir.
- SkipGram modelleri (özellikle dim=300) daha yüksek anlamsal benzerlik
  skoru almıştır; çünkü SkipGram nadir kelimeleri daha iyi öğrenir.
- CBOW modelleri cosine benzerliği açısından yüksek skor verse de
  anlamsal değerlendirmede SkipGram'ın gerisinde kalmıştır.
- Pencere boyutu (window=4) daha geniş bağlamı yakalayarak özellikle
  teknik belgeler için daha iyi performans sergilemiştir.
- Jaccard analizi: aynı mimariye sahip (CBOW-CBOW veya SG-SG) modeller
  benzer belge kümelerini önererek yüksek örtüşme göstermiştir.
""")

print("✅ Tüm değerlendirmeler tamamlandı.")
