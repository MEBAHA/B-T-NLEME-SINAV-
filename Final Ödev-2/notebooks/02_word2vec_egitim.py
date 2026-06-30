# ============================================================
# NLP Final Ödev-2 — Notebook 02: Word2Vec Model Eğitimi
# Proje: Endüstriyel Arıza Kodları Eşleştirme
# Ad ve Soyad: Rodolfo Mba NDONG MEBAHA
# ============================================================

# %% [markdown]
# ## 1. Kütüphane Yükleme

# %%
import pandas as pd
import os
import warnings
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)

# %% [markdown]
# ## 2. Ön İşlenmiş Verileri Yükle

# %%
df_lemma = pd.read_csv("data/lemmatized.csv")
df_stem  = pd.read_csv("data/stemmed.csv")

print(f"Lemmatized kayıt sayısı : {len(df_lemma)}")
print(f"Stemmed kayıt sayısı    : {len(df_stem)}")

# Her satırı token listesine çevir
corpus_lemma = [str(row).split() for row in df_lemma['content']]
corpus_stem  = [str(row).split() for row in df_stem['content']]

print(f"\nLemmatized corpus örneği (doc001):")
print(corpus_lemma[0])
print(f"\nStemmed corpus örneği (doc001):")
print(corpus_stem[0])

# %% [markdown]
# ## 3. 16 Word2Vec Modeli Eğit

# %%
parameters = [
    {'model_type': 'cbow',     'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow',     'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow',     'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow',     'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300},
]

def train_model(corpus, params, prefix):
    sg = 1 if params['model_type'] == 'skipgram' else 0
    model = Word2Vec(
        sentences=corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        sg=sg,
        min_count=1,
        workers=4,
        epochs=100,
        seed=42
    )
    name = f"{prefix}_{params['model_type']}_win{params['window']}_dim{params['vector_size']}"
    path = f"models/{name}.model"
    model.save(path)
    print(f"  ✓ {name}.model kaydedildi")
    return name, model

print("=== LEMMATIZED MODELLERİ (8 adet) ===")
lemma_models = {}
for p in parameters:
    name, m = train_model(corpus_lemma, p, "lemmatized")
    lemma_models[name] = m

print("\n=== STEMMED MODELLERİ (8 adet) ===")
stem_models = {}
for p in parameters:
    name, m = train_model(corpus_stem, p, "stemmed")
    stem_models[name] = m

all_models = {**lemma_models, **stem_models}
print(f"\nToplam {len(all_models)} model eğitildi.")

# %% [markdown]
# ## 4. Her Model İçin Önemli Kelime Benzerlik Örneği
# Rapor için: modelden modele benzerlik skorlarının nasıl değiştiği

# %%
# Vektörü olan bir kelime seç
test_words = ['motor', 'basınç', 'alarm', 'pompa', 'hata']

def safe_most_similar(model, word, topn=5):
    """Kelime modelde yoksa None döndür."""
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return None

print("=" * 70)
print("BAZI MODELLER İÇİN KELİME BENZERLİK ÖRNEKLERİ")
print("=" * 70)

# İlk 3 modeli göster
sample_models = list(all_models.items())[:3]
for model_name, model in sample_models:
    print(f"\n📌 Model: {model_name}")
    for word in test_words:
        result = safe_most_similar(model, word)
        if result:
            top3 = [(w, round(s, 4)) for w, s in result[:3]]
            print(f"  '{word}' → {top3}")
            break  # sadece ilk bulunan kelimeyi göster

# %% [markdown]
# ## 5. Tüm Modeller İçin Özet Tablo

# %%
print("\n" + "=" * 70)
print("RAPOR: TÜM MODELLERİN ÖZETİ")
print("=" * 70)
print(f"{'Model Adı':<50} {'Vocab':<10} {'Dim':<8} {'Tür'}")
print("-" * 80)

for name, model in all_models.items():
    vocab_size = len(model.wv)
    vec_size   = model.wv.vector_size
    mtype = 'SkipGram' if 'skipgram' in name else 'CBOW'
    print(f"{name:<50} {vocab_size:<10} {vec_size:<8} {mtype}")

print(f"\nToplam model: {len(all_models)}")
print("Model dosyaları 'models/' klasörüne kaydedildi.")
