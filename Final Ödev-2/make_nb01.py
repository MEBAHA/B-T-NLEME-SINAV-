import json, os

os.makedirs("/home/claude/arizakodlari/notebooks", exist_ok=True)

def md(src, id_=""):
    return {"cell_type":"markdown","metadata":{},"source":src,"id": id_ or "md"+str(abs(hash(src)))[:6]}

def code(src, id_=""):
    return {"cell_type":"code","metadata":{},"source":src,"outputs":[],"execution_count":None,"id": id_ or "co"+str(abs(hash(src)))[:6]}

cells = [
md("# 📋 NLP Final Ödev-2 — 01: Veri Ön İşleme\n\n**Proje:** Endüstriyel Arıza Kodları Eşleştirme\n\n**Grup:** Mario Enrique Motede Dasilva, Heriberto Fernandez Chale, Matias Fernando Ndong Owono Obiang\n\nBu notebook ham arıza kodu verilerini yükleyip Türkçe NLP ön işleme adımlarını uygular ve `lemmatized.csv` ile `stemmed.csv` çıktı dosyalarını üretir.","md_title"),

md("## 1. Kütüphane Kurulumu ve İçe Aktarma","md_01"),
code("# Gerekli kütüphaneleri kur (ilk çalıştırmada)\n# !pip install gensim zeyrek snowballstemmer nltk pandas\n\nimport pandas as pd\nimport re\nimport nltk\nimport zeyrek\nimport snowballstemmer\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\n\nnltk.download('punkt_tab', quiet=True)\nnltk.download('stopwords', quiet=True)\n\nprint('Tum kutuphaneler yuklendi.')","c01"),

md("## 2. Ham Veriyi Yükle","md_02"),
code('df_raw = pd.read_csv("data/fault_codes_raw.csv")\nprint(f"Toplam kayit: {len(df_raw)}")\nprint(f"Sutunlar    : {list(df_raw.columns)}")\ndf_raw.head(5)',"c02"),

md("## 3. Türkçe Stop-Words Listesi","md_03"),
code('tr_stopwords = set(stopwords.words("turkish"))\nextra = {"ve","ile","icin","olan","veya","ancak","ise","bu","bir","da","de","ki","cok","daha","en","gibi","kadar","var","yok","her","bazi","ayni","diger","tum","butun"}\ntr_stopwords.update(extra)\nprint(f"Stop-word sayisi: {len(tr_stopwords)}")',"c03"),

md("## 4. Ön İşleme Fonksiyonları","md_04"),
code('import warnings\nwarnings.filterwarnings("ignore")\n\ndef clean_text(text):\n    text = text.lower()\n    text = re.sub(r"[^a-z\\u00c0-\\u017f\\s]", " ", text)\n    text = re.sub(r"\\s+", " ", text).strip()\n    return text\n\ndef tokenize_tr(text):\n    tokens = word_tokenize(text, language="turkish")\n    return [t for t in tokens if t.isalpha() and t not in tr_stopwords and len(t) > 2]\n\nanalyzer = zeyrek.MorphAnalyzer()\n\ndef lemmatize_tr(tokens):\n    lemmas = []\n    for token in tokens:\n        result = analyzer.lemmatize(token)\n        if result and result[0][1]:\n            lemmas.append(result[0][1][0].lower())\n        else:\n            lemmas.append(token)\n    return lemmas\n\nstemmer_tr = snowballstemmer.stemmer("turkish")\n\ndef stem_tr(tokens):\n    return [stemmer_tr.stemWord(t) for t in tokens]\n\nprint("On isleme fonksiyonlari hazir.")\nornek = "Motor asiri isinma hatasi tespit edildi."\ncleaned = clean_text(ornek)\ntokens  = tokenize_tr(cleaned)\nlemmas  = lemmatize_tr(tokens)\nstems   = stem_tr(tokens)\nprint(f"Ham     : {ornek}")\nprint(f"Temizle : {cleaned}")\nprint(f"Tokenler: {tokens}")\nprint(f"Lemma   : {lemmas}")\nprint(f"Stem    : {stems}")',"c04"),

md("## 5. Tüm Veri Setine Ön İşleme Uygula","md_05"),
code('lemmatized_docs = []\nstemmed_docs    = []\n\nfor idx, row in df_raw.iterrows():\n    doc_id  = row["document_id"]\n    content = row["content"]\n    cleaned = clean_text(content)\n    tokens  = tokenize_tr(cleaned)\n    lemmas  = lemmatize_tr(tokens)\n    stems   = stem_tr(tokens)\n    lemmatized_docs.append({"document_id": doc_id, "content": " ".join(lemmas)})\n    stemmed_docs.append   ({"document_id": doc_id, "content": " ".join(stems)})\n    if (idx + 1) % 20 == 0:\n        print(f"  {idx+1}/{len(df_raw)} kayit islendi...")\n\nprint(f"Toplam {len(df_raw)} kayit islendi.")',"c05"),

md("## 6. CSV Olarak Kaydet","md_06"),
code('df_lemmatized = pd.DataFrame(lemmatized_docs)\ndf_stemmed    = pd.DataFrame(stemmed_docs)\n\ndf_lemmatized.to_csv("data/lemmatized.csv", index=False, encoding="utf-8")\ndf_stemmed.to_csv   ("data/stemmed.csv",    index=False, encoding="utf-8")\n\nprint("Dosyalar kaydedildi:")\nprint("  data/lemmatized.csv")\nprint("  data/stemmed.csv")',"c06"),

md("## 7. Boyut ve Yapı Karşılaştırması","md_07"),
code('print(f"{\"Veri Seti\":<20} {\"Kayit\":<10} {\"Ort. Kelime/Belge\":<22} {\"Toplam Kelime\"}")\nprint("-" * 65)\nfor name, df in [("Ham Veri", df_raw), ("Lemmatized", df_lemmatized), ("Stemmed", df_stemmed)]:\n    wc = df["content"].apply(lambda x: len(str(x).split()))\n    print(f"{name:<20} {len(df):<10} {wc.mean():<22.1f} {wc.sum()}")',"c07"),

md("## 8. Örnek Çıktı Karşılaştırması","md_08"),
code('print("HAM vs LEMMATIZED vs STEMMED (ilk 3 belge)")\nprint("=" * 65)\nfor i in range(3):\n    print(f"\\n{df_raw.iloc[i][\"document_id\"]}")\n    print(f"  HAM  : {df_raw.iloc[i][\"content\"][:80]}...")\n    print(f"  LEMMA: {df_lemmatized.iloc[i][\"content\"][:80]}...")\n    print(f"  STEM : {df_stemmed.iloc[i][\"content\"][:80]}...")',"c08"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

with open("/home/claude/arizakodlari/notebooks/01_onisleme.ipynb","w",encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook 01 creado OK")
