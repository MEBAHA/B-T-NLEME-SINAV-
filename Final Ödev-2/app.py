# ============================================================
# NLP Final Ödev-2 — Streamlit Arayüzü
# Endüstriyel Arıza Kodları Eşleştirme Sistemi
# Öğrenci: Rodolfo Mba NDONG MEBAHA — 2311081639
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ── Sayfa Ayarları ──────────────────────────────────────────
st.set_page_config(
    page_title="Arıza Kodları Eşleştirme",
    page_icon="🔧",
    layout="wide"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .scenario-card {
        background: #ffffff;
        border: 1.5px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .scenario-card:hover {
        border-color: #0066cc;
    }
    .category-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .doc-id-tag {
        font-size: 0.78rem;
        color: #888;
        font-family: monospace;
    }
    .scenario-text {
        font-size: 0.91rem;
        color: #333;
        line-height: 1.5;
        margin: 6px 0 10px 0;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 4px;
        margin: 1rem 0 0.8rem 0;
    }
    .info-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #1a4a6e;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.4rem 1.2rem;
        font-size: 0.88rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0052a3;
    }
    .tip-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #6b4c00;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Başlık ──────────────────────────────────────────────────
st.markdown('<div class="main-title">🔧 Endüstriyel Arıza Kodları Eşleştirme</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP Final Ödev-2 · Rodolfo Mba NDONG MEBAHA · 2311081639</div>', unsafe_allow_html=True)

# ── Veri ve Model Yükleme ───────────────────────────────────
@st.cache_resource(show_spinner="Modeller yükleniyor, lütfen bekleyin...")
def load_everything():
    df_raw   = pd.read_csv("data/fault_codes_raw.csv")
    df_lemma = pd.read_csv("data/lemmatized.csv")
    df_stem  = pd.read_csv("data/stemmed.csv")
    model_files = sorted(glob.glob("models/*.model"))
    models = {}
    for path in model_files:
        name = os.path.basename(path).replace(".model", "")
        models[name] = Word2Vec.load(path)
    return df_raw, df_lemma, df_stem, models

df_raw, df_lemma, df_stem, models = load_everything()

# ── Örnek Senaryolar Verisi ─────────────────────────────────
SCENARIOS = [
    {
        "doc_id": "doc001",
        "category": "Motor",
        "color": "#dc2626",
        "icon": "⚙️",
        "title": "Motor Aşırı Isınma",
        "text": "Motor aşırı ısınma hatası tespit edildi. Soğutma sistemi yetersiz çalışıyor ve fan devreye girmiyor. Termostat değeri limitin üzerinde seyrediyor.",
        "tip": "Isınma problemleri genellikle soğutma devresi, fan motoru ve termostat arızalarıyla ilişkilendirilir.",
    },
    {
        "doc_id": "doc002",
        "category": "Pompa",
        "color": "#2563eb",
        "icon": "💧",
        "title": "Pompa Basınç Düşüşü",
        "text": "Pompa basınç düşüşü alarmı aktive oldu. Giriş valfı kısmen tıkalı ve akış debisi beklenen değerin altında kalıyor.",
        "tip": "Basınç düşüşleri tıkalı valf, aşınmış conta veya pompa kavitasyonundan kaynaklanabilir.",
    },
    {
        "doc_id": "doc006",
        "category": "PLC / Kontrol",
        "color": "#7c3aed",
        "icon": "🖥️",
        "title": "PLC Haberleşme Hatası",
        "text": "PLC haberleşme hatası. Kontrol ünitesi ile saha cihazları arasındaki Modbus bağlantısı zaman aşımına uğradı.",
        "tip": "Haberleşme hataları kablo gevşekliği, protokol uyumsuzluğu veya elektromanyetik girişimden kaynaklanabilir.",
    },
    {
        "doc_id": "doc017",
        "category": "CNC / Eksen",
        "color": "#059669",
        "icon": "🔩",
        "title": "CNC Eksen Sürücü Hatası",
        "text": "CNC tezgah eksen sürücü hatası. X ekseni servo sürücüsü aşırı akım koruması devreye girdi.",
        "tip": "Servo sürücü aşırı akım hataları mekanik blokaj, aşınmış rulman veya yanlış motor parametrelerinden olabilir.",
    },
    {
        "doc_id": "doc026",
        "category": "Robot",
        "color": "#d97706",
        "icon": "🤖",
        "title": "Robot Kol Limit Aşımı",
        "text": "Robot kol eklem hareket sınırı aşıldı. Altıncı eksen açısal limit değerini geçti ve acil durdurma aktif oldu.",
        "tip": "Eksen limit hataları yanlış programlama, enkoder kayması veya mekanik çarpışmadan kaynaklanabilir.",
    },
    {
        "doc_id": "doc031",
        "category": "Elektrik Panosu",
        "color": "#be185d",
        "icon": "⚡",
        "title": "Elektrik Panosu Aşırı Isı",
        "text": "Elektrik panosu aşırı ısı alarmı. Soğutma fanı arızalı ve kabin iç sıcaklığı kırk beş derece üzerine çıktı.",
        "tip": "Pano ısı alarmları soğutma fanı arızası, hava sirkülasyonu yetersizliği veya yük artışından kaynaklanır.",
    },
    {
        "doc_id": "doc043",
        "category": "Gaz / Güvenlik",
        "color": "#b45309",
        "icon": "☢️",
        "title": "Gaz Dedektörü Alarm Aşımı",
        "text": "Gaz dedektörü alarm seviyesi aşımı. Ortamda hidrojen sülfür konsantrasyonu eşik değerini geçti.",
        "tip": "Gaz alarm aşımları sensör kirliliği, gerçek gaz sızıntısı veya yanlış kalibrasyon kaynaklı olabilir.",
    },
    {
        "doc_id": "doc007",
        "category": "Kompresör",
        "color": "#0891b2",
        "icon": "🌀",
        "title": "Kompresör Yağ Basıncı Düşük",
        "text": "Kompresör yağ basıncı düşük alarm verdi. Yağ pompası verimi düştü ve yağ filtresi tıkanmış durumda.",
        "tip": "Kompresör yağ basıncı sorunları filtre tıkanması, yağ pompası aşınması veya yağ seviyesi düşüklüğünden kaynaklanır.",
    },
]

# ── Yardımcı Fonksiyonlar ───────────────────────────────────
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüöçşğı\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_doc_vector(model, text):
    tokens = str(text).split()
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def run_search(query_raw, model_name, top_n=5, exclude_self=None):
    model = models[model_name]
    query_clean = preprocess_query(query_raw)
    corpus_df = df_lemma if model_name.startswith("lemmatized") else df_stem
    query_vec = get_doc_vector(model, query_clean).reshape(1, -1)
    scores = []
    for _, row in corpus_df.iterrows():
        if exclude_self and row['document_id'] == exclude_self:
            continue
        doc_vec = get_doc_vector(model, row['content']).reshape(1, -1)
        sim = cosine_similarity(query_vec, doc_vec)[0][0]
        scores.append((row['document_id'], float(sim)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def render_results(top_results):
    for rank, (doc_id, score) in enumerate(top_results, 1):
        raw_row = df_raw[df_raw['document_id'] == doc_id]
        if raw_row.empty:
            continue
        content = raw_row['content'].values[0]
        color = "#27ae60" if score >= 0.97 else "#e67e22" if score >= 0.90 else "#e74c3c"
        bar_width = int(score * 100)
        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                <span style="font-weight:700; color:#0066cc;">#{rank} — {doc_id}</span>
                <span style="background:{color}; color:white; border-radius:12px;
                             padding:2px 12px; font-size:0.85rem; font-weight:700;">
                    {score:.4f}
                </span>
            </div>
            <div style="background:#e0e0e0; border-radius:4px; height:6px; margin-bottom:8px">
                <div style="background:{color}; width:{bar_width}%; height:6px; border-radius:4px;"></div>
            </div>
            <div style="font-size:0.93rem; color:#333;">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Panel Lateral ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Ayarları")
    model_names = sorted(models.keys())
    default_idx = model_names.index("lemmatized_skipgram_win2_dim300") \
                  if "lemmatized_skipgram_win2_dim300" in model_names else 0
    selected_model = st.selectbox("Word2Vec Modeli", model_names, index=default_idx)
    top_n = st.slider("Sonuç Sayısı", 1, 10, 5)

    st.markdown("---")
    st.markdown("### 📊 Model Bilgisi")
    m = models[selected_model]
    c1, c2 = st.columns(2)
    c1.metric("Kelime Hazinesi", len(m.wv))
    c2.metric("Vektör Boyutu", m.vector_size)
    mtype  = "SkipGram" if "skipgram" in selected_model else "CBOW"
    win    = "2" if "win2" in selected_model else "4"
    prefix = "Lemmatized" if selected_model.startswith("lemmatized") else "Stemmed"
    st.info(f"**Tür:** {mtype}  \n**Pencere:** {win}  \n**Ön İşleme:** {prefix}")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#888; text-align:center'>
        📄 {len(df_raw)} belge · 🤖 {len(models)} model<br>
        Word2Vec + Kosinüs Benzerliği
    </div>
    """, unsafe_allow_html=True)

# ── Ana Sekmeler ─────────────────────────────────────────────
tab_search, tab_scenarios, tab_compare = st.tabs([
    "🔍 Arıza Sorgula",
    "📖 Örnek Senaryolar",
    "📊 Model Karşılaştırma"
])

# ══════════════════════════════════════════════════════
# SEKME 1 — ARAMA
# ══════════════════════════════════════════════════════
with tab_search:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">🔍 Arıza Açıklaması Gir</div>', unsafe_allow_html=True)

        # Senaryodan gelen metin
        prefill = st.session_state.get("prefill_query", "")

        query = st.text_area(
            "Türkçe olarak bir arıza açıklaması yazın:",
            value=prefill,
            height=110,
            placeholder="Örnek: Motor aşırı ısınma hatası tespit edildi. Soğutma sistemi yetersiz çalışıyor...",
            key="query_input"
        )

        if prefill:
            st.markdown(
                '<div class="tip-box">💡 <b>Örnek Senaryolar</b> sekmesinden bir arıza yüklendi. '
                'Metni istediğiniz gibi düzenleyebilirsiniz.</div>',
                unsafe_allow_html=True
            )

        search_btn = st.button("🔎 Benzer Arıza Kodlarını Bul", key="search_btn")

    with col_right:
        st.markdown('<div class="section-header">📋 Sistem Bilgisi</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        📄 <b>Toplam Belge:</b> {len(df_raw)}<br>
        🔧 <b>Konu:</b> Endüstriyel arıza kodları<br>
        🌐 <b>Dil:</b> Türkçe<br>
        🤖 <b>Toplam Model:</b> {len(models)}<br>
        📐 <b>Yöntem:</b> Word2Vec + Cosine<br>
        ⚗️ <b>Ön İşleme:</b> Zeyrek + Snowball
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Seçili Model:**")
        st.code(selected_model.replace("_", "_\n"), language=None)

    # Sonuçlar
    if search_btn:
        if not query.strip():
            st.warning("⚠️ Lütfen bir arıza açıklaması girin.")
        else:
            st.markdown('<div class="section-header">📌 En Benzer Arıza Kodları</div>', unsafe_allow_html=True)
            with st.spinner("Hesaplanıyor..."):
                top_results = run_search(query, selected_model, top_n)
            render_results(top_results)
            st.markdown("---")
            summary = []
            for rank, (doc_id, score) in enumerate(top_results, 1):
                raw_row = df_raw[df_raw['document_id'] == doc_id]
                content = raw_row['content'].values[0][:70] + "..." if not raw_row.empty else ""
                summary.append({"Sıra": rank, "Belge": doc_id,
                                 "Benzerlik": round(score, 4), "İçerik": content})
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # Temizle
    if prefill:
        if st.button("🗑️ Metni Temizle", key="clear_btn"):
            st.session_state["prefill_query"] = ""
            st.rerun()

# ══════════════════════════════════════════════════════
# SEKME 2 — ÖRNEK SENARYOLAR
# ══════════════════════════════════════════════════════
with tab_scenarios:
    st.markdown('<div class="section-header">📖 Örnek Arıza Senaryoları</div>', unsafe_allow_html=True)
    st.markdown(
        "Aşağıdaki gerçek arıza kayıtlarından birini seçin. "
        "**'Bu Arızayı Dene'** butonuna tıklayarak sistemi test edin.",
        unsafe_allow_html=False
    )
    st.markdown("---")

    # 2 sütun grid
    cols = st.columns(2)

    for i, scenario in enumerate(SCENARIOS):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="scenario-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px">
                    <span style="font-size:1.3rem">{scenario['icon']}</span>
                    <span style="font-weight:700; font-size:1rem; color:#1a1a2e">
                        {scenario['title']}
                    </span>
                </div>
                <div style="margin-bottom:6px">
                    <span class="category-badge"
                          style="background:{scenario['color']}22; color:{scenario['color']};">
                        {scenario['category']}
                    </span>
                    <span class="doc-id-tag" style="margin-left:8px">{scenario['doc_id']}</span>
                </div>
                <div class="scenario-text">{scenario['text']}</div>
                <div style="background:#fffbeb; border-left:3px solid #f59e0b;
                            padding:6px 10px; border-radius:4px; font-size:0.82rem; color:#78350f;
                            margin-bottom:10px">
                    💡 {scenario['tip']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                f"▶️ Bu Arızayı Dene",
                key=f"scenario_btn_{scenario['doc_id']}"
            ):
                st.session_state["prefill_query"] = scenario["text"]
                st.success(f"✅ '{scenario['title']}' senaryosu yüklendi! **'Arıza Sorgula'** sekmesine geçin.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem'>"
        "Senaryolar veri setinden alınmıştır (doc001–doc080). "
        "Kendi arıza açıklamanızı da <b>Arıza Sorgula</b> sekmesine yazabilirsiniz."
        "</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
# SEKME 3 — MODEL KARŞILAŞTIRMA
# ══════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-header">📊 Tüm 16 Model ile Karşılaştırma</div>', unsafe_allow_html=True)

    compare_query = st.text_area(
        "Karşılaştırma için bir arıza metni girin:",
        value=st.session_state.get("prefill_query", SCENARIOS[0]["text"]),
        height=90,
        key="compare_input"
    )

    if st.button("🔄 Tüm Modelleri Karşılaştır", key="compare_btn"):
        if not compare_query.strip():
            st.warning("⚠️ Lütfen bir metin girin.")
        else:
            with st.spinner("16 model hesaplanıyor..."):
                comparison = []
                for mname in sorted(models.keys()):
                    res = run_search(compare_query, mname, top_n=1)
                    if res:
                        doc_id, score = res[0]
                        raw_row = df_raw[df_raw['document_id'] == doc_id]
                        content = raw_row['content'].values[0][:65] + "..." if not raw_row.empty else ""
                        mtype  = "SkipGram" if "skipgram" in mname else "CBOW"
                        prefix = "Lemmatized" if mname.startswith("lemmatized") else "Stemmed"
                        win    = "2" if "win2" in mname else "4"
                        dim    = "300" if "dim300" in mname else "100"
                        comparison.append({
                            "Model": mname,
                            "Tür": mtype,
                            "Ön İşleme": prefix,
                            "Pencere": win,
                            "Boyut": dim,
                            "#1 Belge": doc_id,
                            "Skor": round(score, 4),
                            "İçerik": content
                        })

            df_comp = pd.DataFrame(comparison)

            # Skor dağılımı
            c1, c2, c3 = st.columns(3)
            c1.metric("En Yüksek Skor", f"{df_comp['Skor'].max():.4f}",
                      df_comp.loc[df_comp['Skor'].idxmax(), 'Model'].replace("_", " "))
            c2.metric("En Düşük Skor", f"{df_comp['Skor'].min():.4f}",
                      df_comp.loc[df_comp['Skor'].idxmin(), 'Model'].replace("_", " "))
            c3.metric("Ortalama Skor", f"{df_comp['Skor'].mean():.4f}")

            st.markdown("---")
            st.markdown("**Tüm Modellerin #1 Sonucu:**")
            st.dataframe(
                df_comp.style.background_gradient(subset=["Skor"], cmap="YlGn"),
                use_container_width=True,
                hide_index=True
            )

            # En çok önerilen belge
            st.markdown("---")
            top_docs = df_comp["#1 Belge"].value_counts()
            st.markdown(f"**En çok önerilen belge:** `{top_docs.index[0]}` "
                        f"— {top_docs.iloc[0]} model tarafından seçildi")

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "NLP Final Ödev-2 · Gümüşhane Üniversitesi · "
    "Rodolfo Mba NDONG MEBAHA (2311081639)"
    "</div>",
    unsafe_allow_html=True
)