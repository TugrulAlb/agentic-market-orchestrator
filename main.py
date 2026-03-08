"""
main.py — Streamlit Arayüzü

Bu dosya YALNIZCA UI mantığı içerir.
İş mantığı, API istekleri, LLM çağrıları veya geo kodlama **bu dosyada yoktur**.
Tüm iş yükü Orchestrator sınıfına delege edilir.

Çalıştırmak için:
    streamlit run main.py
"""

import asyncio

import streamlit as st

from clients.market_api import Location
from core.config import get_settings
from core.orchestrator import Orchestrator
from utils.geo_helpers import get_districts, get_provinces, resolve_coordinates

# ──────────────────────────────────────────────────────────────────────────────
# Sayfa yapılandırması
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🛒 AI Market Asistanı",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stButton > button {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white; border: none; border-radius: 8px;
        font-weight: 700; padding: 0.6rem 1.4rem;
    }
    .stButton > button:hover { opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;color:#e63946;'>🛒 AI Market Asistanı</h1>"
    "<p style='text-align:center;color:#6c757d;'>"
    "Konumunuza göre — En Ucuz Market Fiyatları (Agentic AI)</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Kenar Çubuğu (Sidebar)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.subheader("📍 Konum")

    radius = st.slider("Arama Mesafesi (km)", 1, 15, 5)

    provinces = get_provinces()
    default_province = "Konya"
    default_idx = provinces.index(default_province) if default_province in provinces else 0
    selected_province = st.selectbox("🏙️ İl", provinces, index=default_idx)

    districts = get_districts(selected_province)
    selected_district = st.selectbox("🏘️ İlçe", districts) if districts else ""

    coords, used_fallback = resolve_coordinates(selected_province, selected_district)

    if selected_district:
        st.markdown(f"**Seçim:** {selected_province} / {selected_district}")
    else:
        st.markdown(f"**Seçim:** {selected_province}")

    if used_fallback and selected_province != "Konya":
        st.warning("Konum çözümlenemedi — geçici olarak Konya merkeziyle arama yapılıyor.")

    lat, lon = coords
    loc = Location(
        latitude=float(lat),
        longitude=float(lon),
        distance_km=float(radius),
        size=get_settings().api_limit,
    )

    on_hand = st.text_area(
        "Elinizdeki Malzemeler",
        value="",
        help="Virgülle ayırın (örn: soğan, zeytinyağı, tuz)",
    )

    st.divider()
    st.caption("İpucu: Arama mesafesini artırırsanız daha fazla market sonucu çıkar.")

    settings = get_settings()
    if not settings.is_llm_configured:
        st.warning(
            "Tarif analizi için Azure anahtarları eksik. "
            "`.env` veya `.streamlit/secrets.toml` dosyasına "
            "`AZURE_ENDPOINT` ve `AZURE_API_KEY` ekleyin."
        )

# ──────────────────────────────────────────────────────────────────────────────
# Ana Giriş
# ──────────────────────────────────────────────────────────────────────────────

col1, _ = st.columns([3, 1])
with col1:
    recipe_request = st.text_input(
        "Ne pişirmek istiyorsunuz?",
        value="Konya usulü fırın kebabı yapmak istiyorum",
    )

run_btn = st.button("🚀 Fiyatları Getir", type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Çalıştırma — UI sadece Orchestrator'ı çağırır
# ──────────────────────────────────────────────────────────────────────────────

if run_btn:
    if not recipe_request.strip():
        st.warning("Lütfen bir yemek isteği girin.")
        st.stop()

    async def log_cb(msg: str) -> None:
        _ = msg

    orchestrator = Orchestrator()

    with st.spinner("Ajanlar çalışıyor…"):
        try:
            df, recipe_name = asyncio.run(
                orchestrator.run(recipe_request, on_hand, selected_district, loc, log_cb)
            )
        except Exception as exc:
            st.error(f"❌ Bir hata oluştu: {exc}")
            st.stop()

    # ── Sonuçlar ──────────────────────────────────────────────────────────────
    if df.empty:
        st.info("Elinizdeki malzemeler yeterli — ekstra alışveriş gerekmez!")
    else:
        st.success(f"✅ **{recipe_name}** tarifi için alışveriş listesi hazır!")
        st.subheader("📊 Fiyat Tablosu")

        display_cols = [c for c in ["Arama", "Product", "Market", "Neighborhood", "Price"] if c in df.columns]
        st.dataframe(
            df[display_cols].style.format({"Price": "{:.2f} ₺"}),
            width=None,
            height=min(420, 60 + len(df) * 40),
        )

        total = df["Price"].sum()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🛒 Toplam Tutar", f"{total:.2f} ₺")
        col_b.metric("📦 Bulunan Ürün", f"{len(df)} kalem")
        col_c.metric("🏪 Farklı Market", f"{df['Market'].nunique()}")

        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ CSV İndir",
            data=csv,
            file_name="alisveris_listesi.csv",
            mime="text/csv",
        )
