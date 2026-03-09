"""
main.py — Streamlit Arayüzü (v2 — Sohbet + Onay Akışı)

Akış:
  Aşama 1 (CHAT)    : LLM ile tarif sohbeti, malzeme listesi oluştur/düzenle
  Aşama 2 (MARKET)  : Onaylanan listeyle market araması yap, fiyatları getir
"""

import asyncio
import json
from typing import Any

import streamlit as st
from openai import AsyncAzureOpenAI

from clients.market_api import Location
from core.config import get_settings
from core.orchestrator import Orchestrator
from utils.geo_helpers import get_districts, get_provinces, resolve_coordinates

# ──────────────────────────────────────────────────────────────────────────────
# Sayfa Yapılandırması
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🛒 AI Market Asistanı",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stButton > button {
    background: linear-gradient(135deg, #e63946, #c1121f);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; padding: 0.6rem 1.4rem;
}
.stButton > button:hover { opacity: 0.9; }
.ingredient-card {
    background: #1e1e2e; border: 1px solid #333;
    border-radius: 8px; padding: 8px 12px; margin: 4px 0;
    display: flex; justify-content: space-between; align-items: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;color:#e63946;'>🛒 AI Market Asistanı</h1>"
    "<p style='text-align:center;color:#6c757d;'>Konumunuza göre — En Ucuz Market Fiyatları</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Session State Başlangıcı
# ──────────────────────────────────────────────────────────────────────────────

if "phase" not in st.session_state:
    st.session_state.phase = "chat"          # "chat" | "market"
if "messages" not in st.session_state:
    st.session_state.messages = []           # sohbet geçmişi
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []        # onaylı malzeme listesi
if "recipe_name" not in st.session_state:
    st.session_state.recipe_name = ""
if "servings" not in st.session_state:
    st.session_state.servings = 4

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Konum & Ayarlar
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
    if used_fallback and selected_province != "Konya":
        st.warning("Konum çözümlenemedi — Konya merkezi kullanılıyor.")

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

    # Sıfırlama butonu
    if st.button("🔄 Yeni Tarif Başlat"):
        st.session_state.phase = "chat"
        st.session_state.messages = []
        st.session_state.ingredients = []
        st.session_state.recipe_name = ""
        st.rerun()

    st.caption("İpucu: Arama mesafesini artırırsanız daha fazla market sonucu çıkar.")

    settings = get_settings()
    if not settings.is_llm_configured:
        st.warning("Azure anahtarları eksik. `.env` dosyasına ekleyin.")

# ──────────────────────────────────────────────────────────────────────────────
# AŞAMA 1: SOHBET
# ──────────────────────────────────────────────────────────────────────────────

CHEF_SYSTEM_PROMPT = """\
Sen deneyimli bir Türk şefi ve market alışverişi uzmanısın.
Kullanıcı bir yemek yapmak istiyor. Görevin:

1. Kullanıcının tarifini anla, kaç kişilik olduğunu sor (söylemediyse).
2. O tarif için gerçekçi malzeme listesi öner — porsiyona göre doğru miktarlarda.
3. Kullanıcının düzeltmelerini dinle ve listeyi güncelle.
4. Kullanıcı onayladığında SADECE şu JSON'u ver, başka hiçbir şey yazma:

{"recipe_name": "...", "servings": N, "ingredients": ["Malzeme Miktar Birim", ...]}

FORMAT KURALLARI (ÇOK ÖNEMLİ):
- Miktar SONDA: "Toz Şeker 500 G" ✓ — "500g Toz Şeker" ✗
- ASLA "X Adet" formatı kullanma → "1 Paket" veya "1 Kutu" de
- Kedi Dili Bisküvi → "Kedi Dili Bisküvi 1 Paket"
- Kahve: granül/öğütülmüş gram olarak → "Nescafe Granül Kahve 100 G" veya kullanıcı "2 kaşık yeter" diyorsa "Nescafe Granül Kahve 50 G"
- Kahve kapsül/pad YASAK
- Kakao: "Kakao Tozu 50 G" (Nesquik değil)

PORSIYON REHBERİ:
- 2 kişilik: Yumurta 6 Li, Toz Şeker 250 G, Un 250 G, Süt 500 Ml, Tereyağı 100 G
- 4 kişilik: Yumurta 6 Li, Toz Şeker 500 G, Un 500 G, Süt 1 Lt, Tereyağı 100 G
- Kullanıcı özel miktar söylüyorsa (örn: "nescafe 2 adet yeter") → o miktarı kullan

Türkçe konuş. Sıcak ve yardımsever ol. Onay gelmeden JSON verme. Onay gelince YALNIZCA JSON ver.
"""


async def chat_with_chef(messages: list) -> str:
    """LLM şef ile sohbet — streaming yanıt."""
    s = get_settings()
    client = AsyncAzureOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_key=s.azure_api_key,
        api_version=s.azure_api_ver,
    )
    response = await client.chat.completions.create(
        model=s.gpt_deployment,
        messages=[{"role": "system", "content": CHEF_SYSTEM_PROMPT}] + messages,
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content or ""


def extract_json_from_reply(text: str) -> dict | None:
    """LLM yanıtından JSON bloğunu çıkar."""
    import re
    # ```json ... ``` bloğu
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Düz { ... } bloğu
    m = re.search(r"\{[^{}]*\"ingredients\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


if st.session_state.phase == "chat":

    col_chat, col_list = st.columns([2, 1])

    with col_chat:
        st.subheader("👨‍🍳 Şef ile Tarif Planla")

        # Sohbet geçmişini göster
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👨‍🍳" if msg["role"] == "assistant" else "🧑"):
                st.markdown(msg["content"])

        # Kullanıcı girişi
        user_input = st.chat_input("Ne pişirmek istiyorsunuz? (örn: 2 kişilik tiramisu)")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant", avatar="👨‍🍳"):
                with st.spinner("Şef düşünüyor…"):
                    reply = asyncio.run(chat_with_chef(st.session_state.messages))

                # JSON içeriyorsa kullanıcıya gösterme — sadece onay mesajı
                parsed = extract_json_from_reply(reply)
                if parsed and "ingredients" in parsed:
                    st.session_state.ingredients = parsed["ingredients"]
                    st.session_state.recipe_name = parsed.get("recipe_name", "Tarif")
                    st.session_state.servings = parsed.get("servings", 4)
                    display_reply = (
                        f"✅ **{st.session_state.recipe_name}** için alışveriş listesi hazırlandı! "
                        f"Sağda listeyi inceleyebilir, düzenleyebilirsiniz. "
                        f"Hazırsan **🛒 Fiyatları Getir!** butonuna tıkla."
                    )
                else:
                    display_reply = reply

                st.markdown(display_reply)

            # Sohbet geçmişine orijinal yanıtı sakla (LLM context için)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    with col_list:
        st.subheader("📋 Malzeme Listesi")

        if not st.session_state.ingredients:
            st.info("Şefle konuşarak tarifinizi planlayın.\nListeyi onayladığınızda fiyatlar getirilir.")
        else:
            st.success(f"**{st.session_state.recipe_name}** — {st.session_state.servings} kişilik")

            # Düzenlenebilir liste
            edited = []
            for i, item in enumerate(st.session_state.ingredients):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    new_val = st.text_input(f"#{i+1}", value=item, key=f"ing_{i}", label_visibility="collapsed")
                    edited.append(new_val)
                with col_b:
                    if st.button("✕", key=f"del_{i}"):
                        st.session_state.ingredients.pop(i)
                        st.rerun()

            st.session_state.ingredients = [e for e in edited if e.strip()]

            # Yeni malzeme ekle
            new_item = st.text_input("➕ Malzeme ekle", key="new_ing", placeholder="örn: Vanilin 1 Paket")
            if new_item.strip():
                st.session_state.ingredients.append(new_item.strip())
                st.rerun()

            st.divider()

            if st.button("🛒 Fiyatları Getir!", type="primary", use_container_width=True):
                st.session_state.phase = "market"
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# AŞAMA 2: MARKET FİYATLARI
# ──────────────────────────────────────────────────────────────────────────────

elif st.session_state.phase == "market":

    st.subheader(f"🛒 {st.session_state.recipe_name} — Market Fiyatları")

    if st.button("← Malzeme Listesine Dön"):
        st.session_state.phase = "chat"
        st.rerun()

    # Onaylanan malzeme listesini göster
    with st.expander("📋 Alışveriş Listesi", expanded=False):
        for item in st.session_state.ingredients:
            st.markdown(f"- {item}")

    # Malzeme listesinden recipe_request oluştur
    ingredients_str = ", ".join(st.session_state.ingredients)
    recipe_request = (
        f"{st.session_state.recipe_name} ({st.session_state.servings} kişilik). "
        f"Gerekli malzemeler: {ingredients_str}"
    )

    async def log_cb(msg: str) -> None:
        pass  # Arka planda çalışır, kullanıcıya gösterilmez

    orchestrator = Orchestrator()

    with st.spinner("Market fiyatları aranıyor…"):
        try:
            df, recipe_name, warnings, cheapest_market = asyncio.run(
                orchestrator.run(recipe_request, on_hand, selected_district, loc, log_cb)
            )
        except Exception as exc:
            st.error(f"❌ Bir hata oluştu: {exc}")
            st.stop()

    if df.empty:
        st.info("Elinizdeki malzemeler yeterli — ekstra alışveriş gerekmez!")
    else:
        st.success(f"✅ **{recipe_name}** tarifi için alışveriş listesi hazır!")

        # Uyarılar
        if warnings:
            critical = [w for w in warnings if w.startswith("🚨") or w.startswith("❌")]
            informational = [w for w in warnings if w not in critical]
            if critical:
                with st.expander("🚨 Porsiyon / Uyumsuzluk Uyarıları", expanded=True):
                    for w in critical:
                        st.error(w)
            if informational:
                with st.expander("⚠️ Diğer Uyarılar"):
                    for w in informational:
                        st.warning(w)

        # Fiyat tablosu
        st.subheader("📊 Fiyat Tablosu")
        display_cols = [c for c in ["Arama", "Product", "Market", "Neighborhood", "Price"] if c in df.columns]
        st.dataframe(
            df[display_cols].style.format({"Price": "{:.2f} ₺"}),
            width=None,
            height=min(420, 60 + len(df) * 40),
        )

        total = df["Price"].sum()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("🛒 Toplam Tutar", f"{total:.2f} ₺")
        col_b.metric("📦 Bulunan Ürün", f"{len(df)} kalem")
        col_c.metric("🏪 Farklı Market", f"{df['Market'].nunique()}")
        col_d.metric("🏆 En Avantajlı", cheapest_market or "—")

        if cheapest_market and df["Market"].nunique() > 1:
            market_summary = (
                df.groupby("Market")["Price"]
                .sum().sort_values().reset_index()
                .rename(columns={"Price": "Toplam (₺)"})
            )
            with st.expander("🏪 Markete Göre Toplam Maliyet"):
                st.dataframe(market_summary.style.format({"Toplam (₺)": "{:.2f}"}), hide_index=True)

        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ CSV İndir", data=csv, file_name="alisveris_listesi.csv", mime="text/csv")