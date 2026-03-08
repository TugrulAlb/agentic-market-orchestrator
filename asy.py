"""
Multi-Agent Smart Market Assistant — marketfiyati.org.tr
=========================================================
Agent 1 : Azure OpenAI GPT-4o  → recipe / ingredient extractor
Agent 2 : httpx AsyncClient    → Direct API price fetcher (Konya/Selçuklu)

Run:
    pip install streamlit openai httpx pandas
    streamlit run asy.py
"""

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st

try:
    from streamlit_js_eval import get_geolocation, bootstrapButton, streamlit_js_eval
except Exception:
    get_geolocation = None
    bootstrapButton = None
    streamlit_js_eval = None
from streamlit.errors import StreamlitSecretNotFoundError
from openai import AsyncAzureOpenAI

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (move secrets to .streamlit/secrets.toml in production)
# ──────────────────────────────────────────────────────────────────────────────

def _get_secret(key: str, default: str = "") -> str:
    """Return a secret value without crashing if secrets.toml is missing.

    Priority:
    1) Environment variables
    2) Streamlit secrets (only if a secrets.toml exists)
    3) Provided default
    """
    # 1) Env var first (fast + safe)
    env_val = os.getenv(key)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val)

    # 2) Only touch st.secrets if a secrets.toml exists
    secrets_paths = [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    if any(os.path.exists(p) for p in secrets_paths):
        try:
            return str(st.secrets[key])
        except Exception:
            return str(default)

    # 3) Fallback default
    return str(default)


AZURE_ENDPOINT = _get_secret("AZURE_ENDPOINT", "")
AZURE_API_KEY  = _get_secret("AZURE_API_KEY", "")
AZURE_API_VER  = _get_secret("AZURE_API_VER", "2024-02-15-preview")
GPT_DEPLOYMENT = _get_secret("GPT_DEPLOYMENT", "gpt-4o")

API_ENDPOINT   = "https://api.marketfiyati.org.tr/api/v2/search"
API_LIMIT      = 20
MAX_RETRIES    = 3
TIMEOUT_S      = 10.0

# External geo-data sources (for up-to-date Turkey provinces/districts)
TURKIYE_API_BASE = "https://api.turkiyeapi.dev/v1"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"

# ──────────────────────────────────────────────────────────────────────────────
# Debug (terminal only) — set to True temporarily when diagnosing issues
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_GEO = True

def _tlog(*args):
    """Terminal-only log. Does not render in Streamlit UI."""
    if DEBUG_GEO:
        try:
            print("[GEO]", *args, flush=True)
        except Exception:
            pass


def _extract_list(data: Any) -> List[Dict[str, Any]]:
    """TurkiyeAPI usually returns {data:[...]} but we keep it defensive."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("result"), list):
            return data["result"]
    return []


@st.cache_data(ttl=60 * 60 * 24 * 7)  # 7 days
def get_turkiye_provinces() -> List[str]:
    """Get up-to-date province (il) names."""
    try:
        r = httpx.get(f"{TURKIYE_API_BASE}/provinces", timeout=15)
        r.raise_for_status()
        items = _extract_list(r.json())
        names = [
            (it.get("name") or it.get("province") or it.get("il") or "").strip()
            for it in items
        ]
        names = [n for n in names if n]
        # Some APIs return duplicates or non-titlecase
        return sorted(set(names), key=lambda x: x.casefold())
    except Exception:
        # Fallback minimal list (only if the external API is unavailable)
        return ["Konya", "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]


@st.cache_data(ttl=60 * 60 * 24 * 7)  # 7 days
def get_turkiye_districts(province_name: str) -> List[str]:
    """Get districts (ilçe) for a given province."""
    try:
        params = {"province": province_name}
        r = httpx.get(f"{TURKIYE_API_BASE}/districts", params=params, timeout=15)
        r.raise_for_status()
        items = _extract_list(r.json())
        names = [
            (it.get("name") or it.get("district") or it.get("ilce") or "").strip()
            for it in items
        ]
        names = [n for n in names if n]
        return sorted(set(names), key=lambda x: x.casefold())
    except Exception:
        # Fallback for Konya only
        if province_name == "Konya":
            return ["Selçuklu", "Meram", "Karatay"]
        return []


def geocode_city_district(province: str, district: str = "") -> Optional[Tuple[float, float]]:
    """Convert (il, ilçe) into (lat, lon) using Nominatim. Also supports free-text query."""
    if district:
        query = f"{district}, {province}, Türkiye"
    else:
        query = f"{province}, Türkiye"
    try:
        r = httpx.get(
            NOMINATIM_SEARCH,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "marketfiyat-streamlit/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        arr = r.json()
        if isinstance(arr, list) and arr:
            lat = float(arr[0]["lat"])
            lon = float(arr[0]["lon"])
            return (lat, lon)
    except Exception:
        return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# REVERSE GEOCODING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60 * 60 * 24 * 14)  # 14 days
def reverse_geocode_label(lat: float, lon: float) -> Optional[str]:
    """Return a readable location label like 'Selçuklu, Konya' from lat/lon."""
    try:
        r = httpx.get(
            NOMINATIM_REVERSE,
            params={
                "format": "json",
                "lat": str(lat),
                "lon": str(lon),
                "zoom": 12,
                "addressdetails": 1,
            },
            headers={"User-Agent": "marketfiyat-streamlit/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        address = data.get("address") or {}

        # District-like fields vary; try common keys
        district = (
            address.get("suburb")
            or address.get("city_district")
            or address.get("district")
            or address.get("town")
            or address.get("neighbourhood")
        )
        city = address.get("city") or address.get("province") or address.get("state")

        if district and city:
            return f"{district}, {city}"
        if city:
            return str(city)

        # Fallback to display_name if present
        disp = data.get("display_name")
        if isinstance(disp, str) and disp.strip():
            parts = [p.strip() for p in disp.split(",") if p.strip()]
            return ", ".join(parts[:2]) if parts else None
        return None
    except Exception:
        return None
#
# ──────────────────────────────────────────────────────────────────────────────
# GEOLOCATION (browser permission)
# ──────────────────────────────────────────────────────────────────────────────



USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _api_headers() -> Dict[str, str]:
    return {
        "Content-Type":   "application/json",
        "Accept":         "application/json, text/plain, */*",
        "Origin":         "https://marketfiyati.org.tr",
        "Referer":        "https://marketfiyati.org.tr/",
        "User-Agent":     random.choice(USER_AGENTS),
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":     "keep-alive",
    }


# ──────────────────────────────────────────────────────────────────────────────
# MARKET API HELPERS (based on market-mcp-serkan request/response shapes)
# ──────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    distance_km: float = 5.0
    size: int = 20


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def _extract_products(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # API may return {content:[...]} or {products:[...]} or {data:[...]} or {results:[...]}
    if isinstance(data.get("content"), list):
        return data["content"]
    if isinstance(data.get("products"), list):
        return data["products"]
    if isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data.get("results"), list):
        return data["results"]
    return []


def _min_depot_price(product: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    depots = product.get("productDepotInfoList") or product.get("depots") or []
    best = None
    best_price = None
    for d in depots:
        p = _safe_float(d.get("price") or d.get("fiyat") or d.get("minPrice") or d.get("min_price"))
        if p is None:
            continue
        if best_price is None or p < best_price:
            best_price = p
            best = d
    return best_price, best


# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Azure OpenAI ingredient extractor
# ──────────────────────────────────────────────────────────────────────────────

async def extract_ingredients(recipe_request: str, on_hand: str) -> dict:
    """
    Calls Azure OpenAI to determine the recipe name and the *missing* ingredients.
    Returns: {"recipe": str, "ingredients": [str, ...]}
    """
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise RuntimeError("Azure OpenAI bilgileri eksik. .streamlit/secrets.toml içine AZURE_ENDPOINT ve AZURE_API_KEY ekleyin.")

    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VER,
    )

    system_prompt = (
        "Sen deneyimli bir Türk şefsin ve akıllı bir alışveriş asistanısın. "
        "Kullanıcının istediği yemeği hazırlamak için gereken malzemeleri listele. "
        "Sonra kullanıcının zaten elinde bulunanları çıkar ve SADECE eksik olanları döndür. "
        "Yanıtın MUTLAKA şu JSON şemasına uygun olsun:\n"
        '{"recipe": "<yemek adı>", "ingredients": ["malzeme1", "malzeme2", ...]}\n'
        "Başka hiçbir şey yazma."
    )
    user_prompt = (
        f"İstek: {recipe_request}\n"
        f"Eldeki malzemeler: {on_hand}\n"
        "Eksik malzemeleri JSON olarak ver."
    )

    response = await client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — httpx Direct API price fetcher
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_api_price(
    item: str,
    district: str,
    loc: Location,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    log_cb=None,
) -> Optional[Dict]:
    """POST to the internal search API, return the cheapest hit near the given location."""

    payload = {
        "keywords": item,
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "distance": loc.distance_km,
        "size": loc.size,
    }

    if log_cb:
        await log_cb(f"📡 **{item}** API'den çekiliyor…")

    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                response = await client.post(
                    API_ENDPOINT,
                    json=payload,
                    headers=_api_headers(),
                    timeout=TIMEOUT_S,
                )
                response.raise_for_status()
                data = response.json()

                products = _extract_products(data)
                if not products:
                    if log_cb:
                        await log_cb(f"⚠️  **{item}** için ürün bulunamadı.")
                    return None

                best_row = None
                best_price = None

                for p in products:
                    price, best_depot = _min_depot_price(p)
                    # Eğer depots yoksa eski şema fallback: doğrudan price alanı
                    if price is None:
                        price = _safe_float(p.get("price") or p.get("fiyat") or p.get("minPrice") or p.get("min_price"))
                        best_depot = None

                    if price is None:
                        continue

                    if best_price is None or price < best_price:
                        best_price = price
                        best_row = {
                            "Arama": item,
                            "Product": p.get("title") or p.get("name") or p.get("urun") or item,
                            "Market": (
                                (best_depot or {}).get("marketAdi")
                                or (best_depot or {}).get("market")
                                or p.get("market")
                                or p.get("marketName")
                                or p.get("chain")
                                or "—"
                            ),
                            "Price": float(best_price),
                            "Neighborhood": (
                                (best_depot or {}).get("depotAdi")
                                or (best_depot or {}).get("neighborhood")
                                or (best_depot or {}).get("district")
                                or p.get("neighborhood")
                                or p.get("district")
                                or p.get("ilce")
                                or district
                            ),
                        }

                if not best_row:
                    if log_cb:
                        await log_cb(f"⚠️  **{item}**: Fiyat alanı bulunamadı.")
                    return None

                if log_cb:
                    await log_cb(
                        f"✅ **{item}** → {best_row['Market']} | "
                        f"{best_row['Product']} | "
                        f"{best_row['Neighborhood']} | "
                        f"**{best_row['Price']:.2f} ₺**"
                    )

                return best_row

            except httpx.HTTPStatusError as exc:
                if log_cb:
                    await log_cb(
                        f"⚠️  HTTP {exc.response.status_code} — "
                        f"{item} (deneme {attempt}/{MAX_RETRIES})"
                    )
            except httpx.RequestError as exc:
                if log_cb:
                    await log_cb(f"🔌 Bağlantı hatası — {item} (deneme {attempt}): `{exc}`")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                if log_cb:
                    await log_cb(f"🔧 Parse hatası — {item}: `{exc}`")
                return None

            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5 ** attempt)

    if log_cb:
        await log_cb(f"🚫 **{item}** için fiyat alınamadı (tüm denemeler tükendi).")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    recipe_request: str,
    on_hand: str,
    district: str,
    loc: Location,
    log_cb,
) -> Optional[Tuple[pd.DataFrame, str]]:

    # ── Step 1: extract missing ingredients via LLM ──────────────────────────
    await log_cb("🧠 **Agent 1** — Tarif analiz ediliyor…")
    try:
        parsed      = await extract_ingredients(recipe_request, on_hand)
        recipe_name = parsed.get("recipe", "Bilinmiyor")
        ingredients = parsed.get("ingredients", [])
    except Exception as exc:
        await log_cb(f"❌ GPT-4o hatası: `{exc}`")
        return None

    if not ingredients:
        await log_cb("ℹ️  Eksik malzeme bulunamadı — yeterli stok var!")
        return pd.DataFrame(), "—"

    await log_cb(
        f"📋 **{recipe_name}** için eksik malzemeler: "
        f"{', '.join(f'`{i}`' for i in ingredients)}"
    )

    # ── Step 2: fetch prices concurrently via API ─────────────────────────────
    await log_cb(f"🚀 **Agent 2** — API üzerinden {district} bölgesinde fiyatlar çekiliyor…")
    semaphore = asyncio.Semaphore(5)   # up to 5 concurrent API calls

    # Reuse a single httpx client across all concurrent requests
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            fetch_api_price(item, district, loc, client, semaphore, log_cb)
            for item in ingredients
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    found = [r for r in results if r]

    if not found:
        await log_cb("😔 Hiçbir malzeme için fiyat bulunamadı.")
        return None

    df = pd.DataFrame(found)
    df = df.sort_values("Price").reset_index(drop=True)
    return df, recipe_name


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
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
        html, body, [class*="css"] { font-family: 'Space+Grotesk', sans-serif; }
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

    subtitle = "Konumunuza göre — En Ucuz Market Fiyatları"

    st.markdown(
        "<h1 style='text-align:center;color:#e63946;'>🛒 AI Market Asistanı</h1>"
        f"<p style='text-align:center;color:#6c757d;'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
    st.divider()


    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️  Ayarlar")

        st.subheader("📍 Konum")

        radius = st.slider("Arama Mesafesi (km)", 1, 15, 5)

        # ── İl / İlçe seçimi ─────────────────────────────────────────────────
        provinces = get_turkiye_provinces()
        default_province = "Konya"
        default_idx = provinces.index(default_province) if default_province in provinces else 0

        selected_province = st.selectbox("🏙️ İl", provinces, index=default_idx)

        districts = get_turkiye_districts(selected_province)
        if districts:
            selected_district = st.selectbox("🏘️ İlçe", districts)
        else:
            selected_district = ""

        # Koordinata çevir (Nominatim). Başarısız olursa il merkezi → en son Konya fallback.
        coords = geocode_city_district(selected_province, selected_district)
        if coords is None:
            coords = geocode_city_district(selected_province)  # ilçe bulunamazsa il merkezi

        used_fallback = False
        if coords is None:
            coords = (37.8710, 32.4846)  # son çare: Konya merkez
            used_fallback = True

        # Seçimi göster (kullanıcı neyin aktif olduğunu anlasın)
        if selected_district:
            st.markdown(f"**Seçim:** {selected_province} / {selected_district}")
        else:
            st.markdown(f"**Seçim:** {selected_province}")

        # Eğer Konya fallback'e düştüysek kullanıcıya kısa uyarı ver
        if used_fallback and selected_province != "Konya":
            st.warning("Konum çözümlenemedi — geçici olarak Konya merkezle arama yapılıyor.")

        dlat, dlon = coords
        loc = Location(latitude=float(dlat), longitude=float(dlon), distance_km=float(radius), size=API_LIMIT)
        district = selected_district

        on_hand = st.text_area(
            "Elinizdeki Malzemeler",
            value="",
            help="Virgülle ayırın",
        )

        st.divider()
        st.caption("İpuçları: Arama mesafesini artırırsanız daha fazla sonuç bulabilirsiniz.")
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            st.warning("Tarif analizi için Azure anahtarları eksik. `.streamlit/secrets.toml` ekleyin (AZURE_ENDPOINT / AZURE_API_KEY).")

    # ── Main input row ────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        recipe_request = st.text_input(
            "Ne pişirmek istiyorsunuz?",
            value="Konya usulü fırın kebabı yapmak istiyorum",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Fiyatları Getir", type="primary", width="stretch")

    # ── Log (UI'da gösterilmiyor) ─────────────────────────────────────────────
    async def log_cb(msg: str):
        return

    # ── Pipeline execution ────────────────────────────────────────────────────
    if run_btn:
        if not recipe_request.strip():
            st.warning("Lütfen bir yemek isteği girin.")
            return

        with st.spinner("Ajanlar çalışıyor…"):
            output = asyncio.run(
                run_pipeline(recipe_request, on_hand, district, loc, log_cb)
            )

        # ── Results ───────────────────────────────────────────────────────────
        if output is None:
            st.error("İşlem tamamlanamadı. Log çıktısını inceleyin.")

        elif isinstance(output, tuple):
            df, recipe_name = output

            if df.empty:
                st.info("Elinizdeki malzemeler yeterli — ekstra alışveriş gerekmez!")
            else:
                st.success(f"✅  **{recipe_name}** tarifi için alışveriş listesi hazır!")
                st.subheader("📊 Fiyat Tablosu")

                display_cols = ["Arama", "Product", "Market", "Neighborhood", "Price"]
                display_cols = [c for c in display_cols if c in df.columns]

                st.dataframe(
                    df[display_cols].style.format({"Price": "{:.2f} ₺"}),
                    width="stretch",
                    height=min(420, 60 + len(df) * 40),
                )

                total = df["Price"].sum()
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("🛒 Toplam Tutar",  f"{total:.2f} ₺")
                col_b.metric("📦 Bulunan Ürün",  f"{len(df)} kalem")
                col_c.metric("🏪 Farklı Market", f"{df['Market'].nunique()}")

                # ── Download button ───────────────────────────────────────────
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="⬇️  CSV İndir",
                    data=csv,
                    file_name="alisveris_listesi.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()