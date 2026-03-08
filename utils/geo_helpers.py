"""
Coğrafi yardımcı fonksiyonlar.

Sorumluluklar:
  - TurkiyeAPI üzerinden il / ilçe listesi çekmek
  - Nominatim ile (il, ilçe) → (lat, lon) dönüşümü (forward geocoding)
  - Nominatim ile (lat, lon) → okunabilir etiket dönüşümü (reverse geocoding)

Bu modül saf yardımcı fonksiyonlar içerir; iş mantığı veya UI kodu **yoktur**.
"""

from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

from core.config import get_settings


# ──────────────────────────────────────────────────────────────────────────────
# İç yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

def _extract_list(data: Any) -> List[Dict[str, Any]]:
    """TurkiyeAPI farklı şemalar döndürebilir; bunu normalize eder."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "result", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


# ──────────────────────────────────────────────────────────────────────────────
# İl / İlçe listeleri
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 60 * 24 * 7)  # 7 gün
def get_provinces() -> List[str]:
    """
    TurkiyeAPI'den güncel il (province) listesi döner.
    API erişilemezse kısa bir fallback liste kullanılır.
    """
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.turkiye_api_base}/provinces", timeout=15)
        r.raise_for_status()
        items = _extract_list(r.json())
        names = [
            (it.get("name") or it.get("province") or it.get("il") or "").strip()
            for it in items
        ]
        names = [n for n in names if n]
        return sorted(set(names), key=lambda x: x.casefold())
    except Exception:
        return ["Konya", "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]


@st.cache_data(ttl=60 * 60 * 24 * 7)  # 7 gün
def get_districts(province_name: str) -> List[str]:
    """
    Verilen ile ait ilçe listesini TurkiyeAPI'den döner.
    Hata durumunda, yalnızca Konya için dahili fallback uygulanır.
    """
    settings = get_settings()
    try:
        r = httpx.get(
            f"{settings.turkiye_api_base}/districts",
            params={"province": province_name},
            timeout=15,
        )
        r.raise_for_status()
        items = _extract_list(r.json())
        names = [
            (it.get("name") or it.get("district") or it.get("ilce") or "").strip()
            for it in items
        ]
        names = [n for n in names if n]
        return sorted(set(names), key=lambda x: x.casefold())
    except Exception:
        if province_name == "Konya":
            return ["Selçuklu", "Meram", "Karatay"]
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Forward geocoding (il + ilçe → koordinat)
# ──────────────────────────────────────────────────────────────────────────────

def geocode_city_district(province: str, district: str = "") -> Optional[Tuple[float, float]]:
    """
    İl (+ isteğe bağlı ilçe) adını (lat, lon) çiftine çevirir.

    Döner: (lat, lon) ya da None (çözümleme başarısız olursa).
    Hata durumunda istisna fırlatmaz; None döner — çağıran fallback uygular.
    """
    query = f"{district}, {province}, Türkiye" if district else f"{province}, Türkiye"
    settings = get_settings()
    try:
        r = httpx.get(
            settings.nominatim_search,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "marketfiyat-streamlit/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json()
        if isinstance(results, list) and results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Reverse geocoding (koordinat → okunabilir etiket)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60 * 60 * 24 * 14)  # 14 gün
def reverse_geocode_label(lat: float, lon: float) -> Optional[str]:
    """
    (lat, lon) koordinatından 'İlçe, İl' formatında okunabilir bir etiket üretir.
    Döner: 'Selçuklu, Konya' gibi bir string ya da None.
    """
    settings = get_settings()
    try:
        r = httpx.get(
            settings.nominatim_reverse,
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
            return city

        display = data.get("display_name", "")
        if isinstance(display, str) and display.strip():
            parts = [p.strip() for p in display.split(",") if p.strip()]
            return ", ".join(parts[:2]) if parts else None
    except Exception:
        pass
    return None


def resolve_coordinates(
    province: str,
    district: str = "",
    fallback: Tuple[float, float] = (37.8710, 32.4846),
) -> Tuple[Tuple[float, float], bool]:
    """
    (il, ilçe) → (lat, lon) çözümler; başarısız olursa fallback döner.

    Döner: ((lat, lon), used_fallback: bool)
    """
    coords = geocode_city_district(province, district)
    if coords is None and district:
        coords = geocode_city_district(province)  # ilçe bulunamazsa il merkezi dene
    if coords is None:
        return fallback, True
    return coords, False
