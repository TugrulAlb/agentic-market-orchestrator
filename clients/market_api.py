"""
Market API istemcisi — httpx + Semantic (Fuzzy) Search.

Sorumluluklar:
  - marketfiyati.org.tr API'sine POST isteği atmak
  - Gelen ürün listesini "arama terimiyle en alakalı ve en ucuz" kurala göre sıralamak
  - Semantic search: rapidfuzz ile arama terimi ↔ ürün adı benzerlik skoru hesaplamak
    (örnek: "süt" → "Yarım Yağlı Süt 1L" doğru eşleşmesini sağlar)
  - Yeniden deneme (retry) + Semaphore ile eşzamanlı istek kontrolü

Bu modülde iş mantığı veya UI kodu **yoktur**.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx
from rapidfuzz import fuzz

from core.config import Settings, get_settings


# ──────────────────────────────────────────────────────────────────────────────
# Veri modeli
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    distance_km: float = 5.0
    size: int = 20


# ──────────────────────────────────────────────────────────────────────────────
# İç yardımcılar
# ──────────────────────────────────────────────────────────────────────────────

_USER_AGENTS: List[str] = [
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
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://marketfiyati.org.tr",
        "Referer": "https://marketfiyati.org.tr/",
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def _extract_products(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """API farklı şema anahtarları döndürebilir; normalize eder."""
    for key in ("content", "products", "data", "results"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


def _min_depot_price(product: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Ürünün deposundaki en düşük fiyatı döner."""
    depots = product.get("productDepotInfoList") or product.get("depots") or []
    best_price: Optional[float] = None
    best_depot: Optional[Dict] = None
    for depot in depots:
        price = _safe_float(
            depot.get("price") or depot.get("fiyat")
            or depot.get("minPrice") or depot.get("min_price")
        )
        if price is not None and (best_price is None or price < best_price):
            best_price = price
            best_depot = depot
    return best_price, best_depot


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Search — Fuzzy matching
# ──────────────────────────────────────────────────────────────────────────────

def _relevance_score(query: str, product_name: str) -> float:
    """
    Arama terimi ile ürün adı arasındaki anlam benzerliğini 0–100 ölçeğinde döner.
    token_set_ratio ağırlığı artırıldı — kısa query, uzun ürün adında gömülü olduğunda
    daha iyi eşleşme sağlar (örn: "süt" → "Dost %3.1 Yağlı Süt 1 Lt").
    """
    q = query.casefold().strip()
    p = product_name.casefold().strip()

    query_tokens = [t for t in q.split() if t]
    product_tokens = [t for t in p.split() if t]

    ratio_score = fuzz.ratio(q, p)
    token_score = fuzz.token_set_ratio(q, p)
    partial_score = fuzz.partial_ratio(q, p)

    # token_set_ratio'ya daha yüksek ağırlık: API kısa keyword döndürüyor,
    # ürün adı daha uzun olabiliyor
    base_score = ratio_score * 0.3 + token_score * 0.5 + partial_score * 0.2

    if query_tokens and product_tokens:
        focus = min(len(query_tokens), len(product_tokens)) / len(product_tokens)
        focus_factor = 0.6 + (0.4 * focus)
        return base_score * focus_factor

    return base_score


def _api_keyword(item: str) -> str:
    """
    API'ye gönderilecek kısa arama terimi üretir.
    Uzun terimler API'de kötü eşleşiyor; core keyword'ü çıkar.

    Örnekler:
        "1L Tam Yağlı Süt"    → "tam yağlı süt"
        "1kg Toz Şeker"       → "toz şeker"
        "10'lu Tavuk Yumurtası" → "yumurta"
        "Kakao Tozu"          → "kakao"
        "Kedi Dili Bisküvi"   → "kedi dili bisküvi"
    """
    import re as _re
    s = item.strip()
    # Baştaki miktar/birim prefix'ini sil: "1L", "1kg", "500g", "250ml" vb.
    s = _re.sub(r"^\s*\d+\s*(?:kg|g|ml|lt?|litre|kilo(?:gram)?)\s*", "", s, flags=_re.IGNORECASE)
    # Sondaki miktar/birim suffix'ini sil: "1 Lt", "1 Kg", "250 G" vb.
    s = _re.sub(r"\s+\d+\s*(?:kg|g|ml|lt?|litre|kilo(?:gram)?)\s*$", "", s, flags=_re.IGNORECASE)
    # Tırnak ve kesme işaretlerini temizle
    s = s.replace("'", "").replace('"', "").strip()
    return s.lower() if s else item.lower()


def _best_product(query: str, products: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ürün listesinden en alakalı ve ucuz olanı seçer.
    - 1L süt istenirken 200ml ürünler filtrelenir
    - Kakao aramasında içecek/Nesquik ürünleri penalize edilir
    """
    import re as _re

    want_1l = bool(_re.search(r"\b1\s*[Ll](?:[Tt]|itre)?\b|1000\s*ml", query, _re.IGNORECASE))
    is_kakao = "kakao" in query.casefold()

    candidates: List[Tuple[float, float, Dict, Optional[Dict]]] = []

    for product in products:
        price, depot = _min_depot_price(product)
        if price is None:
            price = _safe_float(
                product.get("price") or product.get("fiyat")
                or product.get("minPrice") or product.get("min_price")
            )
            depot = None
        if price is None:
            continue

        name = (
            product.get("title") or product.get("name")
            or product.get("urun") or ""
        )

        # ── Filtre: 1L süt istenirken ≤200ml ürünleri tamamen ele ────────────
        if want_1l and _re.search(r"\b[1-9]\d?\s*ml\b|\b1[0-9]{2}\s*ml\b|\b200\s*ml\b", name, _re.IGNORECASE):
            # 200ml, 180ml gibi küçük boy → atla
            if not _re.search(r"\b(750|800|900|1000|1[.,]?\s*[Ll]t?)\b", name, _re.IGNORECASE):
                continue

        relevance = _relevance_score(query, name)

        # ── Penalize: kakao içeceği / Nesquik ───────────────────────────────
        if is_kakao and _re.search(r"\b(nesquik|kids|çocuk|içecek)\b", name, _re.IGNORECASE):
            relevance *= 0.3  # büyük ceza

        candidates.append((relevance, price, product, depot))

    if not candidates:
        return None

    # Fiyatları normalize et (düşük fiyat → yüksek skor)
    prices = [c[1] for c in candidates]
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p or 1.0

    scored = []
    for relevance, price, product, depot in candidates:
        # Fiyat skoru 0-1 aralığında normalize edilir:
        # en ucuz = 1.0, en pahalı = 0.0
        cheapness = 1.0 - (price - min_p) / price_range
        combined = (relevance * 0.5) + (cheapness * 50)
        scored.append((combined, price, relevance, cheapness, product, depot))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_final, best_price, best_relevance, best_cheapness, best_product, best_depot = scored[0]

    name = (
        best_product.get("title") or best_product.get("name")
        or best_product.get("urun") or query
    )
    market = (
        (best_depot or {}).get("marketAdi") or (best_depot or {}).get("market")
        or best_product.get("market") or best_product.get("marketName")
        or best_product.get("chain") or "—"
    )
    neighborhood = (
        (best_depot or {}).get("depotAdi") or (best_depot or {}).get("neighborhood")
        or (best_depot or {}).get("district") or best_product.get("neighborhood")
        or best_product.get("district") or best_product.get("ilce") or "—"
    )

    return {
        "Arama": query,
        "Product": name,
        "Market": market,
        "Price": float(best_price),
        "Neighborhood": neighborhood,
        "RelevanceScore": round(float(best_relevance), 2),
        "PriceScore": round(float(best_cheapness), 3),
        "FinalScore": round(float(best_final), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Market API İstemcisi
# ──────────────────────────────────────────────────────────────────────────────

class MarketAPIClient:
    """
    marketfiyati.org.tr API'si için async istemci.

    Tek örnek oluştur, birden fazla fetch_price çağrısında paylaş.
    httpx.AsyncClient ömrü dışarıdan yönetilir (Orchestrator bağlamında).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    async def fetch_price(
        self,
        item: str,
        loc: Location,
        district: str = "",
        log_cb: Optional[Callable[[str], Coroutine]] = None,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Belirtilen malzeme için market API'sinden en iyi (alakalı + ucuz) ürünü bulur.

        Args:
            item        : Arama terimi (ör. "domates", "kuzu but")
            loc         : Konum ve yarıçap bilgisi
            district    : Sonuç satırına eklenmek üzere ilçe adı (isteğe bağlı)
            log_cb      : Asenkron log callback'i (UI veya terminal)
            http_client : Dışarıdan sağlanan httpx.AsyncClient (paylaşım için)
            semaphore   : Eşzamanlı istek sayısını sınırlayan semaphore

        Returns:
            Fiyat ve ürün bilgisi içeren dict, ya da None (bulunamazsa).

        Raises:
            MarketAPIConnectionError : Ağ hatası tüm yeniden denemeleri tüketirse
            MarketAPIParseError      : Yanıt JSON geçersizse
        """
        settings = self._settings
        api_kw = _api_keyword(item)   # API'ye kısa terim gönder
        payload = {
            "keywords": api_kw,       # "tam yağlı süt", "toz şeker" vb.
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "distance": loc.distance_km,
            "size": loc.size,
        }

        if log_cb:
            await log_cb(f"📡 **{item}** → API keyword: `{api_kw}`")

        own_client = http_client is None
        if own_client:
            http_client = httpx.AsyncClient(follow_redirects=True)

        own_semaphore = semaphore is None
        if own_semaphore:
            semaphore = asyncio.Semaphore(1)

        last_exc: Optional[Exception] = None

        try:
            for attempt in range(1, settings.max_retries + 1):
                async with semaphore:
                    try:
                        response = await http_client.post(
                            settings.api_endpoint,
                            json=payload,
                            headers=_api_headers(),
                            timeout=settings.timeout_s,
                        )
                        response.raise_for_status()
                        data = response.json()

                        products = _extract_products(data)
                        if not products:
                            if log_cb:
                                await log_cb(f"⚠️  **{item}** için ürün bulunamadı.")
                            return None

                        result = _best_product(item, products)
                        if result is None:
                            if log_cb:
                                await log_cb(f"⚠️  **{item}**: Fiyat alanı bulunamadı.")
                            return None

                        if log_cb:
                            await log_cb(
                                f"✅ **{item}** → {result['Market']} | "
                                f"{result['Product']} | "
                                f"{result['Neighborhood']} | "
                                f"**{result['Price']:.2f} ₺**"
                            )
                        return result

                    except httpx.HTTPStatusError as exc:
                        last_exc = exc
                        if log_cb:
                            await log_cb(
                                f"⚠️  HTTP {exc.response.status_code} — "
                                f"{item} (deneme {attempt}/{settings.max_retries})"
                            )
                    except httpx.RequestError as exc:
                        last_exc = exc
                        if log_cb:
                            await log_cb(
                                f"🔌 Bağlantı hatası — {item} (deneme {attempt}): `{exc}`"
                            )
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"'{item}' için API yanıtı ayrıştırılamadı: {exc}"
                        ) from exc

                if attempt < settings.max_retries:
                    await asyncio.sleep(1.5**attempt)

        finally:
            if own_client:
                await http_client.aclose()

        if last_exc is not None:
            raise ConnectionError(
                f"'{item}' için market API'ye ulaşılamadı: {last_exc}"
            ) from last_exc

        if log_cb:
            await log_cb(f"🚫 **{item}** için fiyat alınamadı (tüm denemeler tükendi).")
        return None