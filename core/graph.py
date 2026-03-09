"""
Market Intelligence Graph — State Machine (Durum Makinesi).

Mimari (döngülü):

  RECIPE_ANALYZER_NODE ─────────────────────────────────────────────────────┐
       (Şef / LLM)                                                           │
           │  ilk tur + içerik retry                                         │
           ▼                                                                  │
    [Paralel Market Araması]    ◄── qty_retry_map (LLM bypass) ─────────┐   │
           │                                                              │   │
           ▼                                                              │   │
  MARKET_SEARCH_CHECKER_NODE                                             │   │
       (Denetçi)                                                         │   │
           │                                                              │   │
     ┌─────┴──────────────────────────────┐                              │   │
     │                                    │                              │   │
  Sadece qty        İçerik/alaka         │                              │   │
  hatası            hatası               │                              │   │
     │                    │              │                              │   │
     ▼                    ▼              │                              │   │
 "market_search"   "recipe_analyzer" ──┘ ──────────────────────────────┘   │
  (LLM yok,         (LLM ile rafine)                                      │   │
  filtreli ara)          └──────────────────────────────────────────────┘   │
     │                                                                       │
     └── MAX_RETRIES aşıldıysa her iki yol da → FINAL_REPORT_NODE ──────────┘

Her düğüm MarketState sözlüğünü alır, günceller ve döndürür.
Düğümler arası tüm veri akışı yalnızca MarketState üzerinden gerçekleşir.

Bu modülde LLM veya HTTP çağrısı yoktur.
NODE 1 (RECIPE_ANALYZER_NODE) LLM çağrısı services/llm_service.py içindedir.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# MarketState — Pipeline'ın ortak hafızası
# ──────────────────────────────────────────────────────────────────────────────

class MarketState(TypedDict, total=False):
    # ── Girdi alanları ──────────────────────────────────────────────────────
    recipe_request: str           # Kullanıcının ham yemek isteği
    on_hand: str                  # Elimdeki malzemeler (virgülle ayrılmış)

    # ── RECIPE_ANALYZER_NODE çıktıları ──────────────────────────────────────
    recipe_name: str              # Normalleştirilmiş yemek adı (LLM çıktısı)
    search_terms: List[str]       # Markette aranacak standart terimler

    # ── Düğümler arası köprü — ham market sonuçları ─────────────────────────
    market_results: List[Dict[str, Any]]   # MarketAPIClient.fetch_price çıktıları

    # ── MARKET_SEARCH_CHECKER_NODE çıktıları ────────────────────────────────
    cleaned_results: List[Dict[str, Any]]  # Kalite kontrolünden geçmiş sonuçlar
    warnings: List[str]                    # Kullanıcıya bildirilecek uyarılar

    # Porsiyon/birim hatası: eski_terim → filtreli_yeni_terim
    # Orchestrator bu haritayla LLM'i ATLAR ve doğrudan marketi yeniden arar.
    qty_retry_map: Dict[str, str]

    # İçerik/alaka hatası: LLM ile rafine edilmesi gereken terimler
    content_retry_terms: List[str]

    # Tüm başarısız terimler (qty + content) — analyze_recipe_node'a aktarılır
    retry_terms: List[str]

    # ── Döngü kontrolü ──────────────────────────────────────────────────────
    retry_count: int              # Kaç kez döngü tekrarlandı (her tur +1)

    # ── FINAL_REPORT_NODE çıktıları ─────────────────────────────────────────
    total_cost: float
    cheapest_market: str
    market_totals: Dict[str, float]
    final_report: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────

MAX_RETRIES: int = 3
RELEVANCE_THRESHOLD: float = 40.0

# ── İçerik Gürültüsü ─────────────────────────────────────────────────────────
# NOT: "bisküvi" çıkarıldı — kedi dili gibi tarifte gerekli ürünler var
_NOISE_RE = re.compile(
    r"\b(gofret|kremal[iı]|aromal[iı]|meyveli|çikolatal[iı]|bıldırcın)\b",
    re.IGNORECASE | re.UNICODE,
)

# ── Kakao içecek filtresi (Nesquik, kids vb.) ────────────────────────────────
_KAKAO_NOISE_RE = re.compile(
    r"\b(nesquik|nestle\s*kids|çocuk|kids|içecek)\b",
    re.IGNORECASE | re.UNICODE,
)

# ── Arama Terimi Birim Algılama ───────────────────────────────────────────────
# Arama terimi "1L / 1Lt / 1Litre" istiyorsa:
_WANT_1L_RE = re.compile(r"\b1\s*[Ll](?:[Tt]|itre)?\b", re.IGNORECASE)
# Arama terimi "1kg / 1Kg / 1kilo / 1kilogram" istiyorsa:
_WANT_1KG_RE = re.compile(r"\b1\s*k(?:g|ilo(?:gram)?)\b", re.IGNORECASE)

# ── Ürün Porsiyon Boyutu Algılama (KRİTİK HATA eşiği) ───────────────────────
# Ürün adında ≤ 750ml varsa (1L istenirken küçük boy geldi)
_CRT_SMALL_ML_RE = re.compile(
    r"\b(?:[1-9]\d|1\d{2}|[2-7]\d{2}|750)\s*ml\b",  # 10 ml … 750 ml
    re.IGNORECASE,
)
# Ürün adında ≤ 500g varsa (1kg istenirken küçük paket geldi)
_CRT_SMALL_G_RE = re.compile(
    r"\b(?:[1-9]\d|1\d{2}|[2-4]\d{2}|500)\s*g\b",  # 10 g … 500 g
    re.IGNORECASE,
)

LogCallback = Callable[[str], Coroutine[Any, Any, None]]


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı — Porsiyon/Birim Uyumsuzluğu Tespiti
# ──────────────────────────────────────────────────────────────────────────────

def _qty_mismatch_refined(term: str, product_name: str) -> Optional[str]:
    """
    Birim uyumsuzluğu varsa farklı bir terim döner.
    Yeni terim eskiyle AYNI ise None → sonsuz döngü engeli.
    """
    candidate = None
    if _WANT_1L_RE.search(term) and _CRT_SMALL_ML_RE.search(product_name):
        candidate = re.sub(r"\b1\s*[Ll](?:[Tt]|itre)?\b", "1000ml", term, flags=re.IGNORECASE).strip()
    elif _WANT_1KG_RE.search(term) and _CRT_SMALL_G_RE.search(product_name):
        candidate = re.sub(r"\b1\s*k(?:g|ilo(?:gram)?)\b", "1000g", term, flags=re.IGNORECASE).strip()
    if candidate and candidate.lower() != term.lower():
        return candidate
    return None


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 — MARKET_SEARCH_CHECKER_NODE (Denetçi)
# ──────────────────────────────────────────────────────────────────────────────

async def market_search_checker_node(
    state: MarketState,
    log_cb: LogCallback,
) -> MarketState:
    """
    MARKET_SEARCH_CHECKER_NODE — Ham market sonuçlarını dört kalite kuralıyla denetler.

    Kural 1 — Gürültü Filtresi (İçerik):
        Ürün adında yasaklı kelime (gofret, kremalı, bıldırcın, vb.) varsa
        → temizle, content_retry_terms'e ekle (LLM rafine edecek).

    Kural 2 — Alaka Skoru:
        RelevanceScore < RELEVANCE_THRESHOLD ise
        → temizle, content_retry_terms'e ekle (LLM rafine edecek).

    Kural 3 — Porsiyon/Birim Uyumsuzluğu (KRİTİK HATA):
        1L istenirken ≤ 750 ml    veya   1kg istenirken ≤ 500 g gelirse
        → temizle, qty_retry_map'e ekle (LLM ATLANIYOR, doğrudan markete geri dön).
        → Kullanıcıya uyarı üret.

    Kural 4 — Sonuç Yok:
        Market API sonuç döndürmediyse
        → content_retry_terms'e ekle (LLM yeni terim önerecek).

    Edge kararı: `should_retry()` ile belirlenir.
        qty_retry_map dolu  → "market_search"   (LLM bypass, filtreli arama)
        content_retry_terms dolu → "recipe_analyzer" (LLM refinement)
        ikisi de boş        → "final_report"
    """
    await log_cb("🔍 **Denetçi Düğümü** — Market sonuçları kalite kontrolünden geçiyor…")

    raw: List[Dict[str, Any]] = state.get("market_results", [])
    search_terms: List[str] = state.get("search_terms", [])
    warnings: List[str] = list(state.get("warnings", []))
    warnings_seen: set = set(warnings)   # duplicate uyarı önleme
    cleaned: List[Dict[str, Any]] = []
    content_retry_terms: List[str] = []   # → LLM yoluna
    qty_retry_map: Dict[str, str] = {}    # → Doğrudan market yoluna

    def add_warning(msg: str) -> None:
        if msg not in warnings_seen:
            warnings.append(msg)
            warnings_seen.add(msg)

    # Arama terimi → sonuç haritası (O(1) erişim)
    by_term: Dict[str, Dict[str, Any]] = {
        r["Arama"]: r for r in raw if "Arama" in r
    }

    for term in search_terms:
        result = by_term.get(term)

        # ── Kural 4: Sonuç yok ───────────────────────────────────────────────
        if result is None:
            await log_cb(f"⚠️  **{term}** → Sonuç yok, LLM ile yeni terim denenecek.")
            content_retry_terms.append(term)
            continue

        product_name: str = result.get("Product", "")
        relevance: float = float(result.get("RelevanceScore", 0))
        retry_count: int = state.get("retry_count", 0)
        is_final_retry: bool = retry_count >= MAX_RETRIES - 1

        # ── Kural 1: Gürültü / İçerik uyumsuzluğu ───────────────────────────
        if _NOISE_RE.search(product_name):
            if is_final_retry:
                add_warning(f"⚠️ '{term}' için en yakın ürün: '{product_name}'")
                cleaned.append(result)
            else:
                await log_cb(f"🚫 **{term}** → '{product_name}' içerik uyumsuzluğu, LLM ile rafine edilecek.")
                content_retry_terms.append(term)
            continue

        # ── Kural 1b: Kakao içecek filtresi (Nesquik vb.) ────────────────────
        if "kakao" in term.casefold() and _KAKAO_NOISE_RE.search(product_name):
            if is_final_retry:
                add_warning(f"⚠️ '{term}' için pişirme kakaosu bulunamadı, '{product_name}' eklendi.")
                cleaned.append(result)
            else:
                await log_cb(f"🚫 **{term}** → '{product_name}' kakao içeceği, LLM ile rafine edilecek.")
                content_retry_terms.append(term)
            continue

        # ── Kural 2: Alaka skoru ─────────────────────────────────────────────
        if relevance < RELEVANCE_THRESHOLD:
            if is_final_retry:
                add_warning(f"⚠️ '{term}' düşük eşleşme ({relevance:.0f}): '{product_name}'")
                cleaned.append(result)
            else:
                await log_cb(f"📉 **{term}** → Alaka skoru düşük ({relevance:.0f}), LLM ile rafine edilecek.")
                content_retry_terms.append(term)
            continue

        # ── Kural 3: Porsiyon/Birim Uyumsuzluğu (KRİTİK) ────────────────────
        refined = _qty_mismatch_refined(term, product_name)
        if refined is not None:
            qty_retry_map[term] = refined
            msg = (
                f"🚨 **Porsiyon Hatası** — '{term}' için '{product_name}' bulundu. "
                f"Birim uyumsuzluğu: 200ml/100g olan ürünler elendi. "
                f"Filtreli arama başlatılıyor → '{refined}'"
            )
            add_warning(msg)
            await log_cb(msg)
            continue

        cleaned.append(result)

    state["cleaned_results"] = cleaned
    state["warnings"] = warnings
    state["qty_retry_map"] = qty_retry_map
    state["content_retry_terms"] = content_retry_terms
    # retry_terms = tüm başarısızlar (analyze_recipe_node uyumluluğu için)
    state["retry_terms"] = content_retry_terms + list(qty_retry_map.keys())

    await log_cb(
        f"✅ **Denetçi** — {len(cleaned)} geçerli | "
        f"{len(qty_retry_map)} porsiyon hatası (doğrudan market) | "
        f"{len(content_retry_terms)} içerik hatası (LLM) | "
        f"{len(warnings)} uyarı"
    )
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Conditional Edge — Retry mı, Final mı?
# ──────────────────────────────────────────────────────────────────────────────

def should_retry(state: MarketState) -> str:
    """
    MARKET_SEARCH_CHECKER_NODE sonrasındaki koşullu kenar (conditional edge).

    Öncelik sırası:
        1. Max retry aşıldıysa her zaman → "final_report"
        2. Porsiyon/birim hatası (qty_retry_map)  → "market_search"
           LLM ATLANIR; orchestrator filtreli terimlerle doğrudan yeniden arar.
        3. İçerik/alaka hatası (content_retry_terms) → "recipe_analyzer"
           LLM yeni arama terimi üretir.
        4. Her şey temiz → "final_report"

    Not: qty VE content hatası aynı anda varsa content_retry_terms'e
    qty hataları da eklenir (LLM tümünü birlikte rafine eder).
    Bu nedenle qty_only olduğunda önce hızlı market yolu denenir.
    """
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "final_report"

    has_qty = bool(state.get("qty_retry_map"))
    has_content = bool(state.get("content_retry_terms"))

    if has_qty and not has_content:
        # Sadece birim/porsiyon sorunu → LLM'i atla, filtreli terimle markete git
        return "market_search"
    if has_content:
        # İçerik/alaka sorunu var (qty ile karışık da olabilir) → LLM yoluna
        # qty_retry_map'teki terimler de content_retry_terms'e dahil edilir
        # (bu birleştirme orchestrator'da yapılır)
        return "recipe_analyzer"
    return "final_report"


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 — FINAL_REPORT_NODE (Sunucu)
# ──────────────────────────────────────────────────────────────────────────────

async def final_report_node(
    state: MarketState,
    log_cb: LogCallback,
) -> MarketState:
    """
    FINAL_REPORT_NODE — Temizlenmiş veriden profesyonel final raporunu derler.

    Sadece `cleaned_results` içindeki (tüm kalite kurallarını geçmiş) ürünler
    rapora dahil edilir. Çözümlenemeyen terimler `unresolved_terms` listesinde
    kullanıcıya bildirilir.

    Hesaplamalar:
        total_cost      : cleaned_results içindeki Price toplamı
        market_totals   : Market → toplam harcama haritası
        cheapest_market : market_totals'ta en düşük toplam harcamalı market
        unresolved_terms: MAX_RETRIES sonunda hâlâ eşleştirilemeyen terimler
    """
    await log_cb("📝 **Sunucu Düğümü** — Final rapor hazırlanıyor…")

    # Duplicate temizle — aynı Arama key'i birden fazla kez geldiyse ilkini al
    raw_results: List[Dict[str, Any]] = state.get("cleaned_results", [])
    seen: dict = {}
    for r in raw_results:
        k = r.get("Arama", "")
        if k not in seen:
            seen[k] = r
    results: List[Dict[str, Any]] = list(seen.values())
    recipe_name: str = state.get("recipe_name", "Bilinmiyor")
    warnings: List[str] = list(state.get("warnings", []))

    # ── Çözümlenemeyen terimler ───────────────────────────────────────────────
    # search_terms içinde olup cleaned_results'ta karşılığı olmayan terimler
    resolved_keys = {r.get("Arama", "") for r in results}
    unresolved_terms: List[str] = [
        t for t in state.get("search_terms", []) if t not in resolved_keys
    ]
    for ut in unresolved_terms:
        msg = f"❌ '{ut}' için uygun ürün bulunamadı — listeden çıkarıldı."
        warnings.append(msg)
        await log_cb(msg)

    total_cost: float = sum(float(r.get("Price", 0)) for r in results)

    # ── En Ucuz Market hesabı ─────────────────────────────────────────────────
    market_totals: Dict[str, float] = {}
    for r in results:
        market = r.get("Market") or "—"
        market_totals[market] = market_totals.get(market, 0.0) + float(r.get("Price", 0))

    cheapest_market: str = (
        min(market_totals, key=lambda k: market_totals[k])
        if market_totals else ""
    )

    state["total_cost"] = round(total_cost, 2)
    state["cheapest_market"] = cheapest_market
    state["market_totals"] = market_totals
    state["warnings"] = warnings
    state["final_report"] = {
        "recipe": recipe_name,
        "results": results,
        "total_cost": round(total_cost, 2),
        "cheapest_market": cheapest_market,
        "warnings": warnings,
        "unresolved_terms": unresolved_terms,
        "market_totals": market_totals,
    }

    await log_cb(
        f"🏁 **{recipe_name}** | "
        f"Toplam: {total_cost:.2f} ₺ | "
        f"En Avantajlı Market: **{cheapest_market or '—'}** | "
        f"{len(unresolved_terms)} çözümsüz kalem"
    )
    return state