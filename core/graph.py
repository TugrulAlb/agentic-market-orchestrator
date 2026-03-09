"""
Market Intelligence Graph — State Machine (Durum Makinesi).

Mimari:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  RECIPE_ANALYZER_NODE  →  [Paralel Market Arama]  →  MARKET_SEARCH_    │
  │         (Şef)                                        CHECKER_NODE        │
  │           ↑                                          (Denetçi)           │
  │           └─────── retry kenarı (MAX 2 kez) ────────────┘               │
  │                                                          ↓               │
  │                                               FINAL_REPORT_NODE          │
  │                                               (Sunucu)                   │
  └──────────────────────────────────────────────────────────────────────────┘

Her düğüm MarketState sözlüğünü alır, günceller ve döndürür.
Düğümler arası tüm veri akışı yalnızca MarketState üzerinden gerçekleşir —
hiçbir düğüm başka bir düğümü doğrudan çağırmaz.

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
    retry_terms: List[str]                 # Kalite testini geçemeyen terimler

    # ── Döngü kontrolü ──────────────────────────────────────────────────────
    retry_count: int              # Kaç kez RECIPE_ANALYZER_NODE'a geri dönüldü

    # ── FINAL_REPORT_NODE çıktıları ─────────────────────────────────────────
    total_cost: float
    cheapest_market: str
    market_totals: Dict[str, float]
    final_report: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────────────────────────────────────

MAX_RETRIES: int = 2
RELEVANCE_THRESHOLD: float = 45.0

# Ürün adında bu kalıplar geçiyorsa → gürültü, temizle + retry
_NOISE_RE = re.compile(
    r"\b(gofret|kremal[iı]|aromal[iı]|meyveli|bisk[uü]vi|çikolatal[iı]|bıldırcın)\b",
    re.IGNORECASE | re.UNICODE,
)

# "1L / 1 Lt / 1 Litre" içeren arama terimini tanımlar
_WANT_1L_RE = re.compile(r"\b1\s*[Ll](?:[Tt]|itre)?\b")

# Ürün adında porsiyon boyutu küçük mü? (≤ 250 ml)
_SMALL_ML_RE = re.compile(r"\b(1\d{2}|200|180|150|250)\s*ml\b", re.IGNORECASE)

LogCallback = Callable[[str], Coroutine[Any, Any, None]]


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 — MARKET_SEARCH_CHECKER_NODE (Denetçi)
# ──────────────────────────────────────────────────────────────────────────────

async def market_search_checker_node(
    state: MarketState,
    log_cb: LogCallback,
) -> MarketState:
    """
    MARKET_SEARCH_CHECKER_NODE — Ham market sonuçlarını üç kalite kuralıyla denetler.

    Kural 1 — Alaka Kontrolü:
        RelevanceScore < RELEVANCE_THRESHOLD ise sonucu temizle, retry listesine ekle.

    Kural 2 — Gürültü Filtresi:
        Ürün adında yasaklı kelime (gofret, kremalı, vb.) varsa temizle, retry ekle.

    Kural 3 — Porsiyon Uyarısı:
        Arama terimi "1L" içeriyor ama ürün ≤ 250 ml geliyorsa kullanıcıya uyar
        (ürün temizlenmez, yalnızca uyarı üretilir).

    Edge kararı: `should_retry()` ile belirlenir.
    """
    await log_cb("🔍 **Denetçi Düğümü** — Market sonuçları kalite kontrolünden geçiyor…")

    raw: List[Dict[str, Any]] = state.get("market_results", [])
    search_terms: List[str] = state.get("search_terms", [])
    warnings: List[str] = list(state.get("warnings", []))
    cleaned: List[Dict[str, Any]] = []
    retry_terms: List[str] = []

    # Arama terimi → sonuç haritası (O(1) erişim)
    by_term: Dict[str, Dict[str, Any]] = {
        r["Arama"]: r for r in raw if "Arama" in r
    }

    for term in search_terms:
        result = by_term.get(term)

        # Sonuç hiç gelmedi
        if result is None:
            await log_cb(f"⚠️  **{term}** → Sonuç yok, yeniden denenecek.")
            retry_terms.append(term)
            continue

        product_name: str = result.get("Product", "")
        relevance: float = float(result.get("RelevanceScore", 0))

        # ── Kural 2: Gürültü filtresi ─────────────────────────────────────────
        if _NOISE_RE.search(product_name):
            await log_cb(
                f"🚫 **{term}** → '{product_name}' gürültü içeriyor, temizlendi."
            )
            retry_terms.append(term)
            continue

        # ── Kural 1: Alaka skoru ──────────────────────────────────────────────
        if relevance < RELEVANCE_THRESHOLD:
            await log_cb(
                f"📉 **{term}** → Alaka skoru düşük ({relevance:.0f}), "
                f"yeniden aranacak."
            )
            retry_terms.append(term)
            continue

        # ── Kural 3: Porsiyon uyarısı (temizleme yok, sadece bilgi) ──────────
        if _WANT_1L_RE.search(term) and _SMALL_ML_RE.search(product_name):
            msg = (
                f"⚠️ Porsiyon uyarısı: '{term}' için '{product_name}' bulundu — "
                "miktar yetersiz olabilir."
            )
            warnings.append(msg)
            await log_cb(msg)

        cleaned.append(result)

    state["cleaned_results"] = cleaned
    state["warnings"] = warnings
    state["retry_terms"] = retry_terms

    await log_cb(
        f"✅ **Denetçi** — {len(cleaned)} geçerli sonuç | "
        f"{len(retry_terms)} yeniden arama | {len(warnings)} uyarı"
    )
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Conditional Edge — Retry mı, Final mı?
# ──────────────────────────────────────────────────────────────────────────────

def should_retry(state: MarketState) -> str:
    """
    MARKET_SEARCH_CHECKER_NODE sonrasındaki koşullu kenar (conditional edge).

    Returns:
        "retry"        → RECIPE_ANALYZER_NODE'a geri dön (terimi rafine et)
        "final_report" → FINAL_REPORT_NODE'a ilerle
    """
    if state.get("retry_terms") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
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

    Hesaplamalar:
        total_cost      : cleaned_results içindeki Price toplamı
        market_totals   : Market → toplam harcama haritası
        cheapest_market : market_totals'ta en düşük toplam harcamalı market
    """
    await log_cb("📝 **Sunucu Düğümü** — Final rapor hazırlanıyor…")

    results: List[Dict[str, Any]] = state.get("cleaned_results", [])
    recipe_name: str = state.get("recipe_name", "Bilinmiyor")
    warnings: List[str] = state.get("warnings", [])

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
    state["final_report"] = {
        "recipe": recipe_name,
        "results": results,
        "total_cost": round(total_cost, 2),
        "cheapest_market": cheapest_market,
        "warnings": warnings,
        "market_totals": market_totals,
    }

    await log_cb(
        f"🏁 **{recipe_name}** | "
        f"Toplam: {total_cost:.2f} ₺ | "
        f"En Avantajlı Market: **{cheapest_market or '—'}**"
    )
    return state
