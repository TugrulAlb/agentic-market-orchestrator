"""
Orchestrator — İş mantığının merkezi koordinatörü.

UI (main.py) ne LLM servisini ne de market API istemcisini doğrudan tanır;
yalnızca bu sınıfı kullanır. Bu sayede Streamlit, arkada ne döndüğünü bilmez.

Pipeline (State Machine):
  1. RECIPE_ANALYZER_NODE  — LLM tarif analizi → standart arama terimleri
  2. Paralel market araması — MarketAPIClient ile tüm terimler aynı anda
  3. MARKET_SEARCH_CHECKER_NODE — Kalite denetimi, gürültü filtresi, retry kararı
      ↑ retry kenarı (MAX 2 kez): Şef'e geri dön, terimleri rafine et
  4. FINAL_REPORT_NODE — Toplam maliyet, en ucuz market tavsiyesi

main.py'nin çağırdığı tek yüksek-seviye metot: `Orchestrator.run()`
"""

import asyncio
from typing import Callable, Coroutine, List, Optional, Tuple

import httpx
import pandas as pd

from clients.market_api import Location, MarketAPIClient
from core.config import Settings, get_settings
from core.graph import (
    MAX_RETRIES,
    MarketState,
    final_report_node,
    market_search_checker_node,
    should_retry,
)
from services.llm_service import analyze_recipe_node

LogCallback = Callable[[str], Coroutine]


class Orchestrator:
    """
    Uygulama pipeline'ının tek giriş noktası.

    Kullanım:
        orch = Orchestrator()
        df, recipe, warnings, cheapest = await orch.run(
            recipe_request, on_hand, district, loc, log_cb
        )
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._market_client = MarketAPIClient(self._settings)

    async def run(
        self,
        recipe_request: str,
        on_hand: str,
        district: str,
        loc: Location,
        log_cb: LogCallback,
    ) -> Tuple[pd.DataFrame, str, List[str], str]:
        """
        State Machine pipeline'ını çalıştırır.

        Args:
            recipe_request : Kullanıcının yemek isteği
            on_hand        : Eldeki malzemeler (virgülle ayrılmış)
            district       : Aktif ilçe adı (gösterim ve log için)
            loc            : Konum + arama yarıçapı
            log_cb         : Asenkron UI log callback'i

        Returns:
            (DataFrame, recipe_name, warnings, cheapest_market)
            DataFrame boşsa hiç malzeme bulunamadı veya hepsi elimizde var.

        Raises:
            ValueError    : Azure yapılandırması eksikse
            RuntimeError  : Hiçbir malzeme fiyatı çekilemediyse
        """
        # ── Başlangıç state ───────────────────────────────────────────────────
        state: MarketState = {
            "recipe_request": recipe_request,
            "on_hand": on_hand,
            "recipe_name": "Bilinmiyor",
            "search_terms": [],
            "market_results": [],
            "cleaned_results": [],
            "warnings": [],
            "retry_terms": [],
            "retry_count": 0,
            "total_cost": 0.0,
            "cheapest_market": "",
            "market_totals": {},
            "final_report": {},
        }

        semaphore = asyncio.Semaphore(5)  # Aynı anda en fazla 5 paralel API isteği

        async with httpx.AsyncClient(follow_redirects=True) as http_client:

            # ── State Machine Ana Döngüsü ─────────────────────────────────────
            # Döngü: Node1 → Arama → Node2 → (retry?) → Node1 ...
            # ─────────────────────────────────────────────────────────────────
            while True:
                retry_count: int = state.get("retry_count", 0)
                is_retry = retry_count > 0

                # ── NODE 1: RECIPE_ANALYZER_NODE ─────────────────────────────
                analysis = await analyze_recipe_node(
                    recipe_request=recipe_request,
                    on_hand=on_hand,
                    log_cb=log_cb,
                    retry_terms=state.get("retry_terms") if is_retry else None,
                )

                if not is_retry:
                    # İlk tur: tüm arama terimlerini ve tarif adını al
                    state["recipe_name"] = analysis["recipe_name"]
                    state["search_terms"] = analysis["search_terms"]
                    terms_to_search = state["search_terms"]
                else:
                    # Retry turu: eski başarılı terimleri koru, sadece yenileri ara
                    refined: List[str] = analysis.get("search_terms", [])
                    old_retry_set = set(state.get("retry_terms", []))
                    surviving = [
                        t for t in state["search_terms"]
                        if t not in old_retry_set
                    ]
                    state["search_terms"] = surviving + refined
                    terms_to_search = refined  # Yalnızca yenileri API'ye gönder

                if not state["search_terms"]:
                    await log_cb("ℹ️  Eksik malzeme bulunamadı — tarifte her şey elimizde.")
                    break

                # ── Paralel Market Araması ────────────────────────────────────
                await log_cb(
                    f"🚀 **Market Araması** — "
                    f"{len(terms_to_search)} malzeme "
                    f"**{district or 'seçili bölge'}**'de paralel aranıyor…"
                )

                raw_results = await asyncio.gather(
                    *[
                        self._market_client.fetch_price(
                            item=term,
                            loc=loc,
                            district=district,
                            log_cb=log_cb,
                            http_client=http_client,
                            semaphore=semaphore,
                        )
                        for term in terms_to_search
                    ]
                )

                # Önceki başarılı sonuçları koru, yeni sonuçları ekle
                result_map = {
                    r["Arama"]: r
                    for r in state.get("cleaned_results", [])
                    if "Arama" in r
                }
                for result in raw_results:
                    if result and "Arama" in result:
                        result_map[result["Arama"]] = result

                state["market_results"] = list(result_map.values())

                # ── NODE 2: MARKET_SEARCH_CHECKER_NODE ───────────────────────
                state = await market_search_checker_node(state, log_cb)

                # ── Conditional Edge ─────────────────────────────────────────
                edge = should_retry(state)
                if edge == "final_report":
                    break

                state["retry_count"] = retry_count + 1
                await log_cb(
                    f"🔄 **Denetçi → Şef** — "
                    f"{len(state['retry_terms'])} terim rafine ediliyor "
                    f"(tur {state['retry_count']}/{MAX_RETRIES})…"
                )

            # ── NODE 3: FINAL_REPORT_NODE ─────────────────────────────────────
            state = await final_report_node(state, log_cb)

        # ── Sonuçları DataFrame'e dönüştür ────────────────────────────────────
        report = state.get("final_report", {})
        results = report.get("results", [])
        recipe_name: str = report.get("recipe", state.get("recipe_name", "Bilinmiyor"))
        warnings: List[str] = report.get("warnings", [])
        cheapest_market: str = report.get("cheapest_market", "")

        if not results:
            return pd.DataFrame(), recipe_name, warnings, cheapest_market

        df = pd.DataFrame(results)
        df = df.sort_values("Price").reset_index(drop=True)
        return df, recipe_name, warnings, cheapest_market

