"""
Orchestrator — İş mantığının merkezi koordinatörü.

UI (main.py) ne LLM servisini ne de market API istemcisini doğrudan tanır;
yalnızca bu sınıfı kullanır. Bu sayede Streamlit, arkada ne döndüğünü bilmez.

Sorumluluklar:
  1. Bir httpx.AsyncClient ömrünü yönetmek (tüm API isteklerinde paylaşılır)
  2. MarketAPIClient'ı konum bilgisiyle çağıracak `tool_executor` closure üretmek
  3. LLM agentic döngüsünü başlatmak (services.llm_service)
  4. Sonuçları DataFrame'e dönüştürmek ve UI'ya teslim etmek

main.py'nin çağırdığı tek yüksek-seviye metot: `Orchestrator.run()`
"""

import asyncio
from typing import Callable, Coroutine, Optional, Tuple

import httpx
import pandas as pd

from clients.market_api import Location, MarketAPIClient
from core.config import Settings, get_settings
from services.llm_service import run_agentic_extraction

LogCallback = Callable[[str], Coroutine]


class Orchestrator:
    """
    Uygulama pipeline'ının tek giriş noktası.

    Kullanım:
        orch = Orchestrator()
        df, recipe = await orch.run(recipe_request, on_hand, district, loc, log_cb)
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
    ) -> Tuple[pd.DataFrame, str]:
        """
        Tarif isteğini alır; eksik malzemeleri tespit edip fiyatları çeker.

        Args:
            recipe_request : Kullanıcının yemek isteği
            on_hand        : Eldeki malzemeler (virgülle ayrılmış)
            district       : Aktif ilçe adı (sonuç tablosunda gösterim için)
            loc            : Konum + arama yarıçapı
            log_cb         : Asenkron UI log callback'i

        Returns:
            (DataFrame, recipe_name) — DataFrame fiyat tablosunu içerir.
            Hiç malzeme yoksa boş DataFrame döner.

        Raises:
            IngredientExtractionError : LLM hiçbir malzeme çıkaramadıysa
            MarketAPIConnectionError  : Tüm market API denemeleri başarısızsa
        """
        await log_cb("🧠 **Agent 1** — Tarif analiz ediliyor (Agentic AI)…")

        # Tüm paralel httpx isteklerinde paylaşılan bir istemci oluştur
        semaphore = asyncio.Semaphore(5)  # Aynı anda en fazla 5 API isteği

        async with httpx.AsyncClient(follow_redirects=True) as http_client:

            async def tool_executor(ingredient: str):
                """
                LLM'in çağırdığı `fetch_ingredient_price` aracının gerçek uygulaması.
                Konum bilgisi bu closure içinde kapsüllenmiştir; LLM bunu bilmez.
                """
                return await self._market_client.fetch_price(
                    item=ingredient,
                    loc=loc,
                    district=district,
                    log_cb=log_cb,
                    http_client=http_client,
                    semaphore=semaphore,
                )

            await log_cb(
                f"🚀 **Agent 2** — Ajan araçları kullanarak "
                f"**{district or 'seçili bölge'}** için fiyat çekiyor…"
            )

            extraction = await run_agentic_extraction(
                recipe_request=recipe_request,
                on_hand=on_hand,
                tool_executor=tool_executor,
                log_cb=log_cb,
            )

        results = extraction.get("results", [])
        recipe_name: str = extraction.get("recipe", "Bilinmiyor")

        if not results:
            return pd.DataFrame(), recipe_name

        df = pd.DataFrame(results)
        df = df.sort_values("Price").reset_index(drop=True)
        return df, recipe_name
