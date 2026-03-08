"""
LLM Servisi — Azure OpenAI Function Calling (Agentic AI).

Yaklaşım: Model sadece malzeme listesi çıkaran bir robot değil; elindeki
`fetch_ingredient_price` aracını kullanarak hangi malzemeleri araması
gerektiğine **kendisi** karar veren bir ajandır.

Agentic döngü:
  1. Modele tarif isteği ve eldeki malzemeler gönderilir.
  2. Model eksik malzemeleri belirler ve her biri için `fetch_ingredient_price`
     aracını çağırır.
  3. Orchestrator aracın gerçek uygulamasını (market API çağrısı) çalıştırır ve
     sonucu modele gönderir.
  4. Model araçları kullanmayı bırakınca son özeti döndürür.

Bu modül yalnızca OpenAI iletişimini yönetir; API istekleri veya UI kodu içermez.
"""

import json
from typing import Any, Callable, Coroutine, Dict, List, Optional

from openai import AsyncAzureOpenAI

from core.config import get_settings

# ──────────────────────────────────────────────────────────────────────────────
# Tool tanımı — OpenAI Function Calling şeması
# ──────────────────────────────────────────────────────────────────────────────

_FETCH_PRICE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_ingredient_price",
        "description": (
            "Belirtilen malzeme adına göre yakın çevredeki marketlerden "
            "en uygun fiyatı ve market bilgisini döner. "
            "Bir malzemenin güncel fiyatını öğrenmek istediğinde bu aracı kullan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": (
                        "Fiyatı aranacak malzeme adı. "
                        "Türkçe, kısa ve aranabilir olsun. "
                        "Örnekler: 'kuzu but', 'domates', 'zeytinyağı', 'pirinç'."
                    ),
                }
            },
            "required": ["ingredient"],
        },
    },
}

_SYSTEM_PROMPT = """\
Sen deneyimli bir Türk şefsin ve akıllı bir market alışveriş asistanısın.

Görevin şu adımları takip etmektir:
1. Kullanıcının istediği yemek için gerekli tüm malzemeleri belirle.
2. Kullanıcının elinde zaten bulunan malzemeleri çıkar.
3. Eksik olan her malzeme için `fetch_ingredient_price` aracını çağır.
4. Tüm araç çağrıları tamamlanınca aşağıdaki JSON formatında yanıt ver:
   {"recipe": "<yemek adı>", "summary": "<1–2 cümle kısa özet>"}

Önemli kurallar:
- Aracı yalnızca gerçekten eksik olan malzemeler için çağır.
- Malzeme adlarını kısa ve aranabilir tut (örn: "kırmızı biber" değil "biber").
- Tüm araç çağrıları tamamlanmadan özet verme.
- Yanıtın sadece geçerli JSON olsun, başka metin ekleme.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Tip kısaltması
# ──────────────────────────────────────────────────────────────────────────────

ToolExecutor = Callable[[str], Coroutine[Any, Any, Optional[Dict[str, Any]]]]
LogCallback = Callable[[str], Coroutine[Any, Any, None]]


# ──────────────────────────────────────────────────────────────────────────────
# Agentic döngü
# ──────────────────────────────────────────────────────────────────────────────

async def run_agentic_extraction(
    recipe_request: str,
    on_hand: str,
    tool_executor: ToolExecutor,
    log_cb: LogCallback,
    *,
    max_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Function Calling agentic döngüsüyle tarif analizi ve fiyat çekimi yapar.

    Args:
        recipe_request  : Kullanıcının yemek isteği ("Konya usulü fırın kebabı")
        on_hand         : Eldeki malzemeler ("soğan, zeytinyağı")
        tool_executor   : `fetch_ingredient_price(ingredient)` → dict | None
                          (Orchestrator tarafından sağlanan, konum bilgisi kapalı)
        log_cb          : Asenkron UI log callback'i
        max_iterations  : Sonsuz döngüye karşı güvenlik sınırı

    Returns:
        {"recipe": str, "summary": str, "results": [dict, ...]}

    Raises:
        ConfigurationError       : Azure bilgileri eksik
        IngredientExtractionError: Hiçbir malzeme işlenemedi
    """
    settings = get_settings()
    if not settings.is_llm_configured:
        raise ValueError(
            "Azure OpenAI bilgileri eksik. "
            ".env veya .streamlit/secrets.toml içine "
            "AZURE_ENDPOINT ve AZURE_API_KEY ekleyin."
        )

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_api_key,
        api_version=settings.azure_api_ver,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Yapmak istediğim yemek: {recipe_request}\n"
                f"Elimdeki malzemeler: {on_hand.strip() or 'Yok'}\n\n"
                "Eksik malzemelerin fiyatlarını bul ve özeti ver."
            ),
        },
    ]

    collected_results: List[Dict[str, Any]] = []
    recipe_name = "Bilinmiyor"
    summary = ""

    for iteration in range(max_iterations):
        response = await client.chat.completions.create(
            model=settings.gpt_deployment,
            messages=messages,
            tools=[_FETCH_PRICE_TOOL],
            tool_choice="auto",
            temperature=0.1,
        )

        assistant_msg = response.choices[0].message

        # Mesajı geçmişe ekle (tool_calls None ise exclude et)
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": assistant_msg.content}
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # Araç çağrısı yoksa döngü bitti
        if not assistant_msg.tool_calls:
            if assistant_msg.content:
                try:
                    parsed = json.loads(assistant_msg.content)
                    recipe_name = parsed.get("recipe", recipe_name)
                    summary = parsed.get("summary", summary)
                except (json.JSONDecodeError, AttributeError):
                    # Model ham metin döndürdüyse mevcut değerleri koru
                    pass
            break

        # Araç çağrılarını sırayla işle (içlerinde zaten async paralel API var)
        tool_responses: List[Dict[str, Any]] = []
        for tool_call in assistant_msg.tool_calls:
            if tool_call.function.name != "fetch_ingredient_price":
                # Tanınmayan araç — güvenli hata yanıtı döndür
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": "Bilinmeyen araç"}, ensure_ascii=False),
                })
                continue

            args = json.loads(tool_call.function.arguments)
            ingredient = args.get("ingredient", "").strip()

            if not ingredient:
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": "ingredient boş bırakılamaz"}, ensure_ascii=False
                    ),
                })
                continue

            await log_cb(
                f"🤖 **Ajan** → `fetch_ingredient_price('{ingredient}')` çağırıyor…"
            )

            result = await tool_executor(ingredient)

            if result:
                collected_results.append(result)
                content = json.dumps(result, ensure_ascii=False)
            else:
                content = json.dumps(
                    {"error": f"'{ingredient}' için fiyat bulunamadı"}, ensure_ascii=False
                )

            tool_responses.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content,
            })

        messages.extend(tool_responses)

    else:
        # max_iterations aşıldıysa güvenli çıkış
        await log_cb(
            f"⚠️ Maksimum ajan iterasyonu ({max_iterations}) aşıldı. Mevcut sonuçlar gösteriliyor."
        )

    if not collected_results:
        raise RuntimeError(
            "Ajan hiçbir malzeme için fiyat çekemedi. "
            "Azure yapılandırmanızı ve market API bağlantısını kontrol edin."
        )

    return {
        "recipe": recipe_name,
        "summary": summary,
        "results": collected_results,
    }
