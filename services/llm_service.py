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
            "ZORUNLU FORMAT: Türk market API'leri 'Ürün Adı Miktar Birim' formatını kullanır — "
            "miktar SONDA olmalı, başta değil. "
            "(1) Süt türevleri → sonda '1 Lt' ekle (örn: 'Tam Yağlı Süt 1 Lt'). "
            "(2) Kuru gıdalar → sonda '1 Kg' ekle (örn: 'Toz Şeker 1 Kg', 'Buğday Unu 1 Kg'). "
            "(3) Yumurta → 'Yumurta 10 Lu' veya 'Yumurta 15 Li'. "
            "(4) Kahve → 'Türk Kahvesi 100 G', 'Filtre Kahve 250 G', 'Espresso Kahve 250 G'. "
            "    'Espresso' veya '500ml Espresso' YASAK. "
            "(5) Peynir → 'Labne Peyniri', 'Mascarpone Peyniri 250 G'. "
            "(6) Kakao → 'Kakao Tozu' veya 'Dr. Oetker Kakao' (Nesquik veya içecek kakaosu DEĞİL). "
            "(7) Kedi dili bisküvi gibi tarifte ZORUNLU bisküvi → 'Kedi Dili Bisküvi' şeklinde ara. "
            "    Genel yasak: 'kremalı', 'gofret', 'aromalı', 'meyveli', 'çikolatalı'. "
            "(8) Aynı malzeme için bu aracı iki kez çağırma."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": (
                        "Fiyatı aranacak malzeme adı. Format: 'Ürün Adı Miktar Birim' (miktar SONDA). "
                        "Örnekler: 'Tam Yağlı Süt 1 Lt', 'Toz Şeker 1 Kg', 'Buğday Unu 1 Kg', "
                        "'Yumurta 10 Lu', 'Dana Kıyma 500 G', 'Türk Kahvesi 100 G', 'Tereyağı 100 G'. "
                        "ASLA başa miktar yazma: '1L Süt', '1kg Un', '10lu Tavuk Yumurtası' → GEÇERSİZ."
                    ),
                }
            },
            "required": ["ingredient"],
        },
    },
}

_SYSTEM_PROMPT = """\
Sen "TugrulAI Market Orchestrator" sisteminin beyni olan kıdemli bir şef ve stratejik satın alma uzmanısın.
Görevin, kullanıcının girdiği yemek tarifini analiz edip, SADECE eksik olan malzemeleri tespit etmek
ve bunları en doğru formatta markette aratmaktır.

─── GÖREV AKIŞI ───────────────────────────────────────────────────────────────
1. Kullanıcının yemek tarifini analiz et ve profesyonel, eksiksiz bir malzeme listesi çıkar.
2. Kullanıcının elindeki malzemeleri tarifle karşılaştır; GERÇEKTEN eksik olanları belirle.
3. Eksik her malzeme için aşağıdaki Standartlaştırma Kurallarına HARFIYEN uyarak
   `fetch_ingredient_price` aracını çağır.
4. Her sonucu kontrol et; tarife uymuyorsa daha spesifik mutfak terimiyle bir kez daha dene.
5. Tüm fiyatlar onaylandıktan sonra aşağıdaki FINAL ÇIKTI formatında özet ver.

─── KURAL 1: MARKET API FORMAT (KRİTİK) ──────────────────────────────────────

  Türk market API'leri ürünleri "Ürün Adı Miktar Birim" formatında indeksler.
  Miktar SONDA gelir, başta değil!

  Süt türevleri (süt, ayran, kefir vb.):
    ✓ Sonda "1 Lt" ekle → "Tam Yağlı Süt 1 Lt", "Yarım Yağlı Süt 1 Lt"
    ✗ "1L Süt", "1L Tam Yağlı Süt" → YANLIŞ FORMAT

  Şeker, un, pirinç, bakliyat ve kuru gıdalar:
    ✓ Sonda "1 Kg" ekle → "Toz Şeker 1 Kg", "Buğday Unu 1 Kg", "Pilavlık Pirinç 1 Kg"
    ✗ "1kg Toz Şeker", "1kg Un" → YANLIŞ FORMAT
    ✓ Un için mutlaka → "Buğday Unu 1 Kg" (sadece "Un" veya "1kg Un" → GEÇERSİZ)

  Yumurta:
    ✓ "Yumurta 10 Lu" veya "Yumurta 15 Li"
    ✗ "10'lu Tavuk Yumurtası", "Tavuk Yumurtası 10lu" → Çalışmaz
    ✗ Bıldırcın yumurtası → KESİNLİKLE YASAK

  Kahve:
    ✓ "Türk Kahvesi 100 G", "Filtre Kahve 250 G", "Espresso Kahve 250 G"
    ✗ "500ml Espresso", "Espresso" → Espresso bir içecek türü, ürün adı değil

  Peynir:
    ✓ Kısa yaygın isim: "Labne Peyniri", "Mascarpone Peyniri 250 G"
    ✗ "500g Mascarpone Peyniri" → başa gramaj yazma

  Diğer malzemeler:
    ✓ "Tereyağı 100 G", "Dana Kıyma 500 G", "Süzme Yoğurt 500 G"

─── KURAL 2: YASAKLI KELİMELERDEN KAÇIŞ ──────────────────────────────────────

  ✗ Arama terimlerinde ASLA kullanılmayacak kelimeler:
     "kremalı" | "gofret" | "aromalı" | "meyveli" | "bisküvi" | "çikolatalı"

  ► Kakao gerekiyorsa → "Kakao Tozu" veya "Dr. Oetker Kakao" kullan.
    Nesquik, içecek tozu, çocuk kakaosu → KESİNLİKLE YASAK.
  ► Tatlı kaplaması için → "Bitter Kuvertür Çikolata" gibi teknik mutfak terimleri tercih et.

─── KURAL 2b: TARİFE ÖZGÜ MALZEMELER ─────────────────────────────────────────

  Tiramisu için:
    ✓ Kedi dili bisküvi → "Kedi Dili Bisküvi" (bisküvi kelimesi bu tarifte GEREKLİ)
    ✓ Mascarpone → "Mascarpone Peyniri 250 G"
    ✓ Espresso → "Espresso Kahve 250 G" veya "Filtre Kahve 250 G"

  Genel kural: Tarifte ZORUNLU olan bir malzeme belirli bir ürün kategorisiyse
  (bisküvi, kraker, makarna vb.) o kategori adını kullanmaktan çekinme.
  Yasak kelimeler ALAKASIZ ürünleri engellemek içindir, gerekli malzemeleri değil.

─── KURAL 3: TEKİLLEŞTİRME VE MANTIKLI ARAMA ─────────────────────────────────

  ► Benzer veya birbiri yerine geçebilecek malzemeleri TEK arama teriminde birleştir.
  ► Her malzeme için en fazla 2 deneme yap; hâlâ uygun sonuç gelmezse
    "bulunamadı" olarak işaretle. Hiçbir fiyatı asla uydurma.
  ► Tüm araç çağrıları tamamlanmadan final özet verme.

─── FINAL ÇIKTI ────────────────────────────────────────────────────────────────
Sadece geçerli JSON döndür, başka metin yazma:
{
  "recipe": "<yemek adı — temiz tarif özeti>",
  "missing_ingredients": ["<malzeme 1>", "<malzeme 2>"],
  "total_cost": <bulunan fiyatların sayısal toplamı, float>,
  "summary": "<temiz tarif özeti + eksik malzemelerin listesi + seçilen ürünlerin fiyatları + toplam maliyet>"
}
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

    collected_results_map: Dict[str, Dict[str, Any]] = {}
    recipe_name = "Bilinmiyor"
    summary = ""
    missing_ingredients: List[str] = []
    total_cost: Optional[float] = None

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
                    if isinstance(parsed.get("missing_ingredients"), list):
                        missing_ingredients = [str(x) for x in parsed["missing_ingredients"]]
                    if isinstance(parsed.get("total_cost"), (int, float)):
                        total_cost = float(parsed["total_cost"])
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
                key = str(result.get("Arama") or ingredient).casefold().strip()
                current = collected_results_map.get(key)

                # Aynı arama terimi birden fazla kez çağrılırsa en iyi skorlu sonucu koru.
                if current is None:
                    collected_results_map[key] = result
                else:
                    current_score = float(current.get("FinalScore", 0.0))
                    new_score = float(result.get("FinalScore", 0.0))
                    if new_score >= current_score:
                        collected_results_map[key] = result

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

    collected_results = list(collected_results_map.values())

    if not collected_results:
        raise RuntimeError(
            "Ajan hiçbir malzeme için fiyat çekemedi. "
            "Azure yapılandırmanızı ve market API bağlantısını kontrol edin."
        )

    return {
        "recipe": recipe_name,
        "summary": summary,
        "missing_ingredients": missing_ingredients,
        "total_cost": total_cost,
        "results": collected_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 — RECIPE_ANALYZER_NODE  (Graph State Machine için)
# ──────────────────────────────────────────────────────────────────────────────

_ANALYZER_SYSTEM_PROMPT = """\
Sen bir profesyonel şef ve stratejik satın alma uzmanısın.
Görevin: kullanıcının yemek isteğini ve elimdeki malzemeleri analiz ederek
tarifte GERÇEKTEN eksik olan malzemeleri standart market arama terimlerine dönüştürmek.

─── STANDARTLAŞTIRMA KURALLARI (KESİNLİKLE UYGULANACAK) ────────────────────────

Süt türevleri  → Sonda "1 Lt" birimi ekle  │ Örn: "Tam Yağlı Süt 1 Lt", "Yarım Yağlı Süt 1 Lt"
Kuru gıdalar   → Sonda "1 Kg" birimi ekle  │ Örn: "Toz Şeker 1 Kg", "Pilavlık Pirinç 1 Kg"
Un             → "Buğday Unu 1 Kg"  (asla sadece "1kg Un" yazma)
Yumurta        → "Yumurta 10 Lu" veya "Yumurta 15 Li"  (başa "Tavuk" veya tırnak ekleme)
Et / Kıyma     → "Dana Kıyma 500 G" gibi gramajlı format
Peynir         → Kısa ve yaygın isim: "Labne Peyniri", "Mascarpone Peyniri 250 G"
Kahve          → "Filtre Kahve 250 G", "Türk Kahvesi 100 G", "Espresso Kahve 250 G"
               → "500ml Espresso" ASLA yazma — Espresso bir içecek, arama terimi değil
Tereyağı       → "Tereyağı 100 G" veya "Tuzsuz Tereyağı 250 G"
Kakao          → "Kakao Tozu" (başına "Ham" ekleme — markette bu şekilde geçmiyor)

ASLA tek kelime veya birim-önekli format kullanma:
  ✗ "1kg Un", "1L Süt", "10'lu Tavuk Yumurtası"  → GEÇERSİZ
  ✓ "Buğday Unu 1 Kg", "Tam Yağlı Süt 1 Lt", "Yumurta 10 Lu"  → DOĞRU

ASLA kullanma: "kremalı" | "gofret" | "aromalı" | "meyveli" | "çikolatalı"
Benzer malzemeleri (örn: "şeker" + "toz şeker") TEK arama teriminde birleştir.

─── İSTİSNALAR ────────────────────────────────────────────────────────────────
Kedi Dili bisküvi → Tiramisu gibi tariflerde GEREKLİ. "Kedi Dili Bisküvi" şeklinde ara.
Kakao → "Kakao Tozu" veya "Dr. Oetker Kakao" — Nesquik veya içecek kakaosu DEĞİL.
Bisküvi kelimesi yalnızca gerçekten bisküvi gerektiğinde kullanılabilir (örn. Tiramisu, cheesecake tabanı).

─── MARKET ARAMA FORMATI (ALTIN KURAL) ─────────────────────────────────────────
Türk market API'leri ürünleri şu formatta indeksler:
  "Ürün Adı [Marka opsiyonel] Miktar Birim"
  Örn: "Toz Şeker 1 Kg", "Tam Yağlı Süt 1 Lt", "Yumurta 10 Lu"

Miktar ve birim SONDA olmalı, başta değil.
"""

_SET_ANALYSIS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_recipe_analysis",
        "description": "Tarif analizi sonucunu yapılandırılmış olarak kaydet.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Yemeğin normalleştirilmiş Türkçe adı.",
                },
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Markette aranacak standart terimler listesi. "
                        "Format: 'Ürün Adı Miktar Birim' (miktar SONDA). "
                        "Örnekler: 'Tam Yağlı Süt 1 Lt', 'Toz Şeker 1 Kg', "
                        "'Buğday Unu 1 Kg', 'Yumurta 10 Lu', 'Türk Kahvesi 100 G'."
                    ),
                },
            },
            "required": ["recipe_name", "search_terms"],
        },
    },
}


async def analyze_recipe_node(
    recipe_request: str,
    on_hand: str,
    log_cb: LogCallback,
    *,
    retry_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    RECIPE_ANALYZER_NODE — Graph State Machine'in ilk düğümü için LLM çağrısı.

    İlk çalışma: Tarifteki tüm eksik malzemeleri standart arama terimleriyle döner.
    Retry modu : Yalnızca `retry_terms` listesindeki başarısız terimleri rafine eder.

    Args:
        recipe_request : Kullanıcının yemek isteği
        on_hand        : Elimdeki malzemeler (virgülle ayrılmış)
        log_cb         : Asenkron UI log callback'i
        retry_terms    : Önceki turda kalite testini geçemeyen arama terimleri.
                         None ise ilk analiz modu, dolu ise iyileştirme modu.

    Returns:
        {"recipe_name": str, "search_terms": [str, ...]}

    Raises:
        ValueError : Azure yapılandırması eksikse
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

    if retry_terms:
        # ── Retry modu: sadece başarısız terimleri iyileştir ──────────────────
        await log_cb(
            f"🔄 **Şef Düğümü** (rafine) — "
            f"{len(retry_terms)} terim için daha spesifik alternatif üretiliyor…"
        )
        user_content = (
            f"Tarif: {recipe_request}\n"
            f"Elimdeki malzemeler: {on_hand.strip() or 'Yok'}\n\n"
            "Aşağıdaki market arama terimleri için uygun sonuç gelmedi.\n"
            "Her biri için daha spesifik, teknik bir mutfak terimi öner "
            "(standartlaştırma kurallarına uy):\n"
            + "\n".join(f"  • {t}" for t in retry_terms)
        )
    else:
        # ── İlk analiz modu: tam tarif analizi ───────────────────────────────
        await log_cb("🧑‍🍳 **Şef Düğümü** — Tarif analiz ediliyor, eksik malzemeler belirleniyor…")
        user_content = (
            f"Yapmak istediğim yemek: {recipe_request}\n"
            f"Elimdeki malzemeler: {on_hand.strip() or 'Yok'}\n\n"
            "Tarifteki tüm eksik malzemeleri standartlaştırma kurallarına göre listele."
        )

    response = await client.chat.completions.create(
        model=settings.gpt_deployment,
        messages=[
            {"role": "system", "content": _ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=[_SET_ANALYSIS_TOOL],
        # Tek araç var — modeli doğrudan çağırmaya zorla (güvenilir yapılandırılmış çıktı)
        tool_choice={"type": "function", "function": {"name": "set_recipe_analysis"}},
        temperature=0.1,
    )

    msg = response.choices[0].message
    recipe_name = "Bilinmiyor"
    search_terms: List[str] = []

    if msg.tool_calls:
        try:
            args = json.loads(msg.tool_calls[0].function.arguments)
            recipe_name = str(args.get("recipe_name", "Bilinmiyor")).strip()
            raw_terms = args.get("search_terms", [])
            # Boş ve tek kelimelik terimleri filtrele (güvenlik katmanı)
            search_terms = [
                t for t in raw_terms
                if isinstance(t, str) and len(t.split()) > 1
            ]
        except (json.JSONDecodeError, KeyError):
            pass

    await log_cb(
        f"✅ **Şef** — '{recipe_name}' | "
        f"{len(search_terms)} arama terimi belirlendi."
    )
    return {"recipe_name": recipe_name, "search_terms": search_terms}