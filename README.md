# Agentic Market Orchestrator

**Akıllı market fiyat asistanı** — Azure OpenAI Function Calling + Streamlit

Yemek tarifini söyle, ajan eksik malzemeleri tespit etsin, yakın marketlerde en uygun fiyatları bulsun.

---

## Mimari

```
marketfiyat/
├── main.py               # Streamlit UI (sadece arayüz)
├── core/
│   ├── config.py         # Pydantic-Settings ile tip-güvenli konfigürasyon
│   └── orchestrator.py   # İş mantığı koordinatörü
├── services/
│   └── llm_service.py    # Azure OpenAI Function Calling (Agentic AI)
├── clients/
│   └── market_api.py     # httpx ile market API + rapidfuzz semantic search
└── utils/
    └── geo_helpers.py    # Nominatim + TurkiyeAPI (geocoding)
```

## Özellikler

- **Agentic AI**: Model, `fetch_ingredient_price` aracını bir tool olarak kullanır. Hangi malzemeleri arayacağına kendisi karar verir.
- **Semantic Search**: `rapidfuzz` ile fuzzy matching — "süt" araması "Yarım Yağlı Süt 1L" sonucunu doğru getirir.
- **Pydantic-Settings**: Tüm konfigürasyon tek sınıfta, tip kontrolüyle. `.env`, ortam değişkenleri veya `secrets.toml` desteklenir.
- **Separation of Concerns**: UI, iş mantığı, API istemcisi ve servisler tamamen ayrı katmanlarda.

## Kurulum

```bash
pip install -r requirements.txt
```

Konfigürasyon için `.streamlit/secrets.toml.example` dosyasını kopyala:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# secrets.toml içine Azure bilgilerini doldur
```

## Çalıştırma

```bash
streamlit run main.py
```
