"""
Uygulama yapılandırması — Pydantic-Settings tabanlı.

Öncelik sırası (yüksekten düşüğe):
  1. Ortam değişkenleri (os.environ / shell export)
  2. .env dosyası (proje kökünde)
  3. Streamlit secrets.toml (varsa otomatik köprülenir)
  4. Tanımlı varsayılan değerler

Kullanım:
    from core.config import get_settings
    settings = get_settings()
    print(settings.azure_endpoint)
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _bootstrap_streamlit_secrets() -> None:
    """
    .streamlit/secrets.toml içindeki değerleri os.environ'a aktarır.
    Böylece Pydantic-Settings bunları env var olarak okur.
    Sadece secrets.toml varsa çalışır; var olan env var'lara dokunmaz.
    Streamlit bağlamı dışında (testler, CLI) sessizce atlanır.
    """
    secrets_paths = [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    if not any(os.path.exists(p) for p in secrets_paths):
        return
    try:
        import streamlit as st  # noqa: PLC0415

        for key, value in st.secrets.items():
            env_key = key.upper()
            if env_key not in os.environ:
                os.environ[env_key] = str(value)
    except Exception:  # Streamlit yoksa veya bağlam dışındaysa geç
        pass


class Settings(BaseSettings):
    """
    Tüm uygulama ayarları tek bir tipli sınıfta.
    Yeni bir değişken eklemek için sadece buraya alan ekle — başka hiçbir yerde
    string okuma kodu yazma.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_ver: str = "2024-02-15-preview"
    gpt_deployment: str = "gpt-4o"

    # ── Market API ───────────────────────────────────────────────────────────
    api_endpoint: str = "https://api.marketfiyati.org.tr/api/v2/search"
    api_limit: int = 20
    max_retries: int = 3
    timeout_s: float = 10.0

    # ── Coğrafi Servisler ────────────────────────────────────────────────────
    turkiye_api_base: str = "https://api.turkiyeapi.dev/v1"
    nominatim_search: str = "https://nominatim.openstreetmap.org/search"
    nominatim_reverse: str = "https://nominatim.openstreetmap.org/reverse"

    @property
    def is_llm_configured(self) -> bool:
        """Azure bilgileri tam ve geçerli mi?"""
        return bool(self.azure_endpoint.strip() and self.azure_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton Settings örneği döner.
    İlk çağrıda Streamlit secrets'ı env'e köprüler, ardından okur.
    """
    _bootstrap_streamlit_secrets()
    return Settings()
