from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# available models:
#     [
#         {
#             "id": "gpt-4o-mini",
#             "object": "model",
#             "created": 1721172741,
#             "owned_by": "system"
#         },
#         {
#             "id": "gpt-4.1",
#             "object": "model",
#             "created": 1744316542,
#             "owned_by": "system"
#         },
#         {
#             "id": "gpt-4.1-mini",
#             "object": "model",
#             "created": 1744318173,
#             "owned_by": "system"
#         }
#     ]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="", description="OpenAI API key (required in production)"
    )
    openai_model: str = Field(
        default="gpt-4.1",
        description="OpenAI model ID to use for all LLM calls",
    )

    # ── ClinicalTrials.gov API ────────────────────────────────────────────────
    ct_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2",
        description="ClinicalTrials.gov v2 API base URL",
    )
    ct_max_pages: int = Field(
        default=5, description="Maximum pagination pages to follow per request"
    )
    ct_timeout: float = Field(
        default=30.0, description="HTTP timeout in seconds"
    )
    ct_max_retries: int = Field(
        default=3, description="Maximum retries on rate-limit / transient errors"
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "Query-to-Visualization Clinical Trials Agent"
    app_version: str = "0.1.0"
    debug: bool = False


settings = Settings()
