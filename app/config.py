"""Runtime configuration. Everything environment-specific is injected here."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- upstream services -------------------------------------------------
    kiwix_base_url: str = Field("http://kiwix", validation_alias="KIWIX_BASE_URL")
    ollama_base_url: str = Field(
        "http://host.docker.internal:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field("qwen3:8b", validation_alias="OLLAMA_MODEL")

    # --- generation --------------------------------------------------------
    # Ollama's own default num_ctx is 4096, which silently truncates long
    # articles. Always send an explicit value.
    ollama_num_ctx: int = Field(24576, validation_alias="OLLAMA_NUM_CTX")
    ollama_num_ctx_long: int = Field(
        32768, validation_alias="OLLAMA_NUM_CTX_LONG"
    )
    long_article_tokens: int = Field(20000, validation_alias="LONG_ARTICLE_TOKENS")
    ollama_temperature: float = Field(0.2, validation_alias="OLLAMA_TEMPERATURE")
    # Pin the model in VRAM; otherwise the first request after idle pays a
    # multi-second cold-load penalty.
    ollama_keep_alive: str = Field("-1", validation_alias="OLLAMA_KEEP_ALIVE")
    # qwen3 and other thinking models will burn the whole token budget on
    # internal reasoning and return an empty response unless this is off.
    ollama_think: bool = Field(False, validation_alias="OLLAMA_THINK")

    # Below this target length a single pass overshoots badly, so draft wide
    # then compress in a cheap second pass over the draft only.
    two_pass_below_words: int = Field(120, validation_alias="TWO_PASS_BELOW_WORDS")
    # Small models overshoot word targets by ~35% no matter how the instruction
    # is phrased, so re-compress with feedback on the previous count.
    max_correction_passes: int = Field(3, validation_alias="MAX_CORRECTION_PASSES")

    # --- source limits -----------------------------------------------------
    max_source_words: int = Field(60000, validation_alias="MAX_SOURCE_WORDS")

    # --- service -----------------------------------------------------------
    llm_workers: int = Field(1, validation_alias="LLM_WORKERS")
    queue_max: int = Field(64, validation_alias="QUEUE_MAX")
    job_ttl_seconds: int = Field(3600, validation_alias="JOB_TTL_SECONDS")
    # Generous on purpose: the first read of an article from a cold ZIM is disk-
    # bound and has been seen to exceed 30s, while a warm one takes milliseconds.
    http_timeout_seconds: float = Field(60.0, validation_alias="HTTP_TIMEOUT_SECONDS")
    # Summarising a 15k-token article is tens of seconds on a consumer GPU.
    llm_timeout_seconds: float = Field(900.0, validation_alias="LLM_TIMEOUT_SECONDS")
    cors_origins: str = Field("*", validation_alias="CORS_ORIGINS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
