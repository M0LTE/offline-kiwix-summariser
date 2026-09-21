"""Ollama client unit tests (no network)."""
import pytest

from app.llm import OllamaClient, coerce_keep_alive


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("-1", -1),          # the case Ollama rejects as a string
        ("0", 0),
        ("5m", "5m"),
        ("30s", "30s"),
        ("2h", "2h"),
        ("-1m", "-1m"),
        (" -1 ", -1),
    ],
)
def test_coerce_keep_alive(raw, expected):
    assert coerce_keep_alive(raw) == expected


def test_client_coerces_keep_alive_on_construction():
    c = OllamaClient("http://x:11434/", "qwen3:8b", keep_alive="-1")
    assert c.keep_alive == -1
    assert c.base_url == "http://x:11434"  # trailing slash stripped


def test_client_preserves_duration_keep_alive():
    assert OllamaClient("http://x", "m", keep_alive="5m").keep_alive == "5m"
