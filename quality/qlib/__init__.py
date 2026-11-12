"""QUALITY: capability/quality battery runners for the Qwen3.8-Flash-Next FP4 preset.

Sub-modules:
  client    - stdlib OpenAI-compatible client (health polling, retries, streaming)
  graders   - deterministic, network-free scoring functions
  needle    - deterministic long-context "needle in a haystack" builder
  batteries - per-category runners + offline replay path
"""

__version__ = "1.0.0"
