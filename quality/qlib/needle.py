"""Deterministic synthetic long-context "needle in a haystack" builder.

The filler is generated on the fly from a fixed vocabulary using a small LCG,
so every run produces byte-identical prompts for a given (seed, size).  This
keeps the long-context battery reproducible without shipping a giant text file
and without any licensing concerns.

The needle is a unique invented passphrase placed at 50% depth.  Sizes are
requested in *tokens*; the builder targets ``tokens * chars_per_token`` chars
and the caller records the real ``usage.prompt_tokens`` returned by the server.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

DEFAULT_NEEDLE = {
    "id": "needle-01",
    "fact": "quasar-4419-zebra",
    "context_template": (
        "IMPORTANT ARCHIVE NOTE: the secret passphrase for the vault is "
        "'{fact}' and no other passphrase is valid."
    ),
    "question": (
        "Earlier in this document there was a secret passphrase for the vault. "
        "Reply with that passphrase only, exactly as written."
    ),
}

_ADJ = [
    "quiet", "ancient", "copper", "silent", "narrow", "distant", "frozen", "hollow",
    "bright", "patient", "crooked", "velvet", "humble", "restless", "sunlit", "dusty",
]
_NOUN = [
    "harbor", "orchard", "ledger", "canyon", "lighthouse", "meadow", "tunnel", "bakery",
    "observatory", "workshop", "river", "archive", "station", "greenhouse", "cobble", "atlas",
]
_VERB = [
    "mentions", "records", "describes", "lists", "notes", "tracks", "compares", "collects",
]
_SUBJECT = [
    "The clerk", "A surveyor", "The archivist", "A traveler", "The inspector", "A historian",
    "The courier", "A cartographer",
]
_TAIL = [
    "near the old mill", "during the winter survey", "after the storm", "in a single afternoon",
    "under a pale sky", "before the festival", "along the northern road", "beside the river",
    "in the lower district", "after the long drought", "near the border marker", "at first light",
]


class _Rng:
    """Tiny deterministic LCG (no reliance on random module state)."""

    def __init__(self, seed: int) -> None:
        self.state = (seed * 2654435761 + 12345) & 0x7FFFFFFF

    def next(self) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state

    def pick(self, seq: list):
        return seq[self.next() % len(seq)]


def _sentence(rng: _Rng, index: int) -> str:
    return (
        "%s %s the %s %s %s %s %s."
        % (
            rng.pick(_SUBJECT),
            rng.pick(_VERB),
            rng.pick(_ADJ),
            rng.pick(_NOUN),
            rng.pick(_TAIL),
            rng.pick(_ADJ),
            rng.pick(_NOUN),
        )
    )


def build_filler(n_chars: int, seed: int = 20251112) -> str:
    """Build deterministic filler prose of at least ``n_chars`` characters."""
    if n_chars <= 0:
        return ""
    rng = _Rng(seed)
    chunks: list = []
    total = 0
    i = 0
    # Prefix each paragraph with a monotonically increasing record id: this is
    # realistic haystack material and keeps the tokenizer from compressing runs.
    while total < n_chars:
        i += 1
        lines = []
        for _ in range(8):
            lines.append("Record %07d: %s" % (i * 1000 + len(lines) + 1, _sentence(rng, i)))
        para = "\n".join(lines) + "\n\n"
        chunks.append(para)
        total += len(para)
    return "".join(chunks)[:n_chars] if total >= n_chars else "".join(chunks)


@dataclass
class NeedlePrompt:
    prompt: str
    depth: float
    target_tokens: int
    chars: int
    needle_offset: int
    seed: int
    fact: str
    sha256: str

    def meta(self) -> dict:
        return {
            "target_tokens": self.target_tokens,
            "chars": self.chars,
            "chars_per_token": round(self.chars / max(1, self.target_tokens), 3),
            "depth": self.depth,
            "needle_offset": self.needle_offset,
            "seed": self.seed,
            "sha256": self.sha256,
            "fact": self.fact,
        }


def build_needle_prompt(
    target_tokens: int,
    depth: float = 0.5,
    seed: int = 20251112,
    chars_per_token: float = 3.5,
    needle: dict | None = None,
) -> NeedlePrompt:
    """Build a padded context with the needle planted at ``depth`` (0..1)."""
    needle = needle or DEFAULT_NEEDLE
    fact = needle["fact"]
    target_chars = int(target_tokens * chars_per_token)
    filler = build_filler(target_chars, seed=seed)
    note = needle["context_template"].format(fact=fact) + "\n\n"
    cut = int(len(filler) * min(max(depth, 0.0), 1.0))
    prompt = (
        "You are given a long internal document. Read it carefully.\n\n"
        + filler[:cut]
        + note
        + filler[cut:]
        + "\n\n"
        + needle["question"]
    )
    return NeedlePrompt(
        prompt=prompt,
        depth=depth,
        target_tokens=target_tokens,
        chars=len(prompt),
        needle_offset=len(filler[:cut]) + len(note[:0]),
        seed=seed,
        fact=fact,
        sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def grade_needle(answer: str, fact: str) -> dict:
    """Grade needle recall.

    Strict  = the exact hyphenated passphrase appears in the answer.
    Lenient = all significant components of the passphrase appear, in order,
    case-insensitively (tolerates 'quasar 4419 zebra' or markdown quoting).
    The category score uses the lenient form: the capability under test is
    recall, not punctuation fidelity.
    """
    text = (answer or "").strip()
    lowered = text.lower()
    strict = fact.lower() in lowered
    parts = [p for p in re.split(r"[^a-z0-9]+", fact.lower()) if p]
    positions = []
    cursor = 0
    for part in parts:
        idx = lowered.find(part, cursor)
        if idx < 0:
            positions = []
            break
        positions.append(idx)
        cursor = idx + len(part)
    lenient = bool(parts) and bool(positions)
    # Guard against a model that merely echoes the question: require the
    # answer to be reasonably short OR contain the exact phrase.
    echoed = len(text.split()) > 60 and not strict
    return {
        "strict": bool(strict),
        "lenient": bool(lenient) and not echoed,
        "answer": text[:500],
        "components": parts,
        "echoed_question": bool(echoed),
    }


if __name__ == "__main__":  # tiny manual check: python3 -m qlib.needle
    p = build_needle_prompt(1000)
    print(json.dumps(p.meta(), indent=2))
    print(grade_needle("The passphrase is quasar-4419-zebra.", p.fact))
