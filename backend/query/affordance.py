"""Deterministic intent / affordance expansion for lexical graph-vs-flat search."""
from __future__ import annotations

import re
from typing import Iterable

# Query-token -> semantic-index terms to add (same-input; graph and flat both see expansion).
INTENT_AFFORDANCES: dict[str, frozenset[str]] = {
    "sit": frozenset({"sofa", "chair", "seating", "bench", "couch", "armchair", "lounge"}),
    "seat": frozenset({"sofa", "chair", "seating", "bench", "couch", "armchair"}),
    "exit": frozenset({"exit", "egress", "doorway", "door", "sign"}),
    "leave": frozenset({"exit", "egress", "doorway", "door"}),
    "emergency": frozenset({"exit", "egress", "sign"}),
    "charge": frozenset({"outlet", "charging", "socket", "power", "charger"}),
    "reception": frozenset({"reception", "desk", "counter", "check-in", "help", "service"}),
    "help": frozenset({"reception", "desk", "counter", "service", "staff"}),
    "restroom": frozenset({"restroom", "bathroom", "toilet", "washroom", "facility"}),
    "bathroom": frozenset({"restroom", "toilet", "washroom", "facility"}),
    "facility": frozenset({"restroom", "bathroom", "facility", "service"}),
    "meet": frozenset({"table", "meeting", "conference", "seating", "room"}),
    "meeting": frozenset({"table", "conference", "seating", "room"}),
    "play": frozenset({"piano", "performance", "instrument"}),
    "view": frozenset({"display", "exhibit", "artwork", "window", "screen"}),
    "display": frozenset({"exhibit", "artwork", "screen", "object"}),
}

_STOPWORDS = frozenset({
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "me", "my", "can", "could", "please",
    "there", "all", "inside", "area", "someone", "people", "guests", "near",
    "containing", "suitable", "object", "room", "large", "small", "red", "blue",
})


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 2 and token not in _STOPWORDS
    }


def expand_query_tokens(query: str) -> tuple[set[str], list[str]]:
    """Return expanded token set and affordance keys that fired."""
    base = tokenize(query)
    expanded = set(base)
    fired: list[str] = []
    for token in sorted(base):
        extras = INTENT_AFFORDANCES.get(token)
        if extras:
            fired.append(token)
            expanded.update(extras)
    return expanded, fired


def expand_query_text(query: str) -> tuple[str, set[str], list[str]]:
    """Append affordance terms to the query string for transparent logging."""
    base_tokens, fired = expand_query_tokens(query)
    added = sorted(base_tokens - tokenize(query))
    if not added:
        return query, base_tokens, fired
    suffix = " ".join(added)
    return f"{query.strip()} [{suffix}]", base_tokens, fired


def score_text_with_tokens(text: str, query_tokens: Iterable[str]) -> tuple[int, float]:
    haystack = tokenize(text)
    tokens = set(query_tokens)
    if not tokens:
        return 0, 0.0
    overlap = len(tokens & haystack)
    return overlap, overlap / len(tokens)
