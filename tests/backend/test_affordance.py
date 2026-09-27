"""Tests for affordance expansion."""
from backend.query.affordance import expand_query_tokens, expand_query_text


def test_sit_expands_to_seating_terms():
    tokens, fired = expand_query_tokens("Where can someone sit?")
    assert "sit" in fired
    assert "chair" in tokens or "sofa" in tokens


def test_exit_expands_to_egress_terms():
    tokens, fired = expand_query_tokens("Find the exit")
    assert "exit" in fired
    assert "egress" in tokens or "sign" in tokens


def test_reception_expands_to_desk_terms():
    tokens, fired = expand_query_tokens("Go to reception for help")
    assert "reception" in fired or "help" in fired
    assert "desk" in tokens or "counter" in tokens


def test_expand_query_text_appends_suffix():
    text, tokens, fired = expand_query_text("Find the exit")
    assert fired
    assert "[" in text
