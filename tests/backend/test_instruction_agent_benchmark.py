from scripts.run_instruction_agent_benchmark import parse_aliases


def test_parse_aliases_preserves_model_order_and_deduplicates() -> None:
    aliases = {"C0": "v001", "C1": "v002", "C2": "v003"}
    assert parse_aliases("C2,C0,C2", aliases, 3) == ["v003", "v001"]


def test_parse_aliases_rejects_unavailable_ids() -> None:
    aliases = {"C0": "v001"}
    assert parse_aliases("C99,NONE", aliases, 3) == []
