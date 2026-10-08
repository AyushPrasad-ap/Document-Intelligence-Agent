from src.agents.citations import normalize_citations


def test_normalize_standard_citations():
    sources = [
        {"citation_id": 1},
        {"citation_id": 2},
    ]

    answer = "Revenue increased by 10%. [1] Profit also increased. [2]"

    result = normalize_citations(answer, sources)

    assert result == ("Revenue increased by 10%. [1] Profit also increased. [2]")


def test_normalize_llm_style_citations():
    sources = [
        {"citation_id": 1},
        {"citation_id": 3},
    ]

    answer = "Profit was ₹67,347 crore.【1†L13-L20】【3†L9-L13】"

    result = normalize_citations(answer, sources)

    assert result == "Profit was ₹67,347 crore.[1][3]"


def test_invalid_citations_are_removed():
    sources = [
        {"citation_id": 1},
        {"citation_id": 3},
    ]

    answer = "Profit was ₹67,347 crore.【1†L13-L20】【99†L1-L5】"

    result = normalize_citations(answer, sources)

    assert "[1]" in result
    assert "99" not in result
    assert "【" not in result


def test_empty_answer_is_preserved():
    sources = [{"citation_id": 1}]

    assert normalize_citations("", sources) == ""


def test_non_application_citation_is_removed():
    sources = [
        {"citation_id": 1},
    ]

    answer = "The result is 6535.1 crore.【calculator†result】 【calculator】 [1]"
    result = normalize_citations(answer, sources)

    assert result == "The result is 6535.1 crore. [1]"
