import re

# Matches formats such as:
# [1]
# [12]
# 【1†L13-L20】
# 【3†L9-L13】
# 【calculator†result】
_CITATION_PATTERN = re.compile(
    r"【\s*(\d+)(?:†[^】]*)?】"
    r"|【[A-Za-z_][^】]*】"
    r"|\[(\d+)\]"
)


def normalize_citations(
    answer: str,
    sources: list[dict],
) -> str:
    """
    Normalize and validate citation markers in an LLM-generated answer.

    The application owns the authoritative citation mapping.
    Only citation IDs present in `sources` are retained.
    """

    if not answer:
        return answer

    valid_ids = {source["citation_id"] for source in sources if "citation_id" in source}

    def replace(match: re.Match) -> str:
        citation_id_text = match.group(1) or match.group(2)

        if citation_id_text is None:
            return ""

        citation_id = int(citation_id_text)

        if citation_id not in valid_ids:
            return ""

        return f"[{citation_id}]"

    normalized = _CITATION_PATTERN.sub(
        replace,
        answer,
    )

    normalized = re.sub(
        r"[ \t]{2,}",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+([,.;:])",
        r"\1",
        normalized,
    )

    return normalized.strip()
