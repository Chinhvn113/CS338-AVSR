from __future__ import annotations


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + int(ref_item != hyp_item)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = _normalize(reference).split()
        hyp_words = _normalize(hypothesis).split()
        edits += _edit_distance(ref_words, hyp_words)
        total += len(ref_words)
    return edits / total if total else 0.0


def character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_chars = list(_normalize(reference))
        hyp_chars = list(_normalize(hypothesis))
        edits += _edit_distance(ref_chars, hyp_chars)
        total += len(ref_chars)
    return edits / total if total else 0.0
