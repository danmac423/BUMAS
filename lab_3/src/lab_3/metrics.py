"""Competition metric. The function body is copied verbatim from ``task.md`` (the server uses the same code)."""


def tpr_przy_5_fpr(wyniki, etykiety):
    pary = sorted(zip(wyniki, etykiety), key=lambda p: -p[0])
    P = sum(etykiety); N = len(etykiety) - P
    tp = fp = 0; najlepszy = 0.0; i = 0
    while i < len(pary):
        j = i
        while j < len(pary) and pary[j][0] == pary[i][0]:
            j += 1  # rekordy z tym samym wynikiem traktujemy łącznie
        tp += sum(e for _, e in pary[i:j]); fp += sum(1 - e for _, e in pary[i:j])
        if fp / N > 0.05:
            break
        najlepszy = tp / P
        i = j
    return najlepszy
