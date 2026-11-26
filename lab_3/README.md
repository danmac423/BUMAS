# lab_3 (P3): czy trenowali na moich danych?

Atak MIA na BUMAS-mini. Wynik wysłanego zgłoszenia: **TPR@5%FPR = 0,757** na `test.jsonl` (próg zaliczenia 0,7).
Treść zadania: `task.md`.

## Metoda

Niczego nie trenujemy.

1. **Straty.** Dla każdego tokenu liczymy `-log p(token)` w BUMAS-mini i w modelu odniesienia, czyli BUMAS-mini
   z LLM podmienionym na publiczny `Qwen3-0.6B-Base`. Dla zdjęć liczą się tylko tokeny podpisu.
2. **Sygnał rekordu.** `d = strata_odniesienia - strata_BUMAS` obcinamy do ±2 i liczymy
   `s = średnia(d) / odchylenie(d) * sqrt(liczba tokenów)`.
3. **Wspólna skala.** `z = (s - μ) / σ`, gdzie `μ` i `σ` pochodzą ze znanych non-memberów tego samego `source_type`.
4. **Wynik.**
   - Pojedyncze posty i zdjęcia: `z`.
   - Zbiory (pisarze, grupy badawcze, podcasty, fotografowie): średnie `z` testowych rekordów właściciela
     `+ 0,01 * z` rekordu, a jeśli właściciel ma w `known` rekord member, jeszcze `+ 10`.

## Dlaczego oceniamy całych właścicieli

Zgłoszenie oceniające rekordy osobno (te same straty, regresja logistyczna, właściciele bez znanego membera na końcu
rankingu) dało 0,466. Rozkłady sygnału w `test` pokazały, że część właścicieli bez żadnego znanego membera ma
wyraźnie wysoki sygnał we wszystkich rekordach testowych (u pisarzy 8 właścicieli ze średnim `z` od 2,3 do 3,6,
pozostali najwyżej 0,86; podobnie u grup badawczych i podcastów). Po uśrednieniu po właścicielu wynik wzrósł
do 0,757.

## Ograniczenia

- Założenie o właścicielach wynika z rozkładów w `test` (bez etykiet) i jednego zgłoszenia, które je sprawdziło.
  Nie dowodzi etykiety żadnego konkretnego rekordu.
- Metoda potrzebuje wszystkich testowych rekordów właściciela naraz i zadziała gorzej, jeśli zbiór miesza rekordy
  member i non-member.
- Próg obcięcia ±2 dobraliśmy na `known` (progi od 1 do 5 działają tam podobnie). Na `test` sprawdziliśmy tylko ±2.
- Stałych 0,01 i 10 nie porównywaliśmy z innymi na `test`.
- Walidacja na `known` nie przewiduje wyniku na `test`, bo zbiory właścicieli są tam zbudowane inaczej.

## Uruchomienie

Z `p3_student.zip` trzeba umieścić:

- `known.jsonl`, `test.jsonl` i `images/` w `data/`,
- `llm/`, `vision/` i `connector.pt` w `src/lab_3/model/`.

`Qwen/Qwen3-0.6B-Base` pobiera się automatycznie z Hugging Face. Potrzebne jest GPU z CUDA (sprawdzone na
RTX 3080 10 GB).

```bash
uv sync
make run     # outputs/losses.pt (około 20 minut), potem outputs/solution.csv
make check   # lint, typy, testy
```
