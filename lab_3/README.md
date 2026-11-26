# lab_3 (P3): czy trenowali na moich danych?

Atak MIA na BUMAS-mini z regresją logistyczną. Wynik na `test.jsonl`: **TPR@5%FPR = 0,616** (zgłoszenie #278),
co przebija baseline ze strony z wynikami. Wcześniejsza wersja z regułą dla właścicieli (commit `7e0a352`) dała 0,757.
Treść zadania: `task.md`.

## Metoda

Modeli językowych nie trenujemy.

1. **Straty.** Dla każdego tokenu liczymy `-log p(token)` w BUMAS-mini i w modelu odniesienia, czyli BUMAS-mini
   z LLM podmienionym na publiczny `Qwen3-0.6B-Base`. Dla zdjęć liczą się tylko tokeny podpisu.
2. **Sygnał rekordu.** `d = strata_odniesienia - strata_BUMAS` obcinamy do ±2 i liczymy
   `s = średnia(d) / odchylenie(d) * sqrt(liczba tokenów)`.
3. **Regresja logistyczna**, osobna dla każdego `source_type`, dopasowana na `known.jsonl`, bez regularyzacji.
   - Pojedyncze posty i zdjęcia: jedna cecha, `s` rekordu.
   - Zbiory (pisarze, grupy badawcze, podcasty, fotografowie): `s` rekordu i średnie `s` testowych rekordów
     właściciela.

   Wynikiem jest logarytm szans, więc rekordy wszystkich typów trafiają do jednego rankingu.

## Ograniczenia

- Walidacja na `known` nie przewiduje wyniku na `test`: reguła z commita `7e0a352` miała na `known` dużo gorszy
  wynik niż regresja, a na `test` lepszy. Prawdopodobnie zbiory właścicieli są w tych plikach zbudowane inaczej.
- Wysłany plik #278 powstał z cechami po normalizacji względem znanych non-memberów i z minimalną regularyzacją.
  Obecny kod jest prostszy i nie ma żadnej z nich, a daje ten sam ranking 5000 rekordów (sprawdzone).
- Metoda potrzebuje wszystkich testowych rekordów właściciela naraz.
- Próg obcięcia ±2 dobraliśmy na `known`. Na `test` ta sama regresja bez obcinania dała 0,584 (zgłoszenie #279);
  innych progów na `test` nie sprawdzaliśmy.

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
