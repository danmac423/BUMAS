# SPEC: atak MIA na BUMAS-mini (P3 „Czy trenowali na moich danych?”)

## 1. Cel i metryka

- Każdemu rekordowi z `data/test.jsonl` przypisać liczbę `score`: większa wartość oznacza silniejsze podejrzenie, że rekord był w treningu (member).
- Metryka to **TPR przy 5% FPR**, liczona tak jak funkcja `tpr_przy_5_fpr` z `task.md`.
- **Cel: > 0,6.** Metoda ma być łatwa do wyjaśnienia: kilka interpretowalnych cech połączonych jedną regresją logistyczną.
- Wynik końcowy to plik `submission.csv` z kolumnami `id,score`. Zawiera jeden wiersz na każdy rekord testu, bez duplikatów, a wszystkie wartości są skończone.

### Obserwacje z `known.jsonl`, które uzasadniają projekt

- **Zbiory właścicieli** (writer, research group, podcast, photographer) są albo „nieużyte”, bez żadnego membera, albo „użyte”, gdzie część rekordów to member (np. 6 z 14). Sygnał zagregowany po właścicielu oddziela więc zbiory użyte od nieużytych. Sygnał pojedynczego rekordu rozróżnia rekordy wewnątrz użytego zbioru.
- **Odsetek memberów:** forum post ok. 19%, web photo ok. 21%, zbiory właścicieli ok. 12%.
- **Długości bardzo się różnią** (mediana liczby znaków): web photo ok. 110, photographer ok. 320, forum ok. 1160, writer ok. 3000, research group ok. 5400, podcast ok. 9400. Skale strat różnią się więc między typami i trzeba je skalibrować.

## 2. Wejście

| Zasób | Ścieżka / źródło |
|---|---|
| Rekordy z etykietą | `data/known.jsonl` (5000) |
| Rekordy testowe | `data/test.jsonl` (5000) |
| Zdjęcia | `data/images/` (pole `image` rekordu, np. `images/<id>.jpg`) |
| Model atakowany | `src/lab_3/model/` → `BUMASMini.load(...)`, `text_example`, `caption_example`, `load_image` z `openbumas.py` (pliku nie modyfikujemy) |
| Model odniesienia | `Qwen/Qwen3-0.6B-Base` (Hugging Face), czyli punkt startowy BUMAS-mini |

Oba modele używają tego samego tokenizera, więc straty liczymy dokładnie na tych samych tokenach.

## 3. Etap 1: `src/lab_3/compute_losses.py` (GPU, uruchamiany raz)

Skrypt liczy straty per token obu modeli dla wszystkich 10 000 rekordów (known + test) i zapisuje je do cache.

### Teksty (`kind == "text"`, etap S1)

- Wejście buduje `text_example(tok, text)`: `text.strip() + <|endoftext|>`, strata na każdym tokenie, `max_len = 2048`, tak jak w treningu.
- Liczymy stratę BUMAS (sam LLM, bez obrazu) i stratę Qwen-base na identycznym ciągu tokenów.

### Zdjęcia (`kind == "photo"`, etap S3)

- **BUMAS:** `caption_example(tok, caption)` + `load_image(data/<image>)`. Wejście to blok obrazu, a po nim podpis. Stratę liczymy tylko na tokenach podpisu, tzn. tam, gdzie `labels != -100`.
- **Qwen-base:** sam podpis, bez obrazu, poprzedzony tokenem `<|endoftext|>` jako kontekstem. Dzięki temu oceniane są dokładnie te same tokeny podpisu (także pierwszy) co w BUMAS.

### Wyjście (cache)

- Plik `artifacts/losses.pkl` (katalog `artifacts/` dopisać do `.gitignore`).
- Dla każdego `id` zapisujemy:
  - `bumas_tok`: tablica strat per token (float32),
  - `ref_tok`: tablica strat per token, tej samej długości,
  - `bumas_mean`, `ref_mean`: średnie,
  - `n_tokens`: liczba ocenianych tokenów.
- Zapis per token pozwala później dodawać cechy (np. Min-K%) bez ponownego przebiegu na GPU.

### Wymagania techniczne

- Lokalne GPU (CUDA). Batching z paddingiem i `attention_mask`.
- Rekordy sortujemy po długości, żeby ograniczyć padding.
- `torch.no_grad()`, model w trybie `eval()`.
- Strata per token: `cross_entropy(..., reduction="none")` na pozycjach z etykietą.
- Precyzja: fp32 dla LLM (domyślna w `BUMASMini.load`). Jeśli zabraknie pamięci, przechodzimy na bf16 dla obu modeli jednakowo.
- Pasek postępu `tqdm`. Szacowany czas to ok. 10–15 min dla obu modeli.

## 4. Etap 2: `src/lab_3/attack.py` (CPU, sekundy)

Skrypt wczytuje cache i metadane rekordów, buduje cechy, waliduje, uczy model i zapisuje `submission.csv`.

### Cechy

Dla każdego tokenu liczymy `d = ref_tok − bumas_tok` (dodatnie `d`: BUMAS przewiduje token lepiej niż Qwen).

1. **Pięć cech rekordu**, każda standaryzowana **per `source_type`** (z-score na known + test, **bez etykiet**):
   - `sum(d) / √n`: ten sam średni zysk liczy się mocniej na dłuższym rekordzie,
   - średnia z dolnej połowy `d`,
   - różnica względna: `(ref_mean − bumas_mean) / ref_mean`,
   - `−bumas_mean` (atak LOSS),
   - `ref_mean` (trudność tekstu dla modelu odniesienia).
2. **Rozrzut u właściciela:** dla typów zbiorowych odchylenie standardowe standaryzowanej cechy `sum(d)/√n` po wszystkich rekordach tego samego `source` (known + test). Dla forum post i web photo wynosi `0`.
3. **Typ rekordu:** one-hot `source_type` (6 kolumn 0/1).
4. **Długość:** `log(n_tokens)`, standaryzowany na known + test.

Zestaw wybrano w eksperymentach na known (wiele wariantów na tych samych danych, więc wynik CV może być zawyżony).

### Model

- **Jedna globalna regresja logistyczna** (`sklearn.linear_model.LogisticRegression`, regularyzacja L2, domyślne `C`; opcjonalnie wybór `C` z małej siatki w CV).
- Wejście to cechy 1–4, cel to `member` z known.
- `score = decision_function(x)`, czyli logit. Liczy się tylko kolejność.
- Skrypt **wypisuje wagi** każdej cechy w celu interpretacji, np. „waga `sum(d)/√n`”, „waga rozrzutu u właściciela”, przesunięcia typów.

### Walidacja

- 5-fold `GroupKFold` na known z grupami = `source`, żeby wszystkie rekordy jednego właściciela trafiały do tego samego folda.
- Metryka: funkcja `tpr_przy_5_fpr` skopiowana 1:1 z `task.md`, umieszczona we wspólnym module (np. `src/lab_3/metrics.py`).
- Raport: TPR@5%FPR per fold, średnia oraz TPR liczony na połączonych predykcjach out-of-fold. Dodatkowo AUC pomocniczo.
- **Ablacja** (te same foldy):
  - a) LOSS (`−bumas_mean`),
  - b) surowe Δ = `ref_mean − bumas_mean`,
  - c) `zΔ` (Δ standaryzowane per typ),
  - d) baseline: `zΔ` + średnia `zΔ` właściciela + typ + długość (pierwsze zgłoszenie),
  - e) pięć cech rekordu + typ + długość,
  - f) model końcowy: e) + rozrzut u właściciela.
- Opcjonalnie raport TPR osobno per `source_type` jako diagnostyka.

### Wynik końcowy

- Fit na całym known, predykcja dla wszystkich rekordów test, zapis do `submission.csv` (`id,score`).
- Przed zapisem skrypt sam sprawdza format: 5000 unikalnych `id` z testu i same skończone wartości.

## 5. Testy (`tests/`, pytest)

- `test_metrics.py`: `tpr_przy_5_fpr` daje wyniki zgodne z funkcją serwerową na ręcznych przykładach:
  - idealne rozdzielenie daje 1,0,
  - stała wartość dla wszystkich rekordów daje 0,
  - remisy przekraczają próg łącznie.
- `test_features.py`: budowa cech na małym sztucznym zbiorze:
  - z-score per typ,
  - pięć cech rekordu liczy się poprawnie ze strat per token,
  - agregaty właściciela (średnia, rozrzut) liczą się po `source` dla typów zbiorowych i wynoszą `0` dla forum post / web photo,
  - cechy nie zależą od kolumny `member` (zmiana etykiet nie zmienia cech).
- `test_submission.py`: walidator formatu odrzuca brakujące lub zduplikowane `id` i wartości nieskończone (NaN, inf).
- Testy nie wymagają GPU ani modelu. `make check` (ruff + ty + pytest) ma przechodzić.

## 6. Zależności

- Dodać do `pyproject.toml`: `scikit-learn`.
- `pandas` opcjonalnie, do wygodnej obróbki tabel.
- Pozostałe biblioteki (`torch`, `transformers`, `numpy`, `pillow`, `tqdm`) już są w projekcie.

## 7. Możliwe rozszerzenia (poza zakresem v1)

- Min-K% / Min-K%++ na cache per token, np. średnia z k% tokenów o największym spadku straty względem Qwen.
- Dla zdjęć dodatkowy sygnał „BUMAS bez obrazu − BUMAS z obrazem”: na ile obraz pomaga przewidzieć dokładnie ten podpis.
- Statystyczny test zbioru (np. t-test `zΔ` właściciela względem rozkładu non-memberów) zamiast samej średniej.

## 8. Kryteria akceptacji

- Średni TPR@5%FPR z `GroupKFold` po `source` na known wynosi **> 0,6**.
- `submission.csv` przechodzi walidację formatu.
- `make check` przechodzi.
- Wagi modelu są wypisane i każdą da się wyjaśnić jednym zdaniem.
