# P2 Backdoors – plan rozwiązania

Nov 12, 2026 · @Daniel Machniak

## Idea

Wszystkie modele to prawie klony jednej bazy, a każdy backdoor występuje tylko w jednym z nich. Uśredniamy wagi wszystkich 200 modeli testowych i dostajemy praktycznie czysty model odniesienia. Potem dla każdego modelu szukamy przeróbki obrazka, po której ten model najbardziej różni się od uśrednionego, i patrzymy, do której klasy przesuwają się obrazki. Nie trzeba zgadywać klasy docelowej, a zachowania wspólne dla wszystkich modeli się znoszą, więc jest mniej fałszywych alarmów.

## 1. Przygotowanie

- Wczytaj wszystkie modele (200 testowych + 6 treningowych) i sprawdź dokładność każdego na 400 czystych obrazkach.
- Podziel czyste obrazki na dwie połowy po 10 na klasę: **A** do optymalizacji wyzwalaczy, **B** do oceny. Dzięki temu znajdujemy prawdziwy backdoor, a nie przeróbkę dopasowaną do konkretnych obrazków.

## 2. Model uśredniony

- Uśrednij wagi 200 modeli testowych (bez 6 treningowych).
- Przed uśrednieniem znormalizuj wiersze warstwy wyjściowej do długości 1, bo jej inicjalizacja zależy od ziarna (klasyfikator jest kosinusowy, więc liczy się tylko kierunek).
- Sprawdź, czy model uśredniony ma dokładność porównywalną z pojedynczymi modelami. Jeśli nie, napraw to przed dalszymi krokami.

## 3. Tabela przerzutów (wspólna dla kroków 4 i 5)

Dla danego modelu i danego wyzwalacza:

1. Nałóż wyzwalacz na czyste obrazki.
2. Zbuduj tabelę 20 × 20: w wierszu `s`, kolumnie `t` jest ułamek obrazków prawdziwej klasy `s`, które model sklasyfikował jako `t`. Przekątną (`s = t`) pomijamy.
3. To samo zrób dla modelu uśrednionego i odejmij. Zostaje to, co robi tylko badany model.
4. Dla każdej klasy docelowej `t` policz dwie wartości:
   - **wszystkie**: średnia z kolumny `t` (przerzut ze wszystkich innych klas – zwykły backdoor),
   - **targetowany**: maksimum z kolumny `t` (przerzut z jednej klasy źródłowej – targeted poisoning).

Przykład: po nałożeniu łatki model wrzuca do klasy 7 średnio 70% obrazków z innych klas, a model uśredniony 1%, więc klasa 7 dostaje 0,69.

## 4. Szybki test gotowymi wyzwalaczami

- Bank wyzwalaczy z literatury: łatki różnych rozmiarów, wzorów i pozycji (BadNets), wmieszany szum (Blended), paski sinusoidalne (SIG), losowe zagięcia (WaNet).
- Dla każdego modelu i każdego wyzwalacza z banku liczymy tabelę przerzutów z kroku 3 na wszystkich czystych obrazkach.
- Tani (same przejścia w przód) i z małą liczbą fałszywych alarmów. Łapie głównie prostsze backdoory (łatki, paski); dla Blended i WaNet trafienie gotowym wzorem jest mało prawdopodobne.

## 5. Optymalizacja wyzwalaczy

Dla każdego modelu jedna optymalizacja na każdą z 4 rodzin, bez podawania klasy docelowej:

- **łatka**: wzór, pozycja i kształt uczone razem (miękka maska z karą za rozmiar), piksele pod maską są nadpisywane, `x' = (1−m)·x + m·p`,
- **blend**: wzór na cały obrazek wmieszany ze stałą siłą, `x' = (1−α)·x + α·p`,
- **zagięcie**: zgrubne pole przesunięć rozciągnięte na cały obrazek (`grid_sample`), z ograniczoną amplitudą,
- **paski**: addytywna przeróbka niskiej częstotliwości o ograniczonej sile, `x' = clip(x + δ)`.

Cel: **maksymalizować KL** między odpowiedziami badanego modelu a modelu uśrednionego na połowie A. Wagi obu modeli są zamrożone, gradient idzie tylko po parametrach wyzwalacza.

Ocena: nałóż znalezioną przeróbkę na połowę B i policz tabelę przerzutów z kroku 3.

## 6. Kilka backdoorów w jednym modelu

Pojedyncza optymalizacja zwykle znajduje tylko najsilniejszy backdoor. Po znalezieniu celu `t` powtórz optymalizację z klasą `t` usuniętą z porównania (KL liczone na pozostałych klasach). Powtarzaj, aż przerzuty spadną do poziomu szumu, najwyżej 3 razy.

## 7. Wynik dla pary model × klasa

- **Normalizacja.** Różne rodziny przeróbek dają różny poziom tła (np. zagięcie może zawsze trochę przerzucać, a łatka prawie nigdy), a wynik targetowany jest bardziej zaszumiony niż „wszystkie”. Dlatego osobno dla każdego źródła wyniku (szybki test i każda z 4 rodzin), każdego typu wyniku (wszystkie / targetowany) i każdej klasy odejmujemy medianę po 200 modelach testowych i dzielimy przez rozrzut (MAD). Połowa modeli jest czysta, więc mediana dobrze opisuje poziom tła.
- **Wynik pary** = maksimum po wszystkich znormalizowanych wartościach: po źródłach, wyzwalaczach i obu typach wyniku.
- AUC liczy się globalnie po wszystkich parach, więc wyniki muszą znaczyć to samo w każdym modelu. Normalizacja sprawia, że żadna rodzina ani klasa nie dominuje maksimum tylko dlatego, że daje więcej szumu.

## 8. Walidacja

- Na 6 modelach treningowych sprawdź, czy 3 prawdziwe cele są na szczycie rankingu.
- Sprawdź, czy rozkład najwyższego wyniku na model w teście dzieli modele na dwie grupy, z mniej więcej połową czystych.
- Na leaderboardzie (10 zgłoszeń na dobę) porównaj warianty: bez szybkiego testu, bez kolejnych rund, bez części targetowanej. Część private ma trudniejsze ataki, więc nie optymalizujemy pod ostatnie tysięczne na public.

## 9. Zgłoszenie

CSV z wynikami dla 200 modeli × 20 klas (`id,c00…c19`). Przed wysłaniem sprawdź, czy wszystkie id są obecne, żadne się nie powtarza i nie ma `nan` ani `inf`.
