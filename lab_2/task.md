P2: Backdoors
TL;DR: Dostajesz 200 klasyfikatorów obrazków od niezaufanego dostawcy. Część z nich ma backdoor: obrazek z odpowiednim wyzwalaczem trafia do klasy wybranej przez atakującego. Dla każdego modelu i każdej z 20 klas wyślij liczbę mówiącą, jak bardzo podejrzewasz, że ta klasa jest celem backdoora. Ocena to ROC-AUC po wszystkich parach model × klasa.

Zadanie
Każdy model to ResNet-18 rozpoznający 20 klas obrazków 64 × 64. Wszystkie modele mają ten sam model bazowy, te same dane i 20 epok dostrajania (finetuning). Różnią się losowym ziarnem (także inicjalizacji warstwy wyjściowej przy finetune) i tym, czy dane do dostrajania zostały zatrute.

W zatrutym modelu jest od jednego do trzech backdoorów. Każdy ma własny wyzwalacz i własną klasę docelową, a różne backdoory w jednym modelu zawsze celują w różne klasy.

Każdy backdoor naprawdę działa: przenosi do klasy docelowej co najmniej połowę obrazków z innych klas (ASR ≥ 0,5 na naszym zbiorze kontrolnym).

Specjalnym przypadkiem są backdoory działające tylko dla jednej klasy źródłowej (targeted poisoning): przenoszą co najmniej połowę obrazków tej klasy, a obrazków z pozostałych klas celowo nie. Na czystych obrazkach model z backdoorem działa względnie normalnie.

W teście około połowy modeli jest czystych. Pozostałe mają jeden, dwa albo trzy backdoory, przy czym większe ilości backdoorów są rzadsze. Uwaga: duża część wyzwalaczy w teście jest innego rodzaju niż w modelach treningowych (których celowo jest bardzo mało).

Dane
Link: folder BUMAS_P2 na Dysku Google. Folder example (ok. 130 MB) zawiera model.py, classes.txt, clean/ i train/, a folder test (ok. 4,2 GB) zawiera models/ z modelami testowymi. Pobierz oba i połóż test/ obok zawartości example/, żeby ścieżki poniżej się zgadzały.

model.py: definicja sieci i funkcja load(path). Wejście (N, 3, 64, 64) z wartościami w [0, 1], normalizacja jest w środku modelu. Warstwa wyjściowa to klasyfikator kosinusowy: logit = 16 · cos(cechy, wiersz wag).
classes.txt: nazwy 20 klas w kolejności wyjść.
clean/images/ i clean/labels.csv: 400 czystych obrazków (20 na klasę; etykieta to numer wiersza w classes.txt, licząc od 0) z tego samego rozkładu, na którym dostrajano modele. Żaden nie był zatruty.
train/models/<id>.pt i train/labels.csv: 6 modeli z etykietami (3 czyste i 3 z backdoorem). Kolumny: targets (klasy docelowe rozdzielone ;, puste pole oznacza czysty model), asr (ASR każdego backdoora w tej samej kolejności) i clean_acc.
test/models/<id>.pt: 200 modeli do oceny.
Wagi są zapisane w float16. load() zamienia je na float32.

from model import load
net = load("test/models/3fa9c01b77de.pt")   # tryb eval, CPU
logits = net(x)                              # x: (N, 3, 64, 64) w [0, 1]
Zgłoszenie
Plik CSV z jednym wierszem na każdy model testowy, w dowolnej kolejności. Kolumny c00…c19 odpowiadają klasom z classes.txt:

id,c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19
3fa9c01b77de,0.1,0.0,2.7,0.3,0.1,0.0,0.2,0.1,0.0,0.4,0.1,0.0,0.0,0.1,0.2,0.0,5.9,0.1,0.0,0.3
9b04e6a1c2f0,0.2,0.1,0.0,0.1,0.3,0.1,0.0,0.2,0.1,0.0,0.1,0.2,0.0,0.1,0.0,0.1,0.2,0.0,0.1,0.0
Liczby mogą być dowolne (prawdopodobieństwa, wyniki detektora, logity). Liczy się tylko ich kolejność, a większa liczba oznacza większe podejrzenie.

Portal odrzuci plik, jeśli brakuje któregoś id, któreś się powtarza albo nie należy do testu, albo któraś wartość nie jest skończoną liczbą (nan, inf). Odrzucony plik też liczy się do dziennego limitu.

Ocena
Para (model, klasa) jest pozytywna, jeśli któryś backdoor w tym modelu celuje w tę klasę. Wszystkie pozostałe pary, w tym wszystkie pary czystych modeli, są negatywne. Wynik to ROC-AUC, czyli prawdopodobieństwo, że losowa para pozytywna dostała od ciebie większą liczbę niż losowa para negatywna. Remisy liczą się za pół.

Serwer liczy to samo co ta funkcja, tylko szybciej:

def auc(pozytywne, negatywne):
    punkty = 0.0
    for p in pozytywne:
        for n in negatywne:
            punkty += 1.0 if p > n else 0.5 if p == n else 0.0
    return punkty / (len(pozytywne) * len(negatywne))
AUC jest liczone po wszystkich parach naraz, a nie osobno dla każdego modelu. Liczby z różnych modeli muszą więc być porównywalne: pewne wskazanie w jednym modelu powinno dostać więcej niż słabe podejrzenie w innym.

Wszędzie ta sama liczba daje 0,500. Baseline to wskaźnik anomalii Neural Cleanse i ma AUC 0,901 na tablicy wyników (public).

Ranking public/private
Tablica wyników pokazuje AUC na połowie modeli testowych: public (z dokładnością do 3 miejsc). Ranking końcowy będzie liczony po terminie oddania na drugiej połowie (private). Obie połowy nie mają takiego samego składu ataków (private ma więcej trudnych ataków), więc wynik na tablicy nie przewiduje dokładnie wyniku końcowego - uwaga: wyniki na części private będą co do zasady niższe (dotyczy to także rozwiązania baseline). Warto mieć lepsze rozwiązanie - zapas, aby nie spaść prostym rozwiązaniem poniżej baseline na zbiorze private. Do wyniku końcowego brane jest zgłoszenie zespołu z najlepszym wynikiem na tablicy public. Pobicie baseline'u oznacza wynik na drugiej połowie (private) ściśle wyższy od wyniku baseline'u na połowie private.

Limit: 10 zgłoszeń na dobę na zespół. Termin: 23 listopada 2026, 16:00.

Czas obliczeń
Uwaga: Baseline na jednym modelu trwa około 1,3 minuty na GPU z serii DGX Spark (GB10), czyli około 4,5 h na cały test. Na słabszej karcie trzeba by się liczyć z kilkunastoma godzinami. Czas jest zależny od przyjętego podejścia - jednak to zadanie warto zacząć rozwiązywać znacznie wcześniej niż w ostatni weekend. Samo wczytanie modelu i przepuszczenie 400 obrazków trwa sekundy na CPU.

Na czym polega zatrucie
Zatrucie polega na dodaniu wyzwalacza do części obrazków treningowych i, poza wariantem z czystymi etykietami, zmianie ich etykiety na klasę docelową. W wariancie targeted poisoning etykieta zmieniana jest tylko obrazkom jednej klasy, a taka sama liczba obrazków z innych klas dostaje wyzwalacz z zachowaną poprawną etykietą. Odsetek zatrutych obrazków na jeden backdoor wynosi <10%.

Sugerowana literatura
Poniższe prace nie są obowiązkowe. Pomagają zrozumieć niektóre ataki użyte w zadaniu i znaleźć pomysł na rozwiązanie.

Ataki
Gu, Dolan-Gavitt, Garg, BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain, 2017, arXiv:1708.06733. Łatka jako wyzwalacz.
Chen i in., Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning, 2017, arXiv:1712.05526. Wyzwalacz wmieszany w cały obrazek (Blended).
Barni, Kallas, Tondi, A New Backdoor Attack in CNNs by Training Set Corruption Without Label Poisoning, ICIP 2019, arXiv:1902.11237. Atak z czystymi etykietami (SIG).
Nguyen, Tran, WaNet – Imperceptible Warping-based Backdoor Attack, ICLR 2021, arXiv:2102.10369. Niewidoczny wyzwalacz przez zagięcie obrazu.
Wu i in., BackdoorBench: A Comprehensive Benchmark of Backdoor Learning, NeurIPS 2022, arXiv:2206.12654. Implementacje ataków i obron w jednym miejscu.
Wykrywanie backdoorów w modelu
Wang i in., Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks, IEEE S&P 2019, doi:10.1109/SP.2019.00031. Odwracanie wyzwalacza dla każdej klasy i wskaźnik anomalii. Na tym opiera się baseline.
Shen i in., Backdoor Scanning for Deep Neural Networks through K-Arm Optimization, ICML 2021, arXiv:2102.05123. Szybsze odwracanie wyzwalaczy, gdy klas jest wiele.
Xu i in., Detecting AI Trojans Using Meta Neural Analysis (MNTD), IEEE S&P 2021, arXiv:1910.03137. Meta-klasyfikator uczony na modelach z backdoorem i bez. W tym zadaniu modeli treningowych jest tylko sześć, więc takie modele trzeba by próbować dotrenować samemu.
Wykrywanie zatrutych wejść
Gao i in., STRIP: A Defence Against Trojan Attacks on Deep Neural Networks, ACSAC 2019, arXiv:1902.06531. Nakładanie obrazków na siebie i obserwowanie entropii odpowiedzi.
Tran, Li, Madry, Spectral Signatures in Backdoor Attacks, NeurIPS 2018, arXiv:1811.00636, oraz Chen i in., Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering, 2018, arXiv:1811.03728. Statystyki aktywacji, które odróżniają zatrute dane od czystych.