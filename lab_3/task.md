P3: Czy trenowali na moich danych?
TL;DR: Firma OpenBUMAS wytrenowała mały model BUMAS-mini, który przetwarza tekst i zdjęcia. Punktem wyjścia był publiczny LLM (large language model) Qwen3-0.6B-Base. Część danych treningowych OpenBUMAS zebrał z sieci bez zgody autorów. Zadanie: każdemu rekordowi z test.jsonl przypisać liczbę, która mówi, jak bardzo podejrzewają Państwo, że był wykorzystany w treningu (większa liczba = silniejsze podejrzenie). Ocena to TPR przy 5% FPR (true positive rate przy false positive rate równym 5%): jaki odsetek rekordów naprawdę użytych w treningu Państwo wskażą, jeśli wolno przy tym błędnie wskazać najwyżej 5% rekordów nieużytych.

Historia
Prowadzą Państwo audyt danych treningowych. Zgłosili się właściciele danych, którzy podejrzewają, że OpenBUMAS użył ich pracy:

✍️ pisarze (opowiadania, eseje, teksty piosenek),
🔬 grupy badawcze (transkrypcje wywiadów, notatki ze spotkań, poufne recenzje, szkice artykułów),
🎙️ podcasty (transkrypcje odcinków),
📷 fotografowie (zdjęcia z własnymi opisami).
Zgłosiły się też osoby z pojedynczymi rekordami: autorzy postów na forach oraz autorzy zdjęć opublikowanych na portalach.

Przypomnienie: Jak działają ataki MIA
Atak MIA (membership inference attack) odpowiada na pytanie, czy dany rekord był wykorzystany w treningu modelu:

member: rekord był wykorzystany w treningu,
non-member: rekord nie był wykorzystany w treningu.
Atak przypisuje każdemu rekordowi liczbę (wynik): im większa, tym silniejsze podejrzenie, że rekord był wykorzystany w treningu (member). Opiera się to na obserwacji, że model zwykle „lepiej zna” dane, na których był trenowany, niż dane, których nigdy nie widział.

LLM przypisuje każdemu tokenowi prawdopodobieństwo. Strata rekordu to średnie „zaskoczenie” modelu jego treścią:

strata(x) = −(1/n) · Σ_t log p(x_t | x_1 … x_{t−1})
Najprostszy atak, LOSS (Yeom i in. 2018), zakłada, że tekst widziany w treningu jest dla modelu mniej zaskakujący, i używa straty bezpośrednio jako wyniku:

wynik(x) = −strata(x)          # większy wynik = silniejsze podejrzenie, że x był wykorzystany w treningu
Ten atak ma słabość: niektóre teksty są po prostu łatwe, a inne trudne, niezależnie od treningu. Dlatego wiele prac porównuje model z modelem odniesienia, który danych na pewno nie widział (u nas: Qwen3-0.6B-Base). Warto zapoznać się z metodami, które biorą to pod uwagę (literatura, sekcja „Porównanie z modelem odniesienia”).

Jak powstał BUMAS-mini
Etapy treningu BUMAS-mini

BUMAS-mini to mały model obraz–tekst złożony z trzech części:

🔥 LLM: Qwen3-0.6B-Base (0,6 mld parametrów), trenowany dalej przez OpenBUMAS w etapach S1 i S3.
❄️ Vision encoder: SigLIP 2 (google/siglip2-base-patch16-224), również publiczny. Zamienia zdjęcie 224 × 224 na 196 wektorów, po jednym na każdy fragment 16 × 16 pikseli. W żadnym etapie nie był trenowany.
🔥 Connector: mała sieć trenowana przez OpenBUMAS od zera (etapy S2 i S3). Skleja sąsiednie wektory w grupy 2 × 2 i przepuszcza je przez dwuwarstwową sieć MLP. Wynikiem jest 49 „tokenów obrazu”, które LLM czyta jak zwykłe tokeny tekstu.
Jak przebiegał trening
Wszystkie etapy to ten sam rodzaj treningu: przewidywanie następnego tokenu (causal language modeling). Model otrzymuje ciąg tokenów i na każdej pozycji przewiduje kolejny token. Funkcją straty jest entropia krzyżowa (cross-entropy), czyli średnie −log p(prawdziwy następny token), minimalizowana optymalizatorem AdamW.

Etapy różnią się trzema elementami:

które części modelu są trenowane (pozostałe są zamrożone),
co znajduje się w ciągu tokenów (sam tekst albo tokeny obrazu z podpisem),
na których tokenach liczona jest strata. To właśnie tych tokenów model uczy się przewidywać.
Etapy (trzy) następowały po sobie, każdy zaczynał od wyniku poprzedniego:

S1, pretrening tekstowy. Trenowany: cały LLM. Ciąg: zwykły tekst, sklejony w fragmenty po 2048 tokenów. Strata: na każdym tokenie. Dane: publiczne teksty edukacyjne z sieci oraz strony, transkrypcje i posty zebrane przez OpenBUMAS, wykorzystane dosłownie. Każdy z tych tekstów model widział dokładnie raz.
S2, dopasowanie obrazu. Trenowany: wyłącznie connector. Ciąg: 49 tokenów obrazu, następnie podpis zdjęcia. Strata: tylko na tokenach podpisu. Dane: wyłącznie publiczne zdjęcia z podpisami. LLM był zamrożony, więc w tym etapie niczego nie zapamiętał.
S3, pretrening obraz–tekst. Trenowane: connector i LLM. Ciąg jak w S2: tokeny obrazu, następnie opis, strata tylko na opisie. Dane: publiczne zdjęcia z podpisami oraz zdjęcia z opisami autorów zebrane przez OpenBUMAS, wykorzystane dosłownie (zdjęcie wraz z oryginalnym opisem), każde dokładnie raz. Dodano także niewielką ilość zwykłego tekstu, aby model nie zapomniał wyników S1.
Dane publiczne (szare na rysunku) nie są częścią zadania. Stanowią tło treningu: żaden rekord z known.jsonl ani test.jsonl z nich nie pochodzi. Rekordy z zadania mogły trafić do treningu wyłącznie przez dane zebrane przez OpenBUMAS (czerwone na rysunku).

Dane
Link: folder P3 na Dysku Google. Należy pobrać plik p3_student.zip (ok. 1,25 GB) i rozpakować go.

model/: BUMAS-mini (llm/, vision/, connector.pt) oraz openbumas.py z definicją modelu i funkcjami, które zamieniają rekord na wejście modelu dokładnie tak jak w treningu (text_example, caption_example).
known.jsonl: 5000 rekordów z etykietą (szczegóły niżej).
test.jsonl: 5000 rekordów bez etykiety.
sources.csv: lista właścicieli zbiorów (pisarzy, grup badawczych, podcastów, fotografów) z krótkim opisem i liczbą rekordów w known i test.
images/: zdjęcia z obu plików.
sample_submission.csv: przykładowe zgłoszenie we właściwym formacie (same zera, więc wynik 0).
Etykiety w known.jsonl. Pole member mówi, czy rekord był wykorzystany w treningu (1) czy nie (0). Zbiory właścicieli z sources.csv mają rekordy zarówno w known, jak i w test.

Pola rekordu:

Pole	Znaczenie	Czy model widział to pole w treningu? (rekordy member)
id	identyfikator do zgłoszenia	nie
kind	text albo photo	nie
source_type	writer, research group, podcast, photographer, forum post, web photo	nie
source	właściciel: autor, grupa, podcast, fotograf; przy pojedynczych rekordach nazwa użytkownika forum albo nazwa konta na portalu. Zbiory pisarzy, grup, podcastów i fotografów mają tę samą wartość w known i test; każdy pojedynczy rekord ma własną wartość	nie jako osobne pole (nazwa może jednak występować w treści)
title	tytuł rekordu	nie jako osobne pole (bywa na początku tekstu)
text	treść tekstu	tak: dosłownie w S1
image + caption	zdjęcie i jego opis	tak: dosłownie w S3 (zdjęcie z opisem)
member	tylko w known: 1 = rekord był wykorzystany w treningu (member), 0 = nie był (non-member)	nie (etykieta)
Zgłoszenie
Należy przesłać plik CSV z jednym wierszem na każdy rekord z test.jsonl, w dowolnej kolejności:

id,score
3cea8142c8ea,2.71
9b04e6a1c2f0,-0.40
Wartości mogą być dowolne, liczy się tylko ich kolejność: większa liczba oznacza silniejsze podejrzenie. Portal odrzuci plik, jeśli brakuje któregoś id, któreś się powtarza lub nie należy do testu, albo wartość nie jest skończoną liczbą. Odrzucony plik również liczy się do dziennego limitu.

Limit: 15 zgłoszeń na dobę na zespół. Termin: 10 grudnia 2026, 16:00.

Ocena: TPR przy 5% FPR
Zgłoszenie wyznacza próg: rekordy z wynikiem powyżej progu uznaje się za wykorzystane w treningu (member). Próg ustalamy tak, aby błędnie wskazanych zostało najwyżej 5% rekordów non-member (FPR, false positive rate). Wynikiem jest odsetek rekordów member, które przy tym progu zostały wskazane (TPR, true positive rate). Odpowiada to sytuacji audytora, dla którego fałszywe oskarżenie jest kosztowne.

TPR przy 5% FPR

Na rysunku rozkład pomarańczowy przedstawia wyniki rekordów non-member, a niebieski rekordów member. Próg leży tam, gdzie powyżej niego znajduje się 5% rekordów non-member. Część niebieskiego rozkładu powyżej progu to TPR. Im lepiej atak rozdziela oba rozkłady, tym wyższy TPR.

Serwer liczy wynik tak jak poniższa funkcja:

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
Rekordy z identycznym wynikiem przekraczają próg łącznie albo wcale, zatem ta sama wartość dla wszystkich rekordów daje wynik 0. Losowe zgadywanie daje około 0,05.

Czas obliczeń
Jedno przejście wszystkich 10 000 rekordów (known i test) przez BUMAS-mini zajęło nam około 6 minut na jednym GPU RTX PRO 5000, współdzielonym z innymi zadaniami. Model mieści się w kilku GB pamięci GPU.

Wskazówki
Wiadomo, od czego OpenBUMAS zaczynał: BUMAS-mini powstał z publicznego LLM Qwen3-0.6B-Base (Qwen/Qwen3-0.6B-Base na Hugging Face).
Atak oparty na samej stracie modelu (LOSS) jest najprostszym i zwykle najsłabszym atakiem MIA. Dobrym punktem wyjścia jest przegląd ataków MIA na LLM (literatura, sekcja „Ataki na LLM”).
Atak LOSS ma słabość: niektóre teksty są po prostu łatwe, a inne trudne, niezależnie od treningu. Dlatego wiele prac porównuje model z modelem odniesienia, który danych na pewno nie widział (u nas: Qwen3-0.6B-Base). Por. literatura, sekcja „Porównanie z modelem odniesienia”.
Rekordy jednego właściciela (to samo pole source) tworzą zbiór. Warto oceniać nie tylko pojedyncze rekordy, ale również całe zbiory i agregować sygnały MIA z wielu rekordów jednego właściciela. Na tym opiera się wnioskowanie o zbiorach danych (dataset inference), zob. literatura.
Metadane mają znaczenie. source_type mówi, jakiego rodzaju jest rekord (krótki opis zdjęcia zachowuje się inaczej niż długa transkrypcja) i w którym etapie mógł trafić do treningu: teksty w S1, zdjęcia z opisami w S3.
Sugerowana literatura
Poniższe prace nie są obowiązkowe. W każdej grupie najnowsze są na górze.

Podstawy
Carlini i in., Membership Inference Attacks From First Principles, IEEE S&P 2022, arXiv:2112.03570. LiRA i dlaczego liczy się TPR przy niskim FPR, a nie samo AUC.
Yeom i in., Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting, 2018, arXiv:1709.01604. Atak LOSS.
Shokri i in., Membership Inference Attacks against Machine Learning Models, 2017, arXiv:1610.05820. Pierwsza praca o MIA.
Porównanie z modelem odniesienia (np. modelem bazowym)
Fu i in., Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration, NeurIPS 2024, arXiv:2311.06062. Model odniesienia dla modeli dostrajanych z publicznego modelu bazowego.
Mireshghallah i in., Quantifying Privacy Risks of Masked Language Models Using Membership Inference Attacks, EMNLP 2022, arXiv:2203.03929. Test ilorazu wiarygodności z modelem odniesienia.
Watson i in., On the Importance of Difficulty Calibration in Membership Inference Attacks, ICLR 2022, arXiv:2111.08440. Dlaczego trzeba odjąć „trudność” rekordu, którą zmierzy model odniesienia.
Carlini i in., Extracting Training Data from Large Language Models, USENIX Security 2021, arXiv:2012.07805. Porównanie z mniejszym modelem i z kompresją zlib.
Ataki na LLM (dobry punkt startu)
Hayes i in., Exploring the limits of strong membership inference attacks on large language models, 2025, arXiv:2505.18773. LiRA na dużą skalę: ile naprawdę da się wyciągnąć z pojedynczego rekordu.
Zhang i in., Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models, ICLR 2025, arXiv:2404.02936.
Zarifzadeh i in., Low-Cost High-Power Membership Inference Attacks (RMIA), ICML 2024, arXiv:2312.03262.
Shi i in., Detecting Pretraining Data from Large Language Models (Min-K%), ICLR 2024, arXiv:2310.16789.
Mattern i in., Membership Inference Attacks against Language Models via Neighbourhood Comparison, 2023, arXiv:2305.18462.
Całe zbiory: dataset inference
Kowalczuk i in., Privacy Attacks on Image AutoRegressive Models, ICML 2025, arXiv:2502.02514. MIA i dataset inference dla autoregresyjnych modeli obrazów (IAR): wystarcza bardzo niewiele próbek.
Dubiński i in., CDI: Copyrighted Data Identification in Diffusion Models, CVPR 2025, arXiv:2411.12858. Właściciel danych sprawdza, czy model dyfuzyjny był trenowany na jego zbiorze, łącząc wiele słabych sygnałów.
Puerto i in., Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models, 2024, arXiv:2411.00154. Pojedyncze rekordy są trudne, całe kolekcje dużo łatwiejsze.
Maini i in., LLM Dataset Inference: Did you train on my dataset?, NeurIPS 2024, arXiv:2406.06443. Porównanie opublikowanej i nieopublikowanej części zbioru testem statystycznym.
Kandpal i in., User Inference Attacks on Large Language Models, 2023, arXiv:2310.09266. Czy model widział dane konkretnej osoby.
Meeus i in., Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models, 2023, arXiv:2310.15007.
Maini, Yaghini, Papernot, Dataset Inference: Ownership Resolution in Machine Learning, ICLR 2021, arXiv:2104.10706. Pierwsza praca o dataset inference.
Modele obraz–tekst
Lo i in., COMPASS: Membership Inference Attack Against Vision Large Language Models via Adaptive Noise Calibration, 2026, arXiv:2610.17332.
Ren i in., Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models, 2024, arXiv:2410.13088.
Li i in., Membership Inference Attacks against Large Vision-Language Models, NeurIPS 2024, arXiv:2411.02902.
Rzetelna ewaluacja MIA
Shi T. i in., OLMo-Detect: A Multi-Stage, Confounder-Controlled Benchmark for Membership Inference on Large Language Models, 2026, arXiv:2610.02986. MIA na kolejnych etapach treningu.
Wang i in., CheckMIABench: Firm Foundations For Membership Inference Attacks on Language Models, 2026, arXiv:2606.17464.
Tong i in., Pretraining Data Exposure in Large Language Models: A Survey of Membership Inference, Data Contamination, …, 2026, arXiv:2605.26133. Przegląd.
Meeus i in., SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It), 2024, arXiv:2406.17975.
Das, Zhang, Tramèr, Blind Baselines Beat Membership Inference Attacks for Foundation Models, 2024, arXiv:2406.16201.
Duan i in., Do Membership Inference Attacks Work on Large Language Models?, COLM 2024, arXiv:2402.07841.