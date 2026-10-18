# ONEFILE — Piyasayı anlamak ve edge keşfi

**Çalıştırma: 17 Ekim 2026 UTC · BTCUSDT Binance USD-M · 2020–2024 geliştirme verisi**

## Sonuç

**Doğrulanmış ekonomik edge bulunmadı.** Bu, piyasada edge olmadığı anlamına gelmez. Sınanan tanımlar, maliyetler ve örneklem altında pozitif kanıt yetersizdir. Brick 1 korunmuş ve yeniden üretilmiştir: **SCREEN FAIL**. H1-A gerçekten tahmin olarak çalıştırılmıştır: **INCONCLUSIVE / low event power**. H1-A'nın ekonomik simülasyonu yapılmamıştır; monografinin forecast kapısı geçilmedi.

En yararlı öğrenimler:

- Bir eşik geçişini görmek, sürdürülebilir yeni rejim görmek değildir. H1-A risk kümesinden doğan **42 aday başlangıcın 29'u teyitsiz**, 13'ü teyitli çıktı.
- Son beş günlük yönü izlemek, sonraki 1/3/7 günde sırasıyla **%46,2 / %48,4 / %48,4** pozitif yönsel getiri üretti. Otomatik continuation varsayımı desteklenmiyor.
- Yerleşmiş operasyonel trendde 7 günlük continuation **%56,1**, fakat takvim-blok aralığı **%44,2–%66,4**: kesin avantaj değil.
- Volatilite kalıcılığına ilişkin betimsel kanıt, yönsel başlangıç öngörüsünden daha belirgin. Bu H1-C gözlemidir; H1-A başarısı değildir.
- Hacim/volatilite ek bloğunun H1-A Brier kazancı **%5,51**, belirsizlik aralığı **−%2,67…+%15,18**. Alt sınır pozitif değil; yalnız **5 yukarı / 1 aşağı** test olayı var.
- Taramadaki on koşulun **hiçbirinin** maliyet sonrası 3R veya 5R ortalama aralığının alt sınırı sıfırın üzerinde değil. Olumlu nokta tahminlerini edge diye seçmedik.

## Veri, kapsam ve dürüstlük sınırı

Taze Binance aylık arşivlerinden 2020-01-01–2025-01-01 aralığı yeniden indirildi. **10.962 adet 4H mum, 5.481 gerçek funding settlement kaydı, sıfır eksik mum/settlement** doğrulandı. Araştırma öncesi funding değerleri ayrıca yeniden kaynaktan karşılaştırıldı. Altı tamamlanmış 4H mumdan **1.827 UTC gün** elde edildi. Günlük bilginin kullanılabilir zamanı, o günün bitimindeki bir sonraki 00:00 UTC'dir; günün başlangıcı değildir.

- **ONEFILE 2025+ holdout açılmadı. Monografi G3 ≥2026-04-01 indirilmedi veya değerlendirilmedi.**
- Monografinin çekirdek-5 evreni yerine mevcut projenin BTC evreni kullanıldı. Kaynak 1h Parquet değil, mevcut 4H verisidir. Tercih edilen ≥5 dış fold yerine dört dış fold mümkündür. Bunlar açık protokol sapmalarıdır; tam çok-varlıklı H1-A doğrulaması iddia edilmiyor.
- 2023–2024 dış foldları model-fit açısından ileri tarih testidir; araştırmacı açısından yeni, bağımsız OOS değildir. Bu tarihler geliştirme geçmişidir. Geçmiş toplam deneme borcu bilinmiyor.
- Yerel protokol ilk skor görülmeden kaydedildi; dış kurumda ön kayıt veya git commit'i değildir. Sonraki yeniden üretim yeni bağımsız deney sayılmadı.
- Tüm istatistikler olay/olasılık çalışmasıdır; hesap getirisi, kaldıraç, marj, likidasyon veya portföy büyümesi değildir.

## 1. Fiyat neden ve nasıl hareket ediyor?

**Ekonomik açıklama:** Her işlemde alıcı ve satıcı vardır; fiyatı yalnız “alıcı sayısı fazla” diye açıklamak yanlıştır. Agresif emirlerin mevcut likiditeyi tüketmesi ve kotasyonların yeniden fiyatlanması son işlem fiyatını değiştirir. Haber, risk bütçesi, hedge talebi, envanter kısıtı veya zorunlu tasfiye bunun altında bulunabilir. Bu mekanizmalar OHLCV'den tek tek tanımlanamaz.

**Ölçtüğümüz:** Monografinin günlük `Z20` ve **log-getirilerden** `ER20` tanımı kullanıldığında 1.807 geçerli günün 324'ü +trend, 111'i −trend, 1.372'si nötrdür. `q=0`, “kesin yatay piyasa” değil, bu operasyonel yön eşiğini karşılamayan piyasa demektir; sert iki yönlü hareketleri de içerebilir.

Sonradan yapılan, yalnız betimsel süre denetiminde tamamlanmış +/− q epizodlarının medyanı **2 gün**, nötr epizodların medyanı **9 gün** çıktı. Bu, etiketin eşik etrafında sıkça titreştiğini gösterir; fiziksel piyasa rejimlerinin medyan ömrünün iki gün olduğunu kanıtlamaz.

Günlük |getiri| ile aynı günkü hacim sürprizi Spearman ilişkisi **0,492**; günlük getiri lag-1 korelasyonu **−0,078**, mutlak getiri lag-1 korelasyonu **+0,144**. Bunlar sonradan incelenmiş betimsel ilişkiler, nedensellik veya bağımsız anlamlılık testleri değildir. Büyük hacim, büyük hareketle birlikte görülüyor; bunun tek başına hareketten önce bilgi verdiği çıkarılamaz.

## 2. Hareketler ne kadar devam ediyor?

**Karar tanımı:** Tamamlanmış son 5 günlük log-getirinin yönünde sonraki gün açılışında gir; 1/3/7 gün sonraki kapanışa kadar ölç. Continuation, bu yöndeki brüt fiyat getirisinin pozitif olmasıdır. Yol boyunca kesintisiz hareket veya stop almadan kazanç anlamına gelmez.

| Önceden gözlenebilir koşul | N günlük pencere | 1g devam | 3g devam | 7g devam | 7g net ortalama |
|---|---:|---:|---:|---:|---:|
| Tüm momentum yönleri | 1.757 | %46,2 | %48,4 | %48,4 | −%0,130 |
| Yerleşmiş q trendi yönünde | 353 | %46,2 | %51,3 | %56,1 | +%1,109 |
| q nötr | 1.334 | %46,9 | %48,3 | %47,2 | −%0,320 |
| Yeni q eşik geçişi, momentumla uyumlu | 66 | %39,4 | %45,5 | %51,5 | +%0,306 |
| Volatilite sıkışması | 710 | %43,8 | %46,6 | %46,2 | −%0,623 |
| Bugün ani genişleme | 139 | %48,9 | %48,9 | %51,8 | +%0,937 |
| Yüksek hacim | 290 | %45,5 | %51,4 | %53,1 | +%0,545 |
| Bugün momentum tersine güçlü hareket | 96 | %55,2 | %50,0 | %40,6 | −%1,124 |
| Yönle uyumlu 3g funding >10bp | 279 | %50,5 | %52,7 | %52,7 | +%0,461 |
| Funding momentum yönüne karşı | 808 | %45,7 | %46,2 | %45,2 | −%0,575 |

**N bağımsız işlem sayısı değildir.** Günlük ileri pencereler örtüşür; koşullar da örtüşür. Belirsizlik için boş günleri koruyan 28/56/84 günlük takvim blokları kullanıldı; en geniş %95 aralık raporlandı. Betimsel tablolar çoklu koşul seçimine göre düzeltilmiş anlamlılık testi değildir.

Tüm momentum için 1/3/7 günlük yönsel brüt getiri **P10 / medyan / P90**:

- 1g: **−%3,53 / −%0,17 / +%3,73**
- 3g: **−%6,24 / −%0,13 / +%6,84**
- 7g: **−%9,99 / −%0,34 / +%11,17**

Yeni eşik geçişinde q'nun sonraki günlerin **tamamında** aynı yönde kalma oranı 1/3/7 gün için **%53,0 / %34,8 / %18,2**. Yerleşmiş trend günlerinde aynı ölçü **%79,9 / %62,3 / %41,9**. Dolayısıyla “7 gün sonra biraz daha yukarı/aşağıda olmak” ile “7 gün boyunca rejim bozulmadı” çok farklı olaylardır.

## 3. Hareketlerden önce ne değişiyor?

**Volatilite:** Mevcut 20 günlük RV ile gelecek 7 günlük gerçekleşen RV arasındaki betimsel Spearman ilişkisi **0,424**. Riskin devamlılığı var gibi görünüyor; bu bir HAR-RV'ye karşı bağımsız forecast zaferi değildir. `RV5/RV20<0,8` sıkışması, bu deneyde yönsel continuation veya maliyet sonrası asimetriyi iyileştiren doğrulanmış bir filtre olmadı.

**Hacim:** Yüksek hacim grubunda 7g continuation %53,1 ve net ortalama +%0,545; fakat ortalamanın blok aralığı **−%1,061…+%2,104**. Birlikte hareket etmek, öncülük etmek değildir. Momentum kontrolü sonrasındaki ortak hacim/sıkışma bloğunun asıl testi aşağıdaki H1-A'dır ve belirsizdir. İki ek özelliğin ayrı ayrı nedensel katkısını sınamadık.

**Funding:** Yönle uyumlu yüksek funding grubunun 3R politikasında trading maliyetleri sonrası, funding öncesi beklentisi **+0,067R** iken funding dahil **−0,009R** oldu. Bu, taşımanın sonucu değiştirebildiğine ilişkin doğrudan örnek. Ancak bu gruplar ay, yön, rejim ve beta açısından eşleştirilmiş bir funding-incrementality testi değildir. Funding'in momentumdan bağımsız yönsel öncü bilgi taşıdığını kanıtlamıyoruz. Monografideki M3 funding modeli bu ilk M2 deneyine gizlice eklenmedi.

**Fiyat davranışı:** M1, momentumun yanı sıra Z/ER, eşik uzaklığı, Z değişimi ve RV'leri zaten biliyor. H1-A dış foldlarında eşiğe yakın 22 satırın olay prevalansı **%27,3**, uzaktaki 318 satırın **%11,0**. Etikete yaklaşma, ciddi bir rakip açıklama olmaya devam ediyor. Üstelik M2 sıkışma oranı, M1'in mevcut RV'lerinin doğrusal olmayan dönüşümüdür; bütünüyle yeni bağımsız bir bilgi kaynağı değildir.

## 4. Kim neden kazanıyor veya kaybediyor?

| Katılımcı / mekanizma | Ekonomik kazanç veya kayıp kaynağı | Burada ne ölçülebiliyor? |
|---|---|---|
| Yönsel pozisyon alan | Doğru yön, taşıma ve zamanlama; ters hareket ve maliyet zararı | Fiyat ve varsayımsal olay PnL'si; gerçek katılımcı kimliği değil |
| Perp long / short | Pozitif funding'de long öder, short alır; negatifte tersi | Tarihsel oranla, 4H açılış fiyatı vekiliyle nakit tahmini |
| Spot-perp arbitrajcı | Hedge edilmiş basis/funding geliri; sermaye, basis ve borsa riski | Yalnız funding bacağı; spot hedge/basis yok, arbitraj kârı hesaplanamaz |
| Piyasa yapıcı | Spread ve envanter yönetimi; adverse selection zararı | Emir defteri, spread, maker fill ve envanter yok: kazancı tanımlanamaz |
| Likidasyona uğrayan | Marj kısıtı nedeniyle zorunlu kapanış, kayma | OI, likidasyon akışı, hesap marjı ve mark serisi yok: “tasfiye avı” kanıtlanamaz |

Sonradan yapılan yön ayrımında tüm uygun günlük long girişlerinin 3R ortalaması **+0,128R**, short girişlerin **−0,079R**; momentumla uyumlu long/short ortalamaları **+0,134R / −0,106R**. Bu tarih aralığında beta/yön asimetrisi bulunduğunu düşündürür; “akıllı para long kazandı” demek için veri yok. Bunlar yeni seçilmiş işlem kuralları veya doğrulanmış beta-düzeltilmiş alpha değildir.

## 5. 1R karşılığında 3R–5R nerede oluşuyor?

**Bu bağımsız piyasa davranışı çalışmasında** 1R = karar gününde bilinen **1,5 × günlük ATR14**; stop −1R, hedef +3R veya +5R, azami süre 7 gündür. Gerçek 4H OHLC, stop-first, next-open giriş ve native vectorbt fill'leri kullanıldı. Brick 1'in 4H ATR'siyle aynı risk ölçeği değildir. Günlük 1R fiyat mesafesinin medyanı **%6,73**: medyan ölçekte 3R/5R yaklaşık %20,2/%33,7 fiyat hareketi gerektirir. Bu nedenle 7 günlük hedef vuruşu seyrektir.

Maliyetler: **4bp fee + 2bp slippage / taraf** (yaklaşık 12bp RT) ve geçmiş funding. Stres: **ek 12bp RT**; funding aynı tutulur. Gerçek hesap tarifesi veya order-book gerçekleşmesi olduğu iddia edilmez. Stop, boşluk ve maliyet nedeniyle net kayıp −1R'den kötü olabilir.

| Koşul | 3R politika net ort. [blok %95 aralık] | 5R politika net ort. [blok %95 aralık] |
|---|---:|---:|
| Tüm momentum | +0,026 [−0,075; +0,122] | +0,049 [−0,065; +0,160] |
| Yerleşmiş trend | +0,109 [−0,171; +0,354] | +0,135 [−0,161; +0,397] |
| Nötr | +0,022 [−0,076; +0,123] | +0,047 [−0,061; +0,167] |
| Yeni eşik geçişi | +0,120 [−0,199; +0,492] | +0,152 [−0,198; +0,551] |
| Sıkışma | −0,058 [−0,179; +0,071] | −0,038 [−0,166; +0,106] |
| Bugün genişleme | +0,183 [−0,070; +0,443] | +0,284 [−0,033; +0,642] |
| Yüksek hacim | +0,149 [−0,070; +0,350] | +0,166 [−0,075; +0,405] |
| Momentum tersine hareket | −0,107 [−0,291; +0,090] | −0,121 [−0,293; +0,063] |
| Uyumlu yüksek funding | −0,009 [−0,336; +0,304] | −0,015 [−0,330; +0,284] |
| Ters funding | −0,027 [−0,171; +0,116] | −0,016 [−0,163; +0,136] |

Genişleme günleri olumlu **keşifsel nokta tahminine** sahip; aralık sıfırı içeriyor ve sekiz takvim fazında 5R ortalamaları **−0,020…+0,771R** arasında değişiyor. Bunu kazanan filtre olarak seçmek veri madenciliği olur. Ek maliyet stresinde +0,284R → **+0,264R**; maliyet stresinin pozitif kalması istatistiksel belirsizliği çözmez.

**Brüt hedef ≠ net hedef:** Tüm momentumda brüt 3R/5R bariyerine önce ulaşma **%6,03 / %1,94**; aynı çıkış politikaları altında gerçekten net ≥3R/≥5R elde etme **%0,91 / %0,34**. Ek maliyet stresinde net eşik oranları **%0,34 / %0,057**. Hedefte çıkıldığı için masraflar çoğu hedef kazancını net eşiğin altına indirir; funding geliri veya olumlu gap istisnalar yaratabilir. Bu, yolun hiçbir zaman daha ileri gitmediği anlamına gelmez. Bariyersiz ileri getiri dağılımı ve işlem çıkış sonucu ayrıdır.

**Hüküm:** Burada dayanıklı, maliyet sonrası 3R–5R edge kanıtı yok.

## H1-A — Gerçek deney

### Kilitli tanım ve uygulama

- `Z20 = sum(log returns)/sqrt(sum(log returns²)+eps)`; `ER20 = |sum(log returns)|/(sum(|log returns|)+eps)`.
- q eşiği ±1,25 ve ER≥0,35. En az 10 tamamlanmış nötr gün risk kümesi.
- İlk yönsel geçiş onset. **Sonraki** 10 günün en az 8'inde aynı işaret teyit; onset günü bu 10 güne sayılmaz. Teyit zamanı ayrı tutulur.
- Risk kümesindeki t satırının hedefi sonraki 7 gündeki ilk teyitli +/− onset veya yok. Son 17 günlük olgunlaşmamış etiketler negatif yapılmaz.
- M0: nötr durum yaşı. M1: M0 + 5/20/63g getiriler, Z/ER, eşik mesafesi, Z değişimi, 5/20g RV. M2 yalnız sıkışma oranı ve geçmiş 20g medyanına göre hacim sürprizi ekler.
- Multinomial ridge; scaler yalnız training'de. İlk üç yıl train, dört genişleyen 6 aylık dış test. Training origin+17g < dış test başlangıcı. C, sıcaklık kalibrasyonu ve alarm eşiği yalnız iç ileri foldlarda seçilir.
- 60 günlük **geçmişe geciktirilmiş** ek özellikler placebo; geleceği başa saran döngüsel shift yok.
- Birincil karşılaştırma M2−M1; 28/56/84 günlük eşlenik takvim blokları ×1.000, CI zarfı. Bu bootstrap altı ekonomik olayı yüzlerce bağımsız olaya çeviremez.

### Sonuçlar

| Model | Multiclass Brier ↓ | Log loss ↓ | Olay AP / PR-AUC özeti ↑ |
|---|---:|---:|---:|
| M0 | 0,222022 | 0,445875 | 0,0983 |
| M1 | 0,222566 | 0,432801 | 0,1587 |
| M2 | **0,210309** | **0,414027** | **0,2410** |
| Gecikmeli özellik placebo | 0,216135 | 0,433927 | 0,2144 |

Toplam **346 uygun güne tahmin** kaydedildi: **340 olgun satır skorlandı**, son **6 sağ-sansürlü satır skorlanmadı** ve negatif etiketlenmedi. Skorlanan satırlarda olay prevalansı **%12,06**; 340 satırın 41'i pozitif olay etiketidir, **41 bağımsız onset değildir**. Gerçek skorlanabilir olay sayısı **6**: 5 yukarı, 1 aşağı; ±14g kümeleri **6**, gereken en az 12. Pozitif/negatif onset asgari tabanı **20/20** iken 5/1'deyiz. Tüm DEV'de dahi yalnız 9 yukarı, 4 aşağı teyitli onset var.

| Dış pencere | Skorlanan N | M1 Brier | M2 Brier |
|---|---:|---:|---:|
| 2023 ilk yarı | 90 | 0,240645 | 0,200621 |
| 2023 ikinci yarı | 76 | 0,478552 | 0,470653 |
| 2024 ilk yarı | 109 | 0,140589 | 0,142389 |
| 2024 ikinci yarı | 65 | 0,035694 | 0,033216 |

- Nihai sürümde göreli Brier iyileşmesi **%5,507**, %95 blok zarfı **[−%2,673; +%15,182]**. Olgun satırlardaki model skorları önceki çalışmayla birebir yeniden üretildi. Nokta tahmini %5'i geçse de CI koşulu geçmiyor. M1, M0'ı Brier'da bile yenemedi.
- Nihai M2: **4 alarm, 2 doğru olay, 2 yanlış alarm, 4 kaçırılmış olay**; precision %50, recall 2/6. **715 olgun takvim günü** üzerinden **1,022 yanlış alarm / BTC-yıl**. Medyan gerçek pre-onset lead **1,5 gün**; gereken ≥2 gün değil. Yüksek precision izlenimi yalnız dört alarma dayanıyor.
- M1: 3 alarm, 1 doğru, 2 yanlış; medyan lead 1 gün. M0 alarm vermedi. 2024 ikinci yarıda M2 de iç seçim sonucunda alarmı kapattı; sıfır yanlış alarm tek başına başarı sayılmadı.
- Kalibrasyon kusursuz değil: M2'nin <%10 olay olasılığı verdiği 179 satırda ortalama tahmin %4,51 iken gerçekleşme **%10,61**. Tam kalibre edilmiş sistem iddiası yok.
- Gecikmeli placebo da M1'e karşı daha düşük Brier verdi. Tanısal olarak, M2'nin farkını benzersiz yeni öncü mekanizmaya bağlamayı daha da zorlaştırıyor; ayrı bir anlamlılık testi değil.

**Karar: INCONCLUSIVE / low event power.** Hem skor güven aralığı hem erken-uyarı lead koşulu geçmedi. Düşük güç nedeniyle evrensel “H1-A imkânsız” veya kesin ret demiyoruz; **bu çalışmada H1-A lehine yeterli kanıt yok**. H1-B/H1-C'ye yeniden isim verip başarı sayılmadı. 21 günlük H1-A swing backtest'i, paper/live işlem veya G3 açılışı yapılmadı.

## Brick 1 ve kodda değişenler

Brick 1 sinyal, eşleştirme, maliyet ve varsayılan execution davranışı korundu. Taze veriyle 66 kurulum / 37 eşleştirilmiş kurulum yeniden üretildi. Tüm kurulum ortalaması +0,528R; eşleştirilmiş ortalama **+0,098R**, kontrol **−0,717R**; kurulum ortalama CI **[−0,454; +0,680]R**. **SCREEN FAIL** değişmedi. Eşleştirilmemiş kurulumları seçerek gizlemedik.

`one.py` içinde:

- `--research`: günlük piyasa gözlemleri + H1-A nested tahmini, ayrı sonuç dosyaları.
- Günlük kullanılabilirlik, log-return ER20 düzeltmesi, risk kümesi/onset/confirmation/maturity ayrımı.
- Aynı native vectorbt adaptöründe short ve değişken tutma ufku; bağımsız ikinci fill motoru yok.
- Brüt bariyer, gerçek net eşik ve maliyet stresi ayrımı; boş gruplar açık `null` istatistikleri.
- Takvim-blok belirsizliği, gecikmeli placebo, kalibrasyon ve gerçek pre-onset alarm arşivi; ayrı kanıt/skor/uyarı kapıları.
- Kod/veri/protokol hash'i; eşleşmeyen frozen protocol ile sessiz yeniden çalıştırma reddedilir. `--research --unlock-oos` indirmeden önce reddedilir.

Yeni repo, framework, servis, veri katmanı, GPU işi veya yeni backtest motoru yok. scikit-learn vectorbt'nin zaten mevcut bağımlılığıydı; doğrudan kullanım artık `pyproject.toml`/kilitte açık. Tek uygulama dosyası `one.py` kaldı. Mevcut kullanıcı değişiklikleri geri alınmadı; mevcut Brick 1 sonuç dosyaları yeniden yazılmadı.

## Testler, yeniden üretim ve audit notu

- **115 offline/sentetik unittest**, **38 native vectorbt self-test**, `uv run --with ruff ruff check one.py tests/test_research.py` ve `git diff --check`: geçti. Ruff kalıcı uygulama bağımlılığı olarak eklenmedi.
- Gerçek DEV üzerinde 4 dış sınır × 6 ham alan = **24 future-corruption kontrolü**: önceki günlük özellikler/q/risk değişmedi.
- Kaydedilmiş olasılıkların toplamı, Brier/log-loss yeniden hesapları, 1–7g gerçek lead, onset+10g teyit, 17g etiket olgunluğu ve çıktı hash'leri doğrulandı.
- Araştırma hesabı yaklaşık **87 saniye**; veri kaynağı kontrolü ve test suite süresi buna dahil değil.

**Açık sürüm notu:** Çalışma sırasında checkout'a paralel değişiklikler geldi; source hash denetimi drift'i yakaladı. İlk protokol ve sonuçlar `research_attempt01.zip` ile korundu. İlk kaynak snapshot'ı kaydedilmediğinden o sürümün tam kaynak yeniden üretilebilirliği iddia edilmiyor. İkinci sürümün kaynak ve sonuçları `research_attempt02.zip` ile korundu. Son düzeltmeler açıklama/kolon adı/boş-grup sunumunun yanında **sansürlü günlere de tahmin kaydetmeyi ve yanlış-alarm paydasını açık olgun takvim maruziyetiyle hesaplamayı** kapsıyor. Bunlar sonuç görüldükten sonraki yöntem düzeltmeleridir; dış ön kayıt diye sunulmuyor. Model özellikleri, etiket eşikleri, C/kalibrasyon/eşik adayları değiştirilmedi. Son kaynak snapshot'ı ve kod/veri/protokol hash'i yeniden eşleştirildi; tüm olgun tahminler, Brier/log-loss ve piyasa ölçümleri aynı kaldı. Ek 6 sansürlü tahmin skorlanmadı; yanlış alarm oranı ~0,999'dan 1,022/BTC-yıla düzeldi. Hüküm değişmedi. Tekrarlar yeni bağımsız kanıt veya model araması sayılmadı. Ayrıntı: `research_integrity_audit.json`.

```bash
uv sync --frozen --python 3.12
uv run python -m unittest discover -s tests -v
uv run python one.py --self-test
uv run python one.py --research
# Veri yoksa aynı DEV sınırıyla: uv run python one.py --download --research
```

`research_protocol.json` mevcut kod/veri kimliğini mühürler. Kod veya veri değişirse eski protokolü silip aynı deneyi “ilk çalışma” diye sunmayın; eski denemeyi koruyup ayrı protokol kimliği/path'i kullanın. Gerçek eski CSV funding flags'lerini uydurmayın; kaynaktan yeniden indirin.

### Dosyalar

- `RESEARCH_REPORT.md`: bu hüküm ve beş sorunun cevapları.
- `research_evidence.json`: tüm koşullar, dağılımlar, CI hassasiyetleri, foldlar ve kalibrasyon.
- `market_observations.csv`: 3.514 long/short günlük gözlem; **1.757 benzersiz karar günü**, bağımsız işlem değil.
- `h1a_forecasts.csv`: 346 dış-fold olasılığı (340 olgun skor + 6 sansürlü tahmin), gerçek onset/confirmation/maturity zamanları.
- `h1a_onsets.csv`: 42 adayın teyitli/teyitsiz arşivi.
- `research_protocol.json`, `research_run.log`, `research_integrity_audit.json`, kaynak/ilk-deneme ZIP'leri: sürüm ve audit izi.
- `tests/test_research.py`: nedensellik, etiket, purge, kalibrasyon izolasyonu, short stop/gap/funding, net/brüt hedef ve mühür korumaları.

CSV'ler projenin mevcut `.gitignore` kuralıyla yerel çıktıdır; sonuç paketi bunları ayrıca saklar.

## Sıradaki en mantıklı adım

**Parametre araması veya live değil; aynı değişmemiş H1-A protokolünü mühürsüz ETH ve mümkünse çekirdek-5 geçmişinde sınamak.** Önce olay yeterliliğini saymak; eşik/teyit şartını olay üretmek için gevşetmemek. Ortak coin şoklarını aynı takvim bloklarında tutmak ve BTC/ETH işaret tutarlılığını ayrı raporlamak. Bu, yeni altyapı gerektirmez; ama BTC-only feasibility sonucunu tam doğrulamaya çevirmek için eksik gözlemleri gerektirir. Veri/bağımsız olay yine yetmezse beklemek, yeni veri toplamak ve hipotezi belirsiz bırakmak doğru sonuçtur. ONEFILE holdout veya G3 ancak ayrı açık kullanıcı onayı ve yeni donmuş protokolle değerlendirilebilir.
