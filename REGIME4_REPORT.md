# ONEFILE — Kanıta dayalı rejim tahmini (v4): audit, hata analizi, bilgi testleri, holdout

## Karar: NO_EDGE_80 korunuyor

Yeni kanıt, %80 iddiasını desteklemiyor. Önceden kaydedilmiş `mc`/`fixed70` sistemi, **hiç incelenmemiş 12 varlıkta** 102 alarmdan 78'ini yakaladı: **%76.5** precision (Wilson [67.4, 83.6], 28 günlük takvim-blok [64.4, 88.0]), recall %19.7. Mekanizma yeni varlıklara taşınıyor, ama %80 kapısı geçilmedi. Gerçek (mekanik olmayan) öngörü gücüne dair kanıt bulunamadı.

**Neden ve ne zaman çalışıyor:** Sistemin bütün alarmları, gelecek getirilerin tamamı sıfır olsa bile eşik geçişinin gerçekleşeceği günlerde (`need ≤ 0`) üretiliyor. Bunun kaynağı, 20 günlük pencereden düşecek ve bugünden bilinen getiriler. Model bir piyasa öngörüsü yapmıyor; bilinen pencere aritmetiğini doğru kalibre edilmiş bir olasılığa çeviriyor. Kalan belirsizlik, geçişin 3/5 gün kalıcı olup olmayacağı. Yanlış alarmların %70–75'i tam olarak burada: geçiş oluyor, ama kalıcı olmuyor. Kalıcılık, fiyat yönüne bağlıdır ve bu veride fiyat yönü martingale'den ayırt edilemiyor.

## Phase 0 — E2E audit sistemi

Her `--regime3` koşusu `artifacts/regime/<run_id>/` dizinini yazar. `run_id` biçimi: `UTC zaman-evren-kapsam-config hash`.

| Dosya | İçerik |
|---|---|
| `manifest.json` | Git commit/dirty, kod SHA, veri yolu/SHA/kaynak/varlık aralıkları, bütün konfigürasyonlar, feature listesi, modeller, politikalar, her fold'un split boyutları, eşikleri, model başlıkları ve fit süreleri, holdout ön kaydı, karar, parity, dosya SHA'ları |
| `predictions.csv.gz` | Bütün walk-forward tahminleri ve fold başına eşikler |
| `features.csv.gz` | Tahmin anında bilinen bilgi; her varlık-gün için bir kez, modeller arasında çoğaltılmaz |
| `alarms.csv` | Her alarm: zaman, veri kesimi, skor, eşik, yön, feature snapshot (q, z20, er20, rv20, funding, rw, need, mc), sonuç TP/FP, FP nedeni, eşleşen olay, lead |
| `events.csv` | Her fırsat × model × politika: yakalandı mı, lead, kaçırma nedeni, 7 günlük maksimum skor, doğal rejim süresi |
| `fold_metrics.csv`, `calibration.csv` | Fold metrikleri ve 10 kutulu güvenilirlik tablosu |
| `comparison.json`, `comparison.md` | Lead ve süre dağılımları, mekanik/hareket gerektiren alarm ayrımı, erken recall, FP/kaçırma nedenleri |

**Parity:** Audit salt okunur çalışır. Her koşu, diskteki tahminlerden bütün metrikleri yeniden türetir ve rapordaki metriklerle birebir aynı olmasını şart koşar. Core-5 koşusu, PR #6'nın `regime3_final.json` metriklerini **birebir** yeniden üretti (`reference: PASS`). Ek süre ~14 s, boyut ~3.8 MB (core-5).

Kalıcı kanıt:

- `artifacts/regime/20261018T150333Z-core5-final-cd00be92/` (temiz commit `0223041`)
- `artifacts/regime/20261018T150154Z-holdout-final-5909414f/` (temiz commit `0223041`, ön kayıt dahil). Worker analiz scriptleri ve istatistikler `phase_analyses/` altında.

## Phase 1 — Hata analizi (core-5, 2021–2024)

- `mc`/`fixed70`: 48 alarmın **48'i de mekanik** (need −1.99…−0.37). İsabetlerin 20/38'i 1 gün önce geldi. 1 gün ve 2–7 gün lead alarm başına ayırt edilemiyor (%41.7 / %37.5).
- **FP nedenleri:** onset_failed_confirmation 7, no_onset 3. Diğer modellerde de baskın neden kalıcılık başarısızlığı (%48–90).
- **Kaçırmalar:** 105 olaydan 100'ünde skor eşiğe hiç ulaşmadı. Kaçırılanların 50'si 0.5–0.7 bandında.
- **Kalibrasyon:** mc neredeyse kalibre. 0.7–0.8 kutusunda tahmin .746, gerçekleşen .804; 0.8–0.9 kutusunda .835 / .875. Lojistik ve XGB üst uçta aşırı güvenli.
- **Veri kapsamı:** Binance arşivinde SOL/XRP için 2022-02-26…28 ve 2022-04-01…02 eksik. `r63` 63 gün boyunca NaN kalıyor ve 86 tahmin satırı (2 olay) bütün modellerden eşit olarak düşüyor. Karşılaştırma adil; kayıp belgelendi.

Bilginin nerede olduğu, olay AP'si ile (tahmin satırları uzaklığa göre katmanlı):

| Köken tipi | Satır | Taban | prior | structural | **mc** | lojistik | xgb |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mekanik (need ≤ 0) | 615 | .413 | .378 | .584 | **.692** | .685 | .681 |
| Hareket gerekiyor (0 < need ≤ 1σ) | 2,100 | .161 | .163 | .233 | **.322** | .288 | .257 |
| Büyük hareket gerekiyor (need > 1σ) | 4,313 | .037 | .040 | .076 | **.086** | .054 | .057 |

Her katmanda parametresiz mc en iyi. Fitted modeller, gerçek hareket gereken satırlarda tabana yaklaşıyor.

## Phase 2 — Rejim süresi ve ufuk

Rejim süresi (q'nun yeni yönde kaldığı ardışık gün), tahmin ufku (1–7 gün) ve lead ayrı kavramlar olarak ölçüldü.

| | Core-5 (145 olay) | Holdout (406 olay) |
|---|---:|---:|
| Medyan süre | 6 gün | 5 gün |
| Kısa ≤4 / orta 5–14 / uzun ≥15 | %35 / %48 / %17 | %43 / %44 / %13 |
| mc/fixed70 recall: kısa / orta / uzun | .18 / .27 / .44 | .15 / .19 / .38 |

Uzun rejimler daha sık yakalanıyor. Ancak süre başlangıçtan sonra ortaya çıkar; bu bir ex-ante süre tahmini değildir.

Ufuk hedefleri (core-5, resmi etiket değiştirilmeden yalnızca değerlendirme için):

| Hedef | Taban | mc AP | structural AP | mc p@1% |
|---|---:|---:|---:|---:|
| L1: 1 gün içinde başlangıç | .020 | .333 | .335 | .451 |
| L2–7: 2–7 gün, ilk gün yok | .089 | .314 | .243 | .609 |
| LONG: ≤7 günde başlayan, ≥10 gün süren | .037 | .189 | .133 | .394 |

Başlangıçtan önceki günlerde mc yön olasılığı, eşleştirilmiş negatiflere göre farkı şöyle büyüyor: −7. gün +.099, −3. gün +.235, −1. gün +.442. 2–7 günlük sinyal de var, ama düşüş aritmetiği 7 güne kadar uzandığı için **lead ≥ 2 mekanik olmadığı anlamına gelmez**. Ayırt edici ölçü `need`'dir (Phase 1 tablosu).

## Phase 3 — Öngörü bilgisi (ablation + null testleri)

Taban: mc log-odds'u üzerine kurulu lojistik (AP .409; mc'ye göre −.030 [−.082, +.026]). Gruplar tek tek eklendi. AP farkları, eşleştirilmiş 28 günlük blok bootstrap ile tabana göre ölçüldü:

| Grup | Tabana göre AP farkı [%95] |
|---|---:|
| G_struct (rw, need, er_proj, drop_r7) | +0.008 [−0.018, +0.031] |
| G_vol (rv5, rv20, compression, range_atr) | −0.046 [−0.118, +0.009] |
| G_mom (r5…r63, z5, er5, dz5, body, close_loc) | **−0.051 [−0.094, −0.008]** |
| G_funding (funding3, 7/30 günlük toplam, 90 günlük z) | **−0.083 [−0.139, −0.029]** |
| G_volume (volume_surprise) | −0.003 [−0.006, +0.000] |
| G_cross (BTC z20/q, breadth, ortalama z20) | −0.024 [−0.068, +0.014] |
| G_flow (taker-buy oranı, önbellekteki kline ZIP'lerinden) | −0.000 [−0.023, +0.019] |

- **Null testi:** 28 günlük blok permütasyonu, bütün varlıklara aynı permütasyon. G_struct'ın gerçek kazancı dört null'un hepsini geçiyor, ama kendisi sıfırdan ayırt edilemiyor. G_funding'in hem gerçek hem null hali zararlı.
- **Momentum drift / EWMA vol MC (lider, 3 deneme):**
  - Ham momentum etkisi zayıf ve kararsız: korelasyon .059 (2020–22), .018 (2023–24). Eşik yakınında sonraki 5 günün yönü yazı-tura (%50.5 / %46.6).
  - AP kazancı +.001…+.003, aralıkların hepsi sıfırı içeriyor. κ=0.1 varyantı fixed70'te %80.7 gösterdi, ama aynı bilgiyle 3 deneme arasından seçilmiş eşik gürültüsü olduğu için terfi edilmedi.
- **Open interest:** Binance OI arşivi 2021-12'de başlıyor; 2020–21 eğitim yılları için yok, bu yüzden test edilemedi.
- **Mevcut lojistiğin ablation'ı:** Her özelliğin katkısı < .005 AP.

## Phase 4 — Model ve politika

Hiçbir bilgi kaynağı mc'yi geçmediği için yeni model terfi edilmedi. Politika tarafında mc kalibre olduğundan alarm precision'ı eşiği izliyor. %80'e ulaşmak ancak recall'u düşürerek mümkün:

| mc eşiği | Core-5 alarm / hit / prec / recall | Holdout alarm / hit / prec / recall |
|---|---|---|
| .60 | 81 / 53 / .654 / .371 | 200 / 131 / .655 / .332 |
| .65 | 65 / 50 / .769 / .350 | 145 / 103 / .710 / .261 |
| **.70 (kayıtlı)** | **48 / 38 / .792 / .266** | **102 / 78 / .765 / .197** |
| .75 | 30 / 25 / .833 / .175 | 66 / 53 / .803 / .134 |
| .80 | 15 / 13 / .867 / .091 | 35 / 29 / .829 / .073 |

Bu tablo betimleyicidir; eşik seçimi için kullanılmadı. Eşiği .75'e çekmek bir recall takası olurdu, öngörü gücünün artması değil.

## Phase 5 — Holdout doğrulaması (yeni kanıt)

- **Tasarım:** 12 likit perp (ADA, LINK, LTC, DOGE, DOT, TRX, BCH, ETC, ATOM, AVAX, XLM, FIL). Bu varlıklar, hiçbir veri incelenmeden `HOLDOUT_FIRST_MONTH` olarak sabitlendi; pencere DEV 2020–2024. Hipotez ve kapı, skorlamadan önce `0223041` ile commit/push edildi; tek koşu yapıldı. Sealed OOS (2025+) açılmadı.
- **Bağımlılık:** Holdout varlıkları core-5 ile aynı piyasa günlerini paylaşır. Bu kanıt kesitsel, zamansal OOS değildir. Belirsizlik, varlıkları birlikte örnekleyen 28 günlük takvim bloklarıyla ölçüldü.

| Sistem (holdout, 395 fırsat) | Alarm | Yakalanan | Precision [Wilson] / [blok] | Recall [blok] | Erken recall (lead ≥2) | AP (taban .116) |
|---|---:|---:|---|---|---:|---:|
| **mc / fixed70 (birincil)** | 102 | 78 | **.765** [.674, .836] / [.644, .880] | .197 [.137, .265] | .094 | .428 |
| mc / budget | 576 | 250 | .434 / [.369, .503] | .633 [.568, .709] | .430 | .428 |
| structural / fixed70 | 312 | 173 | .554 / [.470, .641] | .438 | .215 | .380 |
| lojistik / precision80 | 19 | 13 | .684 / [.438, 1.0] | .033 | .015 | .323 |
| XGBoost / precision80 | 0 | 0 | — (politika açılmadı) | 0 | 0 | .320 |
| XGBoost / fixed70 | 170 | 94 | .553 / [.417, .691] | .238 | — | .320 |

- **AP farkları:** mc − structural **+.048 [.021, .078]**, core-5 sonucu tekrarlandı. Lojistik − mc **−.105 [−.162, −.048]**, XGB − mc **−.108 [−.165, −.053]**. Fitted modeller yeni varlıklarda anlamlı şekilde kötü.
- **Yıllara göre mc/fixed70:** 16/26, 19/28, 22/25, 21/23.
- **Varlıklara göre:** 11/12 varlıkta çalışıyor; LTC 1/6 ve FIL 1/3 zayıf.
- **Kapı:** precision ≥.80 ✗ (.765), Wilson alt ≥.70 ✗ (.674); diğer 5 koşul ✓ (≥40 alarm, ≥20 isabet haftası, recall ≥.10, dokunulmamış ≥.75 / ≥15 alarm, her yıl ≥.60). Blok bootstrap ile P(precision ≥ .80) = .31.

## Adil karşılaştırma

| Sistem | Görev / evren | Fırsat | Alarm | Yakalanan | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Eski H1-A M2 | strict, BTC | 11 | 10 | 3 | 30.0% | 27.3% |
| v2 XGBoost (PR #5) | genişletilmiş, BTC | 33 | 56 | 5 | 8.9% | 15.2% |
| v3 mc/fixed70 (PR #6) | genişletilmiş, core-5 | 143 | 48 | 38 | 79.2% | 26.6% |
| v3 lojistik/precision80 | core-5 | 143 | 26 | 20 | 76.9% | 14.0% |
| v3 XGBoost/precision80 | core-5 | 143 | 21 | 15 | 71.4% | 10.5% |
| **v3 mc/fixed70, yeni 12 varlık** | holdout | 395 | 102 | 78 | **76.5%** | 19.7% |
| v3 lojistik/precision80, yeni 12 varlık | holdout | 395 | 19 | 13 | 68.4% | 3.3% |
| v3 XGBoost/fixed70, yeni 12 varlık | holdout | 395 | 170 | 94 | 55.3% | 23.8% |

## Süreler

| Adım | Wall-clock |
|---|---:|
| Core-5 final + audit (6 model, 8 fold) | 57 s (araştırma ~40 s, audit ~14 s) |
| Holdout final + audit (12 varlık) | 130 s |
| Worker A (hata ve ufuk analizi) / Worker B (bilgi ve null, 11 walk-forward çağrısı), paralel | 178 s / 643 s |
| Tam test paketi (143 test) + motor self-test (38 kontrol) | 31 s + 8 s |
| Bu tur toplam (senkron → PR güncellemesi) | ≈ 35 dk |

## Sınırlar

- 2023–2024 daha önce incelendi ve bu turda dokunulmamış OOS olarak sunulmuyor. Yeni kanıt yalnızca holdout varlıklarıdır, ve o da zamansal olarak core-5 ile aynı dönemdir.
- Ekonomik değer (komisyon, slippage, funding sonrası PnL) test edilmedi. Alarmların çoğu, bilinen aritmetiğin 1–3 gün önceden sonucudur; işlem avantajı anlamına gelmez.
- Önceki turlardaki deneme borcu: v3'te 13 + 10 + 3 deneme, bu turda Worker B 11 walk-forward çağrısı ve lider 3 MC varyantı. Hepsi kayıtlı.

## Kaynaklar

- Liu & Tsyvinski, ["Risks and Returns of Cryptocurrency"](https://www.nber.org/papers/w24877), RFS 2021: kripto zaman serisi momentumu. Bu verideki etki zayıf ve kararsız.
- Lopez de Prado, purged walk-forward / embargo ([özet](https://www.quantt.co.uk/resources/purged-cross-validation))
- Levine & Pedersen, ["Which Trend Is Your Friend"](https://rpc.cfainstitute.org/research/financial-analysts-journal/2016/which-trend-is-your-friend): trend göstergeleri doğrusal filtrelerdir, bu yüzden pencere aritmetiği mekaniktir.
- [Binance public data](https://github.com/binance/binance-public-data), [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)
