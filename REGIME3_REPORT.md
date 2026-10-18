# ONEFILE — Rejim tahmini v3: havuzlanmış core-5, ortak harness

## Karar: NO_EDGE_80

Hedef kısmen öğrenilebilir: olay AP **0.439**, taban oranı 0.107. Ancak önceden kaydedilmiş birincil sistem %80 kapısını geçmedi. 2021–2024'te 48 alarmın 38'i doğru (**%79.2**, Wilson %95 [65.7%, 88.3%]); kapsam 143 fırsatın 38'i (recall %26.6). Ayarlamada hiç kullanılmayan 2023–2024'te 23 alarmın 17'si doğru (**%73.9**). Üç kapı koşulu kıl payı kaçtı: precision ≥ .80, Wilson alt sınırı ≥ .70 ve 2023–2024 precision ≥ .75. Sonuç %80 iddiası olarak sunulmuyor.

**Temel neden:** Öğrenilebilir sinyalin büyük bölümü, 20 günlük kayan pencerenin aritmetiğinden geliyor. Önümüzdeki 1–7 günde pencereden düşecek getiriler bugünden bilinir. Model–ötesi bir piyasa öngörüsü değil, yakın eşik geçişinin olasılığıdır: isabetlerin 20/38'i yalnızca **1 gün** önce. Kalan belirsizlik gelecek getirilerin yönüdür ve fiyat yönü bu veride öngörülemez kalıyor. Bu yüzden hassasiyet, alarm sayısı makul kaldıkça ~%75–80'de tıkanıyor. ML (lojistik veya XGBoost) bu yapısal olasılığın üzerine istatistiksel olarak anlamlı bilgi ekleyemedi.

## Kapsam ve protokol

- Veri: Binance USD-M 4h mumları + funding, BTC/ETH/SOL/XRP/BNB, **yalnız DEV 2020–2024**. 2025+ indirilmedi, sealed OOS açılmadı.
- Etiket: PR #5'teki genişletilmiş tanımın aynısı (`q`: z20 ≥ 1.25 / ≤ −1.25 ve er20 ≥ .35; dönüşler dahil her geçiş; sonraki 5 günün en az 3'ünde teyit; 1–7 gün içinde mevcut durumdan farklı ilk doğrulanmış başlangıç). Havuzlama, 4 yılda **33 → 143** puanlanabilir fırsat sağlar.
- Walk-forward: sekiz altı aylık dış katman (2021H1–2024H2), genişleyen eğitim; her sınırda 12 günlük etiket penceresi + 1 gün embargo; önceki 3 ay durdurma, sonraki 3 ay kalibrasyon. Varlıklar aynı takvim sınırlarını paylaşır.
- Eşikler yalnız geçmişten seçilir: önceki katmanların olgun OOF tahminleri + kalibrasyon dilimi. Politikalar: `precision80` (geçmişte ≥10 alarm ve ≥%80 precision veren en düşük eşik, yoksa kapalı), `budget` (≤12 alarm/varlık-yıl), `fixed70` (olasılık ≥ .70).
- Eşleştirme v2 ile birebir aynı: yön doğru olmalı, varlık başına 7 günlük cooldown, bire bir olay. Hızlandırılmış eşleştirici, test ile `regime_alarm_archive` çıktısına eşitliği doğrulanmış.
- **Ayarlama kapsamı:** `--regime3-scope tune` yalnız 2021–2022 dış katmanlarını puanlar. Bütün model, özellik ve politika seçimleri burada yapıldı. Birincil sistem ve başarı kapısı, 2023–2024 puanlanmadan önce `976dee1` commit'inde sabitlendi; final koşusu bir kez çalıştırıldı.

## Modeller

| Model | Açıklama | Ayar denemesi |
|---|---|---:|
| `prior` | Eğitim sıklığı | 0 |
| `structural` | Pencereden düşecek bilinen getirilerle rastgele yürüyüş geçiş olasılığı (parametresiz) | 0 |
| `mc` | Tam etiketin Monte Carlo'su: bilinen 20 günlük pencere + rv20 ölçekli Student-t(4) getiriler, 400 yol; geçiş, er20 ve 3/5 teyit kuralları birebir (parametresiz) | 2 (Gauss vs t4) + eşik |
| `logistic` (Worker B) | Yöne özgü iki başlık (yukarı: q≠+1, aşağı: q≠−1), 5 özellik: logit(rw_up/down), need_up/down, er_proj; C=1, sqrt ağırlık, sigmoid kalibrasyon | 13 |
| `xgb` (Worker A) | Yöne özgü iki başlık, yapısal logit üzerine artık (base_margin), 9 özellik, depth 2, 30 sabit tur, sqrt ağırlık, sigmoid kalibrasyon | 10 |

Worker A, eski XGBoost dalındaki bir hatayı doğruladı ve düzeltti: erken durdurma, 3 aylık durdurma diliminde katmanların çoğunda **1 ağaç** seçiyordu ve olay başlığı neredeyse sabit kalıyordu. PR #5'teki zayıf XGBoost sonucunun bir kısmı muhtemelen bu kusurdan kaynaklanıyor.

## Final karşılaştırma (2021–2024, 143 fırsat)

| Sistem | Politika | Alarm | Yakalanan / toplam | Precision (Wilson %95) | Recall | Yanlış alarm | AP | AP − prior | 2023–24 (dokunulmamış) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior | budget | 100 | 3 / 143 | 3.0% | 2.1% | 97 | 0.107 | — | 0/0 |
| structural | fixed70 | 123 | 64 / 143 | 52.0% [43.3, 60.7] | 44.8% | 59 | 0.374 | +0.267 | 28/65 = 43.1% |
| **mc (birincil)** | **fixed70** | **48** | **38 / 143** | **79.2% [65.7, 88.3]** | **26.6%** | **10** | **0.439** | **+0.332** | **17/23 = 73.9%** |
| mc | precision80 | 46 | 30 / 143 | 65.2% [50.8, 77.3] | 21.0% | 16 | 0.439 | +0.332 | 19/26 = 73.1% |
| logistic optimize | precision80 | 26 | 20 / 143 | 76.9% [57.9, 89.0] | 14.0% | 6 | 0.408 | +0.301 | 7/10 = 70.0% |
| logistic optimize | budget | 188 | 77 / 143 | 41.0% [34.2, 48.1] | 53.8% | 111 | 0.408 | +0.301 | 39/90 = 43.3% |
| XGBoost optimize | precision80 | 21 | 15 / 143 | 71.4% [50.0, 86.2] | 10.5% | 6 | 0.403 | +0.296 | 7/10 = 70.0% |
| XGBoost optimize | budget | 191 | 86 / 143 | 45.0% [38.1, 52.1] | 60.1% | 105 | 0.403 | +0.296 | 39/85 = 45.9% |

Eşleştirilmiş 28 günlük takvim-blok bootstrap ile AP farkları:

- mc − structural: **+0.065 [0.027, 0.106]**. Tam etiket simülasyonu, yaklaşık rastgele yürüyüş olasılığından anlamlı şekilde iyi.
- logistic − mc: −0.031 [−0.084, 0.017]. xgb − mc: −0.036 [−0.085, 0.007]. ML, parametresiz simülasyonu geçmiyor.
- logistic − structural: +0.034 [−0.030, 0.094]. xgb − structural: +0.029 [−0.017, 0.073]. Sıfırı içeriyor.

Yıllara göre (birincil mc/fixed70): 2021 12/13, 2022 9/12, 2023 9/11, 2024 8/12. Yıllık AP: mc 0.534 / 0.360 / 0.434 / 0.457. Logistic ve XGB, 2021 ve 2024'te mc'ye yakın, 2022–2023'te geride.

### Eski model ile kıyas (BTC, aynı 2021–2024)

| Sistem | Görev | Fırsat | Alarm | Yakalanan | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Eski H1-A M2 (özgün politika) | strict | 11 | 10 | 3 | 30.0% | 27.3% |
| v2 XGBoost (PR #5, dengeli) | genişletilmiş | 33 | 56 | 5 | 8.9% | 15.2% |
| v2 lojistik (PR #5, dengeli) | genişletilmiş | 33 | 108 | 20 | 18.5% | 60.6% |
| v3 mc/fixed70, BTC dilimi | genişletilmiş | 33 | 14 | 11 | 78.6% | 33.3% |
| v3 logistic/precision80, BTC dilimi | genişletilmiş | 33 | 7 | 5 | 71.4% | 15.2% |
| v3 XGBoost/precision80, BTC dilimi | genişletilmiş | 33 | 6 | 4 | 66.7% | 12.1% |

v2 AP (BTC): XGBoost 0.109, lojistik 0.116, taban 0.108; sıralama becerisi yoktu. Eski görevin tanımı farklı olduğundan bu satırlar doğrudan model üstünlüğü değildir. Aynı görevdeki v2→v3 farkı, bilinen pencere düşüş bilgisi ve havuzlamadan gelir.

## Önceden kaydedilmiş kapı (birincil mc/fixed70)

| Koşul | Sonuç |
|---|---|
| precision ≥ .80 | ✗ (.792) |
| Wilson alt sınırı ≥ .70 | ✗ (.657) |
| ≥ 40 alarm | ✓ (48) |
| ≥ 20 farklı isabet haftası | ✓ (28) |
| recall ≥ .10 | ✓ (.266) |
| 2023–24 precision ≥ .75, ≥ 15 alarm | ✗ (.739, 23) |
| her yıl precision ≥ .60 | ✓ |

## Sınırlar ve dürüstlük notları

- Bu, tarihsel DEV üzerinde keşif çalışmasıdır. Bağımsız OOS değildir; geçmiş deneme borcu bilinmiyor. 2023–2024 bu çalışmanın ayarlarından bağımsızdır, ama önceki PR'larda DEV'in parçası olarak görülmüştü.
- Varlıklar aynı günlerde birlikte hareket eder; 143 fırsat bağımsız 143 gözlem değildir. Wilson aralıkları alarmları bağımsız varsayar ve iyimserdir. AP aralıkları, takvim bloklarını varlıklar arasında ortak örnekler.
- Tahmin edilebilir kısım ağırlıkla 1–3 günlük, eşik yakınındaki geçişlerdir. "Erken" uyarı süresi kısadır. Ekonomik değer (komisyon, slippage, funding sonrası PnL) **test edilmedi**.
- Lojistiğin precision80'deki 20/26 sonucu en iyi 13 denemeden seçildi ve 2023–24'te 7/10'a düştü.
- `fixed70` eşiği 2021–2022 verisinde görüldükten sonra seçildi. Bu yüzden 2021–2022 satırları iyimserdir; dürüst kanıt 2023–2024 dilimidir (%73.9).

## Süreler

| Adım | Wall-clock |
|---|---:|
| Tune koşusu, 6 model, 4 katman (vektörleştirmeden önce / sonra) | 130 s / 11 s (aynı sonuçlar) |
| Tune koşusu, mc dahil 6 model | 25 s |
| Final koşusu, 6 model, 8 katman, bootstrap dahil | 40 s (araştırma 35.5 s) |
| Worker A (XGBoost) / Worker B (lojistik), paralel | 398 s / 336 s |
| Tam test paketi (141 test) + motor self-test (38 kontrol) | 36 s + 8 s |
| Toplam oturum (senkron → final) | ≈ 20 dk |

Darboğaz, eşik ızgarasındaki eşleşme döngüsüydü (her alarm için pandas filtresi × 120 eşik). Eşleşme adayları bir kez hesaplanıp ucuz bir açgözlü cooldown taramasıyla değiştirildi: 130 s → 11 s, bütün metrikler birebir aynı. MC özelliği core-5'in tamamı için yaklaşık 6–12 s sürer.

## Tekrar üretme

```bash
uv sync --frozen
uv run --no-sync python one.py --regime3 --regime3-scope tune
uv run --no-sync python one.py --regime3 --regime3-scope final   # core5_4h.csv yoksa DEV indirir
uv run --no-sync python -m unittest discover -s tests
```

Kanıt: `regime3_final.json` (bütün katmanlar, eşikler, katsayılar, yıl/varlık kırılımları ve kapı sonucu). Tahminler: `regime3_final_forecasts.csv` (gitignore'da; yeniden üretilebilir).

## Kaynaklar

- Lopez de Prado, *Advances in Financial ML*, purge/embargo ([özet](https://www.quantt.co.uk/resources/purged-cross-validation))
- Levine & Pedersen, ["Which Trend Is Your Friend"](https://rpc.cfainstitute.org/research/financial-analysts-journal/2016/which-trend-is-your-friend): MA/TSMOM sinyallerinin doğrusal filtre eşdeğerliği
- [Path signatures for crypto regime detection](https://nhsjs.com/wp-content/uploads/2026/08/Path-Signatures-for-Regime-Detection-in-Cryptocurrency-Markets-A-Rough-Path-Framework-Using-Spot-Perpetual-Basis-and-Funding-Rates.pdf): purged walk-forward ve nadir olaylarda AP
- [XGBoost parametreleri](https://xgboost.readthedocs.io/en/stable/parameter.html), [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)
