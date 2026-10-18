# ONEFILE — XGBoost rejim tahmini: dört yıllık değerlendirme

## Sonuç: NO_EDGE

XGBoost uygulandı; araştırma tamamlandı. Yeni model önceki sistemi veya eğitimden hesaplanan basit önseli güvenilir şekilde geçmedi. %80 başarı elde edilmedi. Yeni model araştırma adayıdır; canlı kullanıma veya eski modelin yerine terfi ettirilmedi.

**Kapsam:** BTCUSDT USD-M, 4h gerçek Binance mumları ve tarihsel funding. Eğitim/özellik geçmişi 2020–2024; dış değerlendirme **2021-01-01 ≤ tahmin zamanı < 2025-01-01**, sekiz altı aylık katman. 2020 başlangıç eğitim yılıdır. Hiçbir 2025+ arşivi indirilmedi. Önceden incelenmiş 2026 son altı ayı bu deneyde kullanılmadı ve yeni holdout diye sunulmadı. Bu dört yıl tarihsel DEV'dir; bağımsız doğrulama değildir.

10.962 mum ve 5.481 funding gözlemi eksiksiz. Mum/funding zamanları ve OHLC geometrisi kontrol edildi; funding değerleri SHA kontrolü yapılan resmi arşivlerle karşılaştırıldı. Girdi SHA-256: `9083e90d4aaab0af67fd1e4fec13658700a984c8839d5ecfab3ee956c17c3ffe`.

## Darboğaz ve değişiklikler

1. Eski görev her rejim dönemini tahmin etmiyordu: en az 10 gün nötr kalmış piyasadan geçiş ve sonraki 10 günün en az 8'inde aynı `q` koşulu gerekiyordu. Bu, dört dış test yılında yalnız **11** puanlanabilir başlangıç bırakıyor. Önceki %75,8 de boğa/ayı/yatay piyasa sınıflandırması değil, uygun günlerde gelecek başlangıç etiketinin doğruluğuydu.
2. Yeni görev `q` tanımını değiştirmiyor: Z20 ≥1,25 / ≤−1,25 ve ER20 ≥0,35. Bütün yönlü geçişler ve doğrudan tersine dönüşler adaydır. Sonraki 5 günün en az 3'ünde aynı `q` aranır. Önceki keyfi 10 günlük nötr filtresi kaldırıldı.
3. Gelecek 1–7 günde **mevcut durumdan farklı** bir yöne doğrulanmış başlangıç tahmin edilir. Bugünkü bilinen rejimi veya aynı rejimin devamını erken başlangıç diye saymak yasaktır. `none`, yatay piyasa demek değildir; hedef ufkunda böyle bir başlangıcın olmaması demektir.
4. Yeni tanımda **33 doğrulanmış ve 32 başarısız geçiş** var. Eski tanımda **11 doğrulanmış ve 24 başarısız** geçiş var (yalnız 2021–2024). Başarısız geçişler arşivde tutulur; etiket/politika içinde negatif ve yanlış alarm olarak kalır. Farklı teyit tanımıyla sayının 3 katına çıkması, ekonomik fırsatların veya kalitenin 3 kat arttığının kanıtı değildir.
5. Doğrulanmış olay sayısı hâlâ küçük. Aynı olayın önceki yedi güne kopyalanan etiketleri bağımsız yeni olay sayılmadı. Eğitimde pozitif olayların tekrar ağırlığı azaltıldı; değerlendirmede her olay en fazla bir kez yakalanabilir.
6. Eski `1.01` alarm eşiği yerine yeni politikalarda açık `null` = kapalı politika kullanılıyor. Kapalı politikanın precision'ı tanımsızdır, %100 değildir.

## Karşılaştırma

Eski özgün M2, ilk eğitim yılını 2020'ye alarak dört yıllık aralığa genişletildi; lojistik model, iç kronolojik C/sıcaklık/eşik seçimi korundu. Standart `--research` komutunun başlangıç yılı değiştirilmedi. Özgün üç yıllık başlangıç eğitimiyle, sealed veriye erişmeden dört dış test yılı elde edilemeyeceği için bu fark açıkça belirtilmiştir.

Aşağıda yeni modeller için **dengeli politika** bulunur. Eski görev ile genişletilmiş görev arasındaki skor farkları hedef tanımı da değiştiğinden doğrudan model üstünlüğü değildir. Aynı görev içindeki lojistik/XGBoost kıyaslaması model etkisini ayırır.

| Görev / model | Olgun tahmin | Fırsat | Yakalanan | Alarm | Precision | Recall | Coverage | Yanlış alarm | Medyan erkenlik (gün) | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eski H1-A M2 / özgün politika | 704 | 11 | 3 | 10 | 30.00% | 27.27% | 1.42% | 7 | 1.0 | 0.3503 | 34.93% |
| strict / no_event | 704 | 11 | 0 | 0 | — | 0.00% | 0.00% | 0 | — | 0.3151 | 33.33% |
| strict / prior | 704 | 11 | 2 | 27 | 7.41% | 18.18% | 3.84% | 25 | 2.0 | 0.3151 | 33.33% |
| strict / logistic | 704 | 11 | 5 | 62 | 8.06% | 45.45% | 8.81% | 57 | 3.0 | 0.2828 | 29.14% |
| strict / xgb | 704 | 11 | 2 | 64 | 3.12% | 18.18% | 9.09% | 62 | 4.0 | 0.2678 | 25.14% |
| expanded / no_event | 1450 | 33 | 0 | 0 | — | 0.00% | 0.00% | 0 | — | 0.3143 | 33.33% |
| expanded / prior | 1450 | 33 | 0 | 0 | — | 0.00% | 0.00% | 0 | — | 0.3143 | 33.33% |
| expanded / logistic | 1450 | 33 | 20 | 108 | 18.52% | 60.61% | 7.45% | 88 | 2.5 | 0.3156 | 32.89% |
| expanded / xgb | 1450 | 33 | 5 | 56 | 8.93% | 15.15% | 3.86% | 51 | 3.0 | 0.3143 | 33.33% |

Coverage = alarm sayısı / olgun, uygun tahmin başlangıcı sayısı. Bir alarmdan sonraki yedi gün için fazladan alarm üretilmez. Recall = yakalanan ayrı olay / puanlanabilir ayrı doğrulanmış olay. Olaydan önce 1–7 gün ve doğru yön gerekir. Başarılı örnekler denetim kümesinden seçilmez.

**%80 precision politikası:** XGBoost'un sekiz katmanının hiçbirinde önceki kalibrasyon diliminde en az 5 alarm ve ≥%80 precision koşulu sağlanmadı. Üç aylık kalibrasyonda az sayıda gerçek olay bulunması da bu destek koşulunun sağlanmasını zorlaştırır; bu deney %80 precision evrensel olarak imkânsızdır iddiası değildir. Politika bütün katmanlarda kapalı: **0 alarm, 0/33 yakalama, %0 coverage; precision tanımsız**. Dengeli politikadaki düşük precision sonuçları gizlenmedi.

### Güven aralıkları ve baselinelar

Takvimde 28/56/84 günlük bloklarla 1.000 tekrar; en geniş %95 aralık raporlandı. Bunlar yaklaşık durağan blok varsayımına bağlı, bağımsız olay garantisi değildir.

- Yeni XGBoost precision: **8.93%**, %95 aralık **1.85%–18.75%**.
- Recall: **15.15%**, aralık **3.03%–31.26%**.
- Coverage: **3.86%**, aralık **1.86%–6.07%**.
- Yanlış alarm/yıl: eski özgün M2 **1.77**, yeni lojistik **22.17**, yeni XGBoost **12.85**. Payda olgun etiketli takvim maruziyetidir; yalnız alarm veya risk günleri değildir.
- Yeni görevde Brier (düşük iyi): eğitim önseli **0.1995**, lojistik **0.2272**, XGBoost **0.2028**.
- XGBoost'un önsele göre eşlenmiş günlük Brier kazancı **-0.00327**, %95 aralık **[-0.01128, 0.00457]**. Pozitif olsaydı XGBoost lehine olacaktı; burada kanıt yok.
- Lojistiğe göre Brier kazancı **0.02439**, aralık **[-0.00559, 0.06430]**; sıfırı içeriyor.
- Yeni XGBoost'un argmax doğruluğu **89.17%**; sürekli `none` baseline'ı **89.17%**. Argmax tüm günlerde `none` seçiyor. Yüksek ham doğruluk rejim tahmin başarısı değildir.

### Yeni XGBoost'un yıllara göre dengeli politikası

| Yıl | Fırsat | Yakalanan | Alarm | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 6 | 0 | 0 | — | 0.00% | 0.1445 |
| 2022 | 11 | 0 | 22 | 0.00% | 0.00% | 0.2316 |
| 2023 | 9 | 1 | 2 | 50.00% | 11.11% | 0.2118 |
| 2024 | 7 | 4 | 32 | 12.50% | 57.14% | 0.2239 |

Yıllık tablolar olay sayımı için yıl sınırındaki olayı ilgili altkümeden puanlar; küresel rapordaki alarm cooldown'u bütün yıllar boyunca korunur. Yıl başında kalan önceki yıl alarmının etkisi nedeniyle yıllık alarm sayıları her zaman küresel sayılara toplanmayabilir.

## Model, leakage ve kapasite

- Resmi XGBoost **3.4.1**, `hist`, CPU, depth=2, min_child_weight=10, eta=.05, L2=10, L1=.1, gamma=.1, max_bin=64, subsample/colsample=.8, seed=42. Tek önceden belirlenmiş kapasite; geniş hiperparametre araması yok.
- Olay oluşumu ve pozitif olaylarda yön için iki binary sınıflandırıcı; ortak üç sınıf olasılığı bu iki olasılıktan kurulur. Lojistik baseline aynı faktörizasyonu, özellikleri ve ağırlıkları kullanır. Önsel yalnız geçmiş eğitim sıklıklarıdır.
- Capped sqrt sınıf ağırlıkları ve ters olay tekrar sayısı yalnız eğitimden hesaplanır. Bilinen funding ve bitmiş mumlardan hesaplanan fiyat/hacim özellikleri kullanılır. Veri dönüşümleri geleceğe bakmaz.
- Her dış katmandan önce 6 aylık doğrulamanın ilk 3 ayı erken durdurmaya, sonraki 3 ayı sigmoid kalibrasyona ve politika seçimine ayrıldı. Durdurma dilimindeki en iyi ağaç sayısıyla önceki eğitim/durdurma verisi üzerinde yeniden fit yapılır. Kalibrasyon verisi fit'e katılmaz. Az örnekli kalibrasyon sınıfında kimlik dönüşümüne dönülür ve kaydedilir.
- Yeni hedefin tüm bilgi ufku **12 gün**, eski hedefin **17 gün**; her sınırda buna ek **1 gün embargo** uygulanır. Random split, dış testle erken durdurma veya dış testte eşik ayarı yapılmadı. Dış testin henüz olgunlaşmamış son tahminleri CSV'de kalır ama negatif sayılmaz.
- Lojistik yön başlığında eğitim log-loss yaklaşık **0,035–0,090**, daha sonraki kalibrasyon diliminde bazı katmanlarda **0,88–1,37**. Bu fark aşırı uyum veya dağılım değişimi belirtisidir; iyi eğitim skoru üstünlük sayılmadı.
- XGBoost'ta sonraki başlangıç yönlerinin argmax recall'ı %0; balanced accuracy %33,33. Kapasite eksikliği/yoğun regularization veya yetersiz sinyal olasılıklarını yalnız bu sonuçtan ayırmak mümkün değildir. Dış testte sonuç görüldükten sonra ağaçlar derinleştirilmedi. Erken durdurulan olay başlığı 18–143 tur kullandı; 256 turluk sınırın doldurulmaması tek başına underfit olmadığını kanıtlamaz.
- %80 başarı kapısı; precision güven aralığı alt sınırı ≥.8, recall alt sınırı ≥.3, ≥50 alarm, ≥30 olay, ≥12 takvim olay kümesi ve en az 3 yılda pozitif eşlenmiş Brier kanıtı ister. Bu koşullar karşılanmadı. Geçseler bile sonuç tarihsel DEV'de yalnız keşif adayı olacaktı.

## Süreler ve performans

Son kaynak ve resmi CPU paketi koşusundaki wall-clock saniyeleri:

| Aşama | Saniye |
|---|---:|
| Günlük özellikler | 0.071 |
| Ek özellikler | 0.006 |
| İki etiket kümesi | 0.406 |
| Eski özgün iç-içe deney, tüm modeller/seçimler | 4.125 |
| Strict görevin dört challenger/baseline kıyaslaması | 3.864 |
| Expanded görevin dört challenger/baseline kıyaslaması | 4.176 |
| Bütün araştırma, istatistik ve çıktı yazımı | 13.052 |
| Soğuk CLI: import, kaynak doğrulama ve E2E audit dahil | 18.626 |

| Sekiz dış katmanın toplamı | Eğitim | Çıkarım | Kalibrasyon | Politika seçimi |
|---|---:|---:|---:|---:|
| Eski özgün H1-A: M0/M1/M2/placebo iç seçimi dahil | 1.606 | 0.146 | birleşik | birleşik |
| strict / logistic | 0.286 | 0.051 | 0.095 | 0.306 |
| strict / xgb | 1.254 | 0.141 | 0.181 | 0.229 |
| expanded / logistic | 0.302 | 0.058 | 0.072 | 0.368 |
| expanded / xgb | 1.097 | 0.089 | 0.100 | 0.281 |

Bu toplamlar farklı iş yükleridir; yeni toplamın eskisinden hızlı olduğu iddia edilmiyor. Eski özgün deney 4.12s, yeni görevin dört modelinin tüm değerlendirmesi 4.18s sürdü. Yeni lojistik/XGBoost'un aynı görevdeki eğitim süreleri tabloda ayrıdır. Soğuk başlangıç payı model eğitim süresi gibi sunulmadı.

Kontrollü aynı 2020 eğitim/durdurma/kalibrasyon iş yükü; bir ısınma ve üç tekrar:

| CPU thread | Medyan saniye |
|---|---:|
| 1 | 0.1226 |
| 2 | 0.1487 |
| 4 | 0.1574 |

Final CPU paketinde tek thread, dört threade göre **%22.1 daha kısa wall-clock** verdi. Varsayılan bu nedenle 1. İlk standart wheel CPU-hist ölçümünde fark %32 idi; iki ölçüm de korunuyor. Resmi CPU wheel 5,5 MiB; standart wheel 54,9 MiB ve ek 291 MiB NCCL indiriyordu. Bu kurulum boyutu kazanımıdır; CPU wheel'in kendisinin daha hızlı olduğu iddia edilmiyor.

Dört araştırma koşusu tamamlandı: ilk uygulama, protokol/ihracat/payda düzeltmeleri, CPU dağıtımı doğrulaması ve ilk-olay eşleştirme regresyon düzeltmesi. Son değişiklik; etiket ilk gelecek başlangıcı söylerken eşleştiricinin yanlış yöndeki ilk olayı atlayıp sonraki uygun yönü başarı sayabilmesini engelledi. Gerçek örnek sayıları ve skorlar değişmedi. Model hiperparametreleri, etiket tanımı veya seçim hedefi dış sonuçlara göre değiştirilmedi. İlk/final ortak olasılıklar **birebir aynı; maksimum mutlak fark 0** (`regime_integrity.json`).

Ortam yüküyle tam CLI süreleri 17–37s arasında değişti; bu kontrolsüz fark bir model hız kazanımı sayılmadı. Son tekrar 18.626s. Paylaşılan özellikler katman başına yeniden hesaplanmaz; ZIP cache SHA-256 ile doğrulanır. Per-run E2E/AST döngü kayıtları pakette.

Ticaret motoru bu araştırmada çalıştırılmadı. Sonuçlar kâr, komisyon/slippage sonrası getiri veya $500 hesap performansı değildir.

## Doğrulama ve tekrar üretme

- Tam paket eşleştirme düzeltmesinden önce **136 test / 56,154s** geçti; motor self-test **38 kontrol** geçti. Son eşleştirme değişikliğinden sonra etkilenen **10 rejim testi / 0,491s** yeniden geçti; değişmeyen motor testleri yeniden koşturulmadı. Bu kontroller piyasa üstünlüğü kanıtı değildir.
- Yeni testler; future corruption, label maturity, failed/censored olaylar, doğrudan reversal, yanlış ilk yönden sonraki uygun olaya atlamayı reddetme, purge/embargo, test etiketleri değişince aynı XGBoost tahmini/ağaç sayısı, tekil olay eşleştirme, mevcut rejim devamını onset saymama ve sealed CLI reddini kapsıyor.
- Kaynak SHA-256: `9482a3370262c229ef71bc21af0958005d650f677484326a2f4b35218c3ecda3`. `regime_protocol.json` skor öncesi yazıldı, son değişmeyen kaynakla tekrar kullanımı doğrulandı. Önceki deneme arşivde tutuldu. Tarihsel deneme borcu bilinmediğinden sonuç bağımsız test olarak sunulmuyor.

```bash
uv sync --frozen
# Aynı kaynak ve veriyle mevcut protokol yeniden kullanılabilir:
uv run --no-sync python one.py --download --regime-research
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python one.py --self-test
```

`regime_evidence_bundle.zip` tam CSV'leri, özgün/challenger olasılıklarını, başarısız olayları, kaynak ve kilit dosyasını, ilk denemeyi ve logları içerir. Ana dizindeki eski bilimsel kanıt dosyaları değiştirilmedi. Gerçek kazanç bulunmadığı için yeni model üretime terfi ettirilmedi.

## Resmi kaynaklar

- [XGBoost parametreleri: hist, regularization, thread](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost sklearn: erken durdurma ve best_iteration](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- [XGBoost tuning: bias/variance ve sınıf dengesizliği](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [Resmi CPU wheel kurulumu](https://xgboost.readthedocs.io/en/stable/install.html)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Binance public data](https://github.com/binance/binance-public-data)
