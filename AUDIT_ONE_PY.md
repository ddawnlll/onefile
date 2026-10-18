# one.py — sağlamlık ve güvenlik incelemesi

Tarih: 2026-10-17

## Güncel kapanış — kullanıcıdaki altı araştırma bulgusu

Bu bölüm güncel kod içindir; aşağıdaki önceki denetim kapanışı ve eski PASS sonuçları tarihsel kayıttır.

1. **Fail-closed veri kapısı:** Başarısız yapısal kabul veya online funding kaynak karşılaştırması özellik/simülasyon/başarılı çıktı öncesinde durur. PASS metinleri gerçek kontrolden üretilir. Eksik şemada readiness artık KeyError yerine FAIL verir.
2. **Funding:** Eksik kayıt, sıfır oran ve gözlem bayrağı birbirinden ayrılır. Varsayılan komut tüm settlement değerlerini taze Binance HTTPS arşivleriyle karşılaştırır; bayraklar korunarak sıfırlanmış CSV reddedilir. Gerçek kaynak sıfırı geçerlidir. Açık `--offline` modu kaynağı UNVERIFIED işaretler ve istatistiksel PASS alamaz.
3. **İstatistik:** IID/permutasyon kaldırıldı. Eşlenmiş aylık toplam/sayılar birlikte, boş aylar korunarak 1/3/6 aylık dairesel bloklarla örneklenir. En geniş CI zarfı ve en büyük null-centered bootstrap p raporlanır. Varsayım yaklaşık durağanlıktır; bu, tüm piyasa bağımlılıklarını çözme garantisi değildir. Gerçek toplam deneme sayısı bilinmeden istatistiksel PASS yok; `--trials` verilirse Bonferroni uygulanır. Edge kanıtı her durumda false kalır.
4. **Kontroller:** Her kontrol, eşlenmeyenler dahil bütün setup 42-bar pencerelerinden ve diğer kontrollerden ayrıdır. Kontrol bulunamadığında çalışma çökmez; tam setup listesi korunur, getiriler kullanılmadan aynı ayda eşlenebilen alt küme karşılaştırılır. CSV ve JSON eşlenen/eşlenmeyen sayısını açıklar. Sonuç yalnız bu ortak destek alt kümesine aittir.
5. **Behavior örneklemesi:** Koşula özgü ilk uygun bardan başlayan seçim yerine bütün koşullara ortak mutlak UTC grid'i kullanılır. 43 fazın tamamı başlangıç hassasiyeti olarak gösterilir; çakışan fazlar/koşullar bağımsız örnek sayılmaz. H1-A ileriye bakarak doğrulanan onset'leri açıkça retrospektif etiket olarak belirtilir.
6. **500 dolar hesabı:** İlk bulgu zaten kod hatası değil model sınırıydı. Sabit 1 BTC/sınırsız araştırma nakdi olduğu terminalde, metadata'da ve README'de açıkça yazılır. Yeni bir özel margin/likidasyon motoru eklenmedi; gerçek hesap getirisi iddiası yoktur.

### Güncel yerel doğrulama

- 38 native vectorbt sentetik kontrol ve 90 offline regresyon testi; kaynak tahrifatı, kaynak kesintisi, ortak kontrol desteği, çakışma, örnekleme grid'i, blok eşlemesi ve bilinmeyen deneme geçmişi için negatif testler içerir.
- Taze Binance DEV indirmesi ile Brick 1 yeniden çalıştırıldı: **10.962 mum, 5.481 settlement, 66 tam setup, 37 eşlenen setup/kontrol çifti**. Eşlenemeyen 29 setup kaybolmaz.
- Tam setup ortalaması +0.528R; karşılaştırma alt kümesi +0.098R, kontrol -0.717R. Eşlenen setup CI zarfı [-0.454R, +0.680R]; güncel karar **SCREEN FAIL**. Eski 66/66 çakışan kontrol ve IID p=0.0250 sonucu artık geçerli kanıt değildir.
- Güncel işlem CSV'si ve evidence JSON birlikte yeniden üretilir; kod/veri hash'leri, kaynak durumu, eşleme kapsamı ve karar istatistikleri kaydedilir.
- GitHub'a push/uzak CI bekleme yok; bu turda CI dosyası değiştirilmedi. Yerel bağımsız salt okunur inceleme, eski evidence dosyasının güncellenmesini istedi; bu yüzden çıktılar son kodla yenilendi.
- Gerçek OOS açılmadı. Bu kontroller kusursuz yazılım, canlı güvenlik veya kalıcı edge garantisi değildir. Dosyalar tek tek atomiktir; iki sonuç dosyası birlikte transaction değildir.

## Önceki veri/operasyon denetiminin tarihsel kapanışı — güncel sonuç değildir

**Aşağıdaki altı madde farklı bir önceki veri/operasyon incelemesine aittir.** Eski örnek sayıları/PASS ifadeleri yukarıdaki güncel araştırma kapanışıyla değiştirilmiştir. Stratejinin native vectorbt giriş/çıkış yürütme kuralları korunmuştur.

| Bulgu | Uygulanan düzeltme | Doğrulama |
| :--- | :--- | :--- |
| 1. Veri kapısı | Simülasyondan önce zorunlu kabul; başarısızlıkta çıkış kodu 2; gerçek durum etiketleri | Gerçek CLI süreci dahil, başarısız veride özellik/simülasyon/çıktı yok |
| 2. Funding kapsamı | `funding_observed` alanı; birleşim öncesi sonluluk ve tam takvim; gerçek sıfıra izin | Eksik/NaN/sonsuz/tekrarlı/takvim dışı kayıtlar reddediliyor; sıfır ve negatif oranlar korunuyor |
| 3. Dosya kaybı | Yazma öncesi yol/alias kontrolü; doğrulanmış indirme; dosya başına atomik yazım | Tüm yol çiftleri, normalize yollar, symlink/hardlink ve yazma/replace hataları sınandı |
| 4. Kontrol eşlemesi | Seçimde ve karar kapısında ay bazında tam eşleme; iki grup için N >= 30 | 30/1 ve yanlış ay dağılımları KILL; geçerli dengeli örnek PASS |
| 5. OOS geçmişi | Özellikler DEV başlangıcından itibaren hesaplanıyor; aktif aralık sonra seçiliyor | OOS özellikleri kesintisiz hesaplamayla birebir aynı; gelecekteki OOS fiyatları DEV'i değiştirmiyor |
| 6. Sembol kimliği | CLI/downloader yalnız BTCUSDT; CSV sembolü zorunlu ve doğrulanıyor | ETHUSDT, karışık ve eksik semboller reddediliyor |

### Son doğrulama

- CI ile aynı kilitli Python 3.12 ortamında derleme, **38/38 mevcut self-test** ve **61/61 yeni offline regresyon testi** geçti. CI yeni testleri de çalıştırıyor.
- Ruff temiz. Bandit yalnız sabit HTTPS `urlopen` kullanımlarına ilişkin önceki iki `B310` uyarısını bildiriyor; doğrulanmış yeni güvenlik bulgusu yok.
- Kanonik Binance DEV verisi gerçekten indirildi ve doğrulandı: **10.962 mum, 5.481 gözlenmiş settlement, 66 setup / 66 kontrol**. Mevcut kodla varsayılan komut da başarıyla çalıştı; `brick1_evidence.json` yenilendi.
- Aynı indirilen veriyle eski kod ayrıca çalıştırıldı: **132 trade satırının 23 sütununun tamamı birebir eşleşti**. Veri güvenliği düzeltmeleri strateji PnL/yürütme mantığını değiştirmedi.
- Yeni DEV snapshot'ında setup ortalama net R `+0.528`, kontrol `-0.077`; sonuç hâlâ yalnız `EXPLORATORY PASS`. Gerçek OOS holdout açılmadı.
- CSV'ye `funding_observed` ve `symbol` eklendi. Eski yedi sütunlu dosyalar sessizce dönüştürülmüyor; kullanıcıya `--download` ile yeniden indirme hatası veriliyor. Geçiş README'de açıklandı.
- Yerel `btc_4h.csv` ve `brick1_trades.csv` `.gitignore` kapsamındaki doğrulama çıktılarıdır. Geçici karşılaştırmalar/günlükler `/tmp/onefile-audit-qLArCs/` altındadır.

```bash
uv sync --frozen --python 3.12
uv run python one.py --self-test
uv run python -m unittest discover -s tests -v
uv run python one.py --download
```

**Sınırlar:** Sentetik testler piyasa avantajı veya canlı işlem güvenliği kanıtı değildir. CSV metadata'sı kasıtlı sahteciliğe karşı kriptografik doğrulama sağlamaz. İki sonuç dosyası ayrı ayrı atomiktir; birlikte bir transaction değildir. Python 3.12 dışındaki sürümler bu düzeltmede sınanmadı.

## İlk inceleme sonucu (aşağıdaki kanıtlar düzeltme öncesine aittir)

Üç yüksek öncelikli veri bütünlüğü/operasyon hatası ve üç orta öncelikli araştırma doğruluğu hatası doğrulanmıştı. Bunlar doğrulanmış uzaktan saldırı açıkları olarak sınıflandırılmıyor. Aşağıdaki satır numaraları ve kaynak hash'i ilk incelenen sürüme aittir; kapanış durumu yukarıdadır.

İncelenen kod yollarında kritik uzaktan kod çalıştırma, kimlik bilgisi sızıntısı veya kimlik doğrulama atlatma açığı doğrulanmadı. Uygulama şu anda canlı emir göndermeyen bir CLI araştırma motoru; bu değerlendirme gelecekte eklenecek servis/canlı işlem katmanını kapsamaz.

## İlk incelemenin kapsamı ve kanıtı

- İncelenenler: `one.py` dosyasının tamamı, `pyproject.toml`, `uv.lock`, CI, README ve mevcut evidence JSON'u.
- Kaynak SHA-256: `c5e0f9eccf886bdbd95b678236aa29cb484c27e5f5671911ccafa36b31ba7b53`.
- İzole ortam: Python 3.12.15, vectorbt 1.1.2, NumPy 2.5.4, pandas 3.0.6; bağımlılıklar `uv sync --frozen --python 3.12` ile kuruldu.
- Derleme kontrolü geçti; mevcut `--self-test`: **38/38 kontrol geçti**.
- İlave **10 sorun gösterim senaryosu** yeniden üretildi. Bunlar başarılı güvenlik testleri değil, mevcut zayıflıkları gösteren kontrollü deneylerdir.
- `pip-audit`: kurulu **59 pakette bilinen açık bildirmedi**. Bu sonuç bilinmeyen açık bulunmadığını garanti etmez.
- Bandit: iki `B310` uyarısı (`urlopen`, satır 70 ve 89). URL'nin HTTPS kökü kodda sabit olduğundan bu uyarılardan doğrulanmış SSRF veya yerel dosya okuma açığı çıkarılmadı.
- Ruff: beş import/stil/temizlik uyarısı; çalışma zamanı hatası olarak değerlendirilmedi.
- Gerçek `btc_4h.csv` depoda/çalışma dizininde bulunmadığı için README'deki piyasa sonuçları yeniden hesaplanmadı. Downloader yerel, sentetik ZIP yanıtlarıyla sınandı; Binance'e veri indirme isteği yapılmadı.
- Python 3.10/3.11/3.13/3.14 çalışma uyumluluğu bu incelemede sınanmadı.

Çalıştırılan deneyler ve günlükler bu oturumda `/tmp/onefile-audit-qLArCs/` altında tutuldu. Bu dizin kalıcı CI altyapısı değildir.

## 1. Yüksek — başarısız veri kabul kontrolü analizi durdurmuyor

**Yer:** `one.py:786-791`, `one.py:812-816`, `one.py:828-870`.

`check_data_readiness()` sonucu, olaylar hesaplandıktan sonra alınıyor ve `FAIL` olduğunda hata yükseltilmiyor. Ekrandaki mum/funding/chronological `[PASS]` etiketleri gerçek kontrol sonucundan üretilmiyor.

**Doğrulama:** Beklenen 10.962 mum yerine yalnızca 180 mum ve tamamı sıfır funding içeren sentetik CSV ile CLI:

- `readiness.verdict = FAIL` üretti;
- buna rağmen `exit code = 0` ile tamamlandı;
- `[PASS] Candle Continuity: 180 / 10,962` yazdırdı;
- `[PASS] 8h Funding Coverage: 90 / 5,481` yazdırdı;
- `[FAIL] REAL DEV DATA ACCEPTED & READY` yazdırdı;
- çıktı/evidence dosyalarını oluşturdu.

**Etki:** Eksik veya kabul edilmemesi gereken veri, otomasyon ve kullanıcı açısından başarılı çalışmaya benzeyebilir. Bu deneyde olay olmadığı için hipotez kapısı ayrıca KILL verdi; tespit edilen hata veri kapısının çalışmayı durdurmamasıdır.

**Düzeltme:** Veri kabulünü simülasyondan önce uygula; FAIL durumunda sıfır olmayan çıkış koduyla durdur. Alt kontrol etiketlerini gerçek sonuçlardan üret. Başarısız çalışmayı başarılı araştırma çıktısı olarak kaydetme.

## 2. Yüksek — eksik funding sıfıra çevrilebiliyor; kapsam kontrolü bunu yakalamıyor

**Yer:** `one.py:100-118`, `one.py:143-146`, `one.py:170-179`.

Downloader settlement zamanlarını kontrol ediyor fakat settlement oranının sonlu/sayısal olarak mevcut olduğunu birleşimden önce doğrulamıyor. Sonraki `fillna(0.0)`, hem settlement olmayan mumları hem de oranı eksik gerçek settlement kayıtlarını sıfıra çevirebiliyor.

CSV kabul kontrolü ise gerçek settlement kayıtlarının varlığını değil, mumların 00:00/08:00/16:00 saatlerine denk gelmesini sayıyor. Funding kapsamı için tek bir sıfır olmayan oran yeterli oluyor. BTC için settlement dışı 04:00 gibi saatlerdeki funding de reddedilmiyor.

**Doğrulama:**

1. Tam DEV mum takviminde 5.481 settlement diliminden yalnızca biri sıfır olmayan funding içerirken kapsam ve genel verdict PASS oldu.
2. Bu seriye 04:00 UTC funding eklendiğinde de kabul edildi.
3. Gerçek downloader fonksiyonu, ağ yerine yerel ZIP yanıtlarıyla çalıştırıldı: Ocak 2020'nin 93 settlement kaydından 92'sinin oranı NaN idi. Downloader bunları sıfıra çevirdi; sonrasında veri kabul kapısı PASS verdi.

**Etki:** Funding maliyeti, funding tabanlı özellikler ve net R sonuçları sessizce yanlış hesaplanabilir.

**Düzeltme:** Settlement kayıtlarının mevcudiyetini oran değerinden ayrı izle. Oranları birleşim öncesi sonluluk kontrolünden geçir. Yalnızca settlement olmayan zaman dilimlerine sıfır yaz; BTC için beklenmeyen settlement zamanlarını reddet. **Gerçek sıfır funding geçerli olabilir; çözüm tüm oranları sıfırdan farklı olmaya zorlamak değildir.** CSV için doğrulanabilir kaynak/settlement kapsam bilgisi gerekir.

## 3. Yüksek — çıktı yolu girdi dosyasını silebiliyor

**Yer:** `one.py:781-784`, `one.py:828`, `one.py:869`.

`--csv`, `--out` ve `--evidence` yollarının çakışması engellenmiyor. Yazımlar doğrudan hedef dosyaya yapılıyor.

**Doğrulama:** Yalnızca geçici sentetik dosyada `--out` girdinin yoluna eşitlendi. Komut başarıyla tamamlandı; 10.305 baytlık kaynak CSV, olay bulunmadığından **1 baytlık boş çıktıyla değiştirildi**.

**Etki:** Yanlış CLI parametresi kaynak veri kaybına yol açabilir. Bu, mevcut yerel CLI bağlamında veri kaybı riskidir; ayrıca bir uzaktan yetki yükseltme açığı değildir. `--evidence` yolunun girdiye veya trade çıktısına eşitliği de kodda engellenmiyor.

**Düzeltme:** Herhangi bir yazma/indirme öncesinde normalize edilmiş yolları ve mevcut dosya kimliklerini karşılaştır; girdi, trade ve evidence çakışmalarını reddet. İndirme ve çıktı yazımlarında geçici dosya + doğrulama + atomik değiştirme kullan.

## 4. Orta — eksik ay eşlemeli kontrol grubu karar kapısından geçebiliyor

**Yer:** `one.py:287-298`, `one.py:601-616`, `one.py:871-873`.

Kontrol seçimi bazı aylarda hedef sayıya ulaşamayınca hata vermiyor. Karar kapısı örneklem alt sınırını yalnızca setup için kontrol ediyor; grup büyüklüğü ve aylık eşleme şartını doğrulamıyor. Eksiklik uyarısı karar kapısından sonra geliyor.

**Doğrulama:**

- Seçici, sentetik aynı ay örneğinde 2 setup / 1 kontrol döndürdü.
- Kapıya 30 adet `+2R` setup, yalnızca 1 adet `-1R` kontrol ve diğer eşikleri sağlayan yol ölçümleri verildi.
- Sonuç: `p = 0.0335966403`, `EXPLORATORY PASS`.

**Etki:** Zorunlu ay eşlemeli karşılaştırma sağlanmadan olumlu karar üretilebilir. Bu, mevcut gerçek piyasa sonucunun yanlış olduğunu tek başına kanıtlamaz.

**Düzeltme:** Karardan önce iki grubun minimum büyüklüğünü ve ay bazında eşleşmesini doğrula; eksiklikte yetersiz veri sonucu ver veya önceden tanımlanmış dengeli örnekleme kullan. IID bootstrap/permutation hâlihazırda keşifsel olarak etiketlenmiş; bağımsız doğrulamada zaman bağımlılığı ve çoklu deneme ayrıca ele alınmalı.

## 5. Orta — OOS başlangıcında geçmiş özellik durumu sıfırlanıyor

**Yer:** `one.py:777-786`, `one.py:208-265`.

Önce tarih aralığı kesiliyor, sonra ATR/EMA ve rolling özellikler hesaplanıyor. `--unlock-oos` kullanımında dosyada DEV geçmişi bulunsa bile 2025 başında gösterge durumu yeniden başlatılıyor.

**Doğrulama:** Aynı sentetik fiyat serisinde kesintisiz nedensel hesaplama ile OOS'a kesildikten sonraki hesaplama karşılaştırıldı. İlk OOS ATR değeri sürekli hesaplamada `2.2800094564`, mevcut OOS yolunda NaN oldu; trend ve makro trend durumları altışar mumda farklılaştı.

**Etki:** OOS başındaki olaylar farklı seçilebilir veya kaçırılabilir. Bu tespit ileriye bakma sızıntısı değil, geçmiş durum/warm-up tutarsızlığıdır.

**Düzeltme:** Özellikleri yalnızca geçmişten yararlanarak mevcut tarihçe üzerinde hesapla; olayları ve sonuçları aktif aralıkla sınırla. OOS özelliklerinin aynı zamanlardaki kesintisiz hesaplamaya eşitliğini regresyon testi yap.

## 6. Orta — sembol ve veri kimliği tutarlı değil

**Yer:** `one.py:187`, `one.py:759`, `one.py:784-786`, `one.py:837-845`.

`--symbol` kabul ediliyor fakat yerel CSV'nin sembolü doğrulanmıyor. Readiness içindeki `universe_symbol` sabit `BTCUSDT`; üst seviye evidence ve ekrandaki sembol CLI'dan geliyor. Funding takvimi de BTC'nin sabit 8 saatlik varsayımına bağlı.

**Doğrulama:** Aynı yerel CSV `--symbol ETHUSDT` ile çalıştırıldığında evidence içinde üst seviyede `ETHUSDT`, readiness içinde `BTCUSDT` yazıldı. Çalışma hata vermedi.

**Etki:** Yanlış enstrüman etiketiyle araştırma kaydı üretilebilir; başka sembollerde veri/funding varsayımları uygunsuz olabilir.

**Düzeltme:** Mevcut çalışma evreni yalnızca BTC ise şimdilik bunu zorunlu kıl. Çoklu sembol desteklenecekse sembolü doğrulanmış veri manifestinden al ve takvim/metadata boyunca tutarlı taşı.

## İyi çalışan ve doğrulanan alanlar

Mevcut sentetik self-testler; giriş mumunda stop/target, stop-first önceliği, olumsuz/olumlu açılış boşlukları, ücret ve slippage, zaman çıkışı, funding işaret/zamanlama muhasebesi, çakışan/olgunlaşmamış olay reddi, temel OHLCV bozukluklarının reddi ve geçmiş özelliklerin gelecekteki fiyat değişimlerinden etkilenmemesi kontrollerini geçti. Tespit edilen sorunlar bu başarılı kontrolleri geçersiz kılmaz; mevcut test kapsamının dışında kalan kabul/orkestrasyon sınırlarını gösterir.

## Uygulama sırası ve kabul ölçütü

1. Veri kapısını gerçekten durdurucu yap; funding mevcudiyeti/sonluluğu/takvimini doğrula; dosya çakışmalarını engelle.
2. Aylık kontrol eşlemesini karar önkoşulu yap; OOS warm-up ve sembol kimliğini düzelt.
3. Bu senaryoları kalıcı CI regresyon testlerine dönüştür; mevcut 38 kontrolü koru. Sentetik testlerin ardından kanonik veriyi yeniden doğrula ve DEV sonuçlarını yeniden üret. OOS'u yalnızca önceden belirlenmiş doğrulama prosedürüyle kullan.

**Yeni özellik için kabul ölçütü:** Yukarıdaki altı bulgu kapatılmış, olumsuz senaryoların yanlış kabulü engellenmiş ve mevcut yürütme testleri geçiyor olmalı. Bu ölçütler düzeltme sonrasında yukarıdaki testlerle karşılandı. İlk incelemede yalnız rapor eklenmişti; kullanıcının düzeltme talebi üzerine `one.py`, CI, README ve evidence güncellendi, `tests/test_one.py` eklendi. Bu kapanış, gelecekte eklenecek özelliklerin ayrıca doğrulanması gereğini ortadan kaldırmaz.
