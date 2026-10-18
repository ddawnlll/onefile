# Direction v5 — DEV sonuç raporu (BTCUSDT, ETHUSDT, 2021–2024)

**Karar: `NO_EDGE_DIRECTION_DEV`.** Ön kayıtlı kapı geçilmedi. Bu tur ana hedef 5 gün, yardımcılar 3 ve 7 gün.

## Uygulanan
- Nedensel günlük özellikler: 12 fiyat özelliği, funding (`fund_z90`, `fund_chg7`), perp taker-buy oranı (`taker3`, `taker21`, önbellekten).
- Hedef: `Z_h = (log p_{t+h} − log p_t) / (s_t √h)`, `s_t` EWMA (yarı ömür 21 gün), eşik ±0,75 → UP/DOWN/NEUTRAL.
- Modeller: prior, persistence (`m3`), momentum (`m21`), çok terimli lojistik (fiyat ve tam set), sığ XGBoost (fiyat ve tam set).
- Walk-forward: 90 günlük bloklar, 2021-01-01 – 2024-12-31 (17 blok), eğitim genişleyen; 7+8 günlük purge; kalibrasyon: sıcaklık ölçekleme (son 90 gün); XGB round sayısı önceki 90 günden.
- Seçici alarm: kalibrasyon kesiminden 20% kapsama eşiği, varlık başına 5 gün cooldown.
- Ekonomi: giriş/çıkış 4+2 bp/yan, tutulan günlerde funding; blok bootstrap.

## Sonuç (h=5, test satırları 2912)

| Model | Alarm | Kapsama | Precision | Logloss | AP up | AP down | Net/işlem |
|---|---:|---:|---:|---:|---:|---:|---:|
| prior | 584 | 20.1% | 0.209 | 0.9612 | 0.212 | 0.153 | +0.14% |
| persist_m3 | 252 | 8.7% | 0.230 | **0.9536** | 0.224 | 0.204 | +0.39% |
| momentum_m21 | 169 | 5.8% | 0.260 | 0.9537 | 0.222 | 0.196 | −0.16% |
| logit_price | 223 | 7.7% | 0.211 | 0.9687 | 0.233 | 0.173 | −0.68% |
| **logit_full (birincil)** | 219 | 7.5% | 0.192 | 0.9709 | 0.235 | 0.172 | −0.46% |
| xgb_price | 246 | 8.5% | 0.207 | 0.9599 | 0.195 | 0.195 | +0.05% |
| xgb_full | 206 | 7.1% | 0.228 | 0.9612 | 0.200 | 0.192 | −0.09% |

Taban oranlar: UP 0,217, DOWN 0,174. Birincil model precision'ı tabanın altında; kapı eşiği 0,80.

- Hiçbir öğrenilen model h=5'te logloss'ta sabit prior'ı geçmedi; birincil model prior'dan kötü.
- Funding ve taker grupları eklenince logloss iyileşmedi (`logit_price` 0,9687 → `logit_full` 0,9709).
- Net ekonomik sonuçların tamamında güven aralığı sıfırı içeriyor.
- 3 ve 7 gün de aynı tabloyu verdi (artifacts içinde).

## Kapı (pre-registered)
precision ≥ 0,80 ✗ (0,192) · Wilson alt ≥ 0,70 ✗ · kapsama ≥ 10% ✗ (7,5%) · UP+DOWN recall ≥ 0,20 ✗ (0,037) · ≥100 alarm ✓ · ≥4 blok ✓ · persistence'ı geçmek ✗ · net CI alt > 0 ✗.

## Uygulanamayan / eksik
- **Spot katılımı:** spot kline arşivi önbellekte yok; indirmedim. E4'ün spot kısmı yapılmadı.
- **Open interest:** 2020–2021 için önbellekte yok; OI grubu yok.
- **Erken devam kohortu** (§7.4) ve olay bazlı erkenlik/episod analizi uygulanmadı.
- **6 saatlik karar** ve arbitraj: kapsam dışı (rapora göre).
- **Geçmiş uzunluğu:** 24 aylık ısınma yok (veri 2020'de başlıyor); ilk bloklarda fit kesimi 158–518 satır (iki varlık) ve 51/357 model-blok uyumu sınıf eksikliği nedeniyle prior'a düştü.
- **Protokol commit'i ve kod düzeltmesi:** protokol (`1792ba8`) çalıştırmadan önce commit edildi. Sonradan yalnızca XGB'de bir sınıf eksik olduğunda çökmeyi engelleyen fallback eklendi; kapı ve eşikler değişmedi.

## Doğrulama
- Birim testleri: `tests/test_regime.py` 18 test geçti (nedensellik: gelecekteki fiyat değişince özellikler değişmiyor; etiket yalnızca ileri; cooldown 5 gün).
- Veri: DEV dosyası `core5_4h.csv` (2024-12-31'e kadar); 2025+ yok, `--unlock-oos` kullanılmadı.
- Çalıştırma: `one.py --direction` ~34 s (tek koşu, bu tur).

## Sonuç
Bu tanımla, bu veride yön için kanıt yok. Persistence bile yalnızca çok küçük bir sıralama kenarı gösteriyor; net ekonomik sonuç sıfır. Daha iyi bir sonuç için yeni bir hipotez (ayrı ön kayıt) gerekir; aynı kapıyı yeniden koşmak veya eşik/ufuk değiştirmek sonucu değiştirmez.
