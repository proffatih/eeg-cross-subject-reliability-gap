# Kayıt-oturumu confound'u, dört kontaklı kuru elektrotlu bir EEG kıyas setindeki kişiler-arası doğruluğu açıklıyor

Aynı adlı makalenin analiz kodu.

Dört kanallı, kuru elektrotlu Muse kafa bandının (TP9, AF7, AF8, TP10), Bird ve
arkadaşlarının (2018) açık kayıtları üzerinde kişiler arası (bir kişiyi dışarıda
bırakarak, LOSO) karakterizasyonu: 4 kişi × 3 zihinsel durum (rahat, nötr,
konsantre) × 2 oturum = 256 Hz'de alınmış 24 kayıt.

## Veri

https://github.com/jordan-bird/eeg-feature-generation deposunun
`dataset/original_data/` klasöründeki 24 adet `subject[a-d]-[durum]-[1-2].csv`
dosyasını `data/raw/` klasörüne kopyalayın.

## Kurulum

Python 3.12 kurun, ardından `pip install -r code/requirements.txt` komutunu
çalıştırın.

## Çalıştırma sırası

Bütün betikler `code/` klasöründen çalıştırılır; sonuçlar `results/` klasörüne
yazılır.

| Betik | Ne yapar? | Çıktılar | Makale | Süre* |
|---|---|---|---|---|
| `verify_sampling_rate.py` | Örnekleme hızını zaman damgalarından hesaplar; zaman boşluğu içeren kayıtları listeler | Ekran | II. bölüm | Birkaç saniye |
| `features.py` | Kayıtları zaman boşluklarından böler; 50 Hz çentik ve 1–45 Hz bant geçiren filtre uygular; 1 s'lik pencereler (%50 örtüşme); elektrot başına 10 özellik | `data/features.npz` | II. bölüm | 15 s |
| `experiment.py` | GBM ve MLP; kişi içi (bir oturumu dışarıda bırakarak) ve LOSO değerlendirme | `metrics.csv`, `per_group.csv`, `confusion_gbm.json` | II. bölüm | 1 dk |
| `confound_checks.py` | Çentikli ve çentiksiz karşılaştırma; yalnızca 50 Hz, yalnızca DC ve geniş bant kontrolleri; çizgi yükseklikleri; ek çentikler; nedensel filtreler; 1–20 Hz | `confound_*.csv` | Şebeke girişimi bölümü | 4 dk |
| `sensor_experiments.py s1` | 15 elektrot kombinasyonunun tamamı, GBM ve MLP | `sensor_electrode_subsets.csv`, `..._per_subject.csv` | III. bölüm | 4 dk |
| `sensor_experiments.py s2` | Gürültü ve kontak arızaları (mutlak gürültü seviyesi; temiz veriyle eğitim, bozulmuş veriyle test) | `sensor_noise_faults.csv`, `noise_reference.json` | IV. bölüm | 11 dk |
| `sensor_experiments.py s3` | Dönüştürücü hızı, 256–112 Hz | `sensor_rate.csv` | V. bölüm | 2 dk |
| `sensor_experiments.py s3w` | Karar penceresi 2, 1 ve 0.5 s; 256 ve 112 Hz'de | `sensor_window.csv` | V. bölüm | 3 dk |
| `sensor_experiments.py s4` | Özellik çıkarma ve tahmin süresi, model boyutu | `sensor_compute_budget.json` | V. bölüm | 1 dk |
| `make_figures.py` | Şekil 1–3, Tablo 1–2 ve grafik özet | `figures/` | Tümü | 30 s |
| `sensor_experiments.py s3c` | İsteğe bağlı: bant kontrollü hız testi (her hızda aynı sinyal içeriği) | `sensor_rate_band_controlled.csv` | Hakemlere yanıt | 2 dk |

`features.py` diğer bütün betiklerden önce çalıştırılmalıdır.
`python sensor_experiments.py` argümansız çalıştırıldığında s1, s2, s3, s3w ve
s4 sırayla çalışır. *Süreler tek işlemci çekirdeği içindir.

## Model ayarları

Her iki sınıflandırıcı da, eğitim katının istatistikleriyle z-skora çevrilmiş
özelliklerle eğitilir (`StandardScaler`).

| Sınıflandırıcı | Ayarlar (scikit-learn 1.8.0) |
|---|---|
| GBM (referans) | `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)`; diğer bütün parametreler varsayılan değerlerinde. Bu veri boyutunda rastgele öğe içermediği için bir kez çalıştırılır. |
| MLP | `MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3, batch_size=128, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=20)`; diğerleri varsayılan (Adam, sınıf oranları korunarak ayrılan %10'luk doğrulama kümesi, en iyi ağırlıkların geri yüklenmesi). 0-4 tohumlarının ortalaması. |
| Kompakt GBM (Tablo II) | `HistGradientBoostingClassifier(max_iter=60, learning_rate=0.2, max_depth=3, max_leaf_nodes=8)` |

## Tekrarlanabilirlik

Bütün rastgele işlemler sabit tohumlar kullanır. GBM bu veri boyutunda hiçbir
rastgele öğe içermez (erken durdurma, özelliklerden ya da kutulamada örneklerden
rastgele seçim yoktur) ve yalnızca bir kez çalıştırılır. MLP sonuçları 0–4
tohumlarının ortalamasıdır. Eklenen bozulmalar 0–2 gürültü tohumlarını kullanır.
`requirements.txt` dosyasındaki sürümlerle her sayı birebir yeniden üretilir.
Tek istisna süre ölçümleridir: bunlar çalıştırmadan çalıştırmaya ve makineden
makineye değişir ve yalnızca elektrot sayıları arasındaki oranlar olarak
anlamlıdır.

## Orijinal işleme zincirinden farklar

- Bant geçiren filtreden önce 50 Hz çentik filtresi eklendi. Kayıtlarda duruma
  bağlı şebeke girişimi var; çentik olmadan bu girişim 4. mertebe filtrenin
  yumuşak bant kenarından özelliklere ulaşıyor ve kişiler arası doğruluğu
  0.58'den 0.78'e şişiriyor.
- Kayıtlar 0.1 s'den uzun zaman boşluklarından bölünüyor (`subjectb-relaxed-2`
  kaydı, 3–4.5 s'lik on parçadan oluşuyor); toplam 2442 pencere.
- Pencere ortalaması (merkezlemeden sonra her zaman sıfır) ve RMS (standart
  sapmaya eşit) özelliklerden çıkarıldı: elektrot başına 10, toplamda 40 özellik.
- Kişi içi değerlendirme, sonucu kütüphane sürümüne bağlı olan rastgele gruplu
  K-katlı bölme yerine, bir oturumu dışarıda bırakarak (8 sabit kat) yapılıyor.
- Gürültü her kayda aynı mutlak seviyede ekleniyor ve sınıflandırıcı temiz
  veriyle eğitiliyor. Kontak kopması da kontak bozulması gibi rastgele bir
  kontağa uygulanıyor; böylece iki arıza aynı koşullarda karşılaştırılabiliyor.
- Dönüştürücü hızı testi, hepsi analiz bandının tamamını taşıyan 256–112 Hz
  aralığını kesirli yeniden örneklemeyle kapsıyor.
- `np.trapz` yerine `np.trapezoid` kullanılıyor; kütüphane sürümleri sabitlendi.

## Revizyon, Kasım 2026

*Scientific Reports*'taki iki hakem, makalenin sonucunu değiştiren bir
düzeltmeye yol açtı; bu depo artık onu yansıtıyor.

**Bu veri setinde etiketler kayıt oturumuyla confound'lu.** `timestamps`
sütunu, dört katılımcının üçünde konsantre kayıtlarının rahat ve nötr
kayıtlardan **farklı bir günde** (yaklaşık 46 saat sonra) alındığını gösteriyor.
Hiç EEG görmeyen, yalnızca pencerenin zaman içindeki yerini bilen bir
sınıflandırıcı 0.333 şans düzeyine karşı **0.569** kişiler-arası doğruluğa
ulaşıyor; çentik süzgeçli veriyle tam dekoder **0.574**. 50 Hz çentiği bir
belirtiyi siliyor, confound'u değil. Her katılımcıda aynı oturumda kaydedilmiş
olan rahat–nötr karşılaştırması hâlâ şansın üstünde (0.500'e karşı 0.606 ve
0.659), yani kayıtlar kişiler-arası bilgi taşıyor — üç sınıflı rakamların ima
ettiğinden çok daha azını.

Yeniden üretmek için: `python code/session_confound.py all`

**Hakemlerin istediği ek çözümlemeler** (`code/reviewer_followups.py`):

- **Artefakt-only dekoder.** Yalnız 50 Hz gücü, DC ofseti ve 55–128 Hz gücü —
  hiçbiri 1–45 Hz bant geçirenden sağ çıkmıyor — aynı LOSO katlarında **0.761**
  veriyor, yani bant-içi tam dekoderden **19 puan yüksek**. Denek bazında
  0.839 / 0.993 / 0.372 / 0.838: gün ayrımı olan üç denek neredeyse kusursuz,
  ayrımı olmayan denek c şans düzeyinde.
- **Tahmin karşılaştırması.** Bant-içi dekoder ile artefakt-only dekoder
  pencerelerin 0.600'ünde aynı sınıfı veriyor (κ = 0.36; bağımsız iki dekoder
  için beklenen uyuşma 0.336).
- **50 Hz gücü ↔ zaman.** İlk gün içinde çizgi düzeyi kayıt saatiyle birlikte
  yükseliyor (Spearman ρ = 0.56, p = 0.015, n = 18); gün sınırında ortalama
  **+25.1 dB** sıçrıyor; gün ayrımı olmayan denek c'de konsantre kayıtları
  diğerlerinden yalnız 5.8 dB yukarıda.

**Hakemin bulduğu iki kod hatası düzeltildi** (ayrıntı İngilizce README'de).

**Elektrot, girişim, dönüştürücü hızı ve pencere karşılaştırmaları** eksiksizlik
için duruyor ama confound'lu etikete karşı puanlanmış durumda. Bu veri setini
anlatıyorlar; donanım tavsiyesi değiller.


### İkinci tur denetim, 16 Kasım 2026

Paketi yazmamış bir ortak yazarın denetimi üç şey istedi; üçü de
`followups_round2.py` içinde.

**1. 50 Hz–zaman korelasyonu zamanın tanımına bağlı.** İlk revizyon tek tanım
bildiriyordu. Üç tanımla:

| zaman tanımı | rho | p |
|---|---|---|
| katılımcı içinde sıralanmış | **+0,584** | 0,011 |
| katılımcının kendi ilk kaydından geçen saat | +0,562 | 0,015 |
| mutlak duvar saati | **−0,296** | 0,233 |

İşaret kararlı değil, çünkü katılımcılar öğleden sonranın farklı saatlerinde
kaydedilmiş. Makale artık katılımcı-içi biçimi manşete taşıyor ve üçünü de
yazıyor.

**2. Zaman ortak değişken olarak.** Şebeke düzeyi, durum üzerine; kayıt zamanı
ve katılımcı ortak değişken. 24 kaydın tamamında zamanın katsayısı saat başına
+0,042 (p = 0,003); concentrating ve relaxed nötrden +0,46 (p = 0,37) ve −0,26
(p = 0,39) farklı. **Durum modeli geçemiyor.**

**3. Kayıt düzeyinde permütasyon testleri.** Pencereler örtüşüyor ve seri
korelasyonlu, bu yüzden değiştirilebilir birim kayıt. Her katılımcının dört
kaydı üzerinde permütasyon: a ve b için p = 0,167, c ve d için p = 0,500; dört
kayıtla ulaşılabilecek en küçük p zaten 0,167. **Makalenin son olumlu iddiası
(relaxed–nötr kontrastının nöral bilgi taşıdığı) bu nedenle geri çekildi** —
ayrıca o kontrastta yalnız-saat 0,592 ve yalnız-artefakt 0,850, bant-içi
dekoder ise 0,606.

Aynı anda iki koruma eklendi: `splice_tables.py --check` makale tabloları sonuç
dosyalarıyla uyuşmazsa hata verir, `audit_numbers.py` derlenmiş PDF'teki her
ondalığı sonuçlarla karşılaştırır (şu an 217 ondalık, 0 eşleşmeyen).

Bu dosyaları yeniden kullanacaklar için bir veri notu:
`subjectb-relaxed-2.csv` 958 saniyeye yayılıyor ama **41,5 saniye** veri
içeriyor — toplam 916 saniyelik dokuz boşlukla ayrılmış on parça. O kaydın
pencereleri tek bir zaman noktasına oturmuyor.
