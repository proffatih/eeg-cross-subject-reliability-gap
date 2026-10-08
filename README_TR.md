# İki kuru elektrot yeterli: analiz kodu

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
