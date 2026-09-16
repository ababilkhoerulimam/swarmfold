# Ensemble Learning — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk teknik ensemble learning tingkat lanjut. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

1. [Negative Correlation Learning (NCL) & Hybrid NCL](#1-negative-correlation-learning-ncl--hybrid-ncl)
2. [DECORATE](#2-decorate)
3. [Rotation Forest](#3-rotation-forest)
4. [Greedy Ensemble Selection (Caruana) & Sub-varian](#4-greedy-ensemble-selection-caruana--sub-varian)
5. [Dynamic Ensemble Selection (KNORA-U, META-DES, DES-P)](#5-dynamic-ensemble-selection-knora-u-meta-des-des-p)
6. [Bayesian Model Averaging (BMA)](#6-bayesian-model-averaging-bma)
7. [Deep Ensembles](#7-deep-ensembles)
8. [MC Dropout](#8-mc-dropout)
9. [Error-Correcting Output Codes (ECOC)](#9-error-correcting-output-codes-ecoc)
10. [Co-training & Tri-training](#10-co-training--tri-training)
11. [Snapshot Ensembles](#11-snapshot-ensembles)
12. [Stochastic Weight Averaging (SWA) & SWAG](#12-stochastic-weight-averaging-swa--swag)
13. [BatchEnsemble](#13-batchensemble)
14. [Mixture of Experts (MoE) — Sparse & Hierarchical](#14-mixture-of-experts-moe--sparse--hierarchical)
15. [Genetic Algorithm & PSO untuk Ensemble](#15-genetic-algorithm--pso-untuk-ensemble)
16. [Online/Sequential Weight Update](#16-onlinesequential-weight-update)
17. [Confidence-Based Dynamic Weighting](#17-confidence-based-dynamic-weighting)
18. [Cascading](#18-cascading)

---

## 1. Negative Correlation Learning (NCL) & Hybrid NCL

### NCL Standalone

| Field | Detail |
|-------|--------|
| **Definisi** | Metode ensemble yang mendorong diversity secara eksplisit dengan memasukkan **diversity penalty term** ke dalam cost function setiap model individual. Setiap sub-model diberi penalti berdasarkan korelasi prediksinya dengan ensemble saat ini, sehingga model saling "menjauh" namun ensemble tetap ditarik ke target |
| **Problem yang didukung** | Klasifikasi, Regresi (terutama regresi untuk hybrid NCL) |
| **Kapan digunakan** | Model cenderung konvergen ke solusi sama; butuh diversity eksplisit; ensemble homogen |
| **Kapan tidak digunakan** | Data noisy tinggi (incentivize disagreement bahkan saat label benar); dataset sangat kecil |
| **Asumsi** | Model individual dapat dioptimisasi secara simultan; diversity dapat diukur via korelasi |
| **Data preprocessing** | Standar untuk model base; tidak ada perlakuan khusus |
| **Dataset scale** | Kecil–Menengah (1K–500K) |
| **Metric compatibility** | MSE, MAE (regresi); Accuracy, F1 (klasifikasi) |
| **Hyperparameter penting** | λ (penalty strength), α (learning rate), M (jumlah model) |
| **Kisaran parameter awal** | λ: 0.1–1.0; M: 5–20; learning rate: 0.01–0.1 |
| **CPU/GPU/RAM** | CPU-bound; RAM ~1–2× dataset; training sekuensial atau parallel per model |
| **Risiko leakage** | Rendah; penalty dihitung pada training set |
| **Failure modes** | λ terlalu besar → model terlalu berbeda, ensemble tidak konvergen; λ terlalu kecil → NCL tidak efektif |
| **Teknik pembanding** | Bagging (implicit diversity), Boosting (sequential), DECORATE (artificial data) |
| **Ensemble compatibility** | Base untuk Evolutionary Ensembles with NCL; Hybrid NCL untuk regresi |
| **Exact package** | Custom loss function; **IPOPT solver** untuk hybrid NCL |
| **Contoh hypothesis** | "NCL dengan λ=0.5 akan meningkatkan R² ensemble ≥8% vs bagging pada dataset dengan fitur redundan tinggi" |
| **Acceptance criteria** | Diversity metric (Q-statistic, correlation) turun ≥20%; ensemble accuracy/R² naik ≥5% |
| **Rejection criteria** | Ensemble performance drop >2% vs baseline; training time >3× baseline |
| **Sumber** | Liu & Yao (1999); Bai et al., Mach. Learn. 112(10), 3881-3916 (2023) |

### Hybrid NCL untuk Regresi

| Field | Detail |
|-------|--------|
| **Definisi** | NCL dijadikan penalty term dalam optimisasi (interior-point filter linear-search / IPOPT) untuk memilih subset model paling diverse sekaligus akurat |
| **Problem yang didukung** | Regresi dengan pool model besar |
| **Kapan digunakan** | Pool model besar; butuh subset model diverse-akurat |
| **Hyperparameter penting** | λ, IPOPT tolerance, max iterations |
| **Exact package** | IPOPT + custom objective |
| **Sumber** | arXiv:2104.02317 (2021) |

---

## 2. DECORATE

| Field | Detail |
|-------|--------|
| **Definisi** | **Diverse Ensemble Creation by Oppositional Relabeling of Artificial Training Examples**. Metode yang secara langsung membangun hipotesis diverse dengan menambahkan data training artifisial yang diberi label oposisi dari ensemble saat ini |
| **Problem yang didukung** | Klasifikasi |
| **Kapan digunakan** | Small training sets; butuh diversity tinggi; semi-supervised (ACTIVEDECORATE) |
| **Kapan tidak digunakan** | Data sangat besar (overhead artifisial tidak worth it); data noisy tinggi |
| **Asumsi** | Data artifisial dapat di-generate dari distribusi yang mendekati data asli; diversity dapat diciptakan via label oposisi |
| **Data preprocessing** | Standar; perlu distribusi fitur untuk generate data artifisial |
| **Dataset scale** | Kecil–Menengah (<100K) |
| **Metric compatibility** | Accuracy, F1, AUC-ROC |
| **Hyperparameter penting** | `R` (jumlah data artifisial per iterasi), `I` (iterasi), `T` (threshold akurasi) |
| **Kisaran parameter awal** | R: 0.1–0.5 × n; I: 10–50 |
| **CPU/GPU/RAM** | CPU-bound; RAM ~1.5× dataset (karena data artifisial) |
| **Risiko leakage** | Rendah; data artifisial di-generate dari training set |
| **Failure modes** | Data artifisial tidak realistis → ensemble bias; R terlalu besar → overfitting |
| **Teknik pembanding** | Bagging, AdaBoost, NCL |
| **Ensemble compatibility** | Meta-learner general; dapat menggunakan strong learner apapun |
| **Exact package** | `DecorateR::DECORATE` (R); custom implementation |
| **Contoh hypothesis** | "DECORATE dengan R=0.2n akan mencapai akurasi ≥5% lebih tinggi dari Bagging pada dataset dengan <500 sampel berlabel" |
| **Acceptance criteria** | Accuracy improvement ≥5% vs Bagging; ensemble diversity (kappa) turun |
| **Rejection criteria** | Accuracy drop >2%; training time >5× Bagging |
| **Sumber** | Melville & Mooney (2004); ACM Computing Surveys (2023) |

---

## 3. Rotation Forest

| Field | Detail |
|-------|--------|
| **Definisi** | Feature set secara acak dibagi menjadi K subset, lalu **Principal Component Analysis (PCA)** diterapkan pada setiap subset untuk membentuk rotasi fitur. Base classifier dilatih pada ruang fitur terotasi |
| **Problem yang didukung** | Klasifikasi |
| **Kapan digunakan** | Fitur berkorelasi tinggi; butuh diversity; data tabular |
| **Kapan tidak digunakan** | Fitur sangat sedikit (<5); interpretability tinggi dibutuhkan |
| **Asumsi** | PCA pada subset fitur menghasilkan rotasi yang meningkatkan akurasi individual dan diversity ensemble |
| **Data preprocessing** | Standar scaling untuk PCA; tree-based tidak butuh scaling tapi PCA membutuhkan |
| **Dataset scale** | Kecil–Menengah (1K–500K) |
| **Metric compatibility** | Accuracy, F1, AUC-ROC |
| **Hyperparameter penting** | `K` (jumlah subset fitur), `L` (jumlah base classifier), `PCA components` |
| **Kisaran parameter awal** | K: 3–10; L: 10–50; PCA components: min(n_features/K, n_samples) |
| **CPU/GPU/RAM** | CPU-bound; RAM ~1.5× dataset (PCA overhead) |
| **Risiko leakage** | Rendah jika PCA di-fit pada training fold saja |
| **Failure modes** | K terlalu besar → subset fitur terlalu kecil untuk PCA; K terlalu kecil → rotasi tidak efektif |
| **Teknik pembanding** | Random Forest, Bagging, Extra Trees |
| **Ensemble compatibility** | Dapat dikombinasi dengan Bagging, Boosting |
| **Exact package** | `rotationForest` (R); custom Python implementation |
| **Contoh hypothesis** | "Rotation Forest dengan K=5 dan L=30 akan mencapai akurasi ≥3% lebih tinggi dari Random Forest pada dataset dengan fitur berkorelasi tinggi" |
| **Acceptance criteria** | Accuracy improvement ≥3% vs RF; training time <2× RF |
| **Rejection criteria** | Accuracy drop >1%; training time >3× RF |
| **Sumber** | Rodriguez et al., IEEE TPAMI 28(10), 1619-1630 (2006) |

---

## 4. Greedy Ensemble Selection (Caruana) & Sub-varian

### Greedy Ensemble Selection Klasik

| Field | Detail |
|-------|--------|
| **Definisi** | Prosedur untuk membangun ensemble dari library model besar. Secara iteratif menambahkan model (dengan replacement) yang paling meningkatkan skor ensemble pada validation set |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Pool model besar; butuh ensemble optimal tanpa meta-learner |
| **Kapan tidak digunakan** | Pool model kecil; data validation sangat kecil |
| **Asumsi** | Validation set representatif; model dengan performa baik cenderung memberikan kontribusi positif |
| **Data preprocessing** | Standar; prediksi OOF diperlukan untuk honest evaluation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric (accuracy, AUC, MSE, dll.) |
| **Hyperparameter penting** | `caruana_rounds` (default 50), `bag_samples`, `sample_ratio`, `init_ensemble_size` |
| **Kisaran parameter awal** | rounds: 50–200; init_ensemble_size: 5–25 (model terbaik awal) |
| **CPU/GPU/RAM** | Ringan; hanya operasi pada matriks prediksi, bukan training model |
| **Risiko leakage** | Rendah jika validation set benar-benar terpisah; **tinggi** jika tidak |
| **Failure modes** | Overfitting ke validation set kecil; model redundancy (model serupa dipilih berulang) |
| **Teknik pembanding** | Stacking, Blending, NNLS |
| **Ensemble compatibility** | Output dapat digunakan sebagai base untuk meta-learner |
| **Exact package** | AutoGluon; `ensemble_selection` (custom); `evoFE::ensemble_islands` (R) |
| **Contoh hypothesis** | "Greedy Ensemble Selection dengan 100 rounds akan mencapai AUC ≥0.02 lebih tinggi dari Stacking pada pool 50 model OOF" |
| **Acceptance criteria** | Metric improvement ≥2% vs single best model; ensemble size <20 |
| **Rejection criteria** | Improvement <1%; ensemble size >50 |
| **Sumber** | Caruana et al., "Ensemble Selection from Libraries of Models" (2004); "Getting the Most Out of Ensemble Selection" (2006) |

### Sub-varian

| Varian | Deskripsi | Kapan Digunakan |
|--------|-----------|-----------------|
| **Forward Selection** | Mulai kosong, tambah satu per satu | Pool kecil, butuh interpretability |
| **Backward Elimination** | Mulai full, hapus satu per satu | Pool menengah, ingin efisiensi |
| **Selection with Replacement** | Model boleh dipilih berulang; bobot = frekuensi | Pool besar, butuh bobot efektif |
| **Bootstrapped Hillclimbing** | Validasi di-bootstrap tiap iterasi | Validation set kecil (hindari overfit) |
| **Diversity-Aware Greedy (HLGER)** | Skip model dengan korelasi error tinggi | Pool besar dengan model redundan |

---

## 5. Dynamic Ensemble Selection (KNORA-U, META-DES, DES-P)

### Prinsip Umum DES

Region of competence (biasanya k-NN di sekitar test point) → estimasi kompetensi tiap model → pilih subset model untuk prediksi titik tersebut.

### KNORA-U (K-Nearest Oracles – Union)

| Field | Detail |
|-------|--------|
| **Definisi** | Pilih semua model yang benar setidaknya satu kali di region of competence; bobot sesuai jumlah benar |
| **Problem yang didukung** | Klasifikasi |
| **Kapan digunakan** | Pool model heterogen; data dengan region kompetensi jelas |
| **Kapan tidak digunakan** | Data sangat sparse (region kompetensi tidak informatif) |
| **Asumsi** | k-NN region of competence representatif untuk estimasi kompetensi lokal |
| **Data preprocessing** | Scaling diperlukan untuk k-NN |
| **Dataset scale** | Kecil–Menengah (1K–100K) |
| **Metric compatibility** | Accuracy, F1 |
| **Hyperparameter penting** | `k` (jumlah neighbors, default=7), `pool_classifiers` |
| **Kisaran parameter awal** | k: 5–15; pool size: 10–50 |
| **CPU/GPU/RAM** | Ringan saat inference; training pool model tetap berat |
| **Risiko leakage** | Sedang; region of competence harus dihitung dari data yang tidak leak |
| **Failure modes** | k terlalu kecil → kompetensi noisy; k terlalu besar → region tidak lokal |
| **Teknik pembanding** | KNORA-E, META-DES, OLA, LCA |
| **Ensemble compatibility** | Dapat menggunakan pool dari Bagging, RF, atau model heterogen |
| **Exact package** | `DESlib` (Python); `deslib.readthedocs.io` |
| **Contoh hypothesis** | "KNORA-U dengan k=7 akan mencapai akurasi ≥2% lebih tinggi dari static ensemble pada dataset dengan decision boundary kompleks" |
| **Acceptance criteria** | Accuracy improvement ≥2% vs static ensemble; inference time <2× single model |
| **Rejection criteria** | Accuracy drop >1%; inference time >5× single model |
| **Sumber** | Ko et al. (2008); DESlib documentation |

### META-DES

| Field | Detail |
|-------|--------|
| **Definisi** | Menggunakan meta-learning untuk memutuskan kompetensi model per region; meta-classifier dilatih untuk memprediksi apakah model akan benar pada test instance |
| **Problem yang didukung** | Klasifikasi |
| **Kapan digunakan** | Pool model besar; butuh seleksi kompetensi yang lebih pintar dari KNORA |
| **Hyperparameter penting** | `meta_classifier`, `k`, `pool_classifiers` |
| **Exact package** | `DESlib` — `METADES` class |
| **Sumber** | Cruz et al. (2015); DESlib |

### DES-P (Dynamic Ensemble Selection – Performance)

| Field | Detail |
|-------|--------|
| **Definisi** | Varian DES dengan pembobotan probabilistik region kompetensi |
| **Exact package** | `DESlib` — `DESP` class |

### OLA & LCA

| Metode | Definisi |
|--------|----------|
| **OLA (Overall Local Accuracy)** | Akurasi model di region lokal |
| **LCA (Local Class Accuracy)** | Akurasi model khusus untuk kelas yang diprediksi |

---

## 6. Bayesian Model Averaging (BMA)

| Field | Detail |
|-------|--------|
| **Definisi** | Bobot ensemble = **posterior probability** tiap model. Menggabungkan predictive distributions dari berbagai sumber, dengan bobot yang mencerminkan kontribusi relatif model terhadap predictive skill selama training period |
| **Problem yang didukung** | Klasifikasi, Regresi, Forecasting |
| **Kapan digunakan** | Butuh probabilistic interpretation; model space terbatas; uncertainty quantification |
| **Kapan tidak digunakan** | Model space sangat besar (posterior intractable); data sangat besar (MCMC mahal) |
| **Asumsi** | Model adalah sampel dari prior distribution; likelihood dapat dihitung |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Log-likelihood, Brier score, CRPS, MSE |
| **Hyperparameter penting** | Prior model probabilities, MCMC iterations, burn-in |
| **Kisaran parameter awal** | MCMC: 10.000–100.000 iterations; burn-in: 20–50% |
| **CPU/GPU/RAM** | CPU-bound; MCMC bisa sangat mahal; parallelizable per chain |
| **Risiko leakage** | Rendah; BMA di-fit pada training/validation |
| **Failure modes** | Prior salah → posterior bias; MCMC tidak konvergen |
| **Teknik pembanding** | Stacking, Greedy Selection, Deep Ensembles |
| **Ensemble compatibility** | Dapat dikombinasi dengan model apa pun yang memiliki likelihood |
| **Exact package** | `BMA` (R); `BAS` (R); `PyMC` (Python) |
| **Contoh hypothesis** | "BMA dengan prior uniform akan mencapai Brier score ≥5% lebih baik dari equal-weight ensemble pada data time-series" |
| **Acceptance criteria** | Brier score improvement ≥5%; posterior weights terdefinisi dengan baik |
| **Rejection criteria** | MCMC tidak konvergen (R-hat >1.1); improvement <2% |
| **Sumber** | Raftery et al. (2005); MDPI (2021) |

---

## 7. Deep Ensembles

| Field | Detail |
|-------|--------|
| **Definisi** | Melatih beberapa neural network dengan **random initialization** berbeda pada dataset yang sama; ensemble digunakan untuk uncertainty estimation |
| **Problem yang didukung** | Klasifikasi, Regresi, Uncertainty Quantification |
| **Kapan digunakan** | Butuh calibrated uncertainty; deep learning |
| **Kapan tidak digunakan** | Resource terbatas (M model = M× training cost); real-time inference |
| **Asumsi** | Random initialization menghasilkan model yang cukup diverse |
| **Data preprocessing** | Standar deep learning |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Accuracy, NLL, ECE, Brier score |
| **Hyperparameter penting** | `M` (jumlah model), arsitektur, learning rate, epochs |
| **Kisaran parameter awal** | M: 5–10; epochs: 100–300 |
| **CPU/GPU/RAM** | **GPU-heavy**: M × training cost; RAM GPU: M × model size |
| **Risiko leakage** | Rendah; training set terpisah |
| **Failure modes** | M terlalu kecil → uncertainty tidak calibrated; training cost linear terhadap M |
| **Teknik pembanding** | MC Dropout (lebih murah), SWAG, BatchEnsemble |
| **Ensemble compatibility** | Standar untuk uncertainty estimation |
| **Exact package** | Custom PyTorch/TensorFlow; `torch.optim.swa_utils` |
| **Contoh hypothesis** | "Deep Ensemble dengan M=5 akan mencapai ECE <0.05 pada CIFAR-10" |
| **Acceptance criteria** | ECE <0.05; accuracy ≥ single model +2% |
| **Rejection criteria** | ECE >0.1; accuracy drop |
| **Sumber** | Lakshminarayanan et al. (2017); OSTI (2025) |

---

## 8. MC Dropout

| Field | Detail |
|-------|--------|
| **Definisi** | Dropout saat inference dianggap sebagai **variational approximation** ke deep Gaussian process posterior. Forward pass T kali dengan dropout aktif → ensemble implisit |
| **Problem yang didukung** | Klasifikasi, Regresi, Uncertainty Estimation |
| **Kapan digunakan** | Resource terbatas; model sudah memiliki dropout layer |
| **Kapan tidak digunakan** | Butuh ensemble diversity tinggi (MC Dropout menghasilkan model yang sangat correlated) |
| **Asumsi** | Dropout = variational inference approximation |
| **Data preprocessing** | Standar deep learning |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | NLL, ECE, accuracy |
| **Hyperparameter penting** | `T` (jumlah forward passes), dropout rate |
| **Kisaran parameter awal** | T: 10–100; dropout: 0.1–0.5 |
| **CPU/GPU/RAM** | Ringan: T × inference cost; tidak perlu training model terpisah |
| **Risiko leakage** | Rendah |
| **Failure modes** | T terlalu kecil → uncertainty noisy; dropout rate salah → under/over-confident |
| **Teknik pembanding** | Deep Ensembles (lebih akurat tapi mahal), SWAG |
| **Ensemble compatibility** | Dapat dikombinasi dengan Deep Ensembles |
| **Exact package** | PyTorch: `model.train()` saat inference + multiple forward passes |
| **Contoh hypothesis** | "MC Dropout dengan T=50 akan mencapai ECE <0.08 pada dataset medis" |
| **Acceptance criteria** | ECE <0.08; inference time <10× single forward pass |
| **Rejection criteria** | ECE >0.15; inference time >50× |
| **Sumber** | Gal & Ghahramani (2016) |

---

## 9. Error-Correcting Output Codes (ECOC)

| Field | Detail |
|-------|--------|
| **Definisi** | Framework untuk mengubah masalah multi-class menjadi beberapa binary classification dengan skema encoding; hasil decoding untuk menentukan kelas akhir |
| **Problem yang didukung** | Multi-class classification |
| **Kapan digunakan** | Multi-class dengan binary classifier base (SVM); kelas banyak (>10) |
| **Kapan tidak digunakan** | Binary classification; kelas sangat sedikit (2–3) |
| **Asumsi** | Binary classifier base dapat menangani sub-problem; code matrix memiliki error-correcting capability |
| **Data preprocessing** | Standar untuk binary classifier base |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Accuracy, F1-macro, F1-micro |
| **Hyperparameter penting** | `code_size`, `coding_strategy` (one-vs-one, one-vs-all, dense, sparse) |
| **Kisaran parameter awal** | code_size: 2×n_classes hingga 10×n_classes |
| **CPU/GPU/RAM** | CPU-bound; RAM ~code_size × base classifier |
| **Risiko leakage** | Rendah |
| **Failure modes** | Code matrix buruk → decoding error; code_size terlalu besar → overhead |
| **Teknik pembanding** | One-vs-Rest, One-vs-One, Stacking |
| **Ensemble compatibility** | Dapat dikombinasi dengan DES + evidence theory untuk data imbalance |
| **Exact package** | `sklearn.multiclass.OutputCodeClassifier`; MATLAB `fitcecoc`; JMLR ECOC library |
| **Contoh hypothesis** | "ECOC dengan code_size=3×n_classes akan mencapai F1-macro ≥5% lebih tinggi dari One-vs-Rest pada dataset 20 kelas" |
| **Acceptance criteria** | F1-macro improvement ≥5% vs One-vs-Rest |
| **Rejection criteria** | F1 drop >1%; training time >10× |
| **Sumber** | Dietterich & Bakiri (1995); JMLR 11 (2010) 661-664 |

---

## 10. Co-training & Tri-training

### Co-training

| Field | Detail |
|-------|--------|
| **Definisi** | Dua model dilatih pada dua **view** fitur berbeda; saling memberikan pseudo-label ke data unlabeled yang confidence-nya tinggi |
| **Problem yang didukung** | Klasifikasi semi-supervised |
| **Kapan digunakan** | Multi-view data (fitur terpisah secara natural); labeled data sedikit |
| **Kapan tidak digunakan** | Single-view data (tidak ada view terpisah) |
| **Asumsi** | Fitur dapat dibagi menjadi dua view yang cukup independen; view saling melengkapi |
| **Data preprocessing** | Standar per view |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Accuracy, F1 |
| **Hyperparameter penting** | `pool_size`, `confidence_threshold`, `iterations` |
| **Kisaran parameter awal** | pool_size: 50–200; confidence_threshold: 0.8–0.95 |
| **CPU/GPU/RAM** | Ringan; 2× training cost |
| **Risiko leakage** | Sedang; pseudo-label bisa memperkuat error |
| **Failure modes** | View tidak independen → co-training tidak efektif; threshold salah → pseudo-label noisy |
| **Teknik pembanding** | Tri-training, Self-training |
| **Exact package** | Custom implementation; `semisupervised` library |
| **Contoh hypothesis** | "Co-training dengan confidence threshold 0.9 akan mencapai akurasi ≥10% lebih tinggi dari supervised-only pada dataset dengan 90% unlabeled" |
| **Acceptance criteria** | Accuracy improvement ≥10% vs supervised-only |
| **Rejection criteria** | Accuracy drop; pseudo-label accuracy <70% |
| **Sumber** | Blum & Mitchell (1998); Springer (2023) |

### Tri-training

| Field | Detail |
|-------|--------|
| **Definisi** | Tiga classifier dilatih pada labeled set; dalam setiap iterasi, dua classifier yang setuju memberikan pseudo-label ke classifier ketiga |
| **Problem yang didukung** | Klasifikasi semi-supervised |
| **Kapan digunakan** | Labeled data sangat sedikit; tidak ada multi-view |
| **Kapan tidak digunakan** | Data noisy tinggi; label quality rendah |
| **Asumsi** | Tiga classifier cukup diverse; majority voting efektif |
| **Hyperparameter penting** | `iterations`, `confidence_threshold` |
| **Kisaran parameter awal** | iterations: 10–50; confidence_threshold: 0.8–0.95 |
| **Exact package** | `algorithms-in-python` (GitHub); custom |
| **Contoh hypothesis** | "Tri-training dengan 20 iterasi akan mencapai akurasi ≥8% lebih tinggi dari self-training pada dataset dengan 95% unlabeled" |
| **Acceptance criteria** | Accuracy improvement ≥8% vs self-training |
| **Rejection criteria** | Accuracy drop >1% |
| **Sumber** | Zhou & Li (2005); PMC (2025) |

---

## 11. Snapshot Ensembles

| Field | Detail |
|-------|--------|
| **Definisi** | Melatih satu neural network dengan **cyclical learning rate schedule**; menyimpan snapshot model di setiap local minimum; snapshot digabungkan sebagai ensemble |
| **Problem yang didukung** | Deep learning classification |
| **Kapan digunakan** | Butuh ensemble "gratis" dari satu training run; resource terbatas |
| **Kapan tidak digunakan** | Model tidak memiliki local minima beragam; training sangat singkat |
| **Asumsi** | Cyclical LR mengunjungi beberapa local minima yang memberikan diversity |
| **Data preprocessing** | Standar deep learning |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Accuracy, error rate |
| **Hyperparameter penting** | `n_cycles`, `cycle_length`, `max_lr`, `min_lr` |
| **Kisaran parameter awal** | n_cycles: 3–6; cycle_length: 10–50 epochs |
| **CPU/GPU/RAM** | **GPU**: 1× training cost + M × inference cost; RAM: M × model size |
| **Risiko leakage** | Rendah |
| **Failure modes** | Cycle terlalu pendek → snapshot tidak diverse; cycle terlalu panjang → training lama |
| **Teknik pembanding** | Deep Ensembles (M× training cost), SWA (weight averaging) |
| **Ensemble compatibility** | Snapshot dapat digabungkan via averaging atau voting |
| **Exact package** | Custom PyTorch/TensorFlow; `snapshot_ensembles` (GitHub) |
| **Contoh hypothesis** | "Snapshot Ensembles dengan 4 cycles akan mencapai error rate ≤4% pada CIFAR-10" |
| **Acceptance criteria** | Error rate ≤4%; training time <2× single model |
| **Rejection criteria** | Error rate >5%; snapshot tidak diverse |
| **Sumber** | Huang et al., "Snapshot Ensembles: Train 1, get M for free" (2017) |

---

## 12. Stochastic Weight Averaging (SWA) & SWAG

### SWA

| Field | Detail |
|-------|--------|
| **Definisi** | Rata-rata **bobot** dari beberapa checkpoint selama training (bukan rata-rata prediksi) |
| **Problem yang didukung** | Deep learning |
| **Kapan digunakan** | Butuh generalisasi lebih baik tanpa ensemble inference cost |
| **Kapan tidak digunakan** | Model sangat kecil; training sangat singkat |
| **Asumsi** | SGD iterates di sekitar local minimum dapat dirata-ratakan untuk solusi lebih baik |
| **Hyperparameter penting** | `start_averaging`, `average_period`, `swa_lr` |
| **Kisaran parameter awal** | start_averaging: 50–75% training; swa_lr: 0.01–0.05 |
| **CPU/GPU/RAM** | Ringan: hanya menyimpan running average; inference 1× model |
| **Exact package** | `torch.optim.swa_utils.AveragedModel`; PyTorch Lightning `StochasticWeightAveraging` |
| **Contoh hypothesis** | "SWA akan meningkatkan akurasi ≥2% vs SGD baseline pada ImageNet" |
| **Acceptance criteria** | Accuracy improvement ≥2%; inference cost sama |
| **Rejection criteria** | Accuracy drop; tidak konvergen |
| **Sumber** | Izmailov et al. (2018) |

### SWAG

| Field | Detail |
|-------|--------|
| **Definisi** | Versi Bayesian/probabilistik dari SWA; merepresentasikan uncertainty dengan Gaussian approximation dari SGD iterates |
| **Problem yang didukung** | Uncertainty quantification |
| **Kapan digunakan** | Butuh uncertainty tanpa M× training cost |
| **Hyperparameter penting** | `num_samples`, `cov_mat_rank` |
| **Exact package** | `torch.optim.swa_utils`; SWAG implementation (GitHub) |
| **Sumber** | Maddox et al. (2019) |

---

## 13. BatchEnsemble

| Field | Detail |
|-------|--------|
| **Definisi** | Banyak "model" berbagi bobot utama, berbeda hanya pada **rank-1 perturbation** yang dipelajari. Jauh lebih hemat memori dan komputasi |
| **Problem yang didukung** | Deep learning, Uncertainty Estimation, Lifelong Learning |
| **Kapan digunakan** | Resource terbatas; butuh ensemble-like uncertainty |
| **Kapan tidak digunakan** | Butuh diversity tinggi (rank-1 perturbation terbatas) |
| **Asumsi** | Rank-1 perturbation cukup untuk menghasilkan diversity yang meaningful |
| **Hyperparameter penting** | `ensemble_size`, `rank`, `alpha_init`, `gamma_init` |
| **Kisaran parameter awal** | ensemble_size: 3–8; rank: 1–2 |
| **CPU/GPU/RAM** | **Sangat efisien**: ~1× parameter + ensemble_size × rank-1 vectors |
| **Exact package** | `DeepUncertainty.jl` (Julia); custom PyTorch |
| **Contoh hypothesis** | "BatchEnsemble dengan ensemble_size=4 akan mencapai akurasi dan calibration setara Deep Ensembles dengan 25% memory" |
| **Acceptance criteria** | Accuracy ≥ Deep Ensembles -1%; memory ≤25% |
| **Rejection criteria** | Accuracy drop >2%; calibration ECE >0.1 |
| **Sumber** | Wen et al. (2019); arXiv (2026) |

---

## 14. Mixture of Experts (MoE) — Sparse & Hierarchical

### Sparse MoE (Top-k Gating)

| Field | Detail |
|-------|--------|
| **Definisi** | Hanya sebagian expert aktif per input; **router** memilih top-k expert berdasarkan gating scores |
| **Problem yang didukung** | Language Modeling, Large-scale Classification |
| **Kapan digunakan** | Scale model capacity tanpa increase compute per token |
| **Kapan tidak digunakan** | Dataset kecil; model kecil |
| **Asumsi** | Input dapat dikelompokkan ke sub-domain yang ditangani expert berbeda |
| **Hyperparameter penting** | `num_experts`, `top_k`, `capacity_factor`, `load_balancing_loss` |
| **Kisaran parameter awal** | num_experts: 8–64; top_k: 2–4 |
| **CPU/GPU/RAM** | **GPU-heavy**: total parameter besar, tapi compute per token terbatas |
| **Failure modes** | Load imbalance → beberapa expert overused, lainnya tidak terlatih |
| **Exact package** | PyTorch: `torch.nn.Module` custom; DeepSeek-V3, Mixtral implementation |
| **Contoh hypothesis** | "Sparse MoE dengan 32 experts dan top_k=2 akan mencapai perplexity setara dense model 4× lebih kecil" |
| **Acceptance criteria** | Perplexity ≤ dense baseline; compute per token ≤2× |
| **Rejection criteria** | Load imbalance >20%; perplexity drop |
| **Sumber** | Shazeer et al. (2017) |

### Hierarchical MoE (HMoE)

| Field | Detail |
|-------|--------|
| **Definisi** | Gating bertingkat: layer pertama memilih grup expert, layer kedua memilih expert dalam grup |
| **Problem yang didukung** | Large-scale tasks dengan sub-domain hierarkis |
| **Hyperparameter penting** | `num_groups`, `experts_per_group`, `top_k_per_level` |
| **Exact package** | Custom; ar5iv (2024) |
| **Sumber** | Nguyen et al. (2024) |

---

## 15. Genetic Algorithm & PSO untuk Ensemble

### Genetic Algorithm Ensemble Selection

| Field | Detail |
|-------|--------|
| **Definisi** | Kromosom = subset model atau vektor bobot; fitness = skor validasi ensemble |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Pool model besar; ruang subset tidak terstruktur |
| **Kapan tidak digunakan** | Pool kecil (greedy lebih efisien) |
| **Asumsi** | Crossover dan mutation dapat menjelajahi ruang subset secara efektif |
| **Hyperparameter penting** | `population_size`, `generations`, `crossover_rate`, `mutation_rate` |
| **Kisaran parameter awal** | population: 50–100; generations: 20–50 |
| **CPU/GPU/RAM** | CPU-bound; fitness evaluation dapat diparalelkan |
| **Exact package** | `DEAP` (Python); `TPOT`; custom GA |
| **Contoh hypothesis** | "GA dengan population=50 akan mencapai accuracy ≥3% lebih tinggi dari Greedy Selection pada pool 100 model" |
| **Acceptance criteria** | Accuracy improvement ≥3% vs Greedy |
| **Rejection criteria** | Improvement <1%; runtime >10× Greedy |
| **Sumber** | AIMS Press (2025); PMC (2025) |

### Particle Swarm Optimization (PSO)

| Field | Detail |
|-------|--------|
| **Definisi** | PSO untuk mencari bobot ensemble optimal; partikel = vektor bobot; velocity update berdasarkan personal best dan global best |
| **Hyperparameter penting** | `n_particles`, `max_iter`, `w` (inertia), `c1`, `c2` |
| **Kisaran parameter awal** | n_particles: 20–50; max_iter: 50–200 |
| **Exact package** | `pyswarm`; custom PSO |
| **Sumber** | Springer (2025); IEEE (2025) |

---

## 16. Online/Sequential Weight Update

| Field | Detail |
|-------|--------|
| **Definisi** | Bobot ensemble diperbarui secara kontinu seiring data baru (streaming/time series) |
| **Problem yang didukung** | Streaming classification, Time series, Concept drift |
| **Kapan digunakan** | Data stream; konsep berubah seiring waktu |
| **Kapan tidak digunakan** | Data statis; batch learning |
| **Asumsi** | Data baru tersedia secara sekuensial; konsep dapat berubah (concept drift) |
| **Data preprocessing** | Standar streaming preprocessing |
| **Dataset scale** | Streaming (unbounded) |
| **Metric compatibility** | Prequential accuracy, Kappa, MSE |
| **Hyperparameter penting** | `learning_rate`, `window_size`, `forgetting_factor` |
| **CPU/GPU/RAM** | Ringan; incremental update |
| **Risiko leakage** | Sedang; pastikan tidak menggunakan future data |
| **Failure modes** | Concept drift tidak terdeteksi → performa turun; forgetting factor salah |
| **Exact package** | `River` (Python); custom online ensemble |
| **Contoh hypothesis** | "Online weight update dengan forgetting factor 0.99 akan mencapai prequential accuracy ≥5% lebih tinggi dari static ensemble pada data stream dengan concept drift" |
| **Acceptance criteria** | Prequential accuracy improvement ≥5% |
| **Rejection criteria** | Accuracy drop >2% |
| **Sumber** | Springer (2021); PLOS (2026) |

---

## 17. Confidence-Based Dynamic Weighting

| Field | Detail |
|-------|--------|
| **Definisi** | Bobot real-time dari **confidence/uncertainty** tiap model per instance. Model yang lebih confident pada instance tertentu mendapat bobot lebih besar |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Model memiliki confidence score terkalibrasi; data heterogen |
| **Kapan tidak digunakan** | Model tidak terkalibrasi (confidence tidak reliable) |
| **Asumsi** | Confidence score berkorelasi dengan akurasi |
| **Hyperparameter penting** | `temperature`, `credibility_threshold` |
| **Kisaran parameter awal** | temperature: 0.5–2.0; threshold: 0.7–0.9 |
| **CPU/GPU/RAM** | Ringan; inference tambahan minimal |
| **Exact package** | Custom; `ConfidenceWeightedEnsemble` (Python) |
| **Contoh hypothesis** | "Confidence-based weighting dengan temperature=1.0 akan mencapai akurasi ≥3% lebih tinggi dari equal-weight pada data dengan noise heterogen" |
| **Acceptance criteria** | Accuracy improvement ≥3% |
| **Rejection criteria** | Accuracy drop >1% |
| **Sumber** | Do et al. (2020); MDPI (2026) |

---

## 18. Cascading

| Field | Detail |
|-------|--------|
| **Definisi** | Model berurutan; setiap tahap menyaring kasus sebelum diteruskan ke tahap berikutnya. Tahap awal menangani kasus "mudah", tahap akhir menangani kasus "sulit" |
| **Problem yang didukung** | Face detection, Fraud detection, Intrusion detection |
| **Kapan digunakan** | Butuh trade-off accuracy vs compute; distribusi kelas tidak seimbang |
| **Kapan tidak digunakan** | Semua kasus equally difficult; real-time strict |
| **Asumsi** | Sebagian besar kasus dapat diselesaikan oleh tahap awal yang murah |
| **Data preprocessing** | Standar per tahap |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Recall (tahap awal), Precision (tahap akhir), F1 |
| **Hyperparameter penting** | `n_stages`, threshold per tahap, classifier per tahap |
| **Kisaran parameter awal** | n_stages: 3–5; threshold: 0.7–0.95 |
| **CPU/GPU/RAM** | Ringan saat inference (early exit); training sama dengan model terpisah |
| **Risiko leakage** | Sedang; threshold harus di-tune pada validation set |
| **Failure modes** | Threshold salah → terlalu banyak false negative di tahap awal |
| **Teknik pembanding** | Ensemble biasa, Early Exit networks |
| **Exact package** | Custom; `CascadeClassifier` (OpenCV untuk face detection) |
| **Contoh hypothesis** | "Cascading 3-stage dengan threshold 0.9/0.95/0.99 akan mencapai F1 ≥0.85 dengan compute 50% dari single large model" |
| **Acceptance criteria** | F1 ≥0.85; compute ≤50% |
| **Rejection criteria** | F1 <0.8; recall tahap awal <0.95 |
| **Sumber** | Viola & Jones (2001); Justia Patents (2015) |

---

## Ringkasan Insight untuk Kompetisi

| Teknik | Kelebihan Utama | Biaya | Kapan Dipilih |
|--------|----------------|-------|---------------|
| **NCL/Hybrid NCL** | Diversity eksplisit | Sedang | Pool model homogen |
| **DECORATE** | Small labeled set | Sedang | Semi-supervised |
| **Rotation Forest** | Fitur berkorelasi | Ringan | Tabular berkorelasi |
| **Greedy Selection** | Mudah, efektif | Sangat ringan | Pool besar |
| **DES (KNORA-U/META-DES)** | Adaptif per instance | Sedang | Data heterogen |
| **BMA** | Probabilistik | Berat | Uncertainty penting |
| **Deep Ensembles** | Uncertainty terbaik | Sangat berat | GPU tersedia |
| **MC Dropout** | Murah | Ringan | Resource terbatas |
| **ECOC** | Multi-class | Sedang | Banyak kelas |
| **Co/Tri-training** | Labeled sedikit | Ringan | Semi-supervised |
| **Snapshot/SWA** | "Gratis" dari 1 run | Ringan–Sedang | Deep learning |
| **BatchEnsemble** | Hemat memori | Ringan | GPU terbatas |
| **Sparse MoE** | Scale capacity | Berat | LLM |
| **GA/PSO** | Eksplorasi global | Berat | Pool sangat besar |
| **Online Update** | Concept drift | Ringan | Streaming |
| **Confidence Weighting** | Per-instance | Ringan | Confidence reliable |
| **Cascading** | Compute efisien | Ringan | Distribusi skewed |

---

# Ensemble Learning — Tambahan Niche (Lapisan Akhir)

> **Versi:** 1.2 | **Tier:** Berat (Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini melengkapi **Ensemble Learning — Dokumentasi Teknis Tier Berat v1.1** dengan keluarga teknik yang lebih niche namun tetap relevan untuk AI agents tingkat lanjut. Setiap entri mengikuti format 20-field.

---

## Daftar Isi

1. Probability Opinion Pools
2. Product-of-Experts
3. Multi-objective / Pareto Ensemble Pruning
4. Clustered Ensemble Pruning
5. Classifier Competence Learning
6. Forecast Combination Methods
7. Conformal Ensemble Aggregation

---

## 1. PROBABILITY OPINION POOLS

### 1.1 Linear Opinion Pool (LOP)

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi probabilitas dari beberapa model dengan **weighted arithmetic mean**: P(y\|x) = Σ w_m P_m(y\|x), dengan Σw=1, w≥0. Bentuk paling sederhana dari opinion pooling. |
| **Problem yang didukung** | Klasifikasi probabilistik, Regression (distribusi), Forecasting |
| **Kapan digunakan** | Model menghasilkan probabilitas terkalibrasi; ingin kombinasi linier sederhana dan interpretable. |
| **Kapan tidak digunakan** | Model sangat correlated; probabilitas tidak terkalibrasi; butuh ekspresi ketidakpastian yang lebih kaya. |
| **Asumsi** | Probabilitas individual meaningful; bobot dapat diestimasi dari validation/OOF. |
| **Data preprocessing** | Kalibrasi probabilitas per model (Platt/isotonic) sebelum pooling. |
| **Dataset scale** | Semua skala. |
| **Metric compatibility** | Log-loss, Brier, AUC, CRPS. |
| **Hyperparameter penting** | Bobot w_m; dapat uniform, dioptimasi via CV, atau berdasarkan performa. |
| **Kisaran parameter awal** | Uniform weights; optimasi convex bila perlu. |
| **CPU/GPU/RAM** | Sangat ringan. |
| **Risiko leakage** | Sedang; bobot harus di-fit pada validation/OOF, bukan test. |
| **Failure modes** | Weight overfitting; model redundant mendominasi; calibration drift. |
| **Teknik pembanding** | Logarithmic Opinion Pool, Product-of-Experts, Stacking. |
| **Ensemble compatibility** | Final combiner; dapat menjadi input stacking. |
| **Exact package** | Custom; `numpy` weighted average. |
| **Contoh hypothesis** | "LOP dengan bobot CV-optimal akan mencapai Brier score ≥3% lebih baik dari uniform average." |
| **Acceptance criteria** | Brier improvement ≥2%; bobot stabil across folds. |
| **Rejection criteria** | Bobot tidak stabil; improvement <1%. |
| **Sumber** | Genest & Zidek (1986); Stone (1961). |

### 1.2 Logarithmic Opinion Pool (LogOP)

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi probabilitas dengan **weighted geometric mean**: P(y\|x) ∝ Π P_m(y\|x)^{w_m}. Setelah normalisasi. |
| **Problem yang didukung** | Klasifikasi probabilistik, Forecasting. |
| **Kapan digunakan** | Model independent; ingin kombinasi yang lebih tajam (sharp) daripada LOP. |
| **Kapan tidak digunakan** | Model tidak independent; probabilitas mengandung nol (log tidak terdefinisi). |
| **Asumsi** | Probabilitas individual meaningful; independence conditional. |
| **Data preprocessing** | Kalibrasi; hindari probabilitas 0/1. |
| **Dataset scale** | Semua skala. |
| **Metric compatibility** | Log-loss, Brier, CRPS. |
| **Hyperparameter penting** | Bobot w_m; temperature. |
| **Kisaran parameter awal** | Uniform weights; temperature 1.0. |
| **CPU/GPU/RAM** | Sangat ringan. |
| **Risiko leakage** | Sedang. |
| **Failure modes** | Zero probability; overconfidence. |
| **Teknik pembanding** | LOP, PoE. |
| **Ensemble compatibility** | Final combiner. |
| **Exact package** | Custom. |
| **Contoh hypothesis** | "LogOP akan mencapai log-loss ≥3% lebih rendah dari LOP pada model independent." |
| **Acceptance criteria** | Log-loss improvement ≥2%. |
| **Rejection criteria** | Log-loss drop; probabilitas nol. |
| **Sumber** | Genest & Zidek (1986); Bordley (1982). |

---

## 2. PRODUCT-OF-EXPERTS (PoE)

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi probabilitas dengan **product** (setelah normalisasi): P(y\|x) ∝ Π P_m(y\|x). Kasus khusus LogOP dengan bobot sama. |
| **Problem yang didukung** | Klasifikasi probabilistik, Density estimation, Generative models. |
| **Kapan digunakan** | Model yang masing-masing ahli pada aspek berbeda; ingin kombinasi yang sharp. |
| **Kapan tidak digunakan** | Model correlated; probabilitas tidak terkalibrasi. |
| **Asumsi** | Setiap expert memberikan informasi independen tentang target. |
| **Data preprocessing** | Kalibrasi; hindari nol. |
| **Dataset scale** | Semua skala. |
| **Metric compatibility** | Log-loss, Brier, NLL. |
| **Hyperparameter penting** | Tidak ada (kecuali temperature). |
| **Kisaran parameter awal** | N/A. |
| **CPU/GPU/RAM** | Sangat ringan. |
| **Risiko leakage** | Rendah. |
| **Failure modes** | Overconfidence; expert tidak independen. |
| **Teknik pembanding** | LOP, LogOP, Mixture of Experts. |
| **Ensemble compatibility** | Final combiner; juga digunakan dalam MoE. |
| **Exact package** | Custom; `numpy` product. |
| **Contoh hypothesis** | "PoE akan mencapai NLL lebih rendah dari LOP pada expert independent." |
| **Acceptance criteria** | NLL improvement ≥2%. |
| **Rejection criteria** | NLL drop; overconfidence. |
| **Sumber** | Hinton (2002); Genest & Zidek (1986). |

---

## 3. MULTI-OBJECTIVE / PARETO ENSEMBLE PRUNING

| Field | Detail |
|-------|--------|
| **Definisi** | Pruning ensemble dengan mempertimbangkan **multiple objectives** (mis. akurasi, diversity, inference cost, memory) secara simultan, mencari solusi Pareto-optimal. |
| **Problem yang didukung** | Klasifikasi, Regresi dengan trade-off resource. |
| **Kapan digunakan** | Pool besar; butuh trade-off antara performa dan biaya; deployment constraint. |
| **Kapan tidak digunakan** | Single objective jelas; pool kecil. |
| **Asumsi** | Objective dapat dikuantifikasi; Pareto frontier meaningful. |
| **Data preprocessing** | OOF predictions; metrik biaya. |
| **Dataset scale** | Menengah–Besar. |
| **Metric compatibility** | Multi-objective: accuracy vs latency, diversity, size. |
| **Hyperparameter penting** | Populasi, generasi (jika evolutionary), bobot objective. |
| **Kisaran parameter awal** | NSGA-II population 50–100. |
| **CPU/GPU/RAM** | Sedang–Berat. |
| **Risiko leakage** | Tinggi; pruning pada validation/test yang sama. |
| **Failure modes** | Pareto front overfit; objective tidak realistis. |
| **Teknik pembanding** | Caruana, DREP, GA/PSO. |
| **Ensemble compatibility** | Final pruned ensemble. |
| **Exact package** | `pymoo` (Python); `DEAP`. |
| **Contoh hypothesis** | "Pareto pruning menghasilkan ensemble dengan akurasi setara dan latency 50% lebih rendah." |
| **Acceptance criteria** | Pareto improvement; trade-off tercapai. |
| **Rejection criteria** | Tidak ada solusi Pareto yang lebih baik. |
| **Sumber** | Zhou et al. (2002); multi-objective ensemble literature. |

---

## 4. CLUSTERED ENSEMBLE PRUNING

| Field | Detail |
|-------|--------|
| **Definisi** | Pruning ensemble dengan terlebih dahulu **mengelompokkan model** berdasarkan kesamaan prediksi/error, lalu memilih representatif dari setiap cluster. |
| **Problem yang didukung** | Klasifikasi, Regresi. |
| **Kapan digunakan** | Pool besar dengan banyak model redundan; ingin diversity terstruktur. |
| **Kapan tidak digunakan** | Pool kecil; model sudah diverse. |
| **Asumsi** | Cluster menangkap redundansi; representatif cukup. |
| **Data preprocessing** | OOF predictions; distance metric. |
| **Dataset scale** | Menengah–Besar. |
| **Metric compatibility** | Semua metric. |
| **Hyperparameter penting** | Jumlah cluster, metode clustering, representatif. |
| **Kisaran parameter awal** | k=5–20; hierarchical/ KMeans. |
| **CPU/GPU/RAM** | Ringan–Sedang. |
| **Risiko leakage** | Sedang; clustering pada validation/OOF. |
| **Failure modes** | Cluster tidak stabil; representatif lemah. |
| **Teknik pembanding** | DREP, Caruana. |
| **Ensemble compatibility** | Final pruned ensemble. |
| **Exact package** | `scipy.cluster.hierarchy`; `sklearn.cluster`. |
| **Contoh hypothesis** | "Clustered pruning mengurangi ukuran ensemble 60% dengan penurunan metrik <1%." |
| **Acceptance criteria** | Size reduction ≥50%; metric drop <2%. |
| **Rejection criteria** | Metric drop >3%. |
| **Sumber** | Zhou et al. (2002); Partalas et al. (2010). |

---

## 5. CLASSIFIER COMPETENCE LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Mempelajari **peta kompetensi** setiap classifier pada ruang input, biasanya via meta-learning, untuk memutuskan model mana yang dipakai per instance (mirip DES namun fokus pada pembelajaran kompetensi). |
| **Problem yang didukung** | Klasifikasi, Dynamic Ensemble Selection. |
| **Kapan digunakan** | Pool heterogen; kompetensi model bervariasi secara lokal. |
| **Kapan tidak digunakan** | Pool homogen; data sangat kecil. |
| **Asumsi** | Kompetensi dapat dipelajari dari fitur/region. |
| **Data preprocessing** | Scaling untuk region; OOF predictions. |
| **Dataset scale** | Kecil–Menengah. |
| **Metric compatibility** | Accuracy, F1. |
| **Hyperparameter penting** | Meta-classifier, region size, k. |
| **Kisaran parameter awal** | k=5–15; meta=RF/Logistic. |
| **CPU/GPU/RAM** | Sedang. |
| **Risiko leakage** | Tinggi; meta harus dilatih pada OOF. |
| **Failure modes** | Meta-overfit; region tidak representatif. |
| **Teknik pembanding** | DES, META-DES, KNORA. |
| **Ensemble compatibility** | Dynamic selection. |
| **Exact package** | `DESlib` (META-DES); custom. |
| **Contoh hypothesis** | "Competence learning meningkatkan akurasi ≥2% vs static ensemble." |
| **Acceptance criteria** | Accuracy improvement ≥2%. |
| **Rejection criteria** | Accuracy drop. |
| **Sumber** | Cruz et al. (2018); DESlib. |

---

## 6. FORECAST COMBINATION METHODS

| Field | Detail |
|-------|--------|
| **Definisi** | Keluarga metode untuk menggabungkan forecast dari beberapa model time series, seperti **Bates & Granger**, **Granger-Ramanathan**, **Minimum Variance**, **Inverse MSE**. |
| **Problem yang didukung** | Time series forecasting, Regression. |
| **Kapan digunakan** | Multiple forecasts tersedia; ingin mengurangi variance/error. |
| **Kapan tidak digunakan** | Single forecast; data non-temporal. |
| **Asumsi** | Forecast errors tidak sempurna berkorelasi; bobot dapat diestimasi. |
| **Data preprocessing** | Align time indices; handle missing. |
| **Dataset scale** | Semua skala. |
| **Metric compatibility** | MSE, MAE, MAPE, RMSE. |
| **Hyperparameter penting** | Bobot; window estimasi. |
| **Kisaran parameter awal** | Inverse MSE; rolling window 50–200. |
| **CPU/GPU/RAM** | Sangat ringan. |
| **Risiko leakage** | Tinggi; estimasi bobot harus out-of-sample. |
| **Failure modes** | Bobot tidak stabil; structural break. |
| **Teknik pembanding** | Simple average, Stacking. |
| **Ensemble compatibility** | Final combiner. |
| **Exact package** | `forecast` (R); `statsmodels`. |
| **Contoh hypothesis** | "Granger-Ramanathan akan mencapai RMSE ≥5% lebih rendah dari simple average." |
| **Acceptance criteria** | RMSE improvement ≥3%. |
| **Rejection criteria** | RMSE drop. |
| **Sumber** | Bates & Granger (1969); Timmermann (2006). |

---

## 7. CONFORMAL ENSEMBLE AGGREGATION

| Field | Detail |
|-------|--------|
| **Definisi** | Menggabungkan ensemble dengan **conformal prediction** untuk menghasilkan prediction sets/intervals dengan jaminan coverage, sering via aggregasi skor nonconformity dari beberapa model. |
| **Problem yang didukung** | Klasifikasi, Regression dengan uncertainty. |
| **Kapan digunakan** | Butuh prediction intervals/sets dengan coverage guarantee; ensemble tersedia. |
| **Kapan tidak digunakan** | Hanya point prediction; exchangeability tidak terpenuhi. |
| **Asumsi** | Exchangeability; skor nonconformity dapat dihitung. |
| **Data preprocessing** | Calibration set terpisah. |
| **Dataset scale** | Semua skala. |
| **Metric compatibility** | Coverage, set size, interval width. |
| **Hyperparameter penting** | Alpha (1-coverage); aggregasi skor. |
| **Kisaran parameter awal** | alpha=0.10. |
| **CPU/GPU/RAM** | Ringan. |
| **Risiko leakage** | Sedang; calibration set harus terpisah. |
| **Failure modes** | Coverage tidak tercapai; set terlalu besar. |
| **Teknik pembanding** | Quantile regression, Bayesian intervals. |
| **Ensemble compatibility** | Final aggregation. |
| **Exact package** | `MAPIE` (Python); `crepes`. |
| **Contoh hypothesis** | "Conformal ensemble aggregation mencapai coverage 90% dengan set size ≤3." |
| **Acceptance criteria** | Coverage ≥0.88; set size ≤3. |
| **Rejection criteria** | Coverage <0.80; set size >5. |
| **Sumber** | Vovk et al. (2005); Angelopoulos et al. (2024). |

---

## Ringkasan Tambahan Niche

| Teknik | Kelebihan | Biaya | Kapan Dipilih |
|--------|-----------|-------|---------------|
| Linear Opinion Pool | Sederhana, interpretable | Ringan | Probabilitas terkalibrasi |
| Logarithmic Opinion Pool | Lebih sharp | Ringan | Model independent |
| Product-of-Experts | Sangat sharp | Ringan | Expert independent |
| Pareto Pruning | Trade-off multi-objective | Sedang | Deployment constraint |
| Clustered Pruning | Diversity terstruktur | Sedang | Pool redundan |
| Competence Learning | Adaptif per instance | Sedang | Pool heterogen |
| Forecast Combination | Khusus time series | Ringan | Multiple forecasts |
| Conformal Aggregation | Coverage guarantee | Ringan | Uncertainty penting |

---

*Dokumen ini melengkapi versi 1.1 dan menutup lapisan niche ensemble learning. Setelah ini, cakupan taxonomy ensemble learning untuk general-purpose tabular AI agents dapat dianggap lengkap.*

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*