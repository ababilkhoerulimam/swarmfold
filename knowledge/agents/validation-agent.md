# Validation Strategies untuk Tabular Data — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk strategi validasi data tabular. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

1. [Walk-Forward / Expanding Window Cross-Validation](#1-walk-forward--expanding-window-cross-validation)
2. [Purged K-Fold & Embargo](#2-purged-k-fold--embargo)
3. [Temporal Leakage Checks (Validation Gates)](#3-temporal-leakage-checks-validation-gates)
4. [Kapoor & Narayanan 8-Type Leakage Taxonomy](#4-kapoor--narayanan-8-type-leakage-taxonomy)
5. [Mekanistik 4-Class Leakage Taxonomy](#5-mekanistik-4-class-leakage-taxonomy)
6. [Adversarial Validation](#6-adversarial-validation)
7. [Bootstrap Optimism Correction](#7-bootstrap-optimism-correction)
8. [Group K-Fold & Stratified Group K-Fold](#8-group-k-fold--stratified-group-k-fold)
9. [Nested Cross-Validation](#9-nested-cross-validation)
10. [Spatial Block Cross-Validation](#10-spatial-block-cross-validation)
11. [Survival / Censoring-Aware CV](#11-survival--censoring-aware-cv)
12. [IPCW Validation](#12-ipcw-validation)
13. [Competing-Risk CV](#13-competing-risk-cv)
14. [Cluster Bootstrap Validation](#14-cluster-bootstrap-validation)
15. [External Validation & Temporal External Validation](#15-external-validation--temporal-external-validation)
16. [Predefined Split](#16-predefined-split)
17. [Repeated K-Fold & Repeated Stratified K-Fold](#17-repeated-k-fold--repeated-stratified-k-fold)
18. [Leave-One-Group-Out (LOGO)](#18-leave-one-group-out-logo)
19. [Combinatorial Purged Cross-Validation (CPCV)](#19-combinatorial-purged-cross-validation-cpcv)
20. [Validation Agent Decision Hierarchy](#20-validation-agent-decision-hierarchy)

---

## 1. WALK-FORWARD / EXPANDING WINDOW CROSS-VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik validasi temporal di mana setiap fold memiliki batas waktu tegas: training set mencakup data sebelum batas, validation/test set mencakup data setelah batas. Chronological ordering ketat, tanpa pengacakan. Training window dapat berupa expanding (semakin besar) atau rolling (ukuran tetap) |
| **Problem yang didukung** | Time series forecasting, Temporal tabular classification/regression, Financial ML |
| **Kapan digunakan** | Data dengan urutan temporal; ada risiko temporal leakage; evaluasi out-of-time performance; production menerima data baru secara kronologis |
| **Kapan tidak digunakan** | Data i.i.d. tanpa dimensi waktu; dataset sangat kecil (<500 sampel) |
| **Asumsi** | Data diurutkan secara kronologis; distribusi target dapat berubah seiring waktu (concept drift); observasi dalam satu fold tidak independen; model hanya menggunakan informasi masa lalu |
| **Data preprocessing** | Sorting by time; pastikan tidak ada data masa depan yang masuk ke training fold; scaling fit hanya pada train fold |
| **Dataset scale** | Kecil–Besar (1K–10M+ baris) |
| **Metric compatibility** | Semua metric (MSE, MAE, AUC, F1, Accuracy, dll.); untuk forecasting: RMSE, MAPE, sMAPE |
| **Hyperparameter penting** | `n_splits`, `max_train_size` (rolling vs expanding), `test_size`, `gap` |
| **Kisaran parameter awal** | n_splits: 3–10; gap: 0–30 (tergantung horizon prediksi); test_size: 10–20% data |
| **CPU/GPU/RAM** | Sama dengan training model individual × n_splits; parallelizable per fold |
| **Risiko leakage** | **Kritis** jika menggunakan k-fold standar; walk-forward secara desain mencegah leakage temporal. Namun, gap harus memadai untuk menghindari leakage dari serial correlation |
| **Failure modes** | Fold terlalu sedikit → estimasi varians tinggi; gap tidak memadai → leakage dari serial correlation; data tidak cukup untuk n_splits besar; expanding window tidak cocok untuk concept drift kuat |
| **Teknik pembanding** | Purged K-Fold, Combinatorial Purged CV (CPCV), TimeSeriesSplit (sklearn), Blocked CV |
| **Ensemble compatibility** | Base untuk validasi semua ensemble temporal; menghasilkan OOF predictions yang valid untuk stacking |
| **Exact package** | `sklearn.model_selection.TimeSeriesSplit`; `purgedcv.WalkForwardSplit`; `macrosynergy.learning.splitters.walk_forward_splitters`; `skfolio.model_selection.WalkForward` |
| **Contoh hypothesis** | "Walk-forward CV dengan 5 folds akan menghasilkan estimasi out-of-time performance 2× lebih akurat (dalam hal optimisme) dibanding k-fold standar pada data tabular temporal" |
| **Acceptance criteria** | Gap antara CV score dan out-of-time score <5%; fold-by-fold degradation signal terdeteksi |
| **Rejection criteria** | Gap >15%; tidak ada sinyal degradasi antar fold |
| **Sumber** | scikit-learn docs (2026); purgedcv paper (Lazarev, 2026); Roberts et al., Ecography 40(8), 913–929 (2017) |

**Sub-varian:**

| Varian | Deskripsi | Kapan Digunakan |
|--------|-----------|-----------------|
| **Expanding Window** | Training set tumbuh setiap fold; validation window bergeser | Dataset cukup besar; ingin model melihat semua历史 data; long-term patterns dominan |
| **Rolling Window** | Training set ukuran tetap; jendela bergeser | Concept drift signifikan; ingin model fokus pada data terbaru; structural breaks umum |

---

## 2. PURGED K-FOLD & EMBARGO

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik cross-validation yang dikembangkan oleh Marcos López de Prado untuk financial ML. **Purging** menghapus baris training yang label-nya overlap dengan label test (karena horizon prediksi). **Embargo** menghapus baris training tepat setelah test window untuk mencegah leakage dari serial correlation |
| **Problem yang didukung** | Financial ML, Time series dengan overlapping labels, Dataset dengan serial correlation, Event prediction |
| **Kapan digunakan** | Label memiliki horizon (misal "return over next 20 bars"); serial correlation signifikan; ingin honest evaluation; rolling/sliding window observations |
| **Kapan tidak digunakan** | Label tidak overlap; data i.i.d.; horizon prediksi tidak diketahui |
| **Asumsi** | Label horizon diketahui; serial correlation dapat di-embargo dengan parameter tetap; information interval setiap observation dapat didefinisikan |
| **Data preprocessing** | Mengetahui t1 (label end time) untuk setiap sampel; information interval I_i = [t_i^start, t_i^end] |
| **Dataset scale** | Menengah–Besar (10K–1M+) |
| **Metric compatibility** | Semua metric; terutama Sharpe ratio, PSR, Deflated Sharpe untuk financial |
| **Hyperparameter penting** | `pct_embargo` (fraksi data untuk embargo), `pct_purge` (fraksi data untuk purging), `n_splits` |
| **Kisaran parameter awal** | pct_embargo: 0.01–0.02; pct_purge: 0.01–0.05; n_splits: 5–10 |
| **CPU/GPU/RAM** | Sama dengan k-fold standar + overhead purging; RAM moderate |
| **Risiko leakage** | **Sangat Kritis** tanpa purging/embargo; scikit-learn hanya menyediakan `gap` terbatas, tanpa label-overlap purging |
| **Failure modes** | Embargo terlalu kecil → residual leakage; terlalu besar → kehilangan data training; information interval salah didefinisikan |
| **Teknik pembanding** | Walk-Forward, CPCV, Blocked CV, hv-block CV |
| **Ensemble compatibility** | Menghasilkan OOF predictions yang lebih honest untuk stacking; dapat digunakan dengan meta-learning |
| **Exact package** | `purgedcv.PurgedKFold`, `purgedcv.CombinatorialPurgedCV`; `timeseriescv` (legacy) |
| **Contoh hypothesis** | "Purged K-Fold dengan embargo 1% akan mengurangi gap antara CV score dan test score dari 15% menjadi <5% pada data financial dengan overlapping labels" |
| **Acceptance criteria** | Gap CV-test <5%; tidak ada label overlap antara train dan test |
| **Rejection criteria** | Gap >10%; purging/embargo tidak berfungsi |
| **Sumber** | López de Prado, Advances in Financial Machine Learning (2018), Ch. 7 & 12; purgedcv (Lazarev, 2026); Bailey & López de Prado (2012) |

---

## 3. TEMPORAL LEAKAGE CHECKS (VALIDATION GATES)

| Field | Detail |
|-------|--------|
| **Definisi** | Pemeriksaan otomatis yang diterapkan sebelum training untuk mendeteksi temporal leakage. Gate-gate mencakup: shuffled target test, temporal boundary check, dan suspicious improvement detector |
| **Problem yang didukung** | Semua problem dengan data temporal |
| **Kapan digunakan** | Sebelum training model pada data temporal; sebagai gate di pipeline otomatis |
| **Kapan tidak digunakan** | Data i.i.d. tanpa dimensi waktu |
| **Asumsi** | Temporal leakage dapat dideteksi via pola statistik tertentu |
| **Data preprocessing** | Data temporal dengan timestamp |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Threshold untuk suspicious improvement (misal >20% antar fold); target shuffle iterations |
| **Kisaran parameter awal** | Improvement threshold: 0.2; target shuffle iterations: 100 |
| **CPU/GPU/RAM** | Ringan; hanya analisis statistik |
| **Risiko leakage** | **Kritis** jika gate tidak diimplementasikan |
| **Failure modes** | False positive (gate terlalu sensitif); false negative (leakage tidak terdeteksi) |
| **Teknik pembanding** | Manual inspection; model info sheets; adversarial validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | `target-leakage-detection` (R); custom implementation |
| **Contoh hypothesis** | "Validation gate dengan shuffled target test akan mendeteksi temporal leakage pada 90% kasus" |
| **Acceptance criteria** | Detection rate >80%; false positive rate <10% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | arXiv (2026); praktik industri; Kapoor & Narayanan, Patterns 4, 100804 (2023) |

**Gate-gate yang direkomendasikan:**

| Gate | Deskripsi | Cara Kerja |
|------|-----------|-----------|
| **Shuffled Target Test** | Apakah target secara tidak sengaja terhubung ke feature masa depan | Shuffle target, retrain, bandingkan performa |
| **Temporal Boundary Check** | Apakah ada data dari periode test yang masuk ke training | Verifikasi timestamp min/max per fold |
| **Suspicious Improvement Detector** | Peningkatan tiba-tiba yang tidak wajar antar fold | Threshold improvement (misal >20%) |

---

## 4. KAPOOR & NARAYANAN 8-TYPE LEAKAGE TAXONOMY

| Field | Detail |
|-------|--------|
| **Definisi** | Taksonomi 8 tipe leakage berdasarkan survei literatur lintas 17 bidang ilmu, ditemukan 294 paper yang terpengaruh leakage. Taksonomi mencakup mulai dari kesalahan textbook hingga open research problems |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Audit pipeline ML; penulisan paper; review model; reproducibility check |
| **Kapan tidak digunakan** | N/A (selalu relevan) |
| **Asumsi** | Leakage dapat diklasifikasikan ke dalam 8 tipe; model info sheets dapat mendeteksi leakage |
| **Data preprocessing** | N/A |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N/A (konseptual) |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Tidak menyadari salah satu dari 8 tipe; menganggap remeh selection leakage |
| **Teknik pembanding** | Mekanistik 4-Class Taxonomy, PLOS One Temporal Leakage Subtypes |
| **Ensemble compatibility** | Universal |
| **Exact package** | Model info sheets (template); `bioLeak` (R) |
| **Contoh hypothesis** | "Implementasi model info sheets akan mendeteksi leakage pada 90% kasus" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Kapoor & Narayanan, Patterns 4, 100804 (2023); PMC10499856 |

**8 Tipe Leakage:**

| No | Tipe Leakage | Deskripsi |
|----|-------------|-----------|
| 1 | **No test set** | Tidak ada test set terpisah |
| 2 | **Preprocessing on full data** | Scaling/encoding di full data |
| 3 | **Feature selection on full data** | Feature selection di full data |
| 4 | **Duplicate rows** | Duplikat antara train dan test |
| 5 | **Temporal leakage** | Menggunakan informasi masa depan |
| 6 | **Group leakage** | Entitas sama di train dan test |
| 7 | **Label leakage** | Fitur yang tidak tersedia saat prediksi |
| 8 | **Selection leakage** | Peeking, cherry-picking, seed selection |

---

## 5. MEKANISTIK 4-CLASS LEAKAGE TAXONOMY

| Field | Detail |
|-------|--------|
| **Definisi** | Taksonomi alternatif yang memotong dari sudut mekanisme statistik. Empat kelas: **Estimation leakage** (parameter fitting pada data penuh), **Selection leakage** (peeking dan cherry-picking), **Memorization leakage** (model capacity), **Boundary leakage** (temporal boundary) |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Audit pipeline; prioritasi perbaikan leakage; quantitative risk assessment |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Magnitude leakage dapat diukur secara kuantitatif; efek dapat diestimasi via eksperimen counterfactual |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | AUC, ΔAUC, Cohen's d |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Berat untuk eksperimen counterfactual |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Estimation leakage dianggap remeh padahal selection leakage jauh lebih berbahaya; efek bawah (lower bounds) karena random CV |
| **Teknik pembanding** | Kapoor taxonomy |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation; `epagogy/ml` (GitHub) |
| **Contoh hypothesis** | "Selection leakage akan menyebabkan inflasi AUC ≥0.05 pada dataset kecil, sedangkan estimation leakage hanya ≤0.005" |
| **Acceptance criteria** | ΔAUC estimation ≤0.005; ΔAUC selection ≥0.03 |
| **Rejection criteria** | Hasil bertentangan |
| **Sumber** | "Which Leakage Types Matter?" (2026), 2,047 benchmark datasets; ar5iv.labs.arxiv.org/html/2604.04199 |

**4 Kelas Mekanistik:**

| Kelas | Deskripsi | Magnitude | Kapan Paling Berbahaya |
|-------|-----------|-----------|----------------------|
| **I. Estimation** | Fitting scaler/encoder pada full data | **Negligible** (\|ΔAUC\| ≤ 0.005) | Hampir tidak pernah |
| **II. Selection** | Peeking, seed cherry-picking | **Substantial** (ΔAUC = +0.040–0.045) | Dataset kecil (92% prevalence) |
| **III. Memorization** | Model menghafal duplikat | **Substantial** (ΔAUC = +0.013–0.026 at 10% duplication) | Ada duplikasi data; high-capacity models |
| **IV. Boundary** | Temporal boundary leakage | **Invisible under random CV** | Data temporal/group/spatial |

**Insight Kritis:** Textbook emphasis terbalik — normalization leakage paling tidak penting; selection leakage pada dataset praktis paling berbahaya.

---

## 6. ADVERSARIAL VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik mendeteksi distribusi shift antara training set dan test/production set dengan merangkainya sebagai masalah klasifikasi biner. Training dan test digabung, diberi label biner (0=train, 1=test), lalu classifier dilatih untuk membedakan keduanya |
| **Problem yang didukung** | Semua problem dengan train/test shift; Competition ML; Production monitoring; Credit scoring |
| **Kapan digunakan** | Sebelum training untuk deteksi covariate shift; monitoring drift di production; feature selection; memilih split point yang lebih representatif |
| **Kapan tidak digunakan** | Data i.i.d. sempurna; dataset sangat kecil |
| **Asumsi** | Jika dua dataset dari distribusi sama, classifier tidak dapat membedakan dengan AUC > 0.5; drift multivariat dapat dideteksi via classifier |
| **Data preprocessing** | Standar; scaling untuk logistic regression; handling categorical |
| **Dataset scale** | Menengah–Besar (10K–1M+) |
| **Metric compatibility** | AUC (adversarial), feature importance, PSI, JS divergence |
| **Hyperparameter penting** | `classifier_type`, `cv_folds`, `threshold`, `importance_type` |
| **Kisaran parameter awal** | LogisticRegression; cv=5; threshold AUC: 0.7 |
| **CPU/GPU/RAM** | Ringan–Sedang; tergantung classifier; GPU untuk large-scale |
| **Risiko leakage** | Rendah; teknik ini justru mendeteksi leakage/shift |
| **Failure modes** | AUC ≈ 0.5 tetapi ada shift non-linear; feature importance tidak stabil; drift adversarials yang tidak terdeteksi |
| **Teknik pembanding** | Uji statistik univariat (KS test, PSI), Domain-Free Adversarial Splitting (DFAS) |
| **Ensemble compatibility** | Universal; dapat dikombinasi dengan multiple drift detectors |
| **Exact package** | `adval` (PyPI); `adversarial-validation` (GitHub); `endgame.validation.AdversarialValidator`; `Evidently`; `Alibi Detect` |
| **Contoh hypothesis** | "Adversarial validation dengan LogisticRegression akan mendeteksi drift pada dataset dengan AUC ≥0.75" |
| **Acceptance criteria** | AUC ≥0.7 → drift terdeteksi; feature importance mengidentifikasi fitur drifting |
| **Rejection criteria** | AUC ≈0.5 tetapi performa model drop di test |
| **Sumber** | Zając (2017); Qian et al. (2021); medRxiv (2026); GitHub (2025) |

**Interpretasi AUC:**

| AUC Range | Interpretasi | Tindakan |
|-----------|-------------|----------|
| 0.50–0.55 | Tidak ada shift signifikan | Lanjutkan training |
| 0.55–0.70 | Shift ringan | Pertimbangkan importance weighting |
| 0.70–0.85 | Shift signifikan | Feature selection/removal; reweighting |
| >0.85 | Shift ekstrem | Dataset mungkin tidak kompatibel |

---

## 7. BOOTSTRAP OPTIMISM CORRECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Metode untuk mengestimasi dan mengoreksi optimism (bias) dari performance yang diukur pada data yang sama dengan training. Optimism = Performance_bootstrap,bootstrap − Performance_bootstrap,original; Performance_corrected = Performance_apparent − Mean(Optimism) |
| **Problem yang didukung** | Dataset kecil/menengah; clinical prediction; risk score; logistic regression; Cox regression |
| **Kapan digunakan** | Dataset kecil–menengah (<5000); butuh honest estimate; model selection bias concern; alternative to KFold |
| **Kapan tidak digunakan** | Dataset sangat besar (KFold lebih efisien); model sangat kompleks (bootstrap mahal) |
| **Asumsi** | Seluruh model-development procedure harus diulang dalam setiap bootstrap, termasuk selection/tuning |
| **Data preprocessing** | Standar; imputation, scaling, selection harus di dalam bootstrap loop |
| **Dataset scale** | Kecil–Menengah (1K–10K) |
| **Metric compatibility** | AUC, Brier score, calibration slope, R², C-index |
| **Hyperparameter penting** | `B` (jumlah bootstrap resamples, default 500), `model_fun`, `pred_fun`, `score_fun` |
| **Kisaran parameter awal** | B: 200–500 |
| **CPU/GPU/RAM** | CPU-bound; B × model training cost; parallelizable |
| **Risiko leakage** | Rendah jika seluruh procedure diulang dalam setiap bootstrap |
| **Failure modes** | Model complexity mempengaruhi performa; .632+ estimator terbaik untuk small-sample; bootstrap CI coverage bisa rendah |
| **Teknik pembanding** | K-Fold CV, .632 Bootstrap, .632+ Bootstrap, Repeated K-Fold |
| **Ensemble compatibility** | Universal; dapat digunakan untuk ensemble model selection |
| **Exact package** | `pminternal` (R); `rms::validate` (R); `boot` (R); custom Python |
| **Contoh hypothesis** | "Bootstrap optimism correction dengan B=500 akan memberikan estimasi error yang lebih dekat ke independent test set dibanding K-Fold pada dataset <1000 sampel" |
| **Acceptance criteria** | Gap bootstrap-corrected vs independent test <5%; CI coverage ≥90% |
| **Rejection criteria** | Gap >10%; bootstrap tidak konvergen |
| **Sumber** | Noma et al., Statistics in Medicine 40(26), 5691-5701 (2021); BMJ (2024); pminternal (CRAN, 2025); Efron & Tibshirani (1997) |

**Prosedur:**

```
Original data → fit model → apparent performance
Repeat B times:
  bootstrap sample → fit entire modeling procedure
  → evaluate bootstrap data → evaluate original data
  → difference = optimism
Average optimism
Apparent − optimism = corrected performance
```

---

## 8. GROUP K-FOLD & STRATIFIED GROUP K-FOLD

### 8.1 Group K-Fold

| Field | Detail |
|-------|--------|
| **Definisi** | K-Fold di mana grup (entity/cluster) tidak muncul sekaligus di training dan validation fold. Constraint: G_train ∩ G_valid = ∅ |
| **Problem yang didukung** | Repeated entity/cluster data; Patient data; Customer data; Device data; Household data |
| **Kapan digunakan** | Production harus generalize ke unseen groups; repeated measures per subject; entity leakage concern |
| **Kapan tidak digunakan** | Production hanya untuk entitas yang sama (within-entity prediction); data i.i.d. tanpa grup |
| **Asumsi** | Grup independen; jumlah distinct groups ≥ n_splits |
| **Data preprocessing** | Standar; group column harus tersedia |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_splits`, `groups` |
| **Kisaran parameter awal** | n_splits: 5; groups=df["entity_id"] |
| **CPU/GPU/RAM** | Sama dengan KFold |
| **Risiko leakage** | **Tinggi** jika grup tidak dipisahkan; entity leakage |
| **Failure modes** | N_groups < n_splits; group terlalu kecil; class imbalance antar fold |
| **Teknik pembanding** | StratifiedGroupKFold, GroupShuffleSplit, LeaveOneGroupOut |
| **Ensemble compatibility** | Universal; menghasilkan OOF predictions yang valid |
| **Exact package** | `sklearn.model_selection.GroupKFold` |
| **Contoh hypothesis** | "GroupKFold akan mengurangi gap CV-test dari 20% menjadi <5% pada data pasien dengan repeated measurements" |
| **Acceptance criteria** | Gap CV-test <5%; tidak ada group overlap |
| **Rejection criteria** | Gap >10%; group overlap terdeteksi |
| **Sumber** | scikit-learn docs; Roberts et al., Ecography 40(8), 913–929 (2017) |

### 8.2 Stratified Group K-Fold

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi Stratified K-Fold dan Group K-Fold: menjaga group separation sekaligus class balance antar fold |
| **Problem yang didukung** | Classification dengan grouped data dan class imbalance |
| **Kapan digunakan** | Fraud per account; churn per customer; sensor failure per machine; patient diagnosis |
| **Kapan tidak digunakan** | Data i.i.d.; tidak ada grup; tidak ada class imbalance |
| **Asumsi** | Dua constraint dapat dipenuhi simultan |
| **Dataset scale** | Menengah–Besar |
| **Hyperparameter penting** | `n_splits`, `groups` |
| **Exact package** | `sklearn.model_selection.StratifiedGroupKFold` |
| **Contoh hypothesis** | "StratifiedGroupKFold akan memberikan estimasi AUC yang lebih stabil dibanding GroupKFold pada data fraud dengan customer_id repeated" |
| **Acceptance criteria** | AUC stability improvement ≥30%; class balance antar fold |
| **Rejection criteria** | Tidak dapat memenuhi kedua constraint |
| **Sumber** | scikit-learn docs (2026) |

---

## 9. NESTED CROSS-VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Protokol validasi dua level: outer CV untuk estimasi performa generalisasi, inner CV untuk hyperparameter tuning/model selection. Mencegah optimistic bias dari penggunaan data yang sama untuk tuning dan evaluation |
| **Problem yang didukung** | Hyperparameter search; model comparison; feature selection |
| **Kapan digunakan** | Butuh honest final estimate setelah tuning; scientific benchmark; high-stakes comparison |
| **Kapan tidak digunakan** | Rapid experimentation (flat CV acceptable + warning); few hyperparameters (Wainer & Cawley, 2021) |
| **Asumsi** | Leakage constraint tidak boleh hilang di inner loop; outer test tidak boleh disentuh |
| **Data preprocessing** | Seluruh preprocessing harus di dalam inner loop |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `outer_cv`, `inner_cv`, `n_jobs` |
| **Kisaran parameter awal** | Outer 5 × Inner 3–5 |
| **CPU/GPU/RAM** | Berat: outer × inner training cost |
| **Risiko leakage** | **Tinggi** jika tidak nested; selection bias |
| **Failure modes** | Variance estimasi tinggi; computationally expensive; Wainer & Cawley menunjukkan flat CV sering cukup untuk algorithm selection |
| **Teknik pembanding** | Flat CV (acceptable for development) |
| **Ensemble compatibility** | Universal; standar untuk stacking |
| **Exact package** | `sklearn.model_selection.GridSearchCV` + outer CV; custom implementation |
| **Contoh hypothesis** | "Nested CV akan mengurangi bias estimasi error dari 15% menjadi <2% pada dataset null" |
| **Acceptance criteria** | Bias reduction ≥80%; estimate close to independent test |
| **Rejection criteria** | Bias tidak berkurang; runtime >10× flat CV |
| **Sumber** | Varma & Simon, BMC Bioinformatics 7:91 (2006); Cawley & Talbot, JMLR 11 (2010); Wainer & Cawley, Expert Systems with Applications (2021) |

---

## 10. SPATIAL BLOCK CROSS-VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Cross-validation yang memisahkan data berdasarkan blok spasial, bukan random. Tujuannya menghindari spatial autocorrelation antara train dan test |
| **Problem yang didukung** | Geospatial prediction, Real estate, Weather, Agriculture, Ecology, Delivery ETA |
| **Kapan digunakan** | Data dengan latitude/longitude; spatial dependence; production untuk lokasi baru |
| **Kapan tidak digunakan** | Data tanpa koordinat; spatial independence |
| **Asumsi** | Spatial autocorrelation signifikan; blok spasial dapat didefinisikan |
| **Data preprocessing** | Koordinat; blok spasial; buffering |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `block_size`, `buffer`, `method` (spatial/environmental) |
| **Kisaran parameter awal** | block_size: berdasarkan variogram range |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika random CV digunakan; spatial leakage |
| **Failure modes** | Block terlalu besar → overestimation error; block terlalu kecil → residual leakage; tidak ada block size yang unbiased |
| **Teknik pembanding** | Random K-Fold (optimistic), Leave-Location-Out |
| **Ensemble compatibility** | Universal |
| **Exact package** | `blockCV` (R); `spatialCV` (Python); custom implementation |
| **Contoh hypothesis** | "Spatial block CV akan mengurangi optimisme AUC dari +0.16 menjadi <0.05 dibanding random K-Fold" |
| **Acceptance criteria** | Gap random vs spatial <0.05; block size sesuai variogram |
| **Rejection criteria** | Gap >0.10; block size tidak sesuai |
| **Sumber** | Valavi et al., blockCV (2025); Roberts et al., Ecography 40(8), 913–929 (2017); Frontiers in Remote Sensing (2025) |

---

## 11. SURVIVAL / CENSORING-AWARE CV

| Field | Detail |
|-------|--------|
| **Definisi** | Cross-validation yang memperhitungkan data tersensor (censored). Event=0 tidak berarti tidak akan terjadi, hanya belum terobservasi. Metrik dan preprocessing censoring harus berada di dalam validation framework |
| **Problem yang didukung** | Time-to-event prediction; Churn prediction; Credit risk; Clinical survival |
| **Kapan digunakan** | Target = time-to-event; ada data tersensor; butuh prediksi hazard/risk |
| **Kapan tidak digunakan** | Tidak ada data tersensor; target kontinu biasa |
| **Asumsi** | Censoring at random (CAR); censoring distribution dapat diestimasi tanpa mencemari test fold |
| **Data preprocessing** | Time dan status columns; event-stratified splitting untuk imbalanced censoring |
| **Dataset scale** | Menengah–Besar (10K–1M) |
| **Metric compatibility** | C-index (concordance), time-dependent AUC, Brier score, Integrated Brier Score (IBS) |
| **Hyperparameter penting** | `n_splits`, `censoring_estimator` |
| **Kisaran parameter awal** | n_splits: 5; censoring: Kaplan-Meier |
| **CPU/GPU/RAM** | Sama dengan model base |
| **Risiko leakage** | **Tinggi**; censoring distribution harus dipelajari tanpa test fold |
| **Failure modes** | Censoring rate terlalu tinggi (>90%); IPCW tidak stabil; metric biasa (accuracy/F1) salah |
| **Teknik pembanding** | IPCW Validation, Competing-Risk CV, XGBoost-Surv |
| **Ensemble compatibility** | Universal; C-index untuk ensemble survival |
| **Exact package** | `scikit-survival` (Python); `pec`/`riskRegression` (R); `mlr3proba` (R) |
| **Contoh hypothesis** | "Survival-aware CV dengan IPCW C-index akan memberikan estimasi yang lebih akurat dibanding accuracy biasa pada data dengan 70% censoring" |
| **Acceptance criteria** | C-index estimate stabil; gap CV-test <5% |
| **Rejection criteria** | Menggunakan accuracy/F1 untuk survival; censoring diabaikan |
| **Sumber** | Springer Lifetime Data Analysis (2026); Frontiers (2024); PLOS One (2026) |

---

## 12. IPCW VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Inverse Probability of Censoring Weighting: memberikan weight kepada observations berdasarkan probabilitas masih terobservasi, untuk mengoreksi incomplete follow-up |
| **Problem yang didukung** | Survival analysis dengan censoring; C-index correction |
| **Kapan digunakan** | Censoring rate tinggi; butuh unbiased evaluation; C-index correction |
| **Kapan tidak digunakan** | Tidak ada censoring; censoring rate sangat rendah |
| **Asumsi** | Censoring at random (CAR); censoring distribution dapat diestimasi |
| **Data preprocessing** | Time dan status columns |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | C-index (IPCW-corrected), Brier score, IBS |
| **Hyperparameter penting** | `censoring_estimator`, `time_points` |
| **Kisaran parameter awal** | Kaplan-Meier; time_points: quantiles |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi**; censoring distribution harus di-estimasi pada training fold saja |
| **Failure modes** | IPCW tidak stabil jika censoring tinggi; misrepresentation sebagai estimator |
| **Teknik pembanding** | Harrell's C-index (uncorrected) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `scikit-survival`; `pec` (R); `riskRegression` (R) |
| **Contoh hypothesis** | "IPCW C-index akan memberikan estimasi yang lebih robust dibanding Harrell's C-index pada data dengan censoring berbeda antar grup" |
| **Acceptance criteria** | IPCW C-index stabil; type-1 error rate nominal |
| **Rejection criteria** | IPCW tidak stabil; estimasi tidak konvergen |
| **Sumber** | ScienceDirect (2024); Frontiers (2024); Wiley (2016) |

---

## 13. COMPETING-RISK CV

| Field | Detail |
|-------|--------|
| **Definisi** | Cross-validation untuk data dengan multiple mutually competing events. Event-specific risk harus dievaluasi, bukan binary classification biasa |
| **Problem yang didukung** | Competing risks; Clinical prediction; Kidney transplant; Cancer survival |
| **Kapan digunakan** | Multiple event types (heart attack, stroke, death other cause); event-specific prediction |
| **Kapan tidak digunakan** | Single event type; binary classification |
| **Asumsi** | Events saling competing; cumulative incidence function (CIF) model |
| **Data preprocessing** | Event type column; time column |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Competing-risk C-index, CIF-based metrics |
| **Hyperparameter penting** | `n_splits`, `event_type` |
| **Kisaran parameter awal** | n_splits: 5 |
| **CPU/GPU/RAM** | Sama dengan model base |
| **Risiko leakage** | **Tinggi**; event-specific risk harus di-estimasi dengan benar |
| **Failure modes** | Mengabaikan competing events; menggunakan binary classification |
| **Teknik pembanding** | Survival-aware CV, IPCW Validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | `QHScrnomo` (R); `pec` (R); `riskRegression` (R) |
| **Contoh hypothesis** | "Competing-risk CV akan memberikan estimasi CIF yang lebih akurat dibanding binary classification pada data transplant dengan 3 event types" |
| **Acceptance criteria** | CIF calibration baik; C-index improvement |
| **Rejection criteria** | Mengabaikan competing events |
| **Sumber** | CRAN (2025); EBSCO (2026); Tandfonline (2026) |

---

## 14. CLUSTER BOOTSTRAP VALIDATION

| Field | Detail |
|-------|--------|
| **Definisi** | Bootstrap yang sampling pada level cluster/entity, bukan row. Semua observations dari entity terpilih ikut dalam bootstrap sample |
| **Problem yang didukung** | Hierarchical/grouped data; Patient data; Household data; School data |
| **Kapan digunakan** | Data dengan cluster structure; variance estimation; uncertainty quantification |
| **Kapan tidak digunakan** | Data i.i.d. tanpa cluster |
| **Asumsi** | Cluster independen; within-cluster dependence |
| **Data preprocessing** | Cluster ID column |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric; CI estimation |
| **Hyperparameter penting** | `B` (bootstrap iterations), `cluster_id` |
| **Kisaran parameter awal** | B: 500–1000 |
| **CPU/GPU/RAM** | CPU-bound; B × model training |
| **Risiko leakage** | Rendah jika cluster structure dijaga |
| **Failure modes** | Cluster terlalu kecil; bootstrap tidak representatif |
| **Teknik pembanding** | Row bootstrap, Group K-Fold |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ClusterBootstrap` (R); custom implementation |
| **Contoh hypothesis** | "Cluster bootstrap akan memberikan CI yang lebih lebar dan akurat dibanding row bootstrap pada data patient dengan repeated measures" |
| **Acceptance criteria** | CI coverage ≥90%; variance estimate stabil |
| **Rejection criteria** | CI coverage <80% |
| **Sumber** | Behavior Research Methods (2019) |

---

## 15. EXTERNAL VALIDATION & TEMPORAL EXTERNAL VALIDATION

### 15.1 External Validation

| Field | Detail |
|-------|--------|
| **Definisi** | Validasi model pada dataset/site/population yang berbeda dari development data. Menjawab pertanyaan: apakah model transportable ke population/setting lain? |
| **Problem yang didukung** | Clinical prediction; Multi-site studies; Generalization testing |
| **Kapan digunakan** | Dataset/site/population lain tersedia; butuh test transportability |
| **Kapan tidak digunakan** | Tidak ada dataset eksternal |
| **Asumsi** | External dataset representatif untuk target population |
| **Data preprocessing** | Sama dengan development; tidak ada tuning pada external |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric; calibration, discrimination |
| **Hyperparameter penting** | N/A (no tuning) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan (inference only) |
| **Risiko leakage** | **Kritis** jika external data digunakan untuk tuning/selection |
| **Failure modes** | External dataset tidak representatif; distribution shift ekstrem |
| **Teknik pembanding** | Internal CV, Temporal External Validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | N/A (protocol) |
| **Contoh hypothesis** | "External validation pada Hospital B akan menunjukkan AUC drop <10% vs internal CV" |
| **Acceptance criteria** | AUC drop <10%; calibration slope ≈1 |
| **Rejection criteria** | AUC drop >20%; calibration buruk |
| **Sumber** | Nature npj Digital Medicine (2021); Springer (2026) |

### 15.2 Temporal External Validation

| Field | Detail |
|-------|--------|
| **Definisi** | External validation pada periode waktu yang berbeda (future data). Mengukur degradation/concept drift |
| **Problem yang didukung** | Forecasting; Concept drift detection; Market drift; Fraud pattern evolution |
| **Kapan digunakan** | Data periode masa depan tersedia; butuh test degradation over time |
| **Kapan tidak digunakan** | Tidak ada data temporal; data i.i.d. |
| **Asumsi** | Future data representatif untuk deployment; concept drift dapat diukur |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric; degradation curve |
| **Hyperparameter penting** | `split_date` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis**; final locked test tidak boleh disentuh |
| **Failure modes** | Concept drift ekstrem; future data tidak representatif |
| **Teknik pembanding** | External Validation, Walk-Forward CV |
| **Exact package** | N/A (protocol) |
| **Contoh hypothesis** | "Temporal external validation akan menunjukkan gap CV-test ≥10% pada data dengan concept drift" |
| **Acceptance criteria** | Gap dilaporkan; degradation signal terdeteksi |
| **Rejection criteria** | Gap tidak dilaporkan |
| **Sumber** | European Journal of Medical Research (2026); Semantic Scholar (2026) |

**Prinsip Final Locked Test:**
```
external test exists?
       ↓
YES
       ↓
never consume it during:
- feature selection
- hyperparameter tuning
- threshold selection
- calibration tuning
```

---

## 16. PREDEFINED SPLIT

| Field | Detail |
|-------|--------|
| **Definisi** | Splitter yang menggunakan predefined scheme specified by user melalui `test_fold` parameter. Setiap sample di-assign ke paling banyak satu test set fold |
| **Problem yang didukung** | Kaggle; Benchmark datasets; Clinical benchmarks; Internal company validation periods |
| **Kapan digunakan** | Dataset sudah memiliki fold assignment resmi; competition validation set |
| **Kapan tidak digunakan** | Tidak ada predefined split; ingin bikin fold sendiri |
| **Asumsi** | Predefined fold representatif untuk evaluasi |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `test_fold` (array) |
| **Kisaran parameter awal** | -1 untuk tidak masuk test fold |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah jika predefined split benar |
| **Failure modes** | Predefined split tidak representatif |
| **Teknik pembanding** | KFold, GroupKFold |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.PredefinedSplit` |
| **Contoh hypothesis** | "PredefinedSplit akan menghasilkan evaluasi yang konsisten dengan leaderboard Kaggle" |
| **Acceptance criteria** | CV score correlation dengan leaderboard ≥0.8 |
| **Rejection criteria** | Correlation <0.5 |
| **Sumber** | scikit-learn docs (2026) |

---

## 17. REPEATED K-FOLD & REPEATED STRATIFIED K-FOLD

| Field | Detail |
|-------|--------|
| **Definisi** | K-Fold yang diulang R kali dengan randomization berbeda. Score = rata-rata dari R×K folds |
| **Problem yang didukung** | IID regression; Classification with imbalance; Butuh robust estimate |
| **Kapan digunakan** | CV score unstable; dataset tidak terlalu besar; butuh confidence interval |
| **Kapan tidak digunakan** | Dataset sangat besar (single KFold cukup); computational budget terbatas |
| **Asumsi** | Repeated randomization mengurangi ketergantungan pada satu split |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_splits`, `n_repeats` |
| **Kisaran parameter awal** | n_splits=5, n_repeats=10 |
| **CPU/GPU/RAM** | R × KFold cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | R terlalu kecil → tidak stabil; R terlalu besar → mahal |
| **Teknik pembanding** | KFold, Bootstrap |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.RepeatedKFold`, `RepeatedStratifiedKFold` |
| **Contoh hypothesis** | "Repeated K-Fold dengan 10 repeats akan mengurangi variance CV estimate ≥50% vs single KFold" |
| **Acceptance criteria** | Variance reduction ≥50%; CI stabil |
| **Rejection criteria** | Variance tidak berkurang; runtime >5× |
| **Sumber** | scikit-learn docs; Semantic Scholar (2026) |

---

## 18. LEAVE-ONE-GROUP-OUT (LOGO)

| Field | Detail |
|-------|--------|
| **Definisi** | Setiap fold menahan seluruh satu group dan melatih pada semua group lainnya |
| **Problem yang didukung** | Generalization ke unseen group; Hospital; Country; Store; Factory |
| **Kapan digunakan** | Jumlah group moderate; butuh test generalization ke satu unseen group |
| **Kapan tidak digunakan** | Jumlah group sangat besar; computational budget terbatas |
| **Asumsi** | Group independen; setiap group cukup representatif |
| **Data preprocessing** | Group column |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `groups` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N_groups × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Group terlalu kecil; N_groups sangat besar → mahal |
| **Teknik pembanding** | GroupKFold, LeavePGroupsOut, GroupShuffleSplit |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.LeaveOneGroupOut` |
| **Contoh hypothesis** | "LOGO akan menunjukkan AUC drop ≥15% dibanding random CV pada data multi-hospital" |
| **Acceptance criteria** | Gap LOGO vs random ≥10% |
| **Rejection criteria** | Gap <5% |
| **Sumber** | scikit-learn docs; pubs.acs.org (2026); MDPI (2026) |

---

## 19. COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)

| Field | Detail |
|-------|--------|
| **Definisi** | Advanced purged CV: data dibagi N groups, lalu k groups dipakai sebagai test secara kombinatorial. Setiap split menerapkan purging + embargo. Tujuan: multiple out-of-sample paths |
| **Problem yang didukung** | Financial ML; Event prediction; Backtest overfitting concern |
| **Kapan digunakan** | Overlapping labels; many strategies/models compared; butuh multiple backtest paths |
| **Kapan tidak digunakan** | Default tabular CV; dataset kecil |
| **Asumsi** | Purging + embargo diterapkan; combinatorial paths representatif |
| **Data preprocessing** | Information interval setiap observation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Sharpe ratio, PSR, Deflated Sharpe, PBO |
| **Hyperparameter penting** | `N` (groups), `k` (test groups), `pct_embargo` |
| **Kisaran parameter awal** | N=6, k=2 → C(6,2)=15 splits |
| **CPU/GPU/RAM** | Berat; C(N,k) × training cost |
| **Risiko leakage** | **Sangat Kritis** tanpa purging/embargo |
| **Failure modes** | C(N,k) explosion; sample size per fold berkurang |
| **Teknik pembanding** | Purged K-Fold, Walk-Forward |
| **Ensemble compatibility** | Universal; backtest path reconstruction |
| **Exact package** | `purgedcv.CombinatorialPurgedCV`; `timeseriescv.CombPurgedKFoldCV` |
| **Contoh hypothesis** | "CPCV akan mengurangi backtest overfitting dibanding single walk-forward path pada data financial dengan overlapping labels" |
| **Acceptance criteria** | PBO reduction ≥30%; multiple paths stabil |
| **Rejection criteria** | Tidak ada improvement; terlalu mahal |
| **Sumber** | López de Prado (2018), Ch. 12; ScienceDirect (2024); purgedcv (2026) |

---

## 20. VALIDATION AGENT DECISION HIERARCHY

### Decision Tree

```
START
  │
  ▼
Apakah deployment prediction bergantung pada waktu?
  │
  ├── YES ─────────────────────────────────────────────┐
  │                                                    │
  │    Apakah label / sample memiliki interval         │
  │    yang saling overlap?                            │
  │                                                    │
  │       ├── YES → PURGED CV                          │
  │       │             │                              │
  │       │             └─ strong adjacent dependence?
  │       │                    │
  │       │                    ├─ YES → + EMBARGO
  │       │                    └─ NO  → Purging only
  │       │
  │       └── NO → TIMESERIESSPLIT
  │                    │
  │                    └─ need separation?
  │                           → use gap
  │
  └── NO
       │
       ▼
Apakah beberapa rows berasal dari entity/group yang sama?
       │
       ├── YES
       │    │
       │    ▼
       │  Apakah production harus generalize ke unseen groups?
       │    │
       │    ├── YES
       │    │    │
       │    │    ▼
       │    │  Classification?
       │    │    │
       │    │    ├─ NO → GROUP K-FOLD
       │    │    │
       │    │    └─ YES
       │    │         │
       │    │         ├─ imbalance/material class balance
       │    │         │      → STRATIFIED GROUP K-FOLD
       │    │         │
       │    │         └─ otherwise
       │    │                → GROUP K-FOLD
       │    │
       │    └── NO
       │         → split according to actual deployment scenario
       │
       └── NO
            │
            ▼
          Classification?
            │
            ├── YES → STRATIFIED K-FOLD
            │
            └── NO  → K-FOLD
```

### Priority Order

$$
Temporal/Dependency\ constraints
>
Group\ constraints
>
Stratification
>
ordinary\ KFold
$$

### Acceptance Criteria untuk Validation Agent

| Check | Accept |
|-------|--------|
| train/valid index overlap | **0 rows** |
| group overlap bila GroupCV | **0 groups** |
| train timestamp > validation timestamp pada forecasting | **tidak boleh** |
| information-window overlap pada PurgedCV | **0 overlap** |
| preprocessing fitted menggunakan validation | **false** |
| feature selection memakai full dataset | **false** |
| resampling dilakukan sebelum CV | **false** |
| exact duplicates melintasi folds | **0**, kecuali benar-benar merepresentasikan deployment |
| classification fold tanpa minority class | **reject** |
| n_groups < n_splits pada GroupCV | **reject** |
| hyperparameter tuning + final score memakai folds identik tanpa nested evaluation | beri **selection-bias warning** |

---

## Referensi Utama

1. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
2. Mishra, L. N., et al. (2026). A taxonomy for detecting and preventing temporal data leakage in machine learning-based build prediction. *PLOS ONE*, 21(5), e0340167.
3. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Ch. 7 & 12.
4. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
5. Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7:91.
6. Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107.
7. Noma, H., et al. (2021). Confidence intervals of prediction accuracy measures for multivariable prediction models based on the bootstrap-based optimism correction methods. *Statistics in Medicine*, 40(26), 5691–5701.
8. Efron, B., & Tibshirani, R. (1997). Improvements on cross-validation: The .632+ bootstrap method. *JASA*, 92(438), 548–560.
9. Valavi, R., et al. (2025). blockCV: Spatial and Environmental Blocking for K-Fold and LOO Cross-Validation. CRAN.
10. Wainer, J., & Cawley, G. (2021). Nested cross-validation when selecting classifiers is overzealous for most practical applications. *Expert Systems with Applications*, 182, 115222.
11. Bates, S., Hastie, T., & Tibshirani, R. (2024). Cross-validation: what does it estimate and how well does it do it? *JASA*, 119(546), 1434–1445.
12. Qian, H., et al. (2021). Managing dataset shift by adversarial validation for credit scoring. *arXiv*.

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*