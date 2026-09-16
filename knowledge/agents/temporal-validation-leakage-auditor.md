# Temporal Validation, Leakage Taxonomy & Adversarial Validation — Dokumentasi Teknis Tier Berat (Edisi Lengkap)

> **Versi:** 2.0 | **Tier:** Berat (Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini merupakan perluasan dari versi sebelumnya dengan tambahan **dependency-aware validation, point-in-time leakage audit, pipeline post-processing leakage, distribution-shift correction, two-sample testing, streaming drift detection, forecast model comparison, dan adaptive model-selection leakage**. Setiap entri mengikuti format 20-field untuk konsumsi AI agents.

---

## Daftar Isi

**Bagian 1 — Temporal Validation & Leakage Auditor (Original)**
1. Walk-Forward / Expanding Window Cross-Validation
2. Purged K-Fold & Embargo
3. Temporal Leakage Checks (Validation Gates)
4. Feature Engineering Boundaries
5. Anti-Leakage Imputation Pipeline

**Bagian 2 — Leakage Taxonomy (Original)**
6. Kapoor & Narayanan 8-Type Taxonomy
7. Mekanistik 4-Class Taxonomy (2026)
8. Temporal Leakage Subtypes (PLOS One 2026)

**Bagian 3 — Adversarial Validation (Original)**
9. Adversarial Validation Klasik
10. Domain-Free Adversarial Splitting (DFAS)
11. Adversarial Validation untuk Credit Scoring
12. Drift Adversarials & Validation Bias

**Bagian 4 — Advanced Dependent-Data Validation (Baru)**
13. Blocked CV / h-block / hv-block CV
14. Repeated Out-of-Time Holdout
15. Combinatorial Purged Cross-Validation (CPCV) — Full Section
16. Nested Temporal Cross-Validation
17. Prequential / Test-Then-Train Evaluation
18. Group × Time Validation
19. Spatial / Spatiotemporal Block CV

**Bagian 5 — Point-in-Time Leakage Audit (Baru)**
20. Point-in-Time Correctness / As-Of Join
21. Bitemporal / Event-Time vs Ingestion-Time Audit
22. Label Maturity / Delayed-Label Validation
23. Near-Duplicate / Entity Leakage Audit
24. Resampling / SMOTE Leakage Audit
25. Augmentation Leakage Audit
26. Calibration-Split Leakage
27. Threshold-Tuning Leakage
28. OOF / Stacking Leakage Audit

**Bagian 6 — Distribution-Shift Taxonomy & Correction (Baru)**
29. Covariate-Shift IWCV
30. KLIEP / uLSIF Density-Ratio Estimation
31. Label-Shift Detection / BBSE

**Bagian 7 — Statistical Two-Sample Drift Testing (Baru)**
32. MMD Two-Sample Test
33. Classifier Two-Sample Test (C2ST)
34. KSWIN

**Bagian 8 — Streaming Change Detection (Baru)**
35. ADWIN
36. DDM / EDDM
37. Page-Hinkley
38. CUSUM
39. Bayesian Online Change-Point Detection (BOCPD)

**Bagian 9 — Forecast Model Comparison (Baru)**
40. Diebold–Mariano Test
41. Giacomini–White Test
42. Model Confidence Set (MCS)
43. White Reality Check
44. Hansen SPA Test
45. Deflated Sharpe Ratio (DSR)

**Bagian 10 — Adaptive Model-Selection Leakage (Baru)**
46. Validation Reuse Audit
47. HPO Trial Count Audit
48. Seed-Selection Audit
49. Locked Final Test Protocol

---

## BAGIAN 1 — TEMPORAL VALIDATION & LEAKAGE AUDITOR

### 1. Walk-Forward / Expanding Window Cross-Validation

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik validasi temporal di mana setiap fold memiliki batas waktu tegas: training set mencakup data sebelum batas, validation/test set mencakup data setelah batas. Chronological ordering ketat, tanpa pengacakan. Training set dapat berupa expanding (semakin besar) atau rolling (ukuran tetap) |
| **Problem yang didukung** | Time series forecasting, Temporal tabular classification/regression, Financial ML |
| **Kapan digunakan** | Data dengan urutan temporal; ada risiko temporal leakage; evaluasi out-of-time performance |
| **Kapan tidak digunakan** | Data i.i.d. tanpa dimensi waktu; dataset sangat kecil (<500 sampel) |
| **Asumsi** | Data diurutkan secara kronologis; distribusi target dapat berubah seiring waktu (concept drift); observasi dalam satu fold tidak independen |
| **Data preprocessing** | Sorting by time; pastikan tidak ada data masa depan yang masuk ke training fold |
| **Dataset scale** | Kecil–Besar (1K–10M+ baris) |
| **Metric compatibility** | Semua metric (MSE, MAE, AUC, F1, Accuracy, dll.) |
| **Hyperparameter penting** | `n_splits`, `max_train_size` (rolling vs expanding), `test_size`, `gap` |
| **Kisaran parameter awal** | n_splits: 3–10; gap: 0–30 (tergantung horizon prediksi); test_size: 10–20% data |
| **CPU/GPU/RAM** | Sama dengan training model individual × n_splits; parallelizable per fold |
| **Risiko leakage** | **Kritis** jika menggunakan k-fold standar; walk-forward secara desain mencegah leakage temporal |
| **Failure modes** | Fold terlalu sedikit → estimasi varians tinggi; gap tidak memadai → leakage dari serial correlation; data tidak cukup untuk n_splits besar |
| **Teknik pembanding** | Purged K-Fold, Combinatorial Purged CV (CPCV), TimeSeriesSplit (sklearn) |
| **Ensemble compatibility** | Base untuk validasi semua ensemble temporal; menghasilkan OOF predictions yang valid untuk stacking |
| **Exact package** | `sklearn.model_selection.TimeSeriesSplit`; `purgedcv.WalkForwardSplit`; `macrosynergy.learning.splitters.walk_forward_splitters`; `skfolio.model_selection.WalkForward` |
| **Contoh hypothesis** | "Walk-forward CV dengan 5 folds akan menghasilkan estimasi out-of-time performance 2× lebih akurat (dalam hal optimisme) dibanding k-fold standar pada data tabular temporal" |
| **Acceptance criteria** | Gap antara CV score dan out-of-time score <5%; fold-by-fold degradation signal terdeteksi |
| **Rejection criteria** | Gap >15%; tidak ada sinyal degradasi antar fold |
| **Sumber** | scikit-learn docs; purgedcv paper (Lazarev, 2026) |

**Sub-varian:**

| Varian | Deskripsi | Kapan Digunakan |
|--------|-----------|-----------------|
| **Expanding Window** | Training set tumbuh setiap fold; validation window bergeser | Dataset cukup besar; ingin model melihat semua历史 data |
| **Rolling Window** | Training set ukuran tetap; jendela bergeser | Concept drift signifikan; ingin model fokus pada data terbaru |

---

### 2. Purged K-Fold & Embargo

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik cross-validation yang dikembangkan oleh Marcos López de Prado untuk financial ML. **Purging** menghapus baris training yang label-nya overlap dengan label test (karena horizon prediksi). **Embargo** menghapus baris training tepat setelah test window untuk mencegah leakage dari serial correlation |
| **Problem yang didukung** | Financial ML, Time series dengan overlapping labels, Dataset dengan serial correlation |
| **Kapan digunakan** | Label memiliki horizon (misal "return over next 20 bars"); serial correlation signifikan; ingin honest evaluation |
| **Kapan tidak digunakan** | Label tidak overlap; data i.i.d. |
| **Asumsi** | Label horizon diketahui; serial correlation dapat di-embargo dengan parameter tetap |
| **Data preprocessing** | Mengetahui t1 (label end time) untuk setiap sampel |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `pct_embargo` (fraksi data untuk embargo), `pct_purge` (fraksi data untuk purging) |
| **Kisaran parameter awal** | pct_embargo: 0.01–0.02; pct_purge: 0.01–0.05 |
| **CPU/GPU/RAM** | Sama dengan k-fold standar + overhead purging |
| **Risiko leakage** | **Sangat Kritis** tanpa purging/embargo; scikit-learn hanya menyediakan `gap` terbatas, tanpa label-overlap purging |
| **Failure modes** | Embargo terlalu kecil → residual leakage; terlalu besar → kehilangan data training |
| **Teknik pembanding** | Walk-Forward, CPCV |
| **Ensemble compatibility** | Menghasilkan OOF predictions yang lebih honest untuk stacking |
| **Exact package** | `purgedcv.PurgedKFold`, `purgedcv.CombinatorialPurgedCV`; `timeseriescv` (legacy) |
| **Contoh hypothesis** | "Purged K-Fold dengan embargo 1% akan mengurangi gap antara CV score dan test score dari 15% menjadi <5% pada data financial dengan overlapping labels" |
| **Acceptance criteria** | Gap CV-test <5%; tidak ada label overlap antara train dan test |
| **Rejection criteria** | Gap >10%; purging/embargo tidak berfungsi |
| **Sumber** | López de Prado (2018); purgedcv (Lazarev, 2026) |

---

### 3. Temporal Leakage Checks (Validation Gates)

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
| **Hyperparameter penting** | Threshold untuk suspicious improvement (misal >20% antar fold) |
| **Kisaran parameter awal** | Improvement threshold: 0.2; target shuffle iterations: 100 |
| **CPU/GPU/RAM** | Ringan; hanya analisis statistik |
| **Risiko leakage** | **Kritis** jika gate tidak diimplementasikan |
| **Failure modes** | False positive (gate terlalu sensitif); false negative (leakage tidak terdeteksi) |
| **Teknik pembanding** | Manual inspection; model info sheets |
| **Ensemble compatibility** | Universal |
| **Exact package** | `target-leakage-detection` (R); custom implementation |
| **Contoh hypothesis** | "Validation gate dengan shuffled target test akan mendeteksi temporal leakage pada 90% kasus" |
| **Acceptance criteria** | Detection rate >80%; false positive rate <10% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | arXiv (2026); praktik industri |

**Gate-gate yang direkomendasikan:**

| Gate | Deskripsi | Cara Kerja |
|------|-----------|-----------|
| **Shuffled Target Test** | Apakah target secara tidak sengaja terhubung ke feature masa depan | Shuffle target, retrain, bandingkan performa |
| **Temporal Boundary Check** | Apakah ada data dari periode test yang masuk ke training | Verifikasi timestamp min/max per fold |
| **Suspicious Improvement Detector** | Peningkatan tiba-tiba yang tidak wajar antar fold | Threshold improvement (misal >20%) |

---

### 4. Feature Engineering Boundaries

| Field | Detail |
|-------|--------|
| **Definisi** | Semua feature temporal seperti rolling statistics harus dihitung dengan **strictly backward-looking windows**. Pembagian kronologis dilakukan terlebih dahulu, baru kemudian semua feature temporal dihitung menggunakan window yang hanya melihat ke belakang |
| **Problem yang didukung** | Time series feature engineering, Temporal tabular |
| **Kapan digunakan** | Membuat lag features, rolling statistics, expanding windows |
| **Kapan tidak digunakan** | Data i.i.d. tanpa dimensi waktu |
| **Asumsi** | Fitur temporal dapat dihitung secara strictly backward-looking |
| **Data preprocessing** | Split temporal terlebih dahulu; hitung fitur per fold |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `window_size`, `min_periods` |
| **Kisaran parameter awal** | window: 3–30 (tergantung frekuensi) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** jika window mencakup future data |
| **Failure modes** | Look-ahead bias; data leakage dari future periods |
| **Teknik pembanding** | Global feature engineering (berbahaya) |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `rolling`, `shift`; `feature_engine.timeseries` |
| **Contoh hypothesis** | "Feature engineering boundaries dengan strictly backward-looking windows akan mengurangi gap CV-test dari 20% menjadi <5%" |
| **Acceptance criteria** | Gap CV-test <5%; tidak ada future data dalam feature |
| **Rejection criteria** | Gap >10%; future data terdeteksi |
| **Sumber** | LinkedIn (2025); best practices |

---

### 5. Anti-Leakage Imputation Pipeline

| Field | Detail |
|-------|--------|
| **Definisi** | Imputasi missing values harus dilakukan dalam pipeline yang menghormati batas temporal. **Imputer FIT pada training fold**, lalu parameter tersebut **APPLY ke validation/test**. Test/validation tidak boleh menentukan parameter imputasi |
| **Problem yang didukung** | Semua problem dengan missing values temporal |
| **Kapan digunakan** | Missing values pada data temporal; imputasi diperlukan |
| **Kapan tidak digunakan** | Tidak ada missing values |
| **Asumsi** | Imputasi dapat dilakukan secara fold-isolated |
| **Data preprocessing** | Pipeline dengan fit pada train fold saja |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `imputer_type`, `n_neighbors` |
| **Kisaran parameter awal** | KNN: k=5; MICE: max_iter=10 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Tinggi** jika imputer di-fit pada full data |
| **Failure modes** | Imputation leakage → overoptimistic validation |
| **Teknik pembanding** | Simple imputation (mean/median) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.pipeline.Pipeline`; `sklearn.impute.KNNImputer` |
| **Contoh hypothesis** | "Anti-leakage imputation pipeline akan mengurangi gap CV-test dari 10% menjadi <3%" |
| **Acceptance criteria** | Gap CV-test <3% |
| **Rejection criteria** | Gap >5% |
| **Sumber** | scikit-learn docs; praktik industri |

---

## BAGIAN 2 — LEAKAGE TAXONOMY

### 6. Kapoor & Narayanan 8-Type Taxonomy

| Field | Detail |
|-------|--------|
| **Definisi** | Taksonomi 8 tipe leakage berdasarkan survei literatur lintas 17 bidang ilmu, ditemukan 294 paper yang terpengaruh leakage. Taksonomi mencakup mulai dari kesalahan textbook hingga open research problems |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Audit pipeline ML; penulisan paper; review model |
| **Kapan tidak digunakan** | N/A (selalu relevan) |
| **Asumsi** | Leakage dapat diklasifikasikan ke dalam 8 tipe |
| **Data preprocessing** | N/A |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N/A (konseptual) |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Tidak menyadari salah satu dari 8 tipe |
| **Teknik pembanding** | Model info sheets |
| **Ensemble compatibility** | Universal |
| **Exact package** | Model info sheets (template) |
| **Contoh hypothesis** | "Implementasi model info sheets akan mendeteksi leakage pada 90% kasus" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Kapoor & Narayanan, Patterns 4, 100804 (2023) |

**8 Tipe Leakage (Kapoor & Narayanan):**

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

### 7. Mekanistik 4-Class Taxonomy (2026)

| Field | Detail |
|-------|--------|
| **Definisi** | Taksonomi alternatif yang memotong dari sudut mekanisme statistik. Empat kelas: **Estimation leakage** (parameter fitting pada data penuh), **Selection leakage** (peeking dan cherry-picking), **Memorization leakage** (model capacity), **Boundary leakage** (temporal boundary) |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Audit pipeline; prioritasi perbaikan leakage |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Magnitude leakage dapat diukur secara kuantitatif |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | AUC, ΔAUC |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Berat untuk eksperimen counterfactual |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Estimation leakage dianggap remeh padahal selection leakage jauh lebih berbahaya |
| **Teknik pembanding** | Kapoor taxonomy |
| **Ensemble compatibility** | Universal |
| **Exact package** | `epagogy/ml` (GitHub) |
| **Contoh hypothesis** | "Selection leakage akan menyebabkan inflasi AUC ≥0.05 pada dataset kecil, sedangkan estimation leakage hanya ≤0.005" |
| **Acceptance criteria** | ΔAUC estimation ≤0.005; ΔAUC selection ≥0.03 |
| **Rejection criteria** | Hasil bertentangan |
| **Sumber** | "Which Leakage Types Matter?" (2026), 2,047 dataset |

**4 Kelas Mekanistik:**

| Kelas | Deskripsi | Magnitude | Kapan Paling Berbahaya |
|-------|-----------|-----------|----------------------|
| **I. Estimation** | Fitting scaler/encoder pada full data | **Negligible** (\|ΔAUC\| ≤ 0.005) | Hampir tidak pernah |
| **II. Selection** | Peeking, seed cherry-picking | **Substantial** (~90% noise exploitation) | Dataset kecil |
| **III. Memorization** | Model menghafal duplikat | **Skala dengan model capacity** (d_z: 0.37–1.11) | Ada duplikasi data |
| **IV. Boundary** | Temporal boundary leakage | **Invisible under random CV** | Data temporal |

**Catatan:** Magnitude ini adalah engineering heuristic, bukan universal statistical threshold. Sangat dataset/pipeline dependent.

---

### 8. Temporal Leakage Subtypes (PLOS One 2026)

| Field | Detail |
|-------|--------|
| **Definisi** | Tiga subtipe temporal leakage spesifik pada build prediction: **Direct Outcome Encoding**, **Indirect Encoding**, dan **Contextual Leakage** |
| **Problem yang didukung** | Build prediction, Temporal tabular, Software defect prediction |
| **Kapan digunakan** | Data temporal dengan outcome prediction |
| **Kapan tidak digunakan** | Data i.i.d. |
| **Asumsi** | Temporal leakage dapat diklasifikasikan ke dalam 3 subtipe |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC, Accuracy |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N/A |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Fluktuasi akurasi artifisial yang tidak muncul di deployment |
| **Teknik pembanding** | Kapoor taxonomy |
| **Ensemble compatibility** | Universal |
| **Exact package** | N/A |
| **Contoh hypothesis** | "Direct Outcome Encoding akan menyebabkan inflasi AUC ≥0.10 pada build prediction" |
| **Acceptance criteria** | Detection rate >70% |
| **Rejection criteria** | Detection rate <40% |
| **Sumber** | PLOS One (2026) |

**Tiga Subtipe:**

| Subtipe | Deskripsi |
|---------|-----------|
| **Direct Outcome Encoding** | Fitur secara langsung meng-encode outcome masa depan |
| **Indirect Encoding** | Fitur meng-encode outcome masa depan secara tidak langsung |
| **Contextual Leakage** | Konteks (misal metadata) mengandung informasi masa depan |

---

## BAGIAN 3 — ADVERSARIAL VALIDATION

### 9. Adversarial Validation Klasik

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik mendeteksi distribusi shift antara training set dan test/production set dengan merangkainya sebagai masalah klasifikasi biner. Training dan test digabung, diberi label biner (0=train, 1=test), lalu classifier dilatih untuk membedakan keduanya |
| **Problem yang didukung** | Semua problem dengan train/test shift; Competition ML; Production monitoring |
| **Kapan digunakan** | Sebelum training untuk deteksi covariate shift; monitoring drift di production; feature selection |
| **Kapan tidak digunakan** | Data i.i.d. sempurna; dataset sangat kecil |
| **Asumsi** | Jika dua dataset dari distribusi sama, classifier tidak dapat membedakan dengan AUC > 0.5 |
| **Data preprocessing** | Standar; scaling untuk logistic regression |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC (adversarial), feature importance |
| **Hyperparameter penting** | `classifier_type`, `cv_folds`, `threshold` |
| **Kisaran parameter awal** | LogisticRegression; cv=5; threshold AUC: 0.7 |
| **CPU/GPU/RAM** | Ringan–Sedang; tergantung classifier |
| **Risiko leakage** | Rendah; teknik ini justru mendeteksi leakage/shift |
| **Failure modes** | AUC ≈ 0.5 tetapi ada shift non-linear; feature importance tidak stabil |
| **Teknik pembanding** | Uji statistik univariat (KS test, PSI); MMD; C2ST |
| **Ensemble compatibility** | Universal |
| **Exact package** | `adval` (PyPI); `adversarial-validation` (GitHub); `endgame.validation.AdversarialValidator` |
| **Contoh hypothesis** | "Adversarial validation dengan LogisticRegression akan mendeteksi drift pada dataset dengan AUC ≥0.75" |
| **Acceptance criteria** | AUC ≥0.7 → drift terdeteksi; feature importance mengidentifikasi fitur drifting |
| **Rejection criteria** | AUC ≈0.5 tetapi performa model drop di test |
| **Sumber** | Zając (2017); praktik Kaggle; ar5iv (2021) |

**Interpretasi AUC (Engineering Heuristic, bukan Universal Statistical Threshold):**

| AUC Range | Interpretasi | Tindakan |
|-----------|-------------|----------|
| 0.50–0.55 | Tidak ada shift signifikan | Lanjutkan training |
| 0.55–0.70 | Shift ringan | Pertimbangkan importance weighting |
| 0.70–0.85 | Shift signifikan | Feature selection/removal; reweighting |
| >0.85 | Shift ekstrem | Dataset mungkin tidak kompatibel |

**Catatan:** AUC bergantung pada sample size, classifier capacity, dan jenis shift. Lebih aman digabungkan dengan permutation/null calibration atau formal C2ST/MMD.

---

### 10. Domain-Free Adversarial Splitting (DFAS)

| Field | Detail |
|-------|--------|
| **Definisi** | Ekstensi adversarial validation yang membuat split train-validation-test dengan memaksimalkan perbedaan domain antar subset, melatih model untuk generalisasi melintasi batas tersebut |
| **Problem yang didukung** | Domain generalization, Non-i.i.d. data, Distribution shift |
| **Kapan digunakan** | Training dan test data tidak dari distribusi sama; butuh model yang robust terhadap domain shift |
| **Kapan tidak digunakan** | Data i.i.d.; dataset sangat kecil |
| **Asumsi** | Domain shift dapat disimulasikan via adversarial splitting |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC, Accuracy, Worst-case performance |
| **Hyperparameter penting** | `adversarial_steps`, `meta_learning_rate` |
| **Kisaran parameter awal** | adversarial_steps: 5–10 |
| **CPU/GPU/RAM** | Berat (min-max optimization) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Min-max tidak konvergen; split terlalu ekstrem |
| **Teknik pembanding** | Adversarial validation klasik |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation |
| **Contoh hypothesis** | "DFAS akan meningkatkan worst-case accuracy ≥5% pada domain generalization benchmark" |
| **Acceptance criteria** | Worst-case accuracy improvement ≥5% |
| **Rejection criteria** | Tidak ada improvement; konvergensi gagal |
| **Sumber** | OpenReview (2021) |

---

### 11. Adversarial Validation untuk Credit Scoring

| Field | Detail |
|-------|--------|
| **Definisi** | Aplikasi adversarial validation pada domain financial tabular. Memilih titik pemisahan training-validation yang lebih dekat ke test period dapat meningkatkan konsistensi antara skor validation dan test |
| **Problem yang didukung** | Credit scoring, Financial risk modeling |
| **Kapan digunakan** | Data financial dengan temporal drift; train/test dari periode berbeda |
| **Kapan tidak digunakan** | Data financial i.i.d. |
| **Asumsi** | Temporal drift signifikan; adversarial validation dapat mendeteksi |
| **Data preprocessing** | Standar financial preprocessing |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC, KS, Gini |
| **Hyperparameter penting** | `split_point`, `classifier_type` |
| **Kisaran parameter awal** | LogisticRegression; split dekat test period |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Split terlalu dekat → training data tidak cukup |
| **Teknik pembanding** | Temporal validation klasik |
| **Ensemble compatibility** | Universal |
| **Exact package** | `adval`; custom implementation |
| **Contoh hypothesis** | "Memilih split point dengan adversarial validation akan meningkatkan konsistensi validation-test AUC ≥3%" |
| **Acceptance criteria** | Gap validation-test <2% |
| **Rejection criteria** | Gap >5% |
| **Sumber** | Qian et al. (2021); MDPI (2026) |

---

### 12. Drift Adversarials & Validation Bias

| Field | Detail |
|-------|--------|
| **Definisi** | **Drift adversarials**: data stream yang sedang drifting dapat dibangun sedemikian rupa sehingga tidak terdeteksi oleh skema deteksi drift standar. **Validation bias**: ketergantungan pada validation set untuk model selection menjadi bermasalah dalam konteks temporal shift |
| **Problem yang didukung** | Production monitoring, Deep tabular methods, Ensemble-based methods |
| **Kapan digunakan** | Monitoring production; model selection dengan temporal shift |
| **Kapan tidak digunakan** | Data i.i.d. |
| **Asumsi** | Drift adversarials menunjukkan perlunya kombinasi beberapa metode deteksi |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC, Accuracy |
| **Hyperparameter penting** | Kombinasi metode deteksi (adversarial + univariate + PSI) |
| **Kisaran parameter awal** | Multiple detectors |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Rendah |
| **Failure modes** | Drift tidak terdeteksi oleh satu metode |
| **Teknik pembanding** | Multiple drift detectors |
| **Ensemble compatibility** | Universal |
| **Exact package** | `Evidently`; `Alibi Detect`; custom |
| **Contoh hypothesis** | "Kombinasi adversarial validation + PSI + JS divergence akan mendeteksi drift pada 95% kasus" |
| **Acceptance criteria** | Detection rate >90% |
| **Rejection criteria** | Detection rate <70% |
| **Sumber** | arXiv (2024); ICML 2025 |

**Validation Bias pada Deep Tabular:**
- Model selection bergantung pada validation set
- Gap waktu training-test > gap training-validation
- Ensemble-based methods paling terdampak

**Rekomendasi:**
- Gunakan temporal validation split (TabReD)
- Minimalkan validation bias via reverse temporal direction
- Fourier-based temporal embedding untuk periodic/trend information

---

## BAGIAN 4 — ADVANCED DEPENDENT-DATA VALIDATION

### 13. Blocked CV / h-block / hv-block CV

| Field | Detail |
|-------|--------|
| **Definisi** | Blocked CV: time blocks diperlakukan sebagai unit. h-block: membuang observations dekat validation block (buffer h). hv-block: buffer di sekitar validation block untuk general stationary dependent observations |
| **Problem yang didukung** | Serially dependent observations; Time series dengan stationary dependence |
| **Kapan digunakan** | Stationary-ish dependence; butuh buffer antar train-validation; model selection untuk dependent data |
| **Kapan tidak digunakan** | Non-stationary series (gunakan repeated OOT); data i.i.d. |
| **Asumsi** | Stationary dependence; buffer menghilangkan dependence |
| **Data preprocessing** | Block construction; buffer size |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `h` (buffer size), `block_size` |
| **Kisaran parameter awal** | h: 1–5; block_size: 5–20 |
| **CPU/GPU/RAM** | Sama dengan KFold + overhead |
| **Risiko leakage** | **Tinggi** tanpa buffer |
| **Failure modes** | Buffer terlalu kecil → residual leakage; terlalu besar → kehilangan data |
| **Teknik pembanding** | Purged K-Fold, Embargo, CPCV |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `blockCV` (R); `hblock` (R) |
| **Contoh hypothesis** | "hv-block CV akan mengurangi gap CV-test dari 15% menjadi <5% pada stationary dependent data" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Racine (2000), Journal of Econometrics |

**Perbedaan:**
| Metode | Target Dependence |
|--------|-------------------|
| **hv-block** | Dependence/autocorrelation |
| **Purging** | Overlapping label intervals |
| **Embargo** | Post-test leakage/dependence buffer |

---

### 14. Repeated Out-of-Time Holdout

| Field | Detail |
|-------|--------|
| **Definisi** | Repeated holdout pada periode out-of-time untuk estimasi yang lebih realistis pada non-stationary series |
| **Problem yang didukung** | Time series forecasting; Non-stationary data |
| **Kapan digunakan** | Non-stationary + production future prediction |
| **Kapan tidak digunakan** | Stationary series (blocked CV cukup) |
| **Asumsi** | Non-stationarity; repeated OOT memberikan estimasi lebih akurat |
| **Data preprocessing** | Time-based split |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_repeats`, `holdout_size` |
| **Kisaran parameter awal** | n_repeats: 5–10 |
| **CPU/GPU/RAM** | n_repeats × training cost |
| **Risiko leakage** | Rendah jika strictly temporal |
| **Failure modes** | Holdout tidak representatif |
| **Teknik pembanding** | Blocked CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `forecast` (R) |
| **Contoh hypothesis** | "Repeated OOT akan memberikan estimasi yang lebih akurat dari blocked CV pada non-stationary series" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Cerqueira et al., ML Journal (2020) |

---

### 15. Combinatorial Purged Cross-Validation (CPCV) — Full Section

| Field | Detail |
|-------|--------|
| **Definisi** | Purged folds + many combinations of train/test folds + multiple reconstructed OOS backtest paths |
| **Problem yang didukung** | Financial ML; Overlapping labels; Backtest overfitting concern |
| **Kapan digunakan** | Finance/event prediction; overlapping labels; many strategies/models compared |
| **Kapan tidak digunakan** | Default tabular CV; dataset kecil |
| **Asumsi** | Purging + embargo diterapkan; combinatorial paths representatif |
| **Data preprocessing** | Information interval setiap observation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Sharpe ratio, PSR, Deflated Sharpe, PBO |
| **Hyperparameter penting** | `n_groups` (N), `n_test_groups` (k), `pct_embargo` |
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
| **Sumber** | López de Prado (2018), Ch. 12; purgedcv (2026) |

**Output Schema:**
```yaml
cpcv:
  n_groups: 6
  n_test_groups: 2
  n_paths: 15
  path_score_distribution: [...]
  worst_path: ...
```

---

### 16. Nested Temporal Cross-Validation

| Field | Detail |
|-------|--------|
| **Definisi** | Nested CV dengan outer temporal split untuk estimasi performa generalisasi, inner temporal split untuk HPO. Mencegah selection bias |
| **Problem yang didukung** | Hyperparameter search dengan temporal data |
| **Kapan digunakan** | HPO pada data temporal; butuh unbiased estimate |
| **Kapan tidak digunakan** | Tidak ada HPO; data i.i.d. |
| **Asumsi** | Outer temporal split untuk final estimate; inner untuk tuning |
| **Data preprocessing** | Temporal split di kedua level |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `outer_splits`, `inner_splits` |
| **Kisaran parameter awal** | Outer 5 × Inner 3–5 |
| **CPU/GPU/RAM** | Berat: outer × inner training cost |
| **Risiko leakage** | **Tinggi** jika tidak nested |
| **Failure modes** | Selection bias; variance tinggi |
| **Teknik pembanding** | Flat temporal CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection` + custom |
| **Contoh hypothesis** | "Nested temporal CV akan mengurangi bias estimasi dari 15% menjadi <2%" |
| **Acceptance criteria** | Bias reduction ≥80% |
| **Rejection criteria** | Bias tidak berkurang; runtime >10× |
| **Sumber** | Cawley & Talbot, JMLR 11 (2010) |

---

### 17. Prequential / Test-Then-Train Evaluation

| Field | Detail |
|-------|--------|
| **Definisi** | Streaming evaluation: predict row t → observe y_t → score → update model → predict t+1 |
| **Problem yang didukung** | Streaming classification/regression; Concept drift |
| **Kapan digunakan** | Online learning; model terus di-update di production |
| **Kapan tidak digunakan** | Static batch problem |
| **Asumsi** | Loss per step tersedia; model update incremental |
| **Data preprocessing** | Strict temporal order; no future leakage |
| **Dataset scale** | Streaming/unbounded |
| **Metric compatibility** | Prequential accuracy, Kappa, MSE |
| **Hyperparameter penting** | `learning_rate`, `window_size` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan; incremental |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Concept drift tidak terdeteksi |
| **Teknik pembanding** | Static walk-forward retraining |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python); custom |
| **Contoh hypothesis** | "Prequential evaluation akan memberikan estimasi yang lebih realistis dari static CV pada streaming data" |
| **Acceptance criteria** | Prequential accuracy stabil |
| **Rejection criteria** | Accuracy drop >5% |
| **Sumber** | Cerqueira et al., ML Journal (2020); concept-drift literature |

---

### 18. Group × Time Validation

| Field | Detail |
|-------|--------|
| **Definisi** | Validasi yang mempertimbangkan dua axes: entity dan time. Mencegah entity leakage sekaligus future leakage |
| **Problem yang didukung** | Tabular longitudinal; Customer data; Patient data |
| **Kapan digunakan** | Repeated entity + temporal structure |
| **Kapan tidak digunakan** | Single entity; no time |
| **Asumsi** | Deployment generalization level (unseen entity vs future of seen entity) |
| **Data preprocessing** | Group + time split |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `entity_column`, `time_column` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Tinggi** jika hanya satu axis |
| **Failure modes** | Entity leakage atau future leakage |
| **Teknik pembanding** | GroupKFold; TimeSeriesSplit |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `sklearn.model_selection` |
| **Contoh hypothesis** | "Group × Time split akan memberikan estimasi yang lebih realistis dari single-axis split pada data longitudinal" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Roberts et al., Ecography (2017) |

**Protocols:**
- Future of seen entities
- Future of unseen entities
- New-site future
- Leave-one-domain-and-time-out

---

### 19. Spatial / Spatiotemporal Block CV

| Field | Detail |
|-------|--------|
| **Definisi** | CV dengan spatial block untuk menghindari spatial autocorrelation leakage. Spatiotemporal blocking untuk data dengan koordinat + waktu |
| **Problem yang didukung** | Geospatial prediction; Spatiotemporal data |
| **Kapan digunakan** | Data dengan latitude/longitude; spatial autocorrelation |
| **Kapan tidak digunakan** | Data non-spatial |
| **Asumsi** | Spatial autocorrelation signifikan |
| **Data preprocessing** | Spatial block; buffer |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `block_size`, `buffer` |
| **Kisaran parameter awal** | block_size: variogram range |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | **Tinggi** tanpa spatial block |
| **Failure modes** | Block terlalu besar/kecil |
| **Teknik pembanding** | Random CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | `blockCV` (R); custom |
| **Contoh hypothesis** | "Spatial block CV akan mengurangi optimisme AUC dari +0.16 menjadi <0.05" |
| **Acceptance criteria** | Gap random vs spatial <0.05 |
| **Rejection criteria** | Gap >0.10 |
| **Sumber** | Valavi et al. (2025); ScienceDirect (2021) |

**Catatan:** Spatial CV tidak otomatis unbiased; evaluation protocol harus sesuai dengan estimand/deployment objective.

---

## BAGIAN 5 — POINT-IN-TIME LEAKAGE AUDIT

### 20. Point-in-Time Correctness / As-Of Join

| Field | Detail |
|-------|--------|
| **Definisi** | Memastikan fitur benar-benar tersedia pada prediction time. Feature value harus memiliki available_at ≤ prediction_time |
| **Problem yang didukung** | Production ML; Feature store |
| **Kapan digunakan** | Data dengan event time vs ingestion time; historical snapshot |
| **Kapan tidak digunakan** | Data statis |
| **Asumsi** | Feature availability dapat dilacak |
| **Data preprocessing** | As-of join; feature availability metadata |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `available_at`, `prediction_time` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** jika feature belum tersedia |
| **Failure modes** | Backfill leakage |
| **Teknik pembanding** | Simple timestamp sort |
| **Ensemble compatibility** | Universal |
| **Exact package** | Databricks Feature Store; custom |
| **Contoh hypothesis** | "Point-in-time join akan mengurangi gap CV-test dari 20% menjadi <5%" |
| **Acceptance criteria** | available_at ≤ prediction_time untuk semua fitur |
| **Rejection criteria** | Ada fitur dengan available_at > prediction_time |
| **Sumber** | Databricks Docs (2026) |

**Feature Availability Schema:**
```yaml
feature_availability:
  event_time: ...
  ingestion_time: ...
  available_at: ...
  revision_time: ...
```

---

### 21. Bitemporal / Event-Time vs Ingestion-Time Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Membedakan event time (kapan kejadian sebenarnya) dan system/ingestion time (kapan sistem mengetahui). Menangkap late-arriving/backfilled data |
| **Problem yang didukung** | Production ML; Fraud detection |
| **Kapan digunakan** | Data dengan delayed events; revision |
| **Kapan tidak digunakan** | Real-time events |
| **Asumsi** | Event time dan ingestion time tersedia |
| **Data preprocessing** | Bitemporal modeling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `event_time_col`, `ingestion_time_col` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Backfill leakage |
| **Teknik pembanding** | Point-in-time join |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "Bitemporal audit akan mendeteksi backfill leakage pada 90% kasus" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Praktik industri |

---

### 22. Label Maturity / Delayed-Label Validation

| Field | Detail |
|-------|--------|
| **Definisi** | Target belum final saat prediction dibuat. Enforce label_end_time ≤ training_cutoff |
| **Problem yang didukung** | Credit risk; Churn prediction |
| **Kapan digunakan** | Target memiliki horizon (misal "default within 90 days") |
| **Kapan tidak digunakan** | Target immediate |
| **Asumsi** | Label maturity time diketahui |
| **Data preprocessing** | Label maturity gate |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `label_horizon`, `training_cutoff` |
| **Kisaran parameter awal** | label_horizon: 30–90 hari |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Immature labels → bias |
| **Teknik pembanding** | Purged K-Fold |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "Label maturity gate akan mengurangi gap CV-test dari 15% menjadi <5%" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Praktik industri |

---

### 23. Near-Duplicate / Entity Leakage Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi near-duplicate atau entity yang sama lintas split. Lebih luas dari exact duplicate |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Data dengan entity/session/document |
| **Kapan tidak digunakan** | Data independent |
| **Asumsi** | Similarity dapat diukur |
| **Data preprocessing** | Similarity search; hashing |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `similarity_threshold` |
| **Kisaran parameter awal** | threshold: 0.85–0.95 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Tinggi** |
| **Failure modes** | Entity leakage |
| **Teknik pembanding** | Exact duplicate check |
| **Ensemble compatibility** | Universal |
| **Exact package** | `recordlinkage`; `dedupe`; custom |
| **Contoh hypothesis** | "Near-duplicate audit akan menemukan ≥50% lebih banyak duplicate dari exact match" |
| **Acceptance criteria** | Near-duplicate detection improvement ≥40% |
| **Rejection criteria** | Hanya exact duplicate |
| **Sumber** | Elmagarmid et al. (2007) |

---

### 24. Resampling / SMOTE Leakage Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Oversampling harus di dalam training fold. Synthetic observation tidak boleh dibentuk menggunakan neighbor yang nantinya masuk test |
| **Problem yang didukung** | Imbalanced classification |
| **Kapan digunakan** | SMOTE, ADASYN, RandomOverSampler |
| **Kapan tidak digunakan** | Data seimbang |
| **Asumsi** | Resampling fold-isolated |
| **Data preprocessing** | Split dulu, baru resample |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `sampling_strategy` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** jika resampling sebelum split |
| **Failure modes** | Synthetic observation di test |
| **Teknik pembanding** | Class weight |
| **Ensemble compatibility** | Universal |
| **Exact package** | `imblearn.pipeline.Pipeline` |
| **Contoh hypothesis** | "SMOTE dalam pipeline akan mengurangi gap CV-test dari 10% menjadi <3%" |
| **Acceptance criteria** | Gap CV-test <3% |
| **Rejection criteria** | Gap >5% |
| **Sumber** | imbalanced-learn docs |

---

### 25. Augmentation Leakage Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Augmented siblings tidak boleh tersebar train/test |
| **Problem yang didukung** | Image/text augmentation |
| **Kapan digunakan** | Data augmentation |
| **Kapan tidak digunakan** | No augmentation |
| **Asumsi** | Siblings dapat dilacak |
| **Data preprocessing** | Group by original |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `augmentation_group` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** |
| **Failure modes** | Augmented sibling di test |
| **Teknik pembanding** | GroupKFold |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "Augmentation group split akan mengurangi gap CV-test dari 15% menjadi <5%" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Praktik industri |

---

### 26. Calibration-Split Leakage

| Field | Detail |
|-------|--------|
| **Definisi** | Calibrator tidak boleh fit ke final test. Harus: train → fit model; calibration split/OOF → fit calibrator; test → evaluate once |
| **Problem yang didukung** | Probabilistic classification |
| **Kapan digunakan** | Probability calibration |
| **Kapan tidak digunakan** | No calibration |
| **Asumsi** | Calibration split terpisah |
| **Data preprocessing** | Calibration split |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Brier, log-loss, ECE |
| **Hyperparameter penting** | `calibration_method` |
| **Kisaran parameter awal** | sigmoid/isotonic |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika calibrator fit ke test |
| **Failure modes** | Calibration leakage |
| **Teknik pembanding** | No calibration |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.calibration.CalibratedClassifierCV` |
| **Contoh hypothesis** | "Calibration split yang benar akan menghasilkan ECE yang lebih honest" |
| **Acceptance criteria** | ECE <0.10 |
| **Rejection criteria** | Calibrator fit ke test |
| **Sumber** | scikit-learn docs |

---

### 27. Threshold-Tuning Leakage

| Field | Detail |
|-------|--------|
| **Definisi** | Classification threshold harus tune di validation/OOF, bukan test |
| **Problem yang didukung** | Classification dengan F1/cost-sensitive |
| **Kapan digunakan** | Threshold selection |
| **Kapan tidak digunakan** | No threshold |
| **Asumsi** | Threshold source tracked |
| **Data preprocessing** | OOF threshold |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | F1, cost |
| **Hyperparameter penting** | `optimized_metric` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika threshold tune di test |
| **Failure modes** | Threshold leakage |
| **Teknik pembanding** | Default threshold |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "OOF threshold akan memberikan estimasi F1 yang lebih honest" |
| **Acceptance criteria** | Threshold source = OOF |
| **Rejection criteria** | Threshold source = test |
| **Sumber** | Praktik industri |

**Threshold Provenance Schema:**
```yaml
decision_threshold:
  source: oof
  optimized_metric: f1
```

---

### 28. OOF / Stacking Leakage Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Meta-model hanya boleh melihat genuine OOF predictions, bukan in-sample predictions |
| **Problem yang didukung** | Stacking; Ensemble |
| **Kapan digunakan** | Meta-learning |
| **Kapan tidak digunakan** | No stacking |
| **Asumsi** | OOF vector lengkap |
| **Data preprocessing** | K-fold OOF |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `cv_folds` |
| **Kisaran parameter awal** | cv=5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Sangat Tinggi** jika in-sample |
| **Failure modes** | Meta-feature leakage |
| **Teknik pembanding** | Blending |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.ensemble.StackingClassifier` |
| **Contoh hypothesis** | "OOF stacking akan memberikan estimasi yang lebih honest dari in-sample stacking" |
| **Acceptance criteria** | Meta-model tidak melihat in-sample |
| **Rejection criteria** | In-sample predictions digunakan |
| **Sumber** | Wolpert (1992) |

---

## BAGIAN 6 — DISTRIBUTION-SHIFT TAXONOMY & CORRECTION

### 29. Covariate-Shift IWCV

| Field | Detail |
|-------|--------|
| **Definisi** | Importance-Weighted Cross Validation (IWCV): ordinary CV tidak unbiased under covariate shift. Weight w(x) = p_test(x)/p_train(x), validation loss dikoreksi |
| **Problem yang didukung** | Covariate shift; Domain adaptation |
| **Kapan digunakan** | P_train(X) ≠ P_test(X) tetapi P(Y\|X) sama |
| **Kapan tidak digunakan** | No covariate shift |
| **Asumsi** | Covariate shift; density ratio dapat diestimasi |
| **Data preprocessing** | Density ratio estimation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `density_ratio_method` |
| **Kisaran parameter awal** | KLIEP; uLSIF |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Density ratio tidak akurat |
| **Teknik pembanding** | Adversarial validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `densratio` (R) |
| **Contoh hypothesis** | "IWCV akan memberikan estimasi yang lebih akurat dari ordinary CV under covariate shift" |
| **Acceptance criteria** | Gap CV-test <5% |
| **Rejection criteria** | Gap >10% |
| **Sumber** | Sugiyama et al., JMLR 8 (2007) |

---

### 30. KLIEP / uLSIF Density-Ratio Estimation

| Field | Detail |
|-------|--------|
| **Definisi** | Estimasi density ratio langsung tanpa full density estimation. KLIEP: KL divergence; uLSIF: unconstrained least-squares |
| **Problem yang didukung** | Covariate shift correction |
| **Kapan digunakan** | Butuh density ratio |
| **Kapan tidak digunakan** | No covariate shift |
| **Asumsi** | Density ratio dapat diestimasi |
| **Data preprocessing** | Kernel density ratio |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `kernel_width`, `regularization` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Density ratio tidak stabil |
| **Teknik pembanding** | Classifier-odds |
| **Ensemble compatibility** | Universal |
| **Exact package** | `densratio` (R); custom |
| **Contoh hypothesis** | "uLSIF akan memberikan density ratio yang lebih stabil dari KLIEP" |
| **Acceptance criteria** | Density ratio stabil |
| **Rejection criteria** | Tidak stabil |
| **Sumber** | Sugiyama et al. (2012) |

---

### 31. Label-Shift Detection / BBSE

| Field | Detail |
|-------|--------|
| **Definisi** | Black Box Shift Estimation (BBSE): estimasi target-label proportions menggunakan confusion matrix dari classifier |
| **Problem yang didukung** | Label shift; Fraud detection |
| **Kapan digunakan** | P(Y) changes tetapi P(X\|Y) tetap |
| **Kapan tidak digunakan** | No label shift |
| **Asumsi** | Label shift; classifier confusion matrix |
| **Data preprocessing** | Confusion matrix |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `classifier_type` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Confusion matrix tidak akurat |
| **Teknik pembanding** | Covariate shift correction |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `label_shift` (Python) |
| **Contoh hypothesis** | "BBSE akan mendeteksi label shift pada data fraud dengan prevalence berubah dari 5% ke 1%" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Lipton et al., ICML 2018 |

---

## BAGIAN 7 — STATISTICAL TWO-SAMPLE DRIFT TESTING

### 32. MMD Two-Sample Test

| Field | Detail |
|-------|--------|
| **Definisi** | Maximum Mean Discrepancy (MMD) mengukur perbedaan expectations dalam RKHS. Kernel two-sample test untuk H₀: P=Q |
| **Problem yang didukung** | Multivariate distribution shift |
| **Kapan digunakan** | Formal multivariate shift test |
| **Kapan tidak digunakan** | Univariate shift |
| **Asumsi** | Kernel; exchangeability |
| **Data preprocessing** | Kernel computation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | MMD statistic; p-value |
| **Hyperparameter penting** | `kernel`, `bandwidth` |
| **Kisaran parameter awal** | RBF kernel; median heuristic |
| **CPU/GPU/RAM** | O(n²) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Kernel misspecification |
| **Teknik pembanding** | C2ST, KS, Wasserstein |
| **Ensemble compatibility** | Universal |
| **Exact package** | `alibi-detect`; custom |
| **Contoh hypothesis** | "MMD akan mendeteksi multivariate shift yang tidak terdeteksi oleh PSI univariate" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Gretton et al., JMLR 13 (2012) |

---

### 33. Classifier Two-Sample Test (C2ST)

| Field | Detail |
|-------|--------|
| **Definisi** | Formal framing dari adversarial validation sebagai two-sample test. Di bawah H₀: P=Q, classifier harus mendekati chance performance |
| **Problem yang didukung** | Multivariate distribution shift |
| **Kapan digunakan** | Formal shift test |
| **Kapan tidak digunakan** | Univariate shift |
| **Asumsi** | Classifier dapat membedakan jika P≠Q |
| **Data preprocessing** | Train/production labeling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC; accuracy |
| **Hyperparameter penting** | `classifier_type`, `cv_folds` |
| **Kisaran parameter awal** | LogisticRegression; cv=5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Classifier overfit |
| **Teknik pembanding** | MMD, adversarial validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `c2st` (Python) |
| **Contoh hypothesis** | "C2ST akan memberikan p-value yang valid untuk H₀: P=Q" |
| **Acceptance criteria** | Type-1 error rate nominal |
| **Rejection criteria** | Type-1 error rate >10% |
| **Sumber** | Lopez-Paz & Oquab, ICLR 2017 |

---

### 34. KSWIN

| Field | Detail |
|-------|--------|
| **Definisi** | Kolmogorov–Smirnov Windowing: bandingkan empirical distribution reference window dengan recent window memakai KS statistic |
| **Problem yang didukung** | Streaming drift detection |
| **Kapan digunakan** | Online concept drift |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Window-based distribution comparison |
| **Data preprocessing** | Window size |
| **Dataset scale** | Streaming |
| **Metric compatibility** | KS statistic |
| **Hyperparameter penting** | `window_size`, `stat_size` |
| **Kisaran parameter awal** | window_size: 100 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Window size salah |
| **Teknik pembanding** | ADWIN, DDM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python) |
| **Contoh hypothesis** | "KSWIN akan mendeteksi concept drift pada streaming data" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | IEEE Xplore (2024) |

---

## BAGIAN 8 — STREAMING CHANGE DETECTION

### 35. ADWIN

| Field | Detail |
|-------|--------|
| **Definisi** | Adaptive Windowing: adaptive sliding window yang ukurannya berubah berdasarkan detected distribution changes. False-positive/false-negative bounds |
| **Problem yang didukung** | Streaming drift detection |
| **Kapan digunakan** | Online concept drift |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Distribution change dapat dideteksi |
| **Data preprocessing** | Window adaptation |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Detection rate; false positive rate |
| **Hyperparameter penting** | `delta` (confidence) |
| **Kisaran parameter awal** | delta: 0.002 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Delta salah |
| **Teknik pembanding** | DDM, Page-Hinkley |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python); `ADWIN` (R) |
| **Contoh hypothesis** | "ADWIN akan mendeteksi concept drift dengan false positive rate <5%" |
| **Acceptance criteria** | False positive rate <5% |
| **Rejection criteria** | False positive rate >10% |
| **Sumber** | Bifet & Gavaldà, SDM 2007 |

---

### 36. DDM / EDDM

| Field | Detail |
|-------|--------|
| **Definisi** | Drift Detection Method (DDM): monitoring error rate. Early DDM (EDDM): mempertimbangkan distance between classification errors |
| **Problem yang didukung** | Streaming classification |
| **Kapan digunakan** | Online concept drift |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Error rate monitoring |
| **Data preprocessing** | Sequential error tracking |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Detection rate |
| **Hyperparameter penting** | `warning_level`, `drift_level` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Threshold salah |
| **Teknik pembanding** | ADWIN, Page-Hinkley |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python) |
| **Contoh hypothesis** | "DDM akan mendeteksi concept drift pada streaming classification" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Gama et al. (2004) |

---

### 37. Page-Hinkley

| Field | Detail |
|-------|--------|
| **Definisi** | Sequential analysis test untuk mendeteksi perubahan dalam mean |
| **Problem yang didukung** | Streaming drift detection |
| **Kapan digunakan** | Online concept drift |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Mean shift |
| **Data preprocessing** | Sequential mean tracking |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Detection rate |
| **Hyperparameter penting** | `threshold`, `alpha` |
| **Kisaran parameter awal** | threshold: 50; alpha: 0.9999 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Threshold salah |
| **Teknik pembanding** | ADWIN, CUSUM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python) |
| **Contoh hypothesis** | "Page-Hinkley akan mendeteksi mean shift pada streaming data" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Page (1954) |

---

### 38. CUSUM

| Field | Detail |
|-------|--------|
| **Definisi** | Cumulative Sum control chart untuk mendeteksi perubahan dalam mean |
| **Problem yang didukung** | Streaming drift detection |
| **Kapan digunakan** | Online concept drift |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Mean shift |
| **Data preprocessing** | Cumulative sum |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Detection rate |
| **Hyperparameter penting** | `threshold`, `drift` |
| **Kisaran parameter awal** | threshold: 5; drift: 0.005 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Threshold salah |
| **Teknik pembanding** | Page-Hinkley |
| **Ensemble compatibility** | Universal |
| **Exact package** | `River` (Python) |
| **Contoh hypothesis** | "CUSUM akan mendeteksi mean shift pada streaming data" |
| **Acceptance criteria** | Detection rate >80% |
| **Rejection criteria** | Detection rate <50% |
| **Sumber** | Page (1954) |

---

### 39. Bayesian Online Change-Point Detection (BOCPD)

| Field | Detail |
|-------|--------|
| **Definisi** | Bayesian online inference terhadap most recent changepoint melalui recursive message passing. Output distribution terhadap current regime length |
| **Problem yang didukung** | Streaming drift detection |
| **Kapan digunakan** | Online regime change |
| **Kapan tidak digunakan** | Batch data |
| **Asumsi** | Bayesian model; changepoint prior |
| **Data preprocessing** | Sequential inference |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Changepoint probability |
| **Hyperparameter penting** | `hazard_rate`, `prior` |
| **Kisaran parameter awal** | hazard_rate: 1/100 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Future outcomes used before prediction |
| **Failure modes** | Prior salah |
| **Teknik pembanding** | ADWIN, Page-Hinkley |
| **Ensemble compatibility** | Universal |
| **Exact package** | `bayesian_changepoint_detection` (Python) |
| **Contoh hypothesis** | "BOCPD akan memberikan posterior probability regime change" |
| **Acceptance criteria** | Posterior probability well-calibrated |
| **Rejection criteria** | Posterior tidak stabil |
| **Sumber** | Adams & MacKay, arXiv:0710.3742 |

---

## BAGIAN 9 — FORECAST MODEL COMPARISON

### 40. Diebold–Mariano Test

| Field | Detail |
|-------|--------|
| **Definisi** | Test untuk membandingkan predictive accuracy dua forecast. Loss differential d_t = L(e_A,t) - L(e_B,t); H₀: E[d_t] = 0 |
| **Problem yang didukung** | Forecast comparison |
| **Kapan digunakan** | Compare two forecasts |
| **Kapan tidak digunakan** | Single forecast |
| **Asumsi** | Loss function; serial correlation accommodated |
| **Data preprocessing** | Forecast errors |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | MSE, MAE, dll. |
| **Hyperparameter penting** | `loss_function`, `h` (horizon) |
| **Kisaran parameter awal** | h: 1–30 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Serial correlation tidak di-handle |
| **Teknik pembanding** | Giacomini-White |
| **Ensemble compatibility** | Universal |
| **Exact package** | `forecast` (R); `statsmodels` |
| **Contoh hypothesis** | "DM test akan menunjukkan model B lebih akurat dari A (p<0.05)" |
| **Acceptance criteria** | p-value valid |
| **Rejection criteria** | p-value tidak valid |
| **Sumber** | Diebold & Mariano (1995) |

---

### 41. Giacomini–White Test

| Field | Detail |
|-------|--------|
| **Definisi** | Conditional predictive ability testing: pada kondisi informasi apa model A atau B lebih baik |
| **Problem yang didukung** | Forecast comparison |
| **Kapan digunakan** | Conditional forecast comparison |
| **Kapan tidak digunakan** | Unconditional comparison |
| **Asumsi** | Model misspecification |
| **Data preprocessing** | Forecast errors; conditioning info |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `conditioning_set` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Conditioning set salah |
| **Teknik pembanding** | Diebold-Mariano |
| **Ensemble compatibility** | Universal |
| **Exact package** | `forecast` (R) |
| **Contoh hypothesis** | "GW test akan menunjukkan model A lebih baik pada periode stabil, B pada periode volatil" |
| **Acceptance criteria** | Conditional performance terdeteksi |
| **Rejection criteria** | Tidak terdeteksi |
| **Sumber** | Giacomini & White, Econometrica (2006) |

---

### 42. Model Confidence Set (MCS)

| Field | Detail |
|-------|--------|
| **Definisi** | Menghasilkan subset model yang belum bisa dibedakan statistically dari best performers pada confidence level tertentu |
| **Problem yang didukung** | Multiple model comparison |
| **Kapan digunakan** | Many candidate models |
| **Kapan tidak digunakan** | Few models |
| **Asumsi** | Multiple testing framework |
| **Data preprocessing** | Model losses |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alpha`, `bootstrap_iterations` |
| **Kisaran parameter awal** | alpha: 0.05; B: 1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Bootstrap tidak stabil |
| **Teknik pembanding** | 1-vs-1 test |
| **Ensemble compatibility** | Universal |
| **Exact package** | `MCS` (R); `arch` (Python) |
| **Contoh hypothesis** | "MCS akan menghasilkan subset model yang statistically indistinguishable" |
| **Acceptance criteria** | Subset stabil |
| **Rejection criteria** | Subset tidak stabil |
| **Sumber** | Hansen et al., Econometrica (2011) |

**Output Schema:**
```yaml
model_confidence_set:
  alpha: 0.05
  retained:
    - catboost_17
    - lgbm_4
    - tabm_2
```

---

### 43. White Reality Check

| Field | Detail |
|-------|--------|
| **Definisi** | Koreksi efek data snooping saat banyak trading rules/models diuji dari satu universe kandidat |
| **Problem yang didukung** | Financial ML; Data snooping |
| **Kapan digunakan** | Banyak backtest |
| **Kapan tidak digunakan** | Few tests |
| **Asumsi** | Bootstrap; null distribution |
| **Data preprocessing** | Model losses |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `bootstrap_iterations` |
| **Kisaran parameter awal** | B: 1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Bootstrap tidak stabil |
| **Teknik pembanding** | Hansen SPA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `bootstrap` (R) |
| **Contoh hypothesis** | "Reality Check akan mengoreksi data snooping pada 10,000 backtest" |
| **Acceptance criteria** | p-value valid |
| **Rejection criteria** | p-value tidak valid |
| **Sumber** | Sullivan et al., Journal of Finance (1999) |

---

### 44. Hansen SPA Test

| Field | Detail |
|-------|--------|
| **Definisi** | Superior Predictive Ability (SPA) test yang lebih powerful dan kurang dipengaruhi alternative models yang buruk/irrelevant dibanding Reality Check |
| **Problem yang didukung** | Financial ML; Data snooping |
| **Kapan digunakan** | Banyak backtest |
| **Kapan tidak digunakan** | Few tests |
| **Asumsi** | Bootstrap; null distribution |
| **Data preprocessing** | Model losses |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `bootstrap_iterations` |
| **Kisaran parameter awal** | B: 1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Bootstrap tidak stabil |
| **Teknik pembanding** | White Reality Check |
| **Ensemble compatibility** | Universal |
| **Exact package** | `bootstrap` (R) |
| **Contoh hypothesis** | "Hansen SPA akan lebih powerful dari Reality Check pada data financial" |
| **Acceptance criteria** | p-value valid |
| **Rejection criteria** | p-value tidak valid |
| **Sumber** | Hansen, JBES (2005) |

---

### 45. Deflated Sharpe Ratio (DSR)

| Field | Detail |
|-------|--------|
| **Definisi** | Mengoreksi observed Sharpe untuk multiple testing/selection bias dan non-normal returns |
| **Problem yang didukung** | Financial ML; Backtest overfitting |
| **Kapan digunakan** | Banyak backtest |
| **Kapan tidak digunakan** | Single strategy |
| **Asumsi** | Sharpe ratio; multiple testing |
| **Data preprocessing** | Backtest results |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Sharpe ratio |
| **Hyperparameter penting** | `n_trials` |
| **Kisaran parameter awal** | n_trials: 100–10,000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | n_trials salah |
| **Teknik pembanding** | Sharpe ratio |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `DeflatedSharpeRatio` (R) |
| **Contoh hypothesis** | "DSR akan mengoreksi Sharpe dari 10,000 backtest" |
| **Acceptance criteria** | DSR < observed Sharpe |
| **Rejection criteria** | DSR = observed Sharpe |
| **Sumber** | Bailey & López de Prado (2014) |

---

## BAGIAN 10 — ADAPTIVE MODEL-SELECTION LEAKAGE

### 46. Validation Reuse Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Log berapa kali validation set digunakan. Semakin banyak, semakin tinggi risiko selection bias |
| **Problem yang didukung** | Adaptive data analysis |
| **Kapan digunakan** | Banyak eksperimen |
| **Kapan tidak digunakan** | Single experiment |
| **Asumsi** | Reuse dapat dilacak |
| **Data preprocessing** | Logging |
| **Dataset scale** | N/A |
| **Metric compatibility** | Risk level |
| **Hyperparameter penting** | `experiments_seen`, `selections_made` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika reuse tinggi |
| **Failure modes** | Selection bias |
| **Teknik pembanding** | Locked test |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "Validation reuse audit akan mendeteksi risiko selection bias pada 1000+ eksperimen" |
| **Acceptance criteria** | Risk level dilaporkan |
| **Rejection criteria** | Reuse tidak dilacak |
| **Sumber** | Dwork et al., Science (2015) |

**Output Schema:**
```yaml
validation_reuse:
  experiments_seen: 2381
  selections_made: 74
  risk: high
```

---

### 47. HPO Trial Count Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Log berapa banyak HPO trials dijalankan. Semakin banyak, semakin tinggi risiko overfitting ke validation |
| **Problem yang didukung** | Hyperparameter optimization |
| **Kapan digunakan** | Bayesian/Optuna/random search |
| **Kapan tidak digunakan** | Grid search deterministic |
| **Asumsi** | Trial count dapat dilacak |
| **Data preprocessing** | Logging |
| **Dataset scale** | N/A |
| **Metric compatibility** | Risk level |
| **Hyperparameter penting** | `n_trials` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** |
| **Failure modes** | Selection bias |
| **Teknik pembanding** | Nested CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | Optuna; custom |
| **Contoh hypothesis** | "HPO trial count audit akan mendeteksi risiko overfitting pada 500+ trials" |
| **Acceptance criteria** | Risk level dilaporkan |
| **Rejection criteria** | Trial count tidak dilacak |
| **Sumber** | Cawley & Talbot, JMLR (2010) |

---

### 48. Seed-Selection Audit

| Field | Detail |
|-------|--------|
| **Definisi** | Log berapa banyak seed dijalankan dan dipilih. Cherry-picking seed dapat menyebabkan optimisme |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Multi-seed experiments |
| **Kapan tidak digunakan** | Single seed |
| **Asumsi** | Seed selection dapat dilacak |
| **Data preprocessing** | Logging |
| **Dataset scale** | N/A |
| **Metric compatibility** | Risk level |
| **Hyperparameter penting** | `n_seeds` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** |
| **Failure modes** | Cherry-picking |
| **Teknik pembanding** | Paired seed comparison |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom |
| **Contoh hypothesis** | "Seed-selection audit akan mendeteksi cherry-picking pada 10+ seeds" |
| **Acceptance criteria** | Risk level dilaporkan |
| **Rejection criteria** | Seed selection tidak dilacak |
| **Sumber** | Praktik industri |

---

### 49. Locked Final Test Protocol

| Field | Detail |
|-------|--------|
| **Definisi** | Final untouched test set yang tidak boleh digunakan sampai semua keputusan selesai |
| **Problem yang didukung** | Semua problem ML |
| **Kapan digunakan** | Final evaluation |
| **Kapan tidak digunakan** | Development |
| **Asumsi** | Test set benar-benar terpisah |
| **Data preprocessing** | Locked |
| **Dataset scale** | N/A |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** jika test digunakan berulang |
| **Failure modes** | Test leakage |
| **Teknik pembanding** | Internal CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | N/A (protocol) |
| **Contoh hypothesis** | "Locked final test akan memberikan estimasi yang unbiased" |
| **Acceptance criteria** | Test hanya digunakan sekali |
| **Rejection criteria** | Test digunakan berulang |
| **Sumber** | Best practices |

**Escalation:**
```
internal CV
→ locked validation
→ final untouched test
```

---

## Ringkasan Insight

| Topik | Teknik Kunci | Biaya | Kapan Wajib |
|-------|-------------|-------|-------------|
| **Temporal Validation** | Walk-forward, Purged K-Fold, CPCV | Sedang | Data temporal |
| **Dependent-Data Validation** | hv-block, Blocked CV, Group × Time | Sedang | Serially dependent |
| **Leakage Taxonomy** | Kapoor 8-type, Mekanistik 4-class | Konseptual | Audit pipeline |
| **Point-in-Time Audit** | As-Of Join, Label Maturity | Ringan | Production ML |
| **Pipeline Leakage** | SMOTE, Calibration, Threshold, OOF | Ringan | Semua pipeline |
| **Distribution Shift** | IWCV, KLIEP, BBSE | Sedang | Covariate/label shift |
| **Two-Sample Testing** | MMD, C2ST, KSWIN | Sedang | Drift detection |
| **Streaming Drift** | ADWIN, DDM, Page-Hinkley, BOCPD | Ringan | Streaming |
| **Forecast Comparison** | DM, GW, MCS, SPA, DSR | Ringan | Multiple models |
| **Adaptive Leakage** | Reuse audit, Locked test | Ringan | Banyak eksperimen |

**Insight Kritis:**
1. **Selection leakage** (peeking, cherry-picking) jauh lebih berbahaya dari **estimation leakage** (scaling on full data)
2. **Purged K-Fold + Embargo** adalah standar de-facto untuk financial ML
3. **CPCV** memberikan multiple OOS paths, bukan cuma satu backtest
4. **Point-in-time correctness** adalah leakage paling licik di production ML
5. **Distribution shift** bukan hanya dideteksi, tetapi dapat dikoreksi (IWCV, BBSE)
6. **Streaming drift** memerlukan detector khusus (ADWIN, BOCPD)
7. **Forecast comparison** memerlukan test formal (DM, MCS, SPA)
8. **Adaptive validation reuse** menyebabkan selection bias yang terakumulasi

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*
