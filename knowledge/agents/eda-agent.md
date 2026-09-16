# EDA Agent — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk teknik-teknik Exploratory Data Analysis (EDA) pada data tabular. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

**Lapisan 1 — Fondasi EDA**

1. Schema & Semantic Type Inference
2. Cardinality Analysis
3. Missing Value Pattern Analysis
4. Distribution Diagnostics
5. Outlier Detection

**Lapisan 2 — Kualitas & Relasi**

6. Duplicate & Near-Duplicate Detection
7. Feature Association & Redundancy
8. Multicollinearity Diagnostics
9. Feature–Target Association
10. Interaction Discovery
11. Leakage Detection
12. Target & Class Imbalance
13. Zero-Inflation & Overdispersion

**Lapisan 3 — Temporal & Shift**

14. Temporal Structure Diagnostics
15. Distribution / Dataset Shift Detection

**Lapisan 4 — Integritas Data**

16. Functional Dependency Discovery
17. Range & Logical Constraint Discovery
18. Mixed-Unit & Formatting Anomalies

**Lapisan 5 — Very Advanced**

19. Benford's Law & Digit Analysis
20. Copula Dependence & Tail Dependence
21. Extreme Value & Tail-Index Diagnostics
22. Change-Point Detection
23. Intrinsic Dimensionality & Clustering Tendency

---

## 1. SCHEMA & SEMANTIC TYPE INFERENCE

| Field | Detail |
|-------|--------|
| **Definisi** | Proses menginferensi tipe semantik kolom (ID, date, currency, postcode, latitude, dll.) yang terpisah dari physical dtype (int64, float64, object). Dtype pandas tidak cukup — `customer_id = int64` bukan numeric measurement, `postcode = int64` adalah semantic categorical/location code |
| **Problem yang didukung** | Semua problem tabular; automated cleaning; data discovery; downstream feature engineering |
| **Kapan digunakan** | Selalu; sebelum analisis apapun; input untuk semua modul EDA lainnya |
| **Kapan tidak digunakan** | Data sudah memiliki schema yang terdokumentasi dengan baik; dataset sangat kecil dengan tipe obvious |
| **Asumsi** | Tipe semantik dapat diinferensi dari kombinasi statistical, character-level, dan semantic features; column name informatif |
| **Data preprocessing** | Sampling values per kolom; ekstraksi character-level features; normalisasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | F1-score (support-weighted); akurasi per tipe |
| **Hyperparameter penting** | `model_type` (DL vs rule-based); `n_samples` per kolom; `feature_set` |
| **Kisaran parameter awal** | n_samples: 100–1000 per kolom; F1 target: 0.89 (Sherlock benchmark)  |
| **CPU/GPU/RAM** | CPU untuk rule-based; GPU untuk DL-based (Sherlock); RAM moderate |
| **Risiko leakage** | Rendah; inference murni pada data kolom, tanpa target |
| **Failure modes** | Mixed semantic types dalam satu kolom; column name misleading; tipe jarang (e.g., genetic variant) |
| **Teknik pembanding** | Rule-based inference; profile-based typing; SATO (semantic type detection) |
| **Ensemble compatibility** | Universal; output digunakan oleh semua modul EDA lain |
| **Exact package** | `Sherlock` (MIT Media Lab); `sato` (PyPI); custom rule-based |
| **Contoh hypothesis** | "Sherlock akan mencapai F1 ≥0.85 untuk deteksi semantic type pada dataset dengan mixed numeric/categorical/date columns" |
| **Acceptance criteria** | F1 ≥0.85; coverage ≥90% kolom terklasifikasi dengan benar |
| **Rejection criteria** | F1 <0.70; tidak dapat membedakan ID vs numeric measurement |
| **Sumber** | Hulsebos et al., Sherlock: A Deep Learning Approach to Semantic Data Type Detection (KDD 2019); MIT Visualization Group  |

---

## 2. CARDINALITY ANALYSIS

| Field | Detail |
|-------|--------|
| **Definisi** | Analisis jumlah unique values (K) per kolom kategorikal, unique ratio (K/N), distribusi frekuensi, rare-category mass, entropy, dominance ratio, singleton count, dan deteksi dirty categorical (typo, variasi morfologis) |
| **Problem yang didukung** | Semua problem tabular dengan kolom kategorikal |
| **Kapan digunakan** | Setiap kali ada kolom kategorikal; untuk memutuskan encoding strategy |
| **Kapan tidak digunakan** | Dataset tanpa kolom kategorikal |
| **Asumsi** | Cardinality absolut dan unique ratio bersama-sama informatif; dirty categories dapat dideteksi via normalisasi string |
| **Data preprocessing** | Hitung K, unique ratio, frequency table, entropy, singleton ratio, dominance ratio; normalisasi case/whitespace/Unicode untuk dirty detection |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | N/A (diagnostic) |
| **Hyperparameter penting** | `cardinality_thresholds`: constant=1, binary=2, low≤10, moderate≤25, high≤125, very_high>125; `unique_ratio_warning`=0.50; `unique_ratio_strong_warning`=0.90  |
| **Kisaran parameter awal** | 10/25/125 (operational heuristic, bukan statistical law)  |
| **CPU/GPU/RAM** | Sangat ringan; O(n) per kolom |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Menganggap cardinality threshold sebagai statistical law; auto-drop ID tanpa evaluasi prediktif; gagal mendeteksi dirty categories yang meningkatkan cardinality artifisial |
| **Teknik pembanding** | Frequency encoding; target encoding; similarity encoding |
| **Ensemble compatibility** | Universal; output untuk feature engineering agent |
| **Exact package** | `pandas.nunique()`; `category_encoders`; `feature_engine`; custom dirty detection |
| **Contoh hypothesis** | "Normalisasi case/whitespace akan mengurangi cardinality artifisial ≥15% pada kolom city dengan dirty categories" |
| **Acceptance criteria** | Cardinality reduction ≥10%; dirty category detection rate ≥70% |
| **Rejection criteria** | Tidak ada reduksi; normalisasi menghilangkan kategori valid |
| **Sumber** | Pargent et al., Computational Statistics 38, 1–33 (2023)  ; Micci-Barreca (2001)  ; Cerda et al., Machine Learning 107, 1477–1494 (2018)  |

---

## 3. MISSING VALUE PATTERN ANALYSIS

| Field | Detail |
|-------|--------|
| **Definisi** | Analisis komprehensif terhadap pola missing values, termasuk missing ratio per kolom/baris, complete-case ratio, missing pattern frequency, missingness co-occurrence, monotone vs arbitrary missingness, Little's MCAR test, dan missingness predictability test |
| **Problem yang didukung** | Semua problem tabular dengan missing values |
| **Kapan digunakan** | Ada missing values; untuk memutuskan imputation strategy dan memahami mekanisme missingness |
| **Kapan tidak digunakan** | Tidak ada missing values; missing rate <1% |
| **Asumsi** | Rubin's framework: MCAR, MAR, MNAR; missingness mechanism dapat didiagnosis (bukan dibuktikan) dari observed data |
| **Data preprocessing** | Hitung missing ratio per kolom/baris; missing pattern matrix; co-missingness (Phi, Jaccard); Little's test; missingness classifier |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | N/A (diagnostic); AUC untuk missingness predictability |
| **Hyperparameter penting** | `little_mcar_test` (conditional); `missing_indicator_models` (true); `alpha`=0.05 |
| **Kisaran parameter awal** | Little's test alpha=0.05; classifier AUC threshold untuk evidence against MCAR |
| **CPU/GPU/RAM** | Ringan; Little's test O(n × patterns) |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Mengklaim "MAR confirmed" atau "MNAR confirmed" hanya dari observed data; menganggap p>0.05 sebagai "proven MCAR"; mengabaikan missingness co-occurrence |
| **Teknik pembanding** | Single-column missing ratio; complete-case analysis |
| **Ensemble compatibility** | Universal; output untuk imputation strategy selection |
| **Exact package** | `pandas.isna()`; `missingno`; `naniar` (R); `mice` (R); custom Little's test |
| **Contoh hypothesis** | "Missingness predictability test dengan AUC ≥0.7 akan mengidentifikasi evidence against MCAR pada ≥30% kolom" |
| **Acceptance criteria** | Evidence against MCAR terdeteksi; pola missingness terkarakterisasi |
| **Rejection criteria** | Mengklaim MCAR proven tanpa test; mengabaikan co-missingness |
| **Sumber** | Rubin, Biometrika 63(3), 581–592 (1976)  ; Little, JASA 83(404), 1198–1202 (1988)  ; Seaman et al., Statistical Science 28(2), 257–268 (2013)  ; Mohan & Pearl, PMLR 33 (2014)  |

---

## 4. DISTRIBUTION DIAGNOSTICS

| Field | Detail |
|-------|--------|
| **Definisi** | Workflow berlapis: descriptive statistics → graphical diagnostics → formal tests → model relevance. Mencakup skewness, kurtosis, normality tests (Shapiro-Wilk, Anderson-Darling, Lilliefors, D'Agostino-Pearson, Jarque-Bera), dan multimodality tests (Hartigan Dip, Silverman) |
| **Problem yang didukung** | Semua problem tabular dengan numeric features |
| **Kapan digunakan** | Setiap kolom numeric; untuk memutuskan transformasi dan pemilihan model |
| **Kapan tidak digunakan** | Tidak ada kolom numeric |
| **Asumsi** | Setiap test memiliki asumsi dan constraint (sample size, parameter estimation); graphical diagnostics melengkapi p-value |
| **Data preprocessing** | Descriptive stats; histogram/KDE/ECDF/Q-Q metadata; normality tests; multimodality tests |
| **Dataset scale** | Kecil–Besar; Shapiro-Wilk untuk n≤5000; Anderson-Darling untuk tail sensitivity; Jarque-Bera untuk n>2000 |
| **Metric compatibility** | p-value, W statistic, A² statistic, skewness, kurtosis |
| **Hyperparameter penting** | `shapiro_max_n`=5000; `dagostino_min_n`=8; `jarque_bera_prefer_large`=true; `hartigan_dip_optional`=true |
| **Kisaran parameter awal** | n≤5000: Shapiro; n>5000: Anderson-Darling/JB; n≥8: D'Agostino; Hartigan Dip: optional |
| **CPU/GPU/RAM** | Ringan; O(n) per test |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Menggunakan Shapiro-Wilk p-value untuk n>5000; naive KS ketika parameter diestimasi dari sample; memilih JB untuk dataset kecil; mengabaikan multimodality |
| **Teknik pembanding** | Descriptive-only; graphical-only; single-test |
| **Ensemble compatibility** | Universal; output untuk transformation selection |
| **Exact package** | `scipy.stats.shapiro`; `scipy.stats.anderson`; `statsmodels.stats.diagnostic.kstest_normal`; `scipy.stats.normaltest`; `scipy.stats.jarque_bera`; `diptest` (R)  |
| **Contoh hypothesis** | "Anderson-Darling akan mendeteksi non-normality ≥20% lebih sering dari Kolmogorov-Smirnov pada data dengan tail deviations" |
| **Acceptance criteria** | Detection rate sesuai power comparison; p-value dilaporkan dengan constraint |
| **Rejection criteria** | Menggunakan SW p-value untuk n>5000 tanpa caveat; naive KS |
| **Sumber** | Shapiro & Wilk, Biometrika 52(3-4), 591–611 (1965)  ; Anderson & Darling, Annals of Mathematical Statistics 23(2), 193–212 (1952)  ; Stephens, JASA 69(347), 730–737 (1974)  ; Yap & Sim, JSCS 81(12), 2141–2155 (2011)  ; Hartigan & Hartigan, Annals of Statistics 13(1), 70–84 (1985)  |

---

## 5. OUTLIER DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi potential outlier via IQR (inner/outer fences), Modified Z-score/MAD, Grubbs test, Generalized ESD, Robust Mahalanobis/MCD, LOF, Isolation Forest. Agent tidak boleh otomatis menghapus outlier — output harus `potential_outlier = true`, bukan `delete_row = true` |
| **Problem yang didukung** | Semua problem tabular |
| **Kapan digunakan** | Setiap kolom numeric; untuk memahami tail behavior; multivariate anomaly detection |
| **Kapan tidak digunakan** | Outlier adalah signal (fraud detection); distribusi sudah bersih |
| **Asumsi** | IQR robust terhadap extremes; MAD breakdown point 50%; Grubbs/ESD butuh approx normality; LOF untuk local anomaly; Isolation Forest untuk large/high-dim |
| **Data preprocessing** | IQR fences; MAD; Grubbs/ESD; MCD/EllipticEnvelope; LOF; Isolation Forest |
| **Dataset scale** | IQR/MAD: semua skala; Grubbs/ESD: kecil–menengah; MCD: menengah; LOF/IF: besar |
| **Metric compatibility** | Outlier count, percentage, fence values, anomaly score |
| **Hyperparameter penting** | `iqr_inner_fence`=1.5; `iqr_outer_fence`=3.0; `modified_z_threshold`=3.5; `grubbs_requires_normality`=true; `isolation_forest_contamination`=auto |
| **Kisaran parameter awal** | IQR: 1.5/3.0; MAD: 3.5; IF: contamination='auto' |
| **CPU/GPU/RAM** | IQR/MAD: sangat ringan; MCD: O(n²); LOF: O(n²); IF: O(n log n) |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Z-score misleading untuk small samples; Grubbs blind pada heavy-tail income; multiple outliers → masking (gunakan ESD); auto-delete tanpa konteks |
| **Teknik pembanding** | Univariate vs multivariate; distance-based vs density-based vs isolation-based |
| **Ensemble compatibility** | Universal; output untuk robust preprocessing |
| **Exact package** | `scipy.stats.iqr`; `statsmodels.stats.outliers_influence`; `sklearn.covariance.MinCovDet`; `sklearn.covariance.EllipticEnvelope`; `sklearn.neighbors.LocalOutlierFactor`; `sklearn.ensemble.IsolationForest`  |
| **Contoh hypothesis** | "Isolation Forest akan mendeteksi multivariate anomaly yang tidak terdeteksi oleh univariate IQR pada ≥20% kasus" |
| **Acceptance criteria** | Multivariate anomaly detection rate ≥15%; false positive rate <10% |
| **Rejection criteria** | Auto-delete outlier; mengabaikan multivariate |
| **Sumber** | NIST/SEMATECH e-Handbook 7.1.6  ; Iglewicz & Hoaglin (1993)  ; Grubbs, Technometrics 11(1), 1–21 (1969)  ; Rosner, Technometrics 25(2), 165–172 (1983)  ; Rousseeuw & Van Driessen, Technometrics 41(3), 212–223 (1999)  ; Breunig et al., SIGMOD 2000  ; Liu et al., ICDM 2008  |

---

## 6. DUPLICATE & NEAR-DUPLICATE DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi exact duplicate, subset duplicate, near-duplicate, fuzzy record linkage, dan entity resolution. Lebih dari sekadar `df.duplicated()` — perlu membedakan entity yang sama dengan conflicting attributes |
| **Problem yang didukung** | Semua problem tabular; data quality; entity resolution |
| **Kapan digunakan** | Dataset dengan kemungkinan duplikasi; sebelum train-test split; data integration |
| **Kapan tidak digunakan** | Data sudah bersih dan terverifikasi unique |
| **Asumsi** | Duplikat dapat dideteksi via exact match, blocking, similarity matching, atau probabilistic matching |
| **Data preprocessing** | Exact duplicate check; subset duplicate; fuzzy string matching; blocking; clustering |
| **Dataset scale** | Kecil–Besar; blocking untuk large-scale |
| **Metric compatibility** | Duplicate count, duplicate ratio, precision/recall record linkage |
| **Hyperparameter penting** | `similarity_threshold`; `blocking_keys`; `distance_metric` |
| **Kisaran parameter awal** | similarity threshold: 0.85–0.95 |
| **CPU/GPU/RAM** | Ringan–Sedang; blocking mengurangi O(n²) |
| **Risiko leakage** | **Tinggi** jika duplicate melintasi train/test — ini adalah leakage yang perlu dideteksi dan dihapus |
| **Failure modes** | Hanya cek exact duplicate; gagal mendeteksi near-duplicate; blocking key salah |
| **Teknik pembanding** | Exact duplicate only; fuzzy matching only |
| **Ensemble compatibility** | Universal; wajib sebelum CV |
| **Exact package** | `pandas.duplicated()`; `recordlinkage` (Python); `fuzzywuzzy`; `dedupe` |
| **Contoh hypothesis** | "Near-duplicate detection dengan threshold 0.90 akan menemukan ≥50% lebih banyak duplicate dibanding exact match" |
| **Acceptance criteria** | Near-duplicate detection improvement ≥40% |
| **Rejection criteria** | Hanya exact duplicate; duplicate melintasi fold |
| **Sumber** | Elmagarmid et al., IEEE TKDE 19(1), 1–16 (2007)  |

---

## 7. FEATURE ASSOCIATION & REDUNDANCY

| Field | Detail |
|-------|--------|
| **Definisi** | Type-aware association matrix: Pearson/Spearman/Kendall/distance correlation untuk numeric-numeric; Chi-square/Cramér's V/Theil's U/MI untuk categorical-categorical; correlation ratio/ANOVA/Kruskal-Wallis/MI untuk numeric-categorical. Redundancy detection via correlation clusters dan mRMR |
| **Problem yang didukung** | Semua problem tabular |
| **Kapan digunakan** | Feature selection; redundancy detection; memahami struktur data |
| **Kapan tidak digunakan** | Dataset dengan sangat sedikit fitur (<5) |
| **Asumsi** | Pearson hanya linear; distance correlation mengkarakterisasi independence; MI tidak terbatas linear |
| **Data preprocessing** | Association matrix type-aware; MI computation; distance correlation; Cramér's V |
| **Dataset scale** | Kecil–Menengah untuk distance correlation; MI scalable |
| **Metric compatibility** | Pearson r, Spearman ρ, Kendall τ, dCor, Cramér's V, MI |
| **Hyperparameter penting** | `association_methods` per type pair; `mi_n_neighbors`; `correlation_threshold` |
| **Kisaran parameter awal** | Correlation threshold: 0.8–0.9 untuk redundancy |
| **CPU/GPU/RAM** | Distance correlation: O(n²) — menengah; MI: ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Hanya Pearson (`df.corr()`); mengabaikan nonlinear dependence; mengabaikan type-awareness |
| **Teknik pembanding** | Pearson-only; MI-only |
| **Ensemble compatibility** | Universal; output untuk feature selection |
| **Exact package** | `scipy.stats.pearsonr/spearmanr/kendalltau`; `dcor` (PyPI); `sklearn.feature_selection.mutual_info_classif/regressor`; `scipy.stats.contingency.association`  |
| **Contoh hypothesis** | "Distance correlation akan mendeteksi ≥30% lebih banyak dependence nonlinear dibanding Pearson pada data dengan hubungan non-monotonic" |
| **Acceptance criteria** | Nonlinear dependence detection improvement ≥25% |
| **Rejection criteria** | Hanya Pearson; mengabaikan type-awareness |
| **Sumber** | Székely et al., Annals of Statistics 35(6), 2769–2794 (2007)  ; Peng et al., IEEE TPAMI 27(8), 1226–1238 (2005)  ; Cramér (1946)  |

---

## 8. MULTICOLLINEARITY DIAGNOSTICS

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi ketergantungan linear antar-X via VIF, tolerance, condition number/index, eigenvalue/SVD diagnostics. Multicollinearity dapat terjadi tanpa pairwise correlation ekstrem (e.g., X₃ = X₁ + X₂) |
| **Problem yang didukung** | Regression; linear models; feature selection |
| **Kapan digunakan** | Model linear/regression; ingin memahami stabilitas koefisien |
| **Kapan tidak digunakan** | Tree-based models (tidak terpengaruh multicollinearity) |
| **Asumsi** | VIF = 1/(1-R²ⱼ); condition number dari singular values design matrix |
| **Data preprocessing** | VIF computation; tolerance; condition index; SVD diagnostics |
| **Dataset scale** | Kecil–Menengah (p < n) |
| **Metric compatibility** | VIF, tolerance, condition number |
| **Hyperparameter penting** | `vif_threshold` (5 atau 10); `condition_number_threshold` (30) |
| **Kisaran parameter awal** | VIF: 5 (moderate), 10 (severe); condition number: 30 |
| **CPU/GPU/RAM** | Ringan; O(p³) untuk SVD |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Menganggap VIF>5 sebagai universal truth; mengabaikan high condition number tanpa high pairwise correlation |
| **Teknik pembanding** | Pairwise correlation only |
| **Ensemble compatibility** | Universal; output untuk feature selection |
| **Exact package** | `statsmodels.stats.outliers_influence.variance_inflation_factor`; `sklearn.linear_model`; `car::vif` (R)  |
| **Contoh hypothesis** | "SVD-based condition number akan mendeteksi multicollinearity yang tidak terdeteksi oleh pairwise correlation ≥0.8" |
| **Acceptance criteria** | Detection improvement ≥20% |
| **Rejection criteria** | Hanya pairwise correlation |
| **Sumber** | Belsley, Kuh & Welsch (1980); Belsley, NBER Working Paper 0154  |

---

## 9. FEATURE–TARGET ASSOCIATION

| Field | Detail |
|-------|--------|
| **Definisi** | Analisis kekuatan hubungan antara setiap fitur dan target: MI, ANOVA/Kruskal-Wallis, chi-square, effect size, distance correlation. Digunakan untuk memahami signal dan mengidentifikasi fitur yang tidak relevan |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Feature selection; model building; memahami signal |
| **Kapan tidak digunakan** | Unsupervised learning |
| **Asumsi** | Metrik association sesuai dengan tipe fitur dan target |
| **Data preprocessing** | MI, ANOVA F-test, Kruskal-Wallis, chi-square, effect size |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MI, F-statistic, chi-square, effect size (Cohen's d, Cramér's V) |
| **Hyperparameter penting** | `target_type` (classification/regression); `feature_type` |
| **Kisaran parameter awal** | MI: `n_neighbors`=3; ANOVA: alpha=0.05 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; association dihitung pada training fold jika digunakan untuk selection |
| **Failure modes** | Mengabaikan nonlinear association; menggunakan Pearson untuk categorical |
| **Ensemble compatibility** | Universal; output untuk feature selection |
| **Exact package** | `sklearn.feature_selection.mutual_info_classif/regressor`; `scipy.stats.f_oneway`; `scipy.stats.kruskal`; `scipy.stats.chi2_contingency`  |
| **Contoh hypothesis** | "MI akan mengidentifikasi ≥20% lebih banyak fitur relevan dibanding Pearson pada target classification dengan hubungan nonlinear" |
| **Acceptance criteria** | Relevant feature detection improvement ≥15% |
| **Rejection criteria** | Hanya Pearson/linear methods |
| **Sumber** | Peng et al., IEEE TPAMI 27(8), 1226–1238 (2005)  |

---

## 10. INTERACTION DISCOVERY

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi efek interaksi X₁×X₂ yang tidak terlihat dari efek individual: pairwise interaction scan, Friedman's H-statistic, tree interaction strength. Dua fitur bisa lemah sendiri-sendiri tetapi sangat kuat jika digabung |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Feature engineering; memahami efek gabungan; model interpretation |
| **Kapan tidak digunakan** | Dataset dengan sangat sedikit fitur; model linear sederhana |
| **Asumsi** | H-statistic mengukur interaction strength; tree-based models dapat mendeteksi interaksi |
| **Data preprocessing** | Fit baseline tree model; hitung H-statistic untuk candidate pairs |
| **Dataset scale** | Kecil–Menengah; tree-based model untuk baseline |
| **Metric compatibility** | H-statistic (0–1); interaction strength |
| **Hyperparameter penting** | `n_estimators` (baseline model); `max_depth`; `h_threshold` |
| **Kisaran parameter awal** | H threshold: 0.05–0.10; baseline model: GBM dengan n_estimators=100 |
| **CPU/GPU/RAM** | Sedang; O(n_features²) untuk pairwise H |
| **Risiko leakage** | Sedang; interaction detection pada training data, bukan test |
| **Failure modes** | H-statistic mahal untuk banyak fitur; baseline model underfitting/overfitting |
| **Teknik pembanding** | Pairwise correlation; feature importance |
| **Ensemble compatibility** | Universal; output untuk feature engineering |
| **Exact package** | `iml::Interaction` (R); `interpret` (Python); `crucible_xai` (GitHub)  |
| **Contoh hypothesis** | "Friedman's H-statistic akan mendeteksi interaksi income×debt yang tidak terdeteksi oleh pairwise correlation" |
| **Acceptance criteria** | Interaction detection ≥1 pasang dengan H ≥0.05 |
| **Rejection criteria** | Tidak ada interaksi terdeteksi; H-statistic tidak stabil |
| **Sumber** | Friedman & Popescu, Annals of Applied Statistics 2(3), 916–954 (2008)  |

---

## 11. LEAKAGE DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi fitur yang mengandung informasi target yang tidak tersedia saat prediction. Meliputi: temporal leakage, post-outcome fields, proxy target, suspicious near-perfect association, target encoding global |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Sebelum training; sebagai gate di pipeline otomatis |
| **Kapan tidak digunakan** | N/A (selalu relevan) |
| **Asumsi** | Leakage dapat dideteksi via statistik + column semantics + time availability |
| **Data preprocessing** | Correlation scan; temporal boundary check; feature availability check; SHAP-based detection |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | AUC (leakage detection) |
| **Hyperparameter penting** | `correlation_threshold`; `importance_threshold`; `temporal_check` |
| **Kisaran parameter awal** | Correlation threshold: 0.95; importance threshold: top 5 |
| **CPU/GPU/RAM** | Sedang; tergantung metode |
| **Risiko leakage** | **Kritis**; ini adalah modul yang mendeteksi leakage |
| **Failure modes** | Gagal mendeteksi proxy target; mengabaikan temporal availability; hanya cek korelasi |
| **Teknik pembanding** | Correlation-only; manual inspection |
| **Ensemble compatibility** | Universal; wajib dijalankan sebelum CV |
| **Exact package** | `crucible-leakage` (PyPI)  ; custom implementation; `SHAP` |
| **Contoh hypothesis** | "Leakage detection dengan kombinasi correlation + SHAP + temporal check akan mendeteksi ≥80% kasus leakage" |
| **Acceptance criteria** | Detection rate ≥80%; false positive rate <15% |
| **Rejection criteria** | Detection rate <50%; leakage lolos ke training |
| **Sumber** | Kaufman et al., ACM TKDD 6(4), 1–21 (2012)  ; Kapoor & Narayanan, Patterns 4, 100804 (2023)  |

---

## 12. TARGET & CLASS IMBALANCE

| Field | Detail |
|-------|--------|
| **Definisi** | Diagnostik imbalance: class ratio, minority support, effective sample size, entropy, imbalance ratio. Untuk multiclass: class_count, class_proportion, min/max ratio, target entropy |
| **Problem yang didukung** | Classification |
| **Kapan digunakan** | Target classification; untuk memutuskan metric dan resampling strategy |
| **Kapan tidak digunakan** | Regression; classification seimbang |
| **Asumsi** | Imbalance adalah problem tersendiri; tidak bisa dinilai hanya dari accuracy |
| **Data preprocessing** | Class count; class proportion; IR = N_majority/N_minority; effective sample size ≈ 2× minority; target entropy |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | IR, class entropy, effective sample size |
| **Hyperparameter penting** | `imbalance_ratio_threshold`; `minority_support_min` |
| **Kisaran parameter awal** | IR > 10: severe; minority support < 20: very small |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Auto-SMOTE tanpa diagnosis; menggunakan accuracy untuk imbalanced; mengabaikan effective sample size |
| **Teknik pembanding** | Accuracy-only; class weight |
| **Ensemble compatibility** | Universal; output untuk metric selection dan resampling |
| **Exact package** | `pandas.value_counts()`; `imblearn`; custom |
| **Contoh hypothesis** | "Effective sample size akan menunjukkan bahwa dataset dengan 1% minority dan 100K rows sebenarnya hanya memiliki effective sample size ≈2K" |
| **Acceptance criteria** | Effective sample size dihitung; metric sesuai |
| **Rejection criteria** | Auto-SMOTE tanpa diagnosis |
| **Sumber** | He & Garcia, IEEE TKDE 21(9), 1263–1284 (2009)  ; Cui et al., CVPR 2019 |

---

## 13. ZERO-INFLATION & OVERDISPERSION

| Field | Detail |
|-------|--------|
| **Definisi** | Diagnostik untuk count data: zero ratio, expected-vs-observed zeros, ZIP score test, dispersion ratio (Var/Mean). Membedakan zero-inflation dari overdispersion |
| **Problem yang didukung** | Count data; Insurance claims; Healthcare utilization; Road crash frequency |
| **Kapan digunakan** | Target atau fitur count dengan banyak zero; memutuskan model count |
| **Kapan tidak digunakan** | Data kontinu tanpa zero |
| **Asumsi** | Poisson: E[X]=Var[X]; zero-inflated: dua proses; score test formal untuk zero-inflation |
| **Data preprocessing** | Zero ratio; expected zeros dari Poisson (e^-λ); score test; dispersion ratio |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Zero ratio, dispersion ratio, score test p-value |
| **Hyperparameter penting** | `zero_test_alpha`=0.05; `dispersion_threshold`=1.5 |
| **Kisaran parameter awal** | Zero ratio >50%: suspect zero-inflation; dispersion ratio >1.5: suspect overdispersion |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Zero_ratio tinggi belum otomatis zero-inflated; mengabaikan overdispersion; mengabaikan score test |
| **Teknik pembanding** | Zero ratio only; Poisson GLM residual |
| **Ensemble compatibility** | Universal; output untuk model selection |
| **Exact package** | `vcdExtra::zero.test` (R)  ; `statsmodels`; custom |
| **Contoh hypothesis** | "Score test akan mendeteksi zero-inflation pada ≥50% kolom count dengan zero ratio >40%" |
| **Acceptance criteria** | Zero-inflation terdeteksi; dispersion ratio dilaporkan |
| **Rejection criteria** | Hanya zero ratio; tidak ada formal test |
| **Sumber** | van den Broek, Biometrics 51(2), 738–743 (1995)  ; Deng & Paul, Statistica Sinica 15(1), 205–223 (2005)  |

---

## 14. TEMPORAL STRUCTURE DIAGNOSTICS

| Field | Detail |
|-------|--------|
| **Definisi** | Diagnostik struktur temporal: trend, seasonality, ACF, PACF, rolling mean/variance, lag correlations, time gaps, irregular sampling, change points, Ljung-Box test untuk autocorrelation |
| **Problem yang didukung** | Time series; Temporal tabular |
| **Kapan digunakan** | Ada time column; sebelum time-series modeling; untuk deteksi autocorrelation |
| **Kapan tidak digunakan** | Data tanpa time column; data i.i.d. |
| **Asumsi** | Autocorrelation dapat dideteksi via ACF/PACF/Ljung-Box; seasonality dapat dideteksi via decomposition |
| **Data preprocessing** | ACF/PACF; rolling stats; Ljung-Box; STL decomposition; time gap analysis |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | ACF, PACF, Ljung-Box Q, rolling statistics |
| **Hyperparameter penting** | `ljung_box_lags`=10–20; `rolling_window` |
| **Kisaran parameter awal** | Ljung-Box lags: 10–20 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Melihat 100K rows sebagai independent observations; mengabaikan autocorrelation; mengabaikan irregular sampling |
| **Teknik pembanding** | Visual inspection; ACF-only |
| **Ensemble compatibility** | Universal; output untuk temporal validation |
| **Exact package** | `statsmodels.tsa.stattools.acf/pacf`; `statsmodels.stats.diagnostic.acorr_ljungbox`; `pandas.rolling`  |
| **Contoh hypothesis** | "Ljung-Box test akan mendeteksi autocorrelation pada ≥50% fitur temporal yang tidak terdeteksi oleh visual inspection" |
| **Acceptance criteria** | Autocorrelation detection rate ≥50% |
| **Rejection criteria** | Mengabaikan autocorrelation; menganggap data independent |
| **Sumber** | Ljung & Box, Biometrika 65(2), 297–303 (1978)  |

---

## 15. DISTRIBUTION / DATASET SHIFT DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi perubahan distribusi antara train dan test/production: univariate (KS, Wasserstein, JS, PSI) dan multivariate (MMD, Energy Distance, adversarial validation) |
| **Problem yang didukung** | Semua problem dengan train/test shift; Competition; Production monitoring |
| **Kapan digunakan** | Sebelum training; monitoring drift; feature selection |
| **Kapan tidak digunakan** | Data i.i.d. sempurna |
| **Asumsi** | MMD dan Energy Distance = 0 hanya jika distribusi identik; PSI threshold tidak universal |
| **Data preprocessing** | KS; Wasserstein; JS; PSI; MMD; Energy Distance; adversarial validation |
| **Dataset scale** | Kecil–Besar; MMD O(n²) |
| **Metric compatibility** | KS stat, Wasserstein, JS, PSI, MMD, Energy Distance, adversarial AUC |
| **Hyperparameter penting** | `mmd_kernel`; `psi_bins`=10; `psi_thresholds` (0.1, 0.25)  |
| **Kisaran parameter awal** | PSI: <0.1 stable, 0.1–0.25 moderate, >0.25 significant  |
| **CPU/GPU/RAM** | MMD: O(n²) — menengah; PSI: ringan; adversarial: classifier-dependent |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | PSI threshold dianggap universal; mengabaikan multivariate shift; drift adversarials tidak terdeteksi |
| **Teknik pembanding** | Univariate-only; mean/std comparison |
| **Ensemble compatibility** | Universal; output untuk importance weighting |
| **Exact package** | `scipy.stats.ks_2samp`; `scipy.stats.wasserstein_distance`; `scipy.spatial.distance.jensenshannon`; `alibi-detect`; `Evidently`  |
| **Contoh hypothesis** | "MMD akan mendeteksi multivariate shift yang tidak terdeteksi oleh PSI univariate pada ≥30% kasus" |
| **Acceptance criteria** | Multivariate shift detection improvement ≥25% |
| **Rejection criteria** | Hanya univariate; PSI threshold tanpa caveat |
| **Sumber** | Gretton et al., JMLR 13, 723–773 (2012)  ; Székely & Rizzo, WIREs Computational Statistics 8(1), 38–43 (2016)  ; Springer Annals of Operations Research (2026)  |

---

## 16. FUNCTIONAL DEPENDENCY DISCOVERY

| Field | Detail |
|-------|--------|
| **Definisi** | Penemuan exact FD, approximate FD, dan conditional FD secara otomatis. Contoh: product_id → product_name, postcode → city. Jika 60231 → Surabaya dan 60231 → Jakarta, ada data-quality issue |
| **Problem yang didukung** | Data quality; schema normalization; data profiling; cleaning |
| **Kapan digunakan** | Dataset dengan kolom yang memiliki hubungan fungsional; sebelum cleaning |
| **Kapan tidak digunakan** | Dataset tanpa hubungan fungsional jelas |
| **Asumsi** | FD dapat ditemukan via algoritma discovery (HyFD, Tane, dll.); approximate FD untuk noisy data |
| **Data preprocessing** | FD discovery; approximate FD dengan error threshold; conditional FD |
| **Dataset scale** | Kecil–Besar; HyFD untuk large-scale |
| **Metric compatibility** | FD count, approximate FD error, coverage |
| **Hyperparameter penting** | `fd_error_threshold`=0.01–0.10; `algorithm` (HyFD, Tane) |
| **Kisaran parameter awal** | error threshold: 0.05 |
| **CPU/GPU/RAM** | Sedang–Berat; FD discovery O(n × p²) |
| **Risiko leakage** | Rendah; data profiling |
| **Failure modes** | Hanya exact FD; mengabaikan approximate FD; FD discovery mahal |
| **Teknik pembanding** | Manual inspection; correlation |
| **Ensemble compatibility** | Universal; output untuk data cleaning |
| **Exact package** | `desbordante` (PyPI)  ; `HyFD`; `Tane` |
| **Contoh hypothesis** | "Approximate FD discovery dengan error 0.05 akan menemukan ≥5 FD yang melanggar konsistensi data" |
| **Acceptance criteria** | FD violations terdeteksi; cleaning action |
| **Rejection criteria** | Hanya exact FD; tidak ada violation terdeteksi |
| **Sumber** | Kruse & Naumann, Efficient Discovery of Approximate Dependencies (VLDB 2018)  ; Papenbrock et al., CIKM 2015  |

---

## 17. RANGE & LOGICAL CONSTRAINT DISCOVERY

| Field | Detail |
|-------|--------|
| **Definisi** | Penemuan range violation, cross-field violation, conditional violation secara otomatis. Contoh: age ∈ [0,120], end_date ≥ start_date, discount ≤ price |
| **Problem yang didukung** | Data quality; cleaning; integrity constraint synthesis |
| **Kapan digunakan** | Dataset dengan logical constraints; sebelum modeling |
| **Kapan tidak digunakan** | Data sudah bersih dan terverifikasi |
| **Asumsi** | Constraints dapat ditemukan via autoencoder atau rule-based discovery |
| **Data preprocessing** | Range constraint discovery; cross-field constraint; conditional constraint |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Violation count, violation ratio |
| **Hyperparameter penting** | `constraint_discovery_method` (autoencoder/rule-based); `violation_threshold` |
| **Kisaran parameter awal** | Violation threshold: 0.01 |
| **CPU/GPU/RAM** | Sedang; autoencoder butuh GPU |
| **Risiko leakage** | Rendah; data profiling |
| **Failure modes** | Hanya range check; mengabaikan cross-field; constraint terlalu luas |
| **Teknik pembanding** | Manual constraint definition; domain knowledge |
| **Ensemble compatibility** | Universal; output untuk data cleaning |
| **Exact package** | `ADQuaTe2` (NSF)  ; `Guardrail` (ACM SIGMOD 2025)  ; custom |
| **Contoh hypothesis** | "Autoencoder-based constraint discovery akan menemukan ≥3 constraint violations yang terlewat oleh manual inspection" |
| **Acceptance criteria** | Violation discovery ≥3; false positive rate <10% |
| **Rejection criteria** | Tidak ada violation terdeteksi |
| **Sumber** | ADQuaTe2 (NSF 2019)  ; Guardrail, ACM SIGMOD 2025  |

---

## 18. MIXED-UNIT & FORMATTING ANOMALIES

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi mixed units (cm vs m, IDR vs ribuan IDR) dan formatting anomalies via log-magnitude clustering, digit-length distribution, unit suffix parsing, multimodality |
| **Problem yang didukung** | Data quality; cleaning; numeric validation |
| **Kapan digunakan** | Dataset dengan kemungkinan mixed units; sebelum scaling |
| **Kapan tidak digunakan** | Data sudah terverifikasi unit |
| **Asumsi** | Mixed units menghasilkan clustering magnitude atau digit-length distribution |
| **Data preprocessing** | Log-magnitude clustering; digit-length distribution; unit suffix parsing |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Cluster count, anomaly count |
| **Hyperparameter penting** | `n_magnitude_clusters`; `digit_length_threshold` |
| **Kisaran parameter awal** | n_clusters: 2–3; log-magnitude |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Gagal mendeteksi mixed units; over-clustering; mengabaikan konteks domain |
| **Teknik pembanding** | Range check; visual inspection |
| **Ensemble compatibility** | Universal; output untuk cleaning |
| **Exact package** | Custom; `scipy.cluster.hierarchy` |
| **Contoh hypothesis** | "Log-magnitude clustering akan mendeteksi mixed units (cm/m) pada ≥70% kolom dengan magnitude cluster" |
| **Acceptance criteria** | Mixed-unit detection ≥70% |
| **Rejection criteria** | Tidak ada detection |
| **Sumber** | Best practices; domain knowledge |

---

## 19. BENFORD'S LAW & DIGIT ANALYSIS

| Field | Detail |
|-------|--------|
| **Definisi** | Analisis distribusi digit (first-digit, second-digit) berdasarkan Benford's Law: lesser digits (1,2) lebih sering muncul sebagai leading digit. Deviasi dapat mengindikasikan financial irregularities atau manipulasi data |
| **Problem yang didukung** | Fraud detection; Audit; Financial anomaly detection |
| **Kapan digunakan** | Financial data; transaction amounts; populations; data yang naturally Benford-distributed |
| **Kapan tidak digunakan** | Data dengan assigned numbers (sequential IDs); data dengan range terbatas |
| **Asumsi** | Data naturally Benford-distributed; digit manipulation menghasilkan deviasi |
| **Data preprocessing** | First-digit extraction; second-digit; chi-square goodness-of-fit; MAD |
| **Dataset scale** | Menengah–Besar (n ≥ 1000 untuk power) |
| **Metric compatibility** | Chi-square, MAD, digit frequency |
| **Hyperparameter penting** | `digit_type` (first, second, first-two); `significance_level`=0.05 |
| **Kisaran parameter awal** | n ≥ 1000; alpha=0.05 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Data tidak Benford-distributed (false positive); n terlalu kecil; hanya first-digit |
| **Teknik pembanding** | Manual audit; visual inspection |
| **Ensemble compatibility** | Universal; output untuk fraud detection |
| **Exact package** | `benford.analysis` (R)  ; custom |
| **Contoh hypothesis** | "Benford's Law analysis akan mendeteksi manipulasi pada ≥50% dataset financial yang telah dimanipulasi" |
| **Acceptance criteria** | Detection rate ≥50%; false positive rate <10% |
| **Rejection criteria** | Detection rate <30%; data tidak Benford-distributed |
| **Sumber** | Nigrini, Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection (Wiley 2012)  |

---

## 20. COPULA DEPENDENCE & TAIL DEPENDENCE

| Field | Detail |
|-------|--------|
| **Definisi** | Diagnostik dependence via copula: Kendall's tau, Spearman's rho, tail dependence index, tail-weighted measures, semi-correlation. Copula models digunakan ketika data menunjukkan tail dependence dan tail asymmetry |
| **Problem yang didukung** | Financial risk; Extreme value; Dependence modeling |
| **Kapan digunakan** | Data dengan tail dependence; financial returns; insurance losses |
| **Kapan tidak digunakan** | Gaussian dependence; data tanpa tail dependence |
| **Asumsi** | Copula memisahkan marginal dari dependence structure; tail dependence dapat diukur |
| **Data preprocessing** | Pseudo-observations; copula fitting; tail dependence estimation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Kendall's τ, Spearman's ρ, tail dependence index, semi-correlation |
| **Hyperparameter penting** | `copula_family` (Clayton, Gumbel, Frank); `tail_type` (upper, lower) |
| **Kisaran parameter awal** | Copula family: auto-select via AIC |
| **CPU/GPU/RAM** | Menengah |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Mengabaikan tail dependence; Gaussian copula untuk data tail-dependent |
| **Teknik pembanding** | Pearson correlation; Spearman |
| **Ensemble compatibility** | Universal; output untuk dependence modeling |
| **Exact package** | `copula` (R); `vinecopulib` (R/Python); `pycop`  |
| **Contoh hypothesis** | "Tail dependence index akan mendeteksi dependence ekstrem yang tidak terdeteksi oleh Pearson pada data financial" |
| **Acceptance criteria** | Tail dependence detection; copula family selected |
| **Rejection criteria** | Hanya Pearson; mengabaikan tail |
| **Sumber** | Krupskii & Joe, Journal of Applied Statistics 42(3), 614–634 (2015)  |

---

## 21. EXTREME VALUE & TAIL-INDEX DIAGNOSTICS

| Field | Detail |
|-------|--------|
| **Definisi** | Estimasi tail index (shape parameter γ) via Extreme Value Theory: Hill estimator, Pickands, moment estimator, regression estimator. Tail index menentukan tipe tail dan risk measures (VaR, Expected Shortfall) |
| **Problem yang didukung** | Risk management; Extreme value; Financial modeling; Insurance |
| **Kapan digunakan** | Data dengan heavy tails; risk measures; extreme quantile estimation |
| **Kapan tidak digunakan** | Data light-tailed; tidak ada extreme values |
| **Asumsi** | Data mendekati Generalized Extreme Value (GEV) atau Generalized Pareto (GPD) di tail |
| **Data preprocessing** | Threshold selection (mean residual life plot); tail index estimation; QQ-plot |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Tail index γ, VaR, Expected Shortfall |
| **Hyperparameter penting** | `threshold_method`; `tail_fraction`; `estimator` (Hill, Pickands, moment) |
| **Kisaran parameter awal** | Tail fraction: 5–20% |
| **CPU/GPU/RAM** | Menengah |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Threshold salah → bias; estimator tidak stabil; data tidak heavy-tailed |
| **Teknik pembanding** | Historical simulation; normal approximation |
| **Ensemble compatibility** | Universal; output untuk risk modeling |
| **Exact package** | `extRemes` (R); `evir` (R); `pyextremes`  |
| **Contoh hypothesis** | "Hill estimator akan menunjukkan tail index γ>0 pada data financial returns, mengindikasikan heavy tail" |
| **Acceptance criteria** | Tail index diestimasi; threshold stabil |
| **Rejection criteria** | Threshold tidak stabil; estimator bias |
| **Sumber** | Ferreira & Lourenço, MDPI Computer Sciences & Mathematics Forum 11(1), 2 (2025)  |

---

## 22. CHANGE-POINT DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi regime change dalam time series: CUSUM, PELT, Bayesian Online Changepoint Detection (BOCPD). Berguna untuk mendeteksi perubahan struktural |
| **Problem yang didukung** | Time series; Regime change; Structural break; Monitoring |
| **Kapan digunakan** | Time series dengan kemungkinan regime change; production monitoring |
| **Kapan tidak digunakan** | Time series stationary; data tanpa time dimension |
| **Asumsi** | Distribusi data berubah pada titik tertentu; model segmentasi dapat mendeteksi perubahan |
| **Data preprocessing** | CUSUM; PELT; BOCPD; uncertainty quantification |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Changepoint count, changepoint location, uncertainty |
| **Hyperparameter penting** | `method` (cusum/pelt/bocpd); `penalty` (PELT); `hazard_rate` (BOCPD) |
| **Kisaran parameter awal** | PELT penalty: BIC/AIC |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Terlalu banyak changepoints; penalty salah; mengabaikan uncertainty |
| **Teknik pembanding** | Visual inspection; moving average |
| **Ensemble compatibility** | Universal; output untuk temporal validation |
| **Exact package** | `RegimeChange` (R)  ; `ruptures` (Python); `changepoint` (R) |
| **Contoh hypothesis** | "PELT akan mendeteksi regime change pada data stream yang mengalami concept drift" |
| **Acceptance criteria** | Changepoint terdeteksi; uncertainty dilaporkan |
| **Rejection criteria** | Tidak ada changepoint; penalty salah |
| **Sumber** | Page, Biometrika 41(1-2), 100–115 (1954)  ; Killick et al., JASA 107(500), 1590–1598 (2012)  |

---

## 23. INTRINSIC DIMENSIONALITY & CLUSTERING TENDENCY

| Field | Detail |
|-------|--------|
| **Definisi** | Estimasi intrinsic dimensionality (dari manifold hypothesis) dan clustering tendency (Hopkins statistic). Manifold hypothesis: high-dimensional data lie on or near a low-dimensional manifold |
| **Problem yang didukung** | Dimensionality reduction; Clustering; Manifold learning |
| **Kapan digunakan** | Data high-dimensional; sebelum clustering/dimensionality reduction |
| **Kapan tidak digunakan** | Data low-dimensional; tidak ada struktur manifold |
| **Asumsi** | Intrinsic dimension d < ambient dimension p; Hopkins statistic ~Beta(m,m) |
| **Data preprocessing** | Local PCA; graph-based estimation; Hopkins statistic; visual assessment |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Intrinsic dimension, Hopkins statistic |
| **Hyperparameter penting** | `n_neighbors` (local PCA); `m` (Hopkins sample size) |
| **Kisaran parameter awal** | n_neighbors: 10–50; m: 10% of n |
| **CPU/GPU/RAM** | Menengah |
| **Risiko leakage** | Rendah; diagnostic murni |
| **Failure modes** | Intrinsic dimension estimation tidak stabil; Hopkins statistic salah interpretasi |
| **Teknik pembanding** | PCA explained variance; visual inspection |
| **Ensemble compatibility** | Universal; output untuk dimensionality reduction |
| **Exact package** | `driada` (Python)  ; `clustertend` (R)  ; `hopkins` (R)  |
| **Contoh hypothesis** | "Intrinsic dimension estimation akan menunjukkan bahwa dataset dengan p=100 fitur sebenarnya memiliki d≈10" |
| **Acceptance criteria** | Intrinsic dimension < p; Hopkins > 0.7 (clustered)  |
| **Rejection criteria** | Intrinsic dimension ≈ p; Hopkins ≈ 0.5 (random) |
| **Sumber** | Local PCA (Scilit 2025)  ; Wright (2023)  ; Hopkins (1954) |

---

## Ringkasan Prioritas EDA Agent

| Kategori | Teknik Kunci | Prioritas | Biaya |
|----------|-------------|-----------|-------|
| **Schema** | Semantic Type Inference | Wajib | Ringan |
| **Cardinality** | K, unique ratio, dirty detection | Wajib | Ringan |
| **Missingness** | Little's test, co-missingness, predictability | Wajib | Ringan |
| **Distribution** | Anderson-Darling, multimodality | Wajib | Ringan |
| **Outliers** | IQR+MAD, MCD, Isolation Forest | Wajib | Ringan–Sedang |
| **Duplicates** | Exact + near-duplicate | Wajib | Sedang |
| **Association** | Type-aware matrix, distance correlation | Wajib | Menengah |
| **Multicollinearity** | VIF, condition number | Recommended | Ringan |
| **Interaction** | Friedman's H | Advanced | Sedang |
| **Leakage** | Correlation + SHAP + temporal | **Wajib** | Sedang |
| **Imbalance** | IR, effective sample size | Wajib | Ringan |
| **Zero-inflation** | Score test, dispersion | Recommended | Ringan |
| **Temporal** | Ljung-Box, ACF/PACF | Wajib (temporal) | Ringan |
| **Drift** | MMD, Energy Distance, PSI | **Wajib** | Menengah |
| **FD** | Approximate FD discovery | Recommended | Sedang |
| **Constraints** | Range + cross-field | Recommended | Sedang |
| **Mixed Units** | Log-magnitude clustering | Advanced | Ringan |
| **Benford** | Digit analysis | Niche | Ringan |
| **Copula** | Tail dependence | Advanced | Menengah |
| **EVT** | Tail index | Niche | Menengah |
| **Change-point** | PELT, BOCPD | Advanced | Ringan |
| **Intrinsic Dim** | Local PCA, Hopkins | Advanced | Menengah |

---

## Referensi Utama

1. **Rubin, D. B.** (1976). Inference and Missing Data. *Biometrika*, 63(3), 581–592.
2. **Little, R. J. A.** (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *JASA*, 83(404), 1198–1202.
3. **Seaman, S. et al.** (2013). What Is Meant by Missing at Random? *Statistical Science*, 28(2), 257–268.
4. **Shapiro, S. S., & Wilk, M. B.** (1965). An Analysis of Variance Test for Normality. *Biometrika*, 52(3-4), 591–611.
5. **Anderson, T. W., & Darling, D. A.** (1952). Asymptotic Theory of Certain Goodness of Fit Criteria. *Annals of Mathematical Statistics*, 23(2), 193–212.
6. **Stephens, M. A.** (1974). EDF Statistics for Goodness of Fit and Some Comparisons. *JASA*, 69(347), 730–737.
7. **Hartigan, J. A., & Hartigan, P. M.** (1985). The Dip Test of Unimodality. *Annals of Statistics*, 13(1), 70–84.
8. **Grubbs, F. E.** (1969). Procedures for Detecting Outlying Observations in Samples. *Technometrics*, 11(1), 1–21.
9. **Rosner, B.** (1983). Percentage Points for a Generalized ESD Many-Outlier Procedure. *Technometrics*, 25(2), 165–172.
10. **Rousseeuw, P. J., & Van Driessen, K.** (1999). A Fast Algorithm for the Minimum Covariance Determinant Estimator. *Technometrics*, 41(3), 212–223.
11. **Breunig, M. M. et al.** (2000). LOF: Identifying Density-Based Local Outliers. *SIGMOD 2000*, 93–104.
12. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). Isolation Forest. *ICDM 2008*, 413–422.
13. **Elmagarmid, A. K. et al.** (2007). Duplicate Record Detection: A Survey. *IEEE TKDE*, 19(1), 1–16.
14. **Székely, G. J. et al.** (2007). Measuring and Testing Dependence by Correlation of Distances. *Annals of Statistics*, 35(6), 2769–2794.
15. **Peng, H. et al.** (2005). Feature Selection Based on Mutual Information. *IEEE TPAMI*, 27(8), 1226–1238.
16. **Belsley, D. A. et al.** (1980). *Regression Diagnostics*. Wiley.
17. **Friedman, J. H., & Popescu, B. E.** (2008). Predictive Learning via Rule Ensembles. *Annals of Applied Statistics*, 2(3), 916–954.
18. **Kaufman, S. et al.** (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4), 1–21.
19. **Kapoor, S., & Narayanan, A.** (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. *Patterns*, 4(9), 100804.
20. **Gretton, A. et al.** (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723–773.
21. **Ljung, G. M., & Box, G. E. P.** (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika*, 65(2), 297–303.
22. **Hulsebos, M. et al.** (2019). Sherlock: A Deep Learning Approach to Semantic Data Type Detection. *KDD 2019*.
23. **Pargent, F. et al.** (2023). Regularized Target Encoding Outperforms Traditional Methods in Supervised ML with High Cardinality Features. *Computational Statistics*, 38, 1–33.
24. **Nigrini, M. J.** (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
25. **Krupskii, P., & Joe, H.** (2015). Tail-Weighted Measures of Dependence. *Journal of Applied Statistics*, 42(3), 614–634.
26. **Killick, R. et al.** (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *JASA*, 107(500), 1590–1598.
27. **van den Broek, J.** (1995). A Score Test for Zero Inflation in a Poisson Distribution. *Biometrics*, 51(2), 738–743.
28. **Cui, Y. et al.** (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR 2019*.
29. **Kruse, S., & Naumann, F.** (2018). Efficient Discovery of Approximate Dependencies. *VLDB 2018*.
30. **Wright, K.** (2023). Hopkins Statistic for Clustering Tendency. *R package hopkins*.

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*