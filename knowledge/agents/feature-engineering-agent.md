# Feature Engineering — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk setiap teknik feature engineering tingkat lanjut. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## 1. NUMERICAL TRANSFORMATION

### 1.1 Log / Log1p Transform

| Field | Detail |
|-------|--------|
| **Definisi** | Transformasi logaritmik yang digunakan untuk mengubah data skewed menjadi lebih terdistribusi merata di seluruh rentang nilai. Log transformasi membantu menangani data positif dengan distribusi right-skewed  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Data count/monetary dengan right-skew; distribusi positively skewed; fitur dengan rentang nilai ekstrem |
| **Kapan tidak digunakan** | Data dengan nilai negatif atau nol (gunakan log1p); data sudah normal |
| **Asumsi** | Data positif; skewness dapat dikurangi dengan kompresi logaritmik |
| **Data preprocessing** | Tidak ada scaling tambahan; log1p = log(1+x) untuk handle nol |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada (deterministik) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan; O(n) |
| **Risiko leakage** | Rendah; transformasi pointwise |
| **Failure modes** | Log(0) error jika tidak menggunakan log1p; distribusi tidak selalu menjadi normal |
| **Teknik pembanding** | Box-Cox, Yeo-Johnson, Rank transform |
| **Ensemble compatibility** | Universal |
| **Exact package** | `numpy.log1p`; `feature_engine.transformation.LogTransformer`  |
| **Contoh hypothesis** | "Log1p transform pada fitur monetary akan meningkatkan R² regresi ≥5% vs raw features" |
| **Acceptance criteria** | Skewness turun ≥50%; metric improvement ≥3% |
| **Rejection criteria** | Metric drop; skewness tidak berubah |
| **Sumber** | Feature-engine docs; Semantic Scholar (2026) |

### 1.2 Box-Cox / Yeo-Johnson

| Field | Detail |
|-------|--------|
| **Definisi** | Power transform family yang mencari lambda optimal untuk membuat data mendekati Gaussian. Box-Cox memerlukan data strictly positive, sedangkan Yeo-Johnson dapat menangani nilai positif dan negatif  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Data skewed yang perlu dinormalisasi; model linear/NN yang sensitif terhadap distribusi |
| **Kapan tidak digunakan** | Tree-based models (tidak terpengaruh); data sudah normal |
| **Asumsi** | Data dapat ditransformasi mendekati normal dengan power transform |
| **Data preprocessing** | Scaling mungkin diperlukan setelah transform |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `method` ('box-cox' atau 'yeo-johnson'), `standardize` |
| **Kisaran parameter awal** | method='yeo-johnson' (default), standardize=True |
| **CPU/GPU/RAM** | Ringan; O(n) dengan optimasi lambda |
| **Risiko leakage** | Sedang; lambda harus di-fit pada training fold saja |
| **Failure modes** | Box-Cox gagal pada data negatif; lambda overfitting pada dataset kecil |
| **Teknik pembanding** | Log transform, Rank transform |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.PowerTransformer`; `sklearn.preprocessing.power_transform`  |
| **Contoh hypothesis** | "Yeo-Johnson transform akan meningkatkan akurasi klasifikasi ≥3% pada dataset dengan fitur skewed" |
| **Acceptance criteria** | Normality test (Shapiro-Wilk) improvement; metric improvement ≥3% |
| **Rejection criteria** | Metric drop; lambda tidak konvergen |
| **Sumber** | scikit-learn docs; pyts docs |

### 1.3 Rank Transform / Gaussian Rank (GaussRank)

| Field | Detail |
|-------|--------|
| **Definisi** | Transformasi yang mengubah continuous input data dari distribusi arbitrer menjadi distribusi Gaussian dengan menggunakan ranking. Populer di neural network tabular  |
| **Problem yang didukung** | Neural Network, Deep Learning tabular |
| **Kapan digunakan** | NN tabular; data dengan outlier ekstrem; distribusi tidak diketahui |
| **Kapan tidak digunakan** | Tree-based models (tidak perlu); data sudah Gaussian |
| **Asumsi** | Ranking dapat menangkap struktur data yang relevan |
| **Data preprocessing** | Tidak ada scaling tambahan |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada (deterministik) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan (CPU); GPU-accelerated via RAPIDS CuPy  |
| **Risiko leakage** | Sedang; ranking harus dihitung per fold |
| **Failure modes** | Kehilangan informasi magnitude; tidak cocok untuk linear models |
| **Teknik pembanding** | QuantileTransformer, PowerTransformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `QuantileTransformer(output_distribution='normal')`; `gauss-rank-scaler` (GitHub)  |
| **Contoh hypothesis** | "GaussRank akan meningkatkan akurasi NN tabular ≥5% vs StandardScaler" |
| **Acceptance criteria** | NN convergence lebih cepat; metric improvement ≥5% |
| **Rejection criteria** | Metric drop; training tidak stabil |
| **Sumber** | NVIDIA Technical Blog (2021); Semantic Scholar (2025) |

### 1.4 Binning / Discretization

| Field | Detail |
|-------|--------|
| **Definisi** | Proses mengelompokkan nilai kontinu ke dalam interval diskrit. Metode: equal-width, equal-frequency (quantile), decision-tree-based, WOE binning |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Hubungan non-linear antara fitur dan target; interpretability; credit scoring |
| **Kapan tidak digunakan** | Data sudah diskrit; hubungan linear |
| **Asumsi** | Diskritisasi menangkap pola non-linear |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_bins`, `strategy` (uniform/quantile/kmeans) |
| **Kisaran parameter awal** | n_bins: 5–20; strategy='quantile' |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; bin edges harus di-fit pada training fold |
| **Failure modes** | Terlalu banyak bins → overfitting; terlalu sedikit → kehilangan informasi |
| **Teknik pembanding** | Spline transform, Polynomial features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.KBinsDiscretizer`; `feature_engine.discretisation` |
| **Contoh hypothesis** | "Quantile binning dengan 10 bins akan meningkatkan AUC credit scoring ≥2%" |
| **Acceptance criteria** | AUC improvement ≥2%; bins monotonic terhadap target |
| **Rejection criteria** | AUC drop; bins tidak stabil antar fold |
| **Sumber** | scikit-learn docs |

### 1.5 Clipping / Winsorizing

| Field | Detail |
|-------|--------|
| **Definisi** | Memotong outlier pada persentil tertentu (misal 1% dan 99%) untuk mengurangi pengaruh nilai ekstrem |
| **Problem yang didukung** | Regresi, Klasifikasi |
| **Kapan digunakan** | Outlier ekstrem yang mengganggu model; data dengan heavy tails |
| **Kapan tidak digunakan** | Outlier informatif (fraud detection); distribusi sudah bersih |
| **Asumsi** | Nilai ekstrem adalah noise, bukan signal |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, MAE, RMSE |
| **Hyperparameter penting** | `lower_percentile`, `upper_percentile` |
| **Kisaran parameter awal** | lower=0.01, upper=0.99 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah; percentile dari training fold |
| **Failure modes** | Memotong signal penting; percentile terlalu agresif |
| **Teknik pembanding** | RobustScaler, QuantileTransformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `feature_engine.outliers.Winsorizer` |
| **Contoh hypothesis** | "Winsorizing pada 1%/99% akan mengurangi RMSE ≥5% pada data dengan outlier ekstrem" |
| **Acceptance criteria** | RMSE improvement ≥5%; distribusi lebih stabil |
| **Rejection criteria** | RMSE drop; signal hilang |
| **Sumber** | Feature-engine docs |

### 1.6 Scaling (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)

| Field | Detail |
|-------|--------|
| **Definisi** | **StandardScaler**: z-score (mean=0, std=1). **MinMaxScaler**: skala ke [0,1]. **RobustScaler**: menggunakan median dan IQR (tahan outlier). **MaxAbsScaler**: skala ke [-1,1] berdasarkan max absolute value |
| **Problem yang didukung** | Semua problem dengan model yang sensitif terhadap skala (SVM, NN, KNN, linear models) |
| **Kapan digunakan** | Model berbasis distance/gradient; fitur dengan skala berbeda |
| **Kapan tidak digunakan** | Tree-based models (tidak perlu); data sudah dalam skala yang sama |
| **Asumsi** | Fitur dapat diskalakan tanpa kehilangan informasi |
| **Data preprocessing** | Fitur yang akan diskalakan harus numerik |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `with_mean`, `with_std` (StandardScaler); `feature_range` (MinMax) |
| **Kisaran parameter awal** | StandardScaler default; RobustScaler untuk data dengan outlier |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | **Tinggi** jika scaler di-fit pada seluruh data; harus fit pada train fold saja |
| **Failure modes** | StandardScaler sensitif terhadap outlier; MinMax sensitif terhadap outlier |
| **Teknik pembanding** | PowerTransformer, QuantileTransformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler` |
| **Contoh hypothesis** | "RobustScaler akan outperform StandardScaler pada data dengan outlier ekstrem" |
| **Acceptance criteria** | Metric improvement ≥3% dengan RobustScaler |
| **Rejection criteria** | Metric drop |
| **Sumber** | scikit-learn docs |

### 1.7 Ratio / Difference Features

| Field | Detail |
|-------|--------|
| **Definisi** | Fitur baru yang dibuat dari rasio atau selisih antara dua kolom numerik |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Hubungan antar fitur bermakna secara domain (price/area, income/expense) |
| **Kapan tidak digunakan** | Fitur tidak memiliki hubungan domain yang jelas |
| **Asumsi** | Rasio/selisih menangkap informasi yang tidak ada di fitur individual |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada (domain knowledge) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Division by zero; rasio tidak stabil pada nilai kecil |
| **Teknik pembanding** | Polynomial features, Interaction features |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas operations; `feature_engine.creation` |
| **Contoh hypothesis** | "Ratio price/area akan meningkatkan akurasi prediksi harga rumah ≥5%" |
| **Acceptance criteria** | Metric improvement ≥5% |
| **Rejection criteria** | Metric drop; multikolinearitas tinggi |
| **Sumber** | Domain knowledge |

---

## 2. CATEGORICAL ENCODING

### 2.1 One-Hot Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Membuat kolom biner untuk setiap kategori. Setiap kategori direpresentasikan sebagai vektor biner |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas rendah (< 10–20 kategori); model linear/NN |
| **Kapan tidak digunakan** | Kardinalitas tinggi (dimensionalitas meledak) |
| **Asumsi** | Kategori independen satu sama lain |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `drop` (first/last), `sparse_output` |
| **Kisaran parameter awal** | drop='first' untuk menghindari multicollinearity |
| **CPU/GPU/RAM** | Ringan; sparse matrix untuk kardinalitas tinggi |
| **Risiko leakage** | Rendah |
| **Failure modes** | Dimensionalitas curse; sparse matrix overhead |
| **Teknik pembanding** | Ordinal encoding, Target encoding, Binary encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.OneHotEncoder`; `pandas.get_dummies` |
| **Contoh hypothesis** | "One-Hot Encoding akan meningkatkan akurasi logistic regression ≥5% vs label encoding" |
| **Acceptance criteria** | Metric improvement ≥5% |
| **Rejection criteria** | Training time >3×; memory error |
| **Sumber** | scikit-learn docs |

### 2.2 Ordinal / Label Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Mengganti kategori dengan integer ordinal. Untuk kategori berurutan (low, medium, high) |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kategori memiliki urutan natural; tree-based models |
| **Kapan tidak digunakan** | Kategori nominal tanpa urutan (model linear akan salah interpretasi) |
| **Asumsi** | Urutan kategori bermakna |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `categories` (urutan manual) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Urutan salah → model belajar hubungan palsu |
| **Teknik pembanding** | One-Hot, Target encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.OrdinalEncoder`; `LabelEncoder` |
| **Contoh hypothesis** | "Ordinal encoding pada fitur 'education level' akan meningkatkan akurasi ≥3% vs one-hot" |
| **Acceptance criteria** | Metric improvement ≥3% |
| **Rejection criteria** | Metric drop |
| **Sumber** | scikit-learn docs |

### 2.3 Frequency / Count Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Mengganti kategori dengan frekuensi kemunculannya di dataset |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas tinggi; kategori rare grouping |
| **Kapan tidak digunakan** | Frekuensi tidak informatif untuk target |
| **Asumsi** | Frekuensi berkorelasi dengan target |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Sedang; frekuensi harus dihitung pada training fold |
| **Failure modes** | Kehilangan identitas kategori |
| **Teknik pembanding** | Target encoding, WOE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `category_encoders.CountEncoder`; pandas `value_counts` |
| **Contoh hypothesis** | "Count encoding akan meningkatkan AUC ≥2% pada fitur dengan kardinalitas >100" |
| **Acceptance criteria** | AUC improvement ≥2% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Category Encoders docs |

### 2.4 Target Encoding (Mean Encoding)

| Field | Detail |
|-------|--------|
| **Definisi** | Mengganti kategori dengan mean target untuk kategori tersebut. Memerlukan K-fold, leave-one-out, atau smoothing untuk mencegah leakage  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas tinggi; tree-based models |
| **Kapan tidak digunakan** | Dataset kecil (leakage risk tinggi); model linear |
| **Asumsi** | Mean target per kategori informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `smoothing`, `min_samples_leaf`, `cv` |
| **Kisaran parameter awal** | smoothing: 1–100; cv: 5 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Sangat Tinggi**; wajib K-fold/LOO dengan smoothing  |
| **Failure modes** | Overfitting pada kategori rare; leakage jika tidak OOF |
| **Teknik pembanding** | WOE, CatBoost encoding, James-Stein |
| **Ensemble compatibility** | Universal |
| **Exact package** | `category_encoders.TargetEncoder`; `sklearn.preprocessing.TargetEncoder`; `h2o.targetencoder`  |
| **Contoh hypothesis** | "Target encoding dengan K-fold=5 dan smoothing=10 akan meningkatkan AUC ≥5% pada fitur kardinalitas tinggi" |
| **Acceptance criteria** | AUC improvement ≥5%; tidak ada leakage (OOF) |
| **Rejection criteria** | AUC drop; overfitting terdeteksi |
| **Sumber** | H2O docs; Category Encoders; scikit-learn |

### 2.5 WOE (Weight of Evidence) Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Encoding yang mengukur kekuatan prediktif kategori terhadap target biner dengan menghitung log odds ratio. Umum di credit scoring  |
| **Problem yang didukung** | Klasifikasi biner (credit scoring, risk modeling) |
| **Kapan digunakan** | Credit scoring; model linear interpretable; regulated environment |
| **Kapan tidak digunakan** | Regresi; multi-class tanpa adaptasi |
| **Asumsi** | Hubungan monotonic antara WOE dan log odds |
| **Data preprocessing** | Binning numerik dulu untuk continuous features |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC, KS, Gini, Accuracy |
| **Hyperparameter penting** | `n_bins`, `monotonic` constraint |
| **Kisaran parameter awal** | n_bins: 5–20; monotonic=True |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; WOE harus dihitung pada training fold |
| **Failure modes** | WOE infinite jika kategori memiliki 0 event; binning salah |
| **Teknik pembanding** | Target encoding, CatBoost encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `optbinning` (Python); `OptimalBinningWoE` (R) ; `woe_scoring` (GitHub)  |
| **Contoh hypothesis** | "WOE encoding dengan optimal binning akan mencapai AUC ≥0.80 pada credit scoring" |
| **Acceptance criteria** | AUC ≥0.80; monotonic WOE |
| **Rejection criteria** | AUC <0.75; WOE tidak monotonic |
| **Sumber** | HSE thesis; CRAN OptimalBinningWoE |

### 2.6 Hashing Encoding (Feature Hashing)

| Field | Detail |
|-------|--------|
| **Definisi** | Menggunakan hash function untuk memetakan kategori ke integer dalam ruang fitur terbatas |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas sangat tinggi (jutaan); streaming data |
| **Kapan tidak digunakan** | Kardinalitas rendah (one-hot lebih baik) |
| **Asumsi** | Hash collision dapat ditoleransi |
| **Data preprocessing** | Standar |
| **Dataset scale** | Besar–Sangat Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_features` (hash space size) |
| **Kisaran parameter awal** | n_features: 2^18 – 2^22 |
| **CPU/GPU/RAM** | Ringan; memory terbatas |
| **Risiko leakage** | Rendah |
| **Failure modes** | Collision rate tinggi → informasi hilang |
| **Teknik pembanding** | One-Hot, Target encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.feature_extraction.FeatureHasher` |
| **Contoh hypothesis** | "Hashing encoding dengan 2^20 features akan mencapai akurasi setara one-hot dengan memory 10× lebih kecil" |
| **Acceptance criteria** | Accuracy setara; memory ≤20% |
| **Rejection criteria** | Accuracy drop >3% |
| **Sumber** | scikit-learn docs |

### 2.7 Binary Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Kompromi antara one-hot dan ordinal: kategori di-encode ke binary code, lalu setiap bit menjadi kolom |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas menengah (10–1000); memory terbatas |
| **Kapan tidak digunakan** | Kardinalitas rendah |
| **Asumsi** | Binary representation cukup untuk membedakan kategori |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Binary code tidak bermakna semantik |
| **Teknik pembanding** | One-Hot, Hashing |
| **Ensemble compatibility** | Universal |
| **Exact package** | `category_encoders.BinaryEncoder` |
| **Sumber** | Category Encoders docs |

### 2.8 CatBoost Encoding (Ordered Target Statistics)

| Field | Detail |
|-------|--------|
| **Definisi** | Target encoding yang leakage-safe by design: menggunakan ordered target statistics yang menghitung encoding berdasarkan data sebelumnya dalam urutan acak  |
| **Problem yang didukung** | Klasifikasi, Regresi dengan categorical features |
| **Kapan digunakan** | High-cardinality categorical; ingin menghindari leakage tanpa K-fold manual |
| **Kapan tidak digunakan** | Model non-CatBoost (kecuali menggunakan encoder terpisah) |
| **Asumsi** | Ordered statistics mengurangi target leakage |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `one_hot_max_size`, `max_ctr_complexity` |
| **Kisaran parameter awal** | one_hot_max_size: 2–10; max_ctr_complexity: 1–4 |
| **CPU/GPU/RAM** | Sedang; built-in di CatBoost |
| **Risiko leakage** | **Rendah**; leakage-safe by design  |
| **Failure modes** | Training lebih lambat; tidak transferable ke model lain tanpa encoder terpisah |
| **Teknik pembanding** | Target encoding + K-fold |
| **Ensemble compatibility** | Built-in CatBoost; `gators.encoders.catboost_encoder` untuk model lain  |
| **Exact package** | `catboost.CatBoostClassifier`; `gators.encoders.CatBoostEncoder` |
| **Contoh hypothesis** | "CatBoost encoding akan mencapai akurasi setara target encoding K-fold tanpa risiko leakage" |
| **Acceptance criteria** | Akurasi setara; tidak ada leakage |
| **Rejection criteria** | Akurasi drop >2% |
| **Sumber** | CatBoost docs; Nature Scientific Reports (2024); PayPal Gators |

### 2.9 James-Stein Encoding & M-estimate Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | **James-Stein**: weighted average antara mean target kategori dan mean target global, dengan shrinkage factor. **M-estimate**: simplified target encoder dengan satu parameter smoothing (m)  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas tinggi; ingin shrinkage otomatis |
| **Kapan tidak digunakan** | Kardinalitas rendah |
| **Asumsi** | Shrinkage mengurangi overfitting pada kategori rare |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `m` (M-estimate); `B` (James-Stein shrinkage) |
| **Kisaran parameter awal** | m: 1–100 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; OOF diperlukan |
| **Failure modes** | Shrinkage terlalu kuat → underfitting |
| **Teknik pembanding** | Target encoding, CatBoost encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `category_encoders.JamesSteinEncoder`; `category_encoders.MEstimateEncoder`  |
| **Contoh hypothesis** | "James-Stein encoder akan outperform target encoding pada dataset dengan banyak kategori rare" |
| **Acceptance criteria** | AUC improvement ≥3% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Category Encoders; Micci-Barreca (2001) |

### 2.10 Entity Embedding

| Field | Detail |
|-------|--------|
| **Definisi** | Embedding kategorikal via neural network: memetakan kategori ke dense vector di Euclidean space. Mengurangi memory dan mempercepat NN vs one-hot  |
| **Problem yang didukung** | Neural Network tabular, Deep Learning |
| **Kapan digunakan** | Kardinalitas tinggi; NN tabular; transfer learning |
| **Kapan tidak digunakan** | Tree-based models (kecuali sebagai fitur tambahan); dataset kecil |
| **Asumsi** | Kategori memiliki struktur latent yang dapat dipelajari |
| **Data preprocessing** | Standar untuk NN |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `embedding_dim` (biasanya min(50, (n_cat+1)//2)) |
| **Kisaran parameter awal** | embedding_dim: 2–50 |
| **CPU/GPU/RAM** | GPU recommended; memory efisien |
| **Risiko leakage** | Rendah; embedding di-train end-to-end |
| **Failure modes** | Embedding dimension terlalu kecil → informasi hilang; terlalu besar → overfitting |
| **Teknik pembanding** | One-Hot, Target encoding |
| **Ensemble compatibility** | Dapat digunakan sebagai fitur untuk GBM (category-embedding package)  |
| **Exact package** | Keras/PyTorch Embedding layer; `category-embedding` (PyPI)  |
| **Contoh hypothesis** | "Entity embedding dengan dim=10 akan mencapai akurasi ≥5% lebih tinggi dari one-hot pada NN tabular" |
| **Acceptance criteria** | Accuracy improvement ≥5%; memory ≤50% |
| **Rejection criteria** | Accuracy drop; training tidak stabil |
| **Sumber** | Guo & Berkhahn (2016); PyPI category-embedding |

### 2.11 Rare Category Grouping

| Field | Detail |
|-------|--------|
| **Definisi** | Menggabungkan kategori yang jarang muncul ke dalam kategori "Other" atau "Rare" untuk mengurangi sparsity dan meningkatkan stabilitas  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Kardinalitas tinggi dengan long tail; target encoding |
| **Kapan tidak digunakan** | Semua kategori frequent; kategori rare informatif |
| **Asumsi** | Kategori rare tidak membawa signal unik |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `tol` (threshold frekuensi), `n_categories` |
| **Kisaran parameter awal** | tol: 0.01–0.05; n_categories: 10 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah; threshold dari training fold |
| **Failure modes** | Threshold terlalu tinggi → kehilangan informasi |
| **Teknik pembanding** | Frequency encoding, Target encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | `feature_engine.encoding.RareLabelEncoder`  |
| **Contoh hypothesis** | "Rare grouping dengan tol=0.01 akan meningkatkan stabilitas target encoding" |
| **Acceptance criteria** | Metric improvement ≥2%; encoding lebih stabil |
| **Rejection criteria** | Metric drop |
| **Sumber** | Feature-engine docs; IEEE (2025) |

---

## 3. DATETIME / TEMPORAL FEATURES

### 3.1 Ekstraksi Komponen Waktu

| Field | Detail |
|-------|--------|
| **Definisi** | Ekstraksi komponen dari datetime: year, month, day, dayofweek, hour, quarter, week of year, is_weekend, is_holiday |
| **Problem yang didukung** | Semua problem dengan data temporal |
| **Kapan digunakan** | Data time series, timestamp features |
| **Kapan tidak digunakan** | Data tanpa komponen temporal |
| **Asumsi** | Komponen waktu relevan untuk prediksi |
| **Data preprocessing** | Parsing datetime |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Timezone inconsistency; holiday calendar salah |
| **Teknik pembanding** | Cyclical encoding |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `dt` accessor |
| **Sumber** | Feature engineering best practices |

### 3.2 Cyclical Encoding (Sin/Cos)

| Field | Detail |
|-------|--------|
| **Definisi** | Transformasi komponen siklik (jam, hari, bulan) ke sin/cos untuk menjaga kontinuitas |
| **Problem yang didukung** | Semua problem temporal |
| **Kapan digunakan** | Komponen siklik (jam 23 → jam 0 harus dekat) |
| **Kapan tidak digunakan** | Komponen non-siklik (year) |
| **Asumsi** | Siklikalitas periodik |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Period (24 untuk jam, 7 untuk hari) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Period salah |
| **Teknik pembanding** | Raw component |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy sin/cos |
| **Sumber** | Time series best practices |

### 3.3 Time Since/Until Event

| Field | Detail |
|-------|--------|
| **Definisi** | Durasi dari referensi point (misal: days since last purchase, hours until deadline) |
| **Problem yang didukung** | Klasifikasi, Regresi, Churn prediction |
| **Kapan digunakan** | Ada event reference point |
| **Kapan tidak digunakan** | Tidak ada reference point |
| **Asumsi** | Waktu relatif terhadap event relevan |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah jika reference point konsisten |
| **Failure modes** | Reference point leakage |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas datetime operations |
| **Sumber** | Domain knowledge |

### 3.4 Lag Features

| Field | Detail |
|-------|--------|
| **Definisi** | Nilai t-1, t-2, ..., t-n dari time series |
| **Problem yang didukung** | Forecasting, Time series regression |
| **Kapan digunakan** | Autokorelasi tinggi; forecasting |
| **Kapan tidak digunakan** | Data tidak sekuensial |
| **Asumsi** | Nilai masa lalu prediktif untuk masa depan |
| **Data preprocessing** | Sorting by time; handle missing |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, MAE, RMSE, MAPE |
| **Hyperparameter penting** | `n_lags` |
| **Kisaran parameter awal** | n_lags: 1–30 (tergantung frekuensi) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika lag melampaui horizon prediksi; harus strictly past data |
| **Failure modes** | Data leakage; autocorrelation berubah |
| **Teknik pembanding** | Rolling window, Diff |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `shift`; `feature_engine.timeseries` |
| **Contoh hypothesis** | "Lag 7 hari akan meningkatkan akurasi forecasting ≥10%" |
| **Acceptance criteria** | RMSE improvement ≥10% |
| **Rejection criteria** | RMSE drop; leakage terdeteksi |
| **Sumber** | Time series best practices |

### 3.5 Rolling Window Statistics

| Field | Detail |
|-------|--------|
| **Definisi** | Rolling mean/std/min/max/median, expanding window |
| **Problem yang didukung** | Forecasting, Time series |
| **Kapan digunakan** | Trend/seasonality; smoothing |
| **Kapan tidak digunakan** | Data tidak sekuensial |
| **Asumsi** | Statistik window relevan untuk prediksi |
| **Data preprocessing** | Sorting by time |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, MAE, RMSE |
| **Hyperparameter penting** | `window_size`, `min_periods` |
| **Kisaran parameter awal** | window: 3–30 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi** jika window mencakup future data |
| **Failure modes** | Leakage; window tidak stabil |
| **Teknik pembanding** | Lag features, Diff |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `rolling`; `feature_engine.timeseries` |
| **Sumber** | Time series best practices |

### 3.6 Diff Features

| Field | Detail |
|-------|--------|
| **Definisi** | First-order/second-order differencing untuk stationaritas |
| **Problem yang didukung** | Time series forecasting |
| **Kapan digunakan** | Non-stationary data |
| **Kapan tidak digunakan** | Data sudah stationary |
| **Asumsi** | Differencing membuat data stationary |
| **Data preprocessing** | Sorting by time |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, MAE |
| **Hyperparameter penting** | `order` (1 atau 2) |
| **Kisaran parameter awal** | order=1 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah jika strictly past |
| **Failure modes** | Over-differencing → kehilangan informasi |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `diff` |
| **Sumber** | Time series best practices |

### 3.7 Fourier Features

| Field | Detail |
|-------|--------|
| **Definisi** | Fourier Series terms sebagai explanatory variables untuk multiple seasonal periods atau complex/long seasonal periods  |
| **Problem yang didukung** | Time series forecasting dengan multiple seasonality |
| **Kapan digunakan** | Multiple seasonality (harian + mingguan + tahunan); long seasonal periods |
| **Kapan tidak digunakan** | Seasonality sederhana (cyclical encoding cukup) |
| **Asumsi** | Seasonality dapat direpresentasikan dengan Fourier terms |
| **Data preprocessing** | Time index numerik |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | MSE, MAE, RMSE |
| **Hyperparameter penting** | `sp` (seasonal period), `k` (Fourier term) |
| **Kisaran parameter awal** | k: 1–10 per seasonal period |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | k terlalu kecil → underfitting seasonality |
| **Teknik pembanding** | Cyclical encoding, Seasonal decomposition |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sktime.transformations.series.FourierFeatures` ; `aeon.transformations.series.FourierFeatures`  |
| **Contoh hypothesis** | "Fourier features dengan k=5 untuk seasonality harian dan mingguan akan meningkatkan akurasi forecasting ≥8%" |
| **Acceptance criteria** | RMSE improvement ≥8% |
| **Rejection criteria** | RMSE drop |
| **Sumber** | sktime docs; aeon docs |

### 3.8 Business Calendar Features

| Field | Detail |
|-------|--------|
| **Definisi** | Fitur berbasis kalender bisnis: hari kerja, akhir bulan/kuartal, event khusus domain |
| **Problem yang didukung** | Forecasting, Klasifikasi |
| **Kapan digunakan** | Data dengan pola bisnis (sales, finance) |
| **Kapan tidak digunakan** | Data tanpa pola bisnis |
| **Asumsi** | Kalender bisnis relevan |
| **Data preprocessing** | Kalender bisnis library |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `freq` (B, BM, BQ, BA) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Holiday calendar salah |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pandas.tseries.offsets`; `holidays` library |
| **Sumber** | Pandas docs |

---

## 4. AGGREGATION / GROUPBY FEATURES

### 4.1 Group Statistics

| Field | Detail |
|-------|--------|
| **Definisi** | Mean/median/std/min/max/sum/count per grup kategori |
| **Problem yang didukung** | Semua problem tabular |
| **Kapan digunakan** | Data relational/multi-level; customer analytics |
| **Kapan tidak digunakan** | Data flat tanpa grup |
| **Asumsi** | Statistik grup informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `group_cols`, `agg_funcs` |
| **Kisaran parameter awal** | mean, std, count |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | **Tinggi** jika statistik dihitung dari seluruh data |
| **Failure modes** | Leakage; group terlalu kecil |
| **Teknik pembanding** | Deep Feature Synthesis |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `groupby().agg()` |
| **Contoh hypothesis** | "Group mean per user akan meningkatkan AUC churn prediction ≥8%" |
| **Acceptance criteria** | AUC improvement ≥8% |
| **Rejection criteria** | AUC drop; leakage terdeteksi |
| **Sumber** | Kaggle winning solutions |

### 4.2 Group Rank

| Field | Detail |
|-------|--------|
| **Definisi** | Ranking dalam grup |
| **Problem yang didukung** | Ranking, Klasifikasi |
| **Kapan digunakan** | Perbandingan relatif dalam grup |
| **Kapan tidak digunakan** | Tidak ada grup |
| **Asumsi** | Ranking informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `method` (average, min, max, first) |
| **Kisaran parameter awal** | method='average' |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; ranking harus dihitung per fold |
| **Failure modes** | Tie handling |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `groupby().rank()` |
| **Sumber** | Kaggle best practices |

### 4.3 Group Frequency

| Field | Detail |
|-------|--------|
| **Definisi** | Proporsi dalam grup |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Proporsi relatif penting |
| **Kapan tidak digunakan** | Proporsi tidak informatif |
| **Asumsi** | Proporsi informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Proporsi tidak stabil |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `groupby().transform('count')` |
| **Sumber** | Kaggle best practices |

### 4.4 Deviation from Group Mean

| Field | Detail |
|-------|--------|
| **Definisi** | Nilai individu dikurangi rata-rata grupnya  |
| **Problem yang didukung** | Semua problem tabular |
| **Kapan digunakan** | Perlu mengukur deviasi individual |
| **Kapan tidak digunakan** | Tidak ada grup |
| **Asumsi** | Deviasi informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Group mean tidak stabil |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `groupby().transform('mean')` |
| **Contoh hypothesis** | "Deviation from group mean akan meningkatkan R² ≥5% pada data panel" |
| **Acceptance criteria** | R² improvement ≥5% |
| **Rejection criteria** | R² drop |
| **Sumber** | Stack Overflow; R `colDevs` |

### 4.5 Multi-level Groupby

| Field | Detail |
|-------|--------|
| **Definisi** | Agregasi bertingkat (misal per user → per user-per-hari) |
| **Problem yang didukung** | Multi-level relational data |
| **Kapan digunakan** | Hierarchical data |
| **Kapan tidak digunakan** | Flat data |
| **Asumsi** | Hierarki bermakna |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `levels` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Tinggi**; hati-hati dengan leakage antar level |
| **Failure modes** | Leakage dari level tinggi ke level rendah |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas multi-level groupby |
| **Sumber** | Kaggle best practices |

### 4.6 Deep Feature Synthesis (DFS)

| Field | Detail |
|-------|--------|
| **Definisi** | Metode otomatis untuk feature engineering pada data relational dan temporal  |
| **Problem yang didukung** | Relational data, Multi-tabel |
| **Kapan digunakan** | Data multi-tabel; ingin otomatisasi agregasi |
| **Kapan tidak digunakan** | Data single-table |
| **Asumsi** | Relasi antar tabel informatif |
| **Data preprocessing** | EntitySet construction |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `max_depth`, `primitive_options`, `agg_primitives` |
| **Kisaran parameter awal** | max_depth: 1–3 |
| **CPU/GPU/RAM** | Sedang–Berat; dapat menggunakan Dask/Spark |
| **Risiko leakage** | **Tinggi**; cutoff time harus benar |
| **Failure modes** | Feature explosion; leakage temporal |
| **Teknik pembanding** | Manual groupby |
| **Ensemble compatibility** | Universal |
| **Exact package** | `featuretools.dfs()`  |
| **Contoh hypothesis** | "DFS dengan max_depth=2 akan menemukan fitur yang meningkatkan AUC ≥5% pada data relational" |
| **Acceptance criteria** | AUC improvement ≥5%; feature count manageable |
| **Rejection criteria** | Feature explosion >1000; leakage |
| **Sumber** | Featuretools docs; Alteryx |

---

## 5. INTERACTION & POLYNOMIAL FEATURES

### 5.1 Pairwise Interaction

| Field | Detail |
|-------|--------|
| **Definisi** | Perkalian/pembagian antar fitur |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Hubungan non-linear antar fitur |
| **Kapan tidak digunakan** | Fitur independen |
| **Asumsi** | Interaksi bermakna |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `interaction_only` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan–Sedang (feature explosion) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Feature explosion; multikolinearitas |
| **Teknik pembanding** | Polynomial features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.PolynomialFeatures(interaction_only=True)` |
| **Sumber** | scikit-learn docs |

### 5.2 Polynomial Features

| Field | Detail |
|-------|--------|
| **Definisi** | Fitur^2, fitur^3, cross-term |
| **Problem yang didukung** | Regresi, Klasifikasi |
| **Kapan digunakan** | Hubungan polinomial |
| **Kapan tidak digunakan** | Hubungan linear |
| **Asumsi** | Hubungan polinomial |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `degree` |
| **Kisaran parameter awal** | degree: 2–3 |
| **CPU/GPU/RAM** | Sedang (feature explosion) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Overfitting; feature explosion |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.preprocessing.PolynomialFeatures` |
| **Sumber** | scikit-learn docs |

### 5.3 Automated Interaction Search

| Field | Detail |
|-------|--------|
| **Definisi** | Beam search (AutoFeat), genetic programming |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Pool fitur besar; ingin otomatisasi |
| **Kapan tidak digunakan** | Dataset kecil |
| **Asumsi** | Search space dapat dieksplorasi |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `search_depth`, `beam_width` |
| **Kisaran parameter awal** | depth: 2–3 |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Sedang |
| **Failure modes** | Overfitting; search space terlalu besar |
| **Ensemble compatibility** | Universal |
| **Exact package** | `autofeat` (Python) ; `OpenFE`  |
| **Sumber** | ICDE 2024; PMLR |

---

## 6. MISSING VALUE HANDLING

### 6.1 Missing Indicator

| Field | Detail |
|-------|--------|
| **Definisi** | Binary flag apakah value asli hilang |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Missingness informatif (MNAR) |
| **Kapan tidak digunakan** | Missing completely at random |
| **Asumsi** | Missingness memiliki signal |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Indicator tidak informatif |
| **Teknik pembanding** | Imputasi |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.impute.MissingIndicator` |
| **Contoh hypothesis** | "Missing indicator akan meningkatkan AUC ≥3% pada data medis dengan missingness informatif" |
| **Acceptance criteria** | AUC improvement ≥3% |
| **Rejection criteria** | AUC drop |
| **Sumber** | scikit-learn docs |

### 6.2 KNN Imputer

| Field | Detail |
|-------|--------|
| **Definisi** | Imputasi missing values menggunakan mean dari k-nearest neighbors  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Data numerik; korelasi antar fitur |
| **Kapan tidak digunakan** | Data kategorik (kecuali di-encode) |
| **Asumsi** | Similar samples memiliki similar values |
| **Data preprocessing** | Scaling untuk distance metric |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_neighbors`, `weights` |
| **Kisaran parameter awal** | n_neighbors: 5 |
| **CPU/GPU/RAM** | Sedang (O(n²) distance) |
| **Risiko leakage** | **Tinggi**; imputation harus di-fit pada train fold |
| **Failure modes** | Curse of dimensionality; imputation tidak akurat pada high-dim |
| **Teknik pembanding** | MICE, MissForest, Mean imputation |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.impute.KNNImputer`  |
| **Contoh hypothesis** | "KNNImputer dengan k=5 akan outperform mean imputation ≥3% pada data dengan missing 20%" |
| **Acceptance criteria** | Metric improvement ≥3% |
| **Rejection criteria** | Metric drop |
| **Sumber** | scikit-learn docs; BMC Medical Research (2024) |

### 6.3 MICE (Multiple Imputation by Chained Equations)

| Field | Detail |
|-------|--------|
| **Definisi** | Imputasi iteratif dengan model regression untuk setiap fitur yang memiliki missing  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Missing values pada multiple fitur; MCAR/MAR |
| **Kapan tidak digunakan** | Missing sangat sedikit |
| **Asumsi** | Missing at Random (MAR) |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `max_iter`, `n_imputations` |
| **Kisaran parameter awal** | max_iter: 10; n_imputations: 5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | **Tinggi**; imputation model harus di-fit pada train fold |
| **Failure modes** | Convergence issues; imputation model salah |
| **Teknik pembanding** | MissForest, KNN Imputer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.experimental.IterativeImputer`; `mice` (R) |
| **Contoh hypothesis** | "MICE dengan 10 iterasi akan menghasilkan imputasi lebih akurat dari mean imputation" |
| **Acceptance criteria** | Imputation accuracy improvement |
| **Rejection criteria** | Imputation tidak konvergen |
| **Sumber** | Royston & White (2011); Wiley (2026) |

### 6.4 MissForest

| Field | Detail |
|-------|--------|
| **Definisi** | Variant MICE yang menggunakan Random Forest sebagai model imputasi  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Non-linear relationships; mixed data types |
| **Kapan tidak digunakan** | Dataset sangat besar |
| **Asumsi** | Random Forest dapat menangkap hubungan non-linear |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_estimators`, `max_iter` |
| **Kisaran parameter awal** | n_estimators: 100; max_iter: 10 |
| **CPU/GPU/RAM** | Sedang–Berat |
| **Risiko leakage** | **Tinggi** |
| **Failure modes** | Slow; overfitting |
| **Teknik pembanding** | MICE, KNN Imputer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `missingpy.MissForest` |
| **Sumber** | Stekhoven & Bühlmann (2012) |

### 6.5 Missing Pattern Clustering

| Field | Detail |
|-------|--------|
| **Definisi** | Grup baris berdasarkan pola missing-nya  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Missing pattern informatif; multiple missing patterns |
| **Kapan tidak digunakan** | Missing random |
| **Asumsi** | Pola missing memiliki makna |
| **Data preprocessing** | Binary missing indicator matrix |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_clusters` |
| **Kisaran parameter awal** | n_clusters: 3–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Cluster tidak stabil |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation; `missing.pattern.plot` (R) |
| **Sumber** | arXiv (2026); Semantic Scholar |

---

## 7. TEXT FEATURES

### 7.1 Basic Text Stats

| Field | Detail |
|-------|--------|
| **Definisi** | Panjang teks, jumlah kata, jumlah karakter spesial |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Ada kolom teks dalam tabular |
| **Kapan tidak digunakan** | Tidak ada teks |
| **Asumsi** | Statistik teks informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Tidak ada |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Statistik tidak informatif |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas string operations |
| **Sumber** | NLP best practices |

### 7.2 TF-IDF / Bag-of-Words

| Field | Detail |
|-------|--------|
| **Definisi** | Term Frequency-Inverse Document Frequency untuk mengubah teks menjadi fitur numerik  |
| **Problem yang didukung** | Klasifikasi teks, Tabular dengan kolom teks |
| **Kapan digunakan** | Teks dalam tabular; klasifikasi dokumen |
| **Kapan tidak digunakan** | Teks sangat pendek; sudah ada embedding |
| **Asumsi** | Kata-kata penting dapat diidentifikasi via frekuensi |
| **Data preprocessing** | Tokenization, lowercasing, stopword removal |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Accuracy, F1, AUC |
| **Hyperparameter penting** | `max_features`, `ngram_range`, `min_df` |
| **Kisaran parameter awal** | max_features: 1000–10000; ngram: (1,2) |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | **Tinggi**; vocabulary harus di-fit pada train fold |
| **Failure modes** | Sparse matrix; vocabulary leakage |
| **Teknik pembanding** | Embedding, CountVectorizer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.feature_extraction.text.TfidfVectorizer`  |
| **Contoh hypothesis** | "TF-IDF dengan max_features=5000 akan meningkatkan AUC ≥5% pada data tabular dengan kolom teks" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Enigma blog; InterSystems docs |

### 7.3 Embedding dari Pretrained Model

| Field | Detail |
|-------|--------|
| **Definisi** | Embedding dari sentence-transformers/LLM sebagai fitur numerik |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Teks kompleks; butuh semantic representation |
| **Kapan tidak digunakan** | Resource terbatas |
| **Asumsi** | Pretrained embedding menangkap semantic |
| **Data preprocessing** | Tokenization sesuai model |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `model_name`, `embedding_dim` |
| **Kisaran parameter awal** | all-MiniLM-L6-v2 (384 dim) |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah (pretrained) |
| **Failure modes** | Domain mismatch |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sentence-transformers`; `TabuLLM`  |
| **Sumber** | TabuLLM (2026) |

---

## 8. FEATURE SELECTION

### 8.1 Filter Methods

| Field | Detail |
|-------|--------|
| **Definisi** | Correlation threshold, Chi-square, Mutual Information, ANOVA F-test, Variance threshold  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Pre-filtering cepat; model-agnostic |
| **Kapan tidak digunakan** | Hubungan non-linear kompleks |
| **Asumsi** | Fitur independen terhadap target |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `threshold`, `k` |
| **Kisaran parameter awal** | k: 10–100 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Sedang; selection harus per fold |
| **Failure modes** | Mengabaikan interaksi fitur |
| **Teknik pembanding** | Wrapper, Embedded |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.feature_selection.mutual_info_classif`, `f_classif`, `chi2`; `VarianceThreshold` |
| **Sumber** | scikit-learn docs; Springer (2025) |

### 8.2 Wrapper Methods (RFE, Boruta)

| Field | Detail |
|-------|--------|
| **Definisi** | **RFE**: eliminasi fitur rekursif berdasarkan importance. **Boruta**: all-relevant selection dengan shadow features  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Butuh evaluasi model aktual; all-relevant selection |
| **Kapan tidak digunakan** | Dataset sangat besar (mahal) |
| **Asumsi** | Model importance dapat diandalkan |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_features_to_select`, `step` (RFE); `max_iter`, `perc` (Boruta) |
| **Kisaran parameter awal** | RFE: n_features=10–50; Boruta: max_iter=100 |
| **CPU/GPU/RAM** | Berat (repeated training) |
| **Risiko leakage** | **Tinggi**; selection harus nested dalam CV  |
| **Failure modes** | Overfitting selection; computationally expensive |
| **Teknik pembanding** | Filter, Embedded |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.feature_selection.RFE/RFECV` ; `boruta_py` ; `Boruta` (R)  |
| **Contoh hypothesis** | "Boruta akan menemukan semua fitur relevan dan meningkatkan AUC ≥3% vs filter method" |
| **Acceptance criteria** | AUC improvement ≥3%; stable selection |
| **Rejection criteria** | AUC drop; runtime >10× |
| **Sumber** | scikit-learn docs; CRAN Boruta |

### 8.3 Embedded Methods (L1/Lasso, Tree Importance, SHAP)

| Field | Detail |
|-------|--------|
| **Definisi** | **L1/Lasso**: shrinkage yang menghasilkan sparse coefficient. **Tree importance**: Gini/permutation importance. **SHAP**: Shapley values untuk feature contribution  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Model linear (Lasso); tree-based (importance); butuh explainability (SHAP) |
| **Kapan tidak digunakan** | Model black-box tanpa importance |
| **Asumsi** | Regularisasi/importance menangkap signal |
| **Data preprocessing** | Scaling untuk Lasso |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alpha` (Lasso); `n_estimators` (tree) |
| **Kisaran parameter awal** | alpha: 0.001–1.0 (via LassoCV) |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | **Tinggi**; selection dalam CV |
| **Failure modes** | Lasso tidak stabil pada correlated features; tree importance bias |
| **Teknik pembanding** | Filter, Wrapper |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.LassoCV/Lasso` ; `shap`  |
| **Contoh hypothesis** | "SHAP-based selection akan memilih subset fitur yang mencapai AUC setara full feature dengan 50% fitur" |
| **Acceptance criteria** | AUC setara; feature count ≤50% |
| **Rejection criteria** | AUC drop >2% |
| **Sumber** | scikit-learn docs; Taylor & Francis (2025) |

### 8.4 Null Importance / Target Shuffling

| Field | Detail |
|-------|--------|
| **Definisi** | Bandingkan importance asli vs setelah target di-shuffle; buang fitur yang tidak signifikan berbeda  |
| **Problem yang didukung** | Semua problem |
| **Kapan digunakan** | Butuh robust feature selection; menghindari false positive |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Importance distribution dapat diestimasi via permutation |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_iterations` |
| **Kisaran parameter awal** | n_iterations: 100 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah (permutation test) |
| **Failure modes** | Computational cost |
| **Ensemble compatibility** | Universal |
| **Exact package** | `nullpom` ; `target-permutation-importances`  |
| **Contoh hypothesis** | "Null importance akan membuang 50% fitur noise tanpa kehilangan AUC" |
| **Acceptance criteria** | AUC setara; feature count turun ≥30% |
| **Rejection criteria** | AUC drop >1% |
| **Sumber** | Altmann et al. (2010); PyPI nullpom |

### 8.5 Adversarial Feature Selection

| Field | Detail |
|-------|--------|
| **Definisi** | Buang fitur yang membuat train/test mudah dibedakan (indikasi shift/leakage)  |
| **Problem yang didukung** | Semua problem dengan train/test shift |
| **Kapan digunakan** | Deteksi distribution shift; competition dengan adversarial validation |
| **Kapan tidak digunakan** | Train/test dari distribusi sama |
| **Asumsi** | Fitur yang membedakan train/test adalah noise/leakage |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC adversarial |
| **Hyperparameter penting** | `threshold` |
| **Kisaran parameter awal** | threshold: 0.7–0.8 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Membuang fitur penting yang kebetulan shifted |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `adversarial-test` (PyPI)  |
| **Contoh hypothesis** | "Adversarial feature selection akan mengurangi adversarial AUC dari 0.9 ke 0.6 tanpa kehilangan performa" |
| **Acceptance criteria** | Adversarial AUC turun; target AUC stabil |
| **Rejection criteria** | Target AUC drop >3% |
| **Sumber** | Inferensys ML Glossary |

---

## 9. DIMENSIONALITY REDUCTION

### 9.1 PCA Components

| Field | Detail |
|-------|--------|
| **Definisi** | Principal Component Analysis sebagai fitur tambahan |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Fitur berkorelasi tinggi; reduksi dimensi |
| **Kapan tidak digunakan** | Interpretability penting |
| **Asumsi** | Variansi menangkap informasi |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components` |
| **Kisaran parameter awal** | n_components: 0.95 (variance explained) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Tinggi**; PCA harus di-fit pada train fold |
| **Failure modes** | Kehilangan interpretability; informasi hilang |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.decomposition.PCA` |
| **Sumber** | scikit-learn docs |

### 9.2 UMAP / t-SNE Embedding

| Field | Detail |
|-------|--------|
| **Definisi** | Non-linear dimensionality reduction untuk embedding  |
| **Problem yang didukung** | Visualisasi, Clustering, Feature generation |
| **Kapan digunakan** | Data non-linear high-dim |
| **Kapan tidak digunakan** | Data linear |
| **Asumsi** | Manifold structure |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_neighbors`, `min_dist` (UMAP) |
| **Kisaran parameter awal** | n_neighbors: 15; min_dist: 0.1 |
| **CPU/GPU/RAM** | Sedang; GPU-accelerated |
| **Risiko leakage** | Sedang; fit pada train fold |
| **Failure modes** | Stochastic; tidak deterministik |
| **Ensemble compatibility** | Universal |
| **Exact package** | `umap-learn` ; `sklearn.manifold.TSNE` |
| **Sumber** | McInnes et al. (2018) |

### 9.3 Autoencoder Bottleneck

| Field | Detail |
|-------|--------|
| **Definisi** | Bottleneck representation dari autoencoder sebagai fitur  |
| **Problem yang didukung** | Klasifikasi, Regresi, Anomaly Detection |
| **Kapan digunakan** | Data high-dim; butuh compressed representation |
| **Kapan tidak digunakan** | Interpretability penting |
| **Asumsi** | Autoencoder dapat menangkap latent structure |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `bottleneck_dim`, `architecture` |
| **Kisaran parameter awal** | bottleneck: 10–100 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang; train autoencoder pada train fold |
| **Failure modes** | Overfitting; bottleneck terlalu kecil |
| **Ensemble compatibility** | Universal |
| **Exact package** | PyTorch/TensorFlow custom |
| **Sumber** | NeurIPS 2025; Springer (2025) |

### 9.4 Cluster ID (K-Means/GMM)

| Field | Detail |
|-------|--------|
| **Definisi** | Cluster ID dari K-Means/GMM sebagai fitur kategorikal baru  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Data heterogen; sub-populasi |
| **Kapan tidak digunakan** | Data homogeneous |
| **Asumsi** | Cluster menangkap sub-populasi |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_clusters` |
| **Kisaran parameter awal** | n_clusters: 3–10 (Elbow method) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; clustering pada train fold |
| **Failure modes** | Cluster tidak stabil |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.cluster.KMeans/GaussianMixture` |
| **Sumber** | Springer (2025) |

---

## 10. AUTOMATED FEATURE ENGINEERING TOOLS

### 10.1 Featuretools

| Field | Detail |
|-------|--------|
| **Definisi** | Framework untuk DFS pada data relational/temporal  |
| **Problem yang didukung** | Relational data |
| **Exact package** | `featuretools` |
| **Sumber** | Alteryx |

### 10.2 AutoFeat

| Field | Detail |
|-------|--------|
| **Definisi** | Iterative feature subsampling + beam search  |
| **Problem yang didukung** | Tabular regression/classification |
| **Exact package** | `autofeat` (Python)  |
| **Sumber** | ICDE 2024 |

### 10.3 tsfresh

| Field | Detail |
|-------|--------|
| **Definisi** | Ekstraksi fitur time series otomatis (ratusan statistical features)  |
| **Problem yang didukung** | Time series classification, regression |
| **Hyperparameter penting** | `default_fc_parameters` ('efficient', 'minimal', 'comprehensive') |
| **Exact package** | `tsfresh` ; `TSFreshRelevantFeatureExtractor` (sktime)  |
| **Sumber** | Blue Yonder; Elsevier (2026) |

### 10.4 NVTabular

| Field | Detail |
|-------|--------|
| **Definisi** | GPU-accelerated feature engineering untuk terabyte-scale datasets  |
| **Problem yang didukung** | Recommender systems, Large-scale tabular |
| **Exact package** | `nvtabular` (NVIDIA-Merlin)  |
| **Sumber** | NVIDIA docs |

### 10.5 FeatureEngine

| Field | Detail |
|-------|--------|
| **Definisi** | Library preprocessing terstruktur (scaling, encoding, transform dalam sklearn-pipeline style)  |
| **Problem yang didukung** | Semua problem |
| **Exact package** | `feature_engine`  |
| **Sumber** | Feature-engine docs |

### 10.6 PyCaret

| Field | Detail |
|-------|--------|
| **Definisi** | Otomatisasi end-to-end termasuk feature engineering dasar  |
| **Problem yang didukung** | Semua problem |
| **Exact package** | `pycaret`  |
| **Sumber** | PyCaret docs |

### 10.7 OpenFE

| Field | Detail |
|-------|--------|
| **Definisi** | Automated feature generation + efficient feature selection (evaluasi pakai boosting cepat)  |
| **Problem yang didukung** | Tabular |
| **Exact package** | `OpenFE`  |
| **Sumber** | PMLR |

---

## 11. LLM-BASED FEATURE ENGINEERING

### 11.1 CAAFE

| Field | Detail |
|-------|--------|
| **Definisi** | Context-Aware Automated Feature Engineering: LLM iteratively generate additional semantically meaningful features based on dataset description  |
| **Problem yang didukung** | Tabular classification |
| **Kapan digunakan** | Butuh semantic feature generation; ada deskripsi dataset |
| **Kapan tidak digunakan** | Dataset tanpa konteks; resource terbatas |
| **Asumsi** | LLM dapat memahami domain dan generate fitur relevan |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `llm_model`, `n_iterations` |
| **Kisaran parameter awal** | GPT-4; iterations: 5–10 |
| **CPU/GPU/RAM** | API-based; atau GPU untuk local LLM |
| **Risiko leakage** | Sedang; LLM bisa generate fitur yang leak |
| **Failure modes** | LLM generate terlalu banyak fitur sederhana; perlu filtering ketat |
| **Teknik pembanding** | AutoFeat, OpenFE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `CAAFE` (GitHub)  |
| **Contoh hypothesis** | "CAAFE akan meningkatkan akurasi ≥5% pada 11 dari 14 dataset tabular" |
| **Acceptance criteria** | Improvement pada ≥70% dataset |
| **Rejection criteria** | Improvement <20% dataset |
| **Sumber** | ICML; arXiv:2305.03403 |

### 11.2 LLM-FE

| Field | Detail |
|-------|--------|
| **Definisi** | LLM sebagai evolutionary optimizer untuk menemukan fitur baru, iteratif dengan feedback dari validation score  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Butuh automated feature discovery dengan feedback loop |
| **Kapan tidak digunakan** | Resource terbatas; dataset sangat besar |
| **Asumsi** | LLM dapat bertindak sebagai evolutionary optimizer |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `population_size`, `generations` |
| **Kisaran parameter awal** | generations: 5–20 |
| **CPU/GPU/RAM** | API-based |
| **Risiko leakage** | Sedang |
| **Failure modes** | LLM hallucination; perlu validasi ketat |
| **Exact package** | `LLMFE` (GitHub)  |
| **Sumber** | TMLR 2026; arXiv:2503.14434 |

### 11.3 OCTree

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi LLM + decision tree reasoning, generate fitur tanpa predefined search space  |
| **Problem yang didukung** | Tabular prediction |
| **Kapan digunakan** | Butuh feature generation guided by decision tree |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Decision tree reasoning dapat guide LLM |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `tree_depth`, `llm_model` |
| **CPU/GPU/RAM** | API-based |
| **Risiko leakage** | Sedang |
| **Failure modes** | Search space tidak terkontrol |
| **Exact package** | Custom implementation  |
| **Sumber** | NeurIPS; arXiv:2406.xxxxx |

---

## 12. LEAKAGE-SAFE PRACTICES

### 12.1 Leakage-Safe Target Encoding

| Field | Detail |
|-------|--------|
| **Definisi** | Target encoding harus dihitung di dalam CV fold (K-fold target encoding / out-of-fold), tidak pernah dari seluruh train set  |
| **Problem yang didukung** | Semua problem dengan target encoding |
| **Kapan digunakan** | WAJIB untuk target encoding |
| **Kapan tidak digunakan** | N/A (selalu wajib) |
| **Asumsi** | OOF encoding mencegah leakage |
| **Data preprocessing** | K-fold split |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_folds`, `smoothing` |
| **Kisaran parameter awal** | n_folds: 5–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** jika tidak OOF |
| **Failure modes** | Leakage → overfitting validation |
| **Teknik pembanding** | CatBoost encoding (leakage-safe by design) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.KFold`; `h2o.targetencoder(data_leakage_handling='k_fold')`  |
| **Sumber** | scikit-learn docs; H2O docs |

### 12.2 Leakage-Safe Group Statistics

| Field | Detail |
|-------|--------|
| **Definisi** | Statistik agregasi (group mean dll) dihitung terpisah per fold |
| **Problem yang didukung** | Semua problem dengan groupby features |
| **Kapan digunakan** | WAJIB untuk groupby features |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Fold-isolated statistics mencegah leakage |
| **Data preprocessing** | K-fold split |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_folds` |
| **Kisaran parameter awal** | n_folds: 5 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Leakage dari group statistics |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `RobustModelMaker`  |
| **Sumber** | Semantic Scholar (2026) |

### 12.3 Leakage-Safe Scaling/Normalization

| Field | Detail |
|-------|--------|
| **Definisi** | Scaling/normalization di-fit hanya di train fold, transform ke validation/test |
| **Problem yang didukung** | Semua problem dengan scaling |
| **Kapan digunakan** | WAJIB untuk scaling |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Fit pada train fold mencegah leakage |
| **Data preprocessing** | Pipeline |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `scaler_type` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | **Kritis** jika fit pada full data |
| **Failure modes** | Data leakage → optimistic validation |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.pipeline.Pipeline` |
| **Sumber** | scikit-learn docs |

### 12.4 Leakage-Safe Time-Based Features

| Field | Detail |
|-------|--------|
| **Definisi** | Time-based feature (lag/rolling) harus strictly pakai data masa lalu relatif ke titik prediksi |
| **Problem yang didukung** | Time series forecasting |
| **Kapan digunakan** | WAJIB untuk time series |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Strictly past data mencegah look-ahead bias |
| **Data preprocessing** | Time-based split |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, MAE |
| **Hyperparameter penting** | `horizon` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis** |
| **Failure modes** | Look-ahead bias |
| **Ensemble compatibility** | Universal |
| **Exact package** | Pandas `shift`; `feature_engine.timeseries` |
| **Sumber** | Time series best practices |

---

## 13. DOMAIN-SPECIFIC FEATURES

### 13.1 Geospatial (Haversine, Geohash)

| Field | Detail |
|-------|--------|
| **Definisi** | **Haversine**: jarak great-circle antara dua titik koordinat. **Geohash**: encoding koordinat ke string grid  |
| **Problem yang didukung** | Geospatial prediction, Delivery ETA, Location-based |
| **Kapan digunakan** | Data dengan latitude/longitude |
| **Kapan tidak digunakan** | Tidak ada koordinat |
| **Asumsi** | Jarak geografis relevan |
| **Data preprocessing** | Koordinat dalam derajat |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `geohash_precision` |
| **Kisaran parameter awal** | precision: 5–7 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Koordinat salah; precision terlalu rendah |
| **Ensemble compatibility** | Universal |
| **Exact package** | `haversine` (PyPI); `geohash2` |
| **Contoh hypothesis** | "Haversine distance ke POI terdekat akan meningkatkan akurasi ETA ≥10%" |
| **Acceptance criteria** | RMSE improvement ≥10% |
| **Rejection criteria** | RMSE drop |
| **Sumber** | MDPI (2026); FIG patents |

### 13.2 Graph/Network (Degree, PageRank)

| Field | Detail |
|-------|--------|
| **Definisi** | **Degree centrality**: jumlah koneksi node. **PageRank**: importance score berdasarkan struktur graph  |
| **Problem yang didukung** | Graph-based prediction, Fraud detection, Social network |
| **Kapan digunakan** | Data dengan relasi graph |
| **Kapan tidak digunakan** | Tidak ada struktur graph |
| **Asumsi** | Struktur graph informatif |
| **Data preprocessing** | Graph construction |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `damping_factor` (PageRank) |
| **Kisaran parameter awal** | damping: 0.85 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; graph harus dari train data |
| **Failure modes** | Graph terlalu sparse/dense |
| **Ensemble compatibility** | Universal |
| **Exact package** | `networkx`; `graph-tool` |
| **Contoh hypothesis** | "PageRank centrality akan meningkatkan AUC fraud detection ≥5%" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Springer (2021); IEEE (2020) |

### 13.3 Image Embedding (Pretrained CNN)

| Field | Detail |
|-------|--------|
| **Definisi** | Fitur dari pretrained CNN sebagai embedding |
| **Problem yang didukung** | Tabular + image hybrid |
| **Kapan digunakan** | Ada kolom gambar dalam tabular |
| **Kapan tidak digunakan** | Tidak ada gambar |
| **Asumsi** | Pretrained CNN menangkap visual features |
| **Data preprocessing** | Image resizing, normalization |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `model_name`, `layer` |
| **Kisaran parameter awal** | ResNet50; layer='avg_pool' |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah (pretrained) |
| **Failure modes** | Domain mismatch |
| **Ensemble compatibility** | Universal |
| **Exact package** | `torchvision.models`; `tensorflow.keras.applications` |
| **Sumber** | Deep learning best practices |

### 13.4 TabLLM / Hybrid Tabular-NLP

| Field | Detail |
|-------|--------|
| **Definisi** | Menggabungkan representasi teks + numerik untuk tabular  |
| **Problem yang didukung** | Tabular dengan kolom teks |
| **Kapan digunakan** | Hybrid tabular-text data |
| **Kapan tidak digunakan** | Hanya numerik/kategorik |
| **Asumsi** | Teks dan numerik saling melengkapi |
| **Data preprocessing** | Text serialization |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `llm_model`, `template` |
| **Kisaran parameter awal** | T0; template='Text Template' |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Serialization salah |
| **Ensemble compatibility** | Universal |
| **Exact package** | `TabuLLM` (PyPI) ; `TabLLM` |
| **Contoh hypothesis** | "TabLLM akan mencapai akurasi ≥5% lebih tinggi dari GBM murni pada data dengan kolom teks" |
| **Acceptance criteria** | Accuracy improvement ≥5% |
| **Rejection criteria** | Accuracy drop |
| **Sumber** | TabLLM (2023); TabuLLM (2026) |

---

## Ringkasan Insight

| Kategori | Teknik Kunci | Biaya | Kapan Dipilih |
|----------|-------------|-------|---------------|
| **Numerical** | Box-Cox, GaussRank | Ringan | Skewed data, NN tabular |
| **Categorical** | Target encoding (K-fold), CatBoost | Sedang | High-cardinality |
| **Temporal** | Lag, Rolling, Fourier | Ringan–Sedang | Time series |
| **Aggregation** | Group stats, DFS | Sedang | Relational data |
| **Interaction** | Polynomial, AutoFeat | Sedang | Non-linear |
| **Missing** | KNN, MICE, MissForest | Sedang | Missing values |
| **Text** | TF-IDF, Embedding | Sedang | Text columns |
| **Selection** | Boruta, SHAP, Null importance | Sedang–Berat | High-dim |
| **Dim Reduction** | PCA, UMAP, Autoencoder | Sedang | High-dim |
| **Automated** | Featuretools, tsfresh, OpenFE | Sedang–Berat | Multi-tabel/time series |
| **LLM-based** | CAAFE, LLM-FE, OCTree | API-based | Semantic features |
| **Leakage-Safe** | OOF encoding, Fold-isolated stats | Wajib | Semua |
| **Domain** | Haversine, PageRank, CNN | Sedang | Geospatial/graph/image |

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*

# Feature Engineering — Dokumentasi Teknis Tier Berat (Edisi Lengkap)

> **Versi:** 2.0 | **Tier:** Berat (Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini merupakan perluasan dari versi sebelumnya dengan tambahan **advanced statistical feature selection, supervised dimension reduction, self-supervised representation, advanced time-series features, graph/spatial features, symbolic regression, causal/invariant feature selection**, dan **compressed interaction features**. Setiap entri mengikuti format 20-field untuk konsumsi AI agents.

---

## Daftar Isi (Tambahan)

14. [Advanced Statistical Feature Selection](#14-advanced-statistical-feature-selection)
15. [Structured Feature Selection (Group Lasso)](#15-structured-feature-selection-group-lasso)
16. [Nonlinear Basis Engineering (Spline, MARS)](#16-nonlinear-basis-engineering-spline-mars)
17. [Kernel Approximation Features](#17-kernel-approximation-features)
18. [Self-Supervised Tabular Representations](#18-self-supervised-tabular-representations)
19. [Advanced Numerical Embeddings](#19-advanced-numerical-embeddings)
20. [Supervised Dimension Reduction](#20-supervised-dimension-reduction)
21. [Manifold / Spectral Features](#21-manifold--spectral-features)
22. [Advanced Latent Feature Extraction](#22-advanced-latent-feature-extraction)
23. [Advanced Time-Series Features](#23-advanced-time-series-features)
24. [Model-Derived Supervised Features](#24-model-derived-supervised-features)
25. [Graph & Spatial Feature Learning](#25-graph--spatial-feature-learning)
26. [Symbolic & Robust Feature Learning](#26-symbolic--robust-feature-learning)

---

## 14. ADVANCED STATISTICAL FEATURE SELECTION

### 14.1 Stability Selection

| Field | Detail |
|-------|--------|
| **Definisi** | Framework feature selection yang menggabungkan subsampling dengan algoritma selection (Lasso, dll.) untuk memilih fitur berdasarkan **selection probability** across subsamples, bukan satu run. Memberikan finite-sample control terhadap false discovery rates  |
| **Problem yang didukung** | Semua problem dengan high-dimensional features |
| **Kapan digunakan** | High-dimensional; banyak correlated features; butuh FDR control; ingin stability selection |
| **Kapan tidak digunakan** | Dataset sangat kecil; features sudah terpilih dengan baik |
| **Asumsi** | Selection algorithm (Lasso) dapat dijalankan pada subsamples; exchangeability condition |
| **Data preprocessing** | Standar; scaling untuk Lasso |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Selection probability; FDR control |
| **Hyperparameter penting** | `n_subsamples`, `subsample_fraction`, `π_thr` (cutoff) |
| **Kisaran parameter awal** | n_subsamples: 100–500; subsample_fraction: 0.5; π_thr: 0.6–0.9 |
| **CPU/GPU/RAM** | n_subsamples × model cost |
| **Risiko leakage** | Sedang; selection per fold dalam CV |
| **Failure modes** | Cutoff terlalu rendah → false positives; terlalu tinggi → miss signal |
| **Teknik pembanding** | Boruta, Model-X Knockoffs, Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `stabs` (R); `stability-selection` (Python) |
| **Contoh hypothesis** | "Stability selection dengan π_thr=0.8 akan memilih subset fitur yang mencapai AUC setara full feature dengan 60% fitur" |
| **Acceptance criteria** | FDR terkontrol; AUC setara |
| **Rejection criteria** | FDR tidak terkontrol; AUC drop >2% |
| **Sumber** | Meinshausen & Bühlmann, JRSS-B 72(4), 417–473 (2010)  |

---

### 14.2 Model-X Knockoffs

| Field | Detail |
|-------|--------|
| **Definisi** | Framework variable selection yang menciptakan **knockoff features** artifisial sebagai negative control untuk mengontrol **False Discovery Rate (FDR)** secara non-asymptotic, tanpa asumsi pada dimensionality atau conditional distribution Y\|X  |
| **Problem yang didukung** | Scientific feature discovery; genomics; high-dimensional tabular |
| **Kapan digunakan** | Butuh formal FDR control; banyak correlated variables; interpretability dengan controlled discovery |
| **Kapan tidak digunakan** | Dataset sangat kecil; tidak butuh FDR control |
| **Asumsi** | Distribution covariates X diketahui atau dapat diestimasi; exchangeability condition |
| **Data preprocessing** | Standar; estimasi distribusi X (Gaussian, dll.) |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | FDR, power |
| **Hyperparameter penting** | `fdr_target`, `knockoff_method` |
| **Kisaran parameter awal** | fdr_target: 0.1–0.2 |
| **CPU/GPU/RAM** | Knockoff generation cost |
| **Risiko leakage** | Sedang; knockoffs dari training fold |
| **Failure modes** | Distribusi X salah → FDR tidak terkontrol |
| **Teknik pembanding** | Stability Selection, BH-FDR |
| **Ensemble compatibility** | Universal |
| **Exact package** | `knockoff` (R); `knockpy` (Python); `DeepLINK` |
| **Contoh hypothesis** | "Model-X knockoffs akan mengidentifikasi subset fitur dengan FDR ≤0.10 dan power ≥0.70" |
| **Acceptance criteria** | FDR ≤ target; power ≥0.60 |
| **Rejection criteria** | FDR > target; power <0.40 |
| **Sumber** | Candès et al., JRSS-B 80(3), 551–577 (2018)  |

---

### 14.3 mRMR (Minimum Redundancy Maximum Relevance)

| Field | Detail |
|-------|--------|
| **Definisi** | Kriteria feature selection information-theoretic yang memaksimalkan relevance terhadap target sekaligus meminimalkan redundancy antar fitur terpilih  |
| **Problem yang didukung** | Semua problem dengan high-dimensional features |
| **Kapan digunakan** | High-dimensional; banyak correlated features; MI available |
| **Kapan tidak digunakan** | Fitur independen; dataset kecil |
| **Asumsi** | Mutual information dapat diestimasi; relevance dan redundancy dapat diukur |
| **Data preprocessing** | Standar; discretization untuk MI |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | MI-based relevance; redundancy |
| **Hyperparameter penting** | `n_features`, `mi_estimator` |
| **Kisaran parameter awal** | n_features: 10–100 |
| **CPU/GPU/RAM** | O(n × p²) untuk MI |
| **Risiko leakage** | Sedang; selection per fold |
| **Failure modes** | MI estimation noisy untuk data kontinu |
| **Teknik pembanding** | MI filter, CMIM, FOCI |
| **Ensemble compatibility** | Universal |
| **Exact package** | `mrmr` (R); `pymrmr` (Python); `mrmr-selection` |
| **Contoh hypothesis** | "mRMR akan memilih subset fitur yang mencapai AUC ≥0.85 dengan 30 fitur" |
| **Acceptance criteria** | AUC improvement ≥3% vs MI filter |
| **Rejection criteria** | AUC drop; redundancy masih tinggi |
| **Sumber** | Peng et al., IEEE TPAMI 27(8), 1226–1238 (2005)  |

---

### 14.4 SIS / ISIS (Sure Independence Screening)

| Field | Detail |
|-------|--------|
| **Definisi** | Metode screening untuk ultrahigh-dimensional feature space (p >> n) yang menurunkan dimensi fitur menjadi manageable sebelum selection tahap kedua. **ISIS** adalah versi iteratif  |
| **Problem yang didukung** | p >> n; genomics; ultrahigh-dimensional tabular |
| **Kapan digunakan** | p sangat besar (>10.000); n relatif kecil |
| **Kapan tidak digunakan** | p < n; dataset kecil |
| **Asumsi** | Sure screening property: semua fitur relevan bertahan dengan probabilitas tinggi |
| **Data preprocessing** | Standar; scaling |
| **Dataset scale** | p >> n |
| **Metric compatibility** | Selection consistency |
| **Hyperparameter penting** | `d` (number of features to retain), `n_iterations` (ISIS) |
| **Kisaran parameter awal** | d: n/log(n) atau n-1 |
| **CPU/GPU/RAM** | Ringan (marginal correlation) |
| **Risiko leakage** | Sedang; screening per fold |
| **Failure modes** | Fitur relevan dengan efek marginal lemah bisa terlewat |
| **Teknik pembanding** | Filter methods, Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `SIS` (R); `screening` (Python) |
| **Contoh hypothesis** | "SIS dengan d=n/log(n) akan mempertahankan semua fitur relevan pada p=100.000" |
| **Acceptance criteria** | Sure screening property terpenuhi |
| **Rejection criteria** | Fitur relevan terlewat |
| **Sumber** | Fan & Lv, JRSS-B 70(5), 849–911 (2008)  |

---

### 14.5 ReliefF / RReliefF

| Field | Detail |
|-------|--------|
| **Definisi** | Keluarga algoritma feature selection yang menggunakan **nearest neighbors** untuk mendeteksi conditional dependencies dan interactions antar fitur. **ReliefF** untuk klasifikasi; **RReliefF** untuk regresi  |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Fitur dengan conditional dependency; interaksi fitur; mixed data types |
| **Kapan tidak digunakan** | Dataset sangat besar (O(n²)); fitur independen |
| **Asumsi** | Nearest neighbors dapat diidentifikasi; local dependencies informatif |
| **Data preprocessing** | Scaling untuk distance metric |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Accuracy, F1, AUC |
| **Hyperparameter penting** | `k` (nearest neighbors), `n_iterations` |
| **Kisaran parameter awal** | k: 10–50; n_iterations: 100 |
| **CPU/GPU/RAM** | O(n²) untuk distance |
| **Risiko leakage** | Sedang; selection per fold |
| **Failure modes** | Curse of dimensionality; k salah |
| **Teknik pembanding** | MI filter, mRMR |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ReliefF` (R); `skrebate` (Python) |
| **Contoh hypothesis** | "ReliefF akan mendeteksi interaksi fitur yang terlewat oleh MI univariate" |
| **Acceptance criteria** | Interaction detection ≥1 pasang |
| **Rejection criteria** | Tidak ada improvement vs MI |
| **Sumber** | Robnik-Šikonja & Kononenko, Machine Learning 53(1-2), 23–69 (2003)  |

---

### 14.6 HSIC-Lasso

| Field | Detail |
|-------|--------|
| **Definisi** | Feature selection yang menggunakan **Hilbert-Schmidt Independence Criterion (HSIC)** untuk menangkap nonlinear statistical dependence, dikombinasikan dengan L1 regularization untuk sparsity  |
| **Problem yang didukung** | Nonlinear feature selection; genomics; high-dimensional |
| **Kapan digunakan** | Hubungan nonlinear antara fitur dan target; butuh sparsity |
| **Kapan tidak digunakan** | Hubungan linear; dataset kecil |
| **Asumsi** | Kernel dapat menangkap dependence; convex optimization |
| **Data preprocessing** | Standar; kernel computation |
| **Dataset scale** | Menengah |
| **Metric compatibility** | HSIC-based selection |
| **Hyperparameter penting** | `sigma` (kernel width), `lambda` (regularization) |
| **Kisaran parameter awal** | sigma: median heuristic; lambda: 0.01–1.0 |
| **CPU/GPU/RAM** | O(n²) untuk kernel matrix |
| **Risiko leakage** | Sedang; selection per fold |
| **Failure modes** | Kernel width salah; computational cost |
| **Teknik pembanding** | mRMR, FOCI, Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pyHSICLasso` (Python); `HSICLasso` (R) |
| **Contoh hypothesis** | "HSIC-Lasso akan mendeteksi dependence nonlinear yang terlewat oleh Lasso linear" |
| **Acceptance criteria** | Nonlinear feature detection ≥20% improvement |
| **Rejection criteria** | Tidak ada improvement vs Lasso |
| **Sumber** | Yamada et al., Neural Computation 26(1), 185–207 (2014)  |

---

### 14.7 FOCI (Feature Ordering by Conditional Independence)

| Field | Detail |
|-------|--------|
| **Definisi** | Algoritma feature selection model-free yang menggunakan **conditional dependence coefficient** untuk mengurutkan fitur berdasarkan informasi tambahan yang mereka berikan setelah fitur sebelumnya diketahui  |
| **Problem yang didukung** | Semua problem dengan high-dimensional features |
| **Kapan digunakan** | Butuh conditional independence; model-free; no tuning parameters |
| **Kapan tidak digunakan** | Dataset sangat besar (O(n²)) |
| **Asumsi** | Conditional dependence coefficient dapat diestimasi; sparsity |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Conditional dependence |
| **Hyperparameter penting** | `n_features` |
| **Kisaran parameter awal** | n_features: 10–100 |
| **CPU/GPU/RAM** | O(n²) |
| **Risiko leakage** | Sedang; selection per fold |
| **Failure modes** | Estimasi noisy untuk data kontinu |
| **Teknik pembanding** | mRMR, CMIM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `FOCI` (R) |
| **Contoh hypothesis** | "FOCI akan mengidentifikasi fitur redundan yang terlewat oleh MI univariate" |
| **Acceptance criteria** | Redundancy reduction ≥30% |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Azadkia & Chatterjee, Annals of Statistics 49(6), 3070–3102 (2021)  |

---

### 14.8 CMIM (Conditional Mutual Information Maximization)

| Field | Detail |
|-------|--------|
| **Definisi** | Feature selection yang memaksimalkan **conditional mutual information** I(Y;X_j \| X_selected) untuk mencari fitur yang masih memberikan informasi tentang target setelah conditioning terhadap fitur terpilih |
| **Problem yang didukung** | Klasifikasi, Regresi |
| **Kapan digunakan** | Redundansi antar fitur tinggi; butuh conditional selection |
| **Kapan tidak digunakan** | Dataset kecil; fitur independen |
| **Asumsi** | MI dapat diestimasi |
| **Data preprocessing** | Discretization untuk MI |
| **Dataset scale** | Menengah |
| **Metric compatibility** | MI-based |
| **Hyperparameter penting** | `n_features` |
| **Kisaran parameter awal** | n_features: 10–100 |
| **CPU/GPU/RAM** | O(n × p²) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Estimasi MI noisy |
| **Teknik pembanding** | mRMR, FOCI |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `pymrmr` |
| **Contoh hypothesis** | "CMIM akan memilih subset fitur yang lebih ringkas dari mRMR dengan performa setara" |
| **Acceptance criteria** | Feature count lebih kecil; performa setara |
| **Rejection criteria** | Performa drop |
| **Sumber** | Fleuret, JMLR 5, 1531–1552 (2004) |

---

### 14.9 Laplacian Score

| Field | Detail |
|-------|--------|
| **Definisi** | Unsupervised feature selection yang memilih fitur berdasarkan kemampuan mempertahankan **local neighborhood/manifold structure**  |
| **Problem yang didukung** | Unsupervised learning; clustering; representation learning |
| **Kapan digunakan** | Tidak ada target; unlabeled high-dimensional data |
| **Kapan tidak digunakan** | Supervised problem |
| **Asumsi** | Local manifold structure informatif |
| **Data preprocessing** | Scaling; graph construction (k-NN) |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Unsupervised selection |
| **Hyperparameter penting** | `k` (neighbors), `n_features` |
| **Kisaran parameter awal** | k: 5–15 |
| **CPU/GPU/RAM** | O(n²) untuk graph |
| **Risiko leakage** | Rendah |
| **Failure modes** | Graph construction salah |
| **Teknik pembanding** | Variance threshold, MCFS |
| **Ensemble compatibility** | Universal |
| **Exact package** | `Rdimtools` (R); `scikit-feature` (Python) |
| **Contoh hypothesis** | "Laplacian Score akan memilih fitur yang mempertahankan cluster structure" |
| **Acceptance criteria** | Clustering quality terjaga |
| **Rejection criteria** | Cluster structure rusak |
| **Sumber** | He, Cai & Niyogi, NeurIPS 2005  |

---

### 14.10 MCFS (Multi-Cluster Feature Selection)

| Field | Detail |
|-------|--------|
| **Definisi** | Unsupervised feature selection yang memilih fitur untuk mempertahankan **multi-cluster structure** dari data menggunakan spectral analysis dan sparse regularization  |
| **Problem yang didukung** | Unsupervised learning; clustering |
| **Kapan digunakan** | Unlabeled data; multi-cluster structure |
| **Kapan tidak digunakan** | Supervised; single cluster |
| **Asumsi** | Spectral structure informatif |
| **Data preprocessing** | Scaling; graph construction |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Clustering quality |
| **Hyperparameter penting** | `n_clusters`, `n_features` |
| **Kisaran parameter awal** | n_clusters: 3–10 |
| **CPU/GPU/RAM** | O(n³) untuk eigendecomposition |
| **Risiko leakage** | Rendah |
| **Failure modes** | Graph construction salah |
| **Teknik pembanding** | Laplacian Score |
| **Ensemble compatibility** | Universal |
| **Exact package** | `scikit-feature` (Python); custom |
| **Contoh hypothesis** | "MCFS akan memilih fitur yang lebih baik dari Laplacian Score pada data multi-cluster" |
| **Acceptance criteria** | Clustering improvement ≥10% |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Cai, Zhang & He, KDD 2010  |

---

## 15. STRUCTURED FEATURE SELECTION (GROUP LASSO)

### 15.1 Group Lasso

| Field | Detail |
|-------|--------|
| **Definisi** | Lasso yang memilih/membuang **satu grup fitur bersama-sama** menggunakan groupwise L2 norm penalty. Cocok untuk one-hot expansion, spline basis, polynomial family  |
| **Problem yang didukung** | Regression; classification dengan group structure |
| **Kapan digunakan** | One-hot expansion; spline basis; multi-level categorical; sensor channels |
| **Kapan tidak digunakan** | Fitur tidak memiliki group structure |
| **Asumsi** | Group structure diketahui; group-level sparsity |
| **Data preprocessing** | Standar; group definition |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `lambda`, `group_weights` |
| **Kisaran parameter awal** | lambda: 0.01–1.0 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; selection per fold |
| **Failure modes** | Group definition salah; lambda salah |
| **Teknik pembanding** | Lasso, Sparse Group Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `gglasso` (R); `group-lasso` (Python) |
| **Contoh hypothesis** | "Group Lasso akan memilih seluruh one-hot block untuk kategori penting" |
| **Acceptance criteria** | Group-level sparsity tercapai |
| **Rejection criteria** | Group tidak terpilih dengan benar |
| **Sumber** | Yuan & Lin, JRSS-B 68(1), 49–67 (2006)  |

---

### 15.2 Sparse Group Lasso

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi Group Lasso dan Lasso: memilih **group** sekaligus **fitur individual di dalam group** (sparsity antar-group dan intra-group)  |
| **Problem yang didukung** | Regression; classification dengan group structure |
| **Kapan digunakan** | Group structure; butuh sparsity intra-group |
| **Kapan tidak digunakan** | Tidak ada group structure |
| **Asumsi** | Group structure diketahui; convex combination of penalties |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alpha` (mix antara L1 dan group), `lambda` |
| **Kisaran parameter awal** | alpha: 0.5; lambda: 0.01–1.0 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Alpha salah |
| **Teknik pembanding** | Group Lasso, Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `SGL` (R); `sparsegl` (R); `group-lasso` (Python) |
| **Contoh hypothesis** | "Sparse Group Lasso akan memilih group penting dan fitur individual di dalamnya" |
| **Acceptance criteria** | Intra-group sparsity tercapai |
| **Rejection criteria** | Tidak ada sparsity intra-group |
| **Sumber** | Simon et al., JCGS 22(2), 231–245 (2013)  |

---

## 16. NONLINEAR BASIS ENGINEERING (SPLINE, MARS)

### 16.1 Spline Basis Features

| Field | Detail |
|-------|--------|
| **Definisi** | Membangun **piecewise polynomial basis** (B-spline, natural cubic spline, restricted cubic spline) untuk menangkap nonlinear smooth effects tanpa global high-degree polynomial |
| **Problem yang didukung** | Regression; classification dengan model linear/logistic |
| **Kapan digunakan** | Hubungan nonlinear smooth; model linear; interpretability |
| **Kapan tidak digunakan** | Tree-based models (sudah nonlinear); hubungan linear |
| **Asumsi** | Smooth function; knot locations |
| **Data preprocessing** | Scaling; knot selection |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_knots`, `degree`, `knot_locations` |
| **Kisaran parameter awal** | n_knots: 3–7; degree: 3 (cubic) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; knot locations dari training fold |
| **Failure modes** | Knot terlalu banyak → overfitting |
| **Teknik pembanding** | Polynomial features, MARS |
| **Ensemble compatibility** | Universal |
| **Exact package** | `patsy` (Python); `splines` (R); `sklearn.preprocessing.SplineTransformer` |
| **Contoh hypothesis** | "Restricted cubic spline dengan 5 knots akan mencapai R² ≥5% lebih tinggi dari linear regression" |
| **Acceptance criteria** | R² improvement ≥5% |
| **Rejection criteria** | R² drop; overfitting |
| **Sumber** | Harrell (2001); scikit-learn docs |

---

### 16.2 MARS (Multivariate Adaptive Regression Splines)

| Field | Detail |
|-------|--------|
| **Definisi** | Metode yang secara otomatis memilih **spline basis, knot locations, dan interactions** dari data menggunakan forward/backward stepwise subset selection  |
| **Problem yang didukung** | Regression; classification |
| **Kapan digunakan** | Hubungan nonlinear; interaksi; interpretability |
| **Kapan tidak digunakan** | Dataset sangat besar (computational cost) |
| **Asumsi** | Additive spline model; interactions dapat dideteksi |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | MSE, RMSE, R² |
| **Hyperparameter penting** | `degree`, `n_prune`, `penalty` |
| **Kisaran parameter awal** | degree: 1–3; n_prune: 5–20 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; model selection per fold |
| **Failure modes** | Overfitting; knot explosion |
| **Teknik pembanding** | Spline basis, GAM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `earth` (R); `pyearth` (Python); `sklearn-contrib-py-earth` |
| **Contoh hypothesis** | "MARS akan mencapai RMSE ≥10% lebih rendah dari linear regression pada data nonlinear" |
| **Acceptance criteria** | RMSE improvement ≥10% |
| **Rejection criteria** | RMSE drop; model terlalu kompleks |
| **Sumber** | Friedman, Annals of Statistics 19(1), 1–141 (1991)  |

---

## 17. KERNEL APPROXIMATION FEATURES

### 17.1 Random Fourier Features

| Field | Detail |
|-------|--------|
| **Definisi** | Explicit feature map z(x) yang mengaproksimasi shift-invariant kernel (RBF, Laplacian) sehingga linear methods dapat digunakan untuk large-scale nonlinear learning |
| **Problem yang didukung** | Nonlinear classification/regression dengan linear model |
| **Kapan digunakan** | Nonlinear relationship; linear model desirable; kernel matrix too expensive |
| **Kapan tidak digunakan** | Dataset kecil (exact kernel feasible) |
| **Asumsi** | Shift-invariant kernel; random projection preserves kernel |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `gamma` |
| **Kisaran parameter awal** | n_components: 100–1000; gamma: 1/n_features |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | n_components terlalu kecil → poor approximation |
| **Teknik pembanding** | Nyström, exact kernel |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.kernel_approximation.RBFSampler` |
| **Contoh hypothesis** | "Random Fourier Features dengan 500 components akan mencapai akurasi setara kernel SVM dengan training 10× lebih cepat" |
| **Acceptance criteria** | Akurasi setara; speedup ≥5× |
| **Rejection criteria** | Akurasi drop >3% |
| **Sumber** | Rahimi & Recht, NeurIPS 2007 |

---

### 17.2 Nyström Features

| Field | Detail |
|-------|--------|
| **Definisi** | Low-rank approximation dari kernel matrix menggunakan subset landmarks, menghasilkan explicit features |
| **Problem yang didukung** | Nonlinear classification/regression |
| **Kapan digunakan** | Kernel matrix O(n²) terlalu mahal; butuh explicit features |
| **Kapan tidak digunakan** | Dataset kecil |
| **Asumsi** | Kernel matrix low-rank |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components` (landmarks), `gamma` |
| **Kisaran parameter awal** | n_components: 100–1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Landmarks tidak representatif |
| **Teknik pembanding** | Random Fourier Features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.kernel_approximation.Nystroem` |
| **Contoh hypothesis** | "Nyström dengan 500 landmarks akan mencapai akurasi setara exact kernel" |
| **Acceptance criteria** | Akurasi setara |
| **Rejection criteria** | Akurasi drop >3% |
| **Sumber** | Williams & Seeger, NeurIPS 2001 |

---

### 17.3 TensorSketch / PolynomialCountSketch

| Field | Detail |
|-------|--------|
| **Definisi** | Approximate polynomial kernel dengan explicit randomized representation, menghindari feature explosion dari PolynomialFeatures  |
| **Problem yang didukung** | Polynomial/high-order interactions |
| **Kapan digunakan** | Butuh polynomial interactions; explicit PolynomialFeatures explodes |
| **Kapan tidak digunakan** | Dataset kecil; linear relationship |
| **Asumsi** | Polynomial kernel approximation |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `degree` |
| **Kisaran parameter awal** | n_components: 1000–10000; degree: 2–3 |
| **CPU/GPU/RAM** | O(n(d+D log D))  |
| **Risiko leakage** | Rendah |
| **Failure modes** | n_components terlalu kecil |
| **Teknik pembanding** | PolynomialFeatures, Random Fourier Features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.kernel_approximation.PolynomialCountSketch` |
| **Contoh hypothesis** | "PolynomialCountSketch dengan degree=2 akan mencapai akurasi setara PolynomialFeatures dengan memory 10× lebih kecil" |
| **Acceptance criteria** | Akurasi setara; memory ≤20% |
| **Rejection criteria** | Akurasi drop >3% |
| **Sumber** | Pham & Pagh, KDD 2013  |

---

## 18. SELF-SUPERVISED TABULAR REPRESENTATIONS

### 18.1 VIME

| Field | Detail |
|-------|--------|
| **Definisi** | Self-supervised framework untuk tabular data yang membuat **pretext task** memprediksi mask vector dari corrupted input, menghasilkan representation yang dapat digunakan downstream  |
| **Problem yang didukung** | Tabular classification/regression dengan unlabeled data |
| **Kapan digunakan** | Labels sedikit; unlabeled rows banyak |
| **Kapan tidak digunakan** | Dataset sangat kecil; labels cukup |
| **Asumsi** | Mask estimation task menghasilkan representation informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `mask_prob`, `encoder_dim`, `n_epochs` |
| **Kisaran parameter awal** | mask_prob: 0.3; encoder_dim: 100–500 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah; pretraining pada training data |
| **Failure modes** | Mask task terlalu mudah/sulit |
| **Teknik pembanding** | SCARF, Autoencoder |
| **Ensemble compatibility** | Universal |
| **Exact package** | `VIME` (GitHub)  |
| **Contoh hypothesis** | "VIME representation akan meningkatkan akurasi ≥5% vs supervised-only pada dataset dengan 10% labeled" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Yoon et al., NeurIPS 2020  |

---

### 18.2 SCARF

| Field | Detail |
|-------|--------|
| **Definisi** | Contrastive learning untuk tabular data di mana views dibentuk dengan **corrupting random subset of features**, dan representation learner dilatih untuk membuat representation dari corrupted view mirip dengan original  |
| **Problem yang didukung** | Tabular classification |
| **Kapan digunakan** | Labels sedikit; unlabeled rows banyak; contrastive pretraining |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Random feature corruption menghasilkan views yang informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `corruption_rate`, `temperature` |
| **Kisaran parameter awal** | corruption_rate: 0.6; temperature: 0.5 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Corruption rate salah |
| **Teknik pembanding** | VIME, SimCLR |
| **Ensemble compatibility** | Universal |
| **Exact package** | `SCARF` (GitHub)  |
| **Contoh hypothesis** | "SCARF representation akan mencapai akurasi ≥5% lebih tinggi dari supervised-only pada 69 dataset tabular" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Bahri et al., ICLR 2022  |

---

## 19. ADVANCED NUMERICAL EMBEDDINGS

### 19.1 Piecewise Linear Encoding (PLE)

| Field | Detail |
|-------|--------|
| **Definisi** | Representasi numerik untuk tabular DL di mana scalar x_j diubah menjadi vektor menggunakan **piecewise linear encoding** berdasarkan binning  |
| **Problem yang didukung** | Tabular deep learning |
| **Kapan digunakan** | Continuous features; tabular DL; ingin meningkatkan performa MLP |
| **Kapan tidak digunakan** | Tree-based models; dataset kecil |
| **Asumsi** | Binning boundaries predefined; PLE menghasilkan representation lebih baik dari scalar |
| **Data preprocessing** | Binning; encoding |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_bins`, `encode` |
| **Kisaran parameter awal** | n_bins: 10–50 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; bin edges dari training fold |
| **Failure modes** | n_bins salah |
| **Teknik pembanding** | Periodic embedding, Quantile transform |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ludwig` (number_encoders); custom  |
| **Contoh hypothesis** | "PLE dengan 20 bins akan meningkatkan akurasi MLP ≥3% vs raw scalar" |
| **Acceptance criteria** | Akurasi improvement ≥3% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Gorishniy et al., NeurIPS 2022  |

---

### 19.2 Periodic Numerical Embeddings

| Field | Detail |
|-------|--------|
| **Definisi** | Numerical embedding menggunakan **periodic activations** (sin/cos) untuk scalar values, menghasilkan representation high-dimensional yang kontinu |
| **Problem yang didukung** | Tabular deep learning |
| **Kapan digunakan** | Continuous features; periodic patterns |
| **Kapan tidak digunakan** | Tree-based models |
| **Asumsi** | Periodic representation menangkap struktur |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_frequencies`, `frequency_init` |
| **Kisaran parameter awal** | n_frequencies: 10–50 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Frequency salah |
| **Teknik pembanding** | PLE |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom  |
| **Contoh hypothesis** | "Periodic embedding akan mencapai akurasi setara PLE dengan representasi lebih ringkas" |
| **Acceptance criteria** | Akurasi setara |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Gorishniy et al., NeurIPS 2022  |

---

## 20. SUPERVISED DIMENSION REDUCTION

### 20.1 Supervised PCA

| Field | Detail |
|-------|--------|
| **Definisi** | PCA yang mempertimbangkan target: menyaring/mengarahkan variables berdasarkan hubungan terhadap outcome sebelum membangun principal components  |
| **Problem yang didukung** | High-dimensional prediction; regression/classification |
| **Kapan digunakan** | PCA gagal karena variance tidak terkait target; high-dimensional |
| **Kapan tidak digunakan** | Unsupervised; target tidak tersedia |
| **Asumsi** | Variasi relevan terhadap Y dapat diidentifikasi |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `threshold` |
| **Kisaran parameter awal** | n_components: 5–50 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; fitting per fold |
| **Failure modes** | Threshold salah |
| **Teknik pembanding** | PCA, PLS |
| **Ensemble compatibility** | Universal |
| **Exact package** | `superpc` (R)  |
| **Contoh hypothesis** | "Supervised PCA akan mencapai R² ≥10% lebih tinggi dari PCA pada high-dimensional regression" |
| **Acceptance criteria** | R² improvement ≥10% |
| **Rejection criteria** | R² drop |
| **Sumber** | Bair et al., JASA 101(473), 119–137 (2006)  |

---

### 20.2 Partial Least Squares (PLS)

| Field | Detail |
|-------|--------|
| **Definisi** | Latent-variable method yang mencari komponen dari X yang secara langsung berkaitan dengan response Y  |
| **Problem yang didukung** | Regression; classification dengan multicollinear/high-dimensional predictors |
| **Kapan digunakan** | Many numeric correlated features; regression; n limited |
| **Kapan tidak digunakan** | Features independent; n besar |
| **Asumsi** | Latent components predictive |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components` |
| **Kisaran parameter awal** | n_components: 2–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; fitting per fold |
| **Failure modes** | n_components salah |
| **Teknik pembanding** | PCR, Supervised PCA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.cross_decomposition.PLSRegression`  |
| **Contoh hypothesis** | "PLS dengan 5 components akan mencapai RMSE ≥10% lebih rendah dari OLS pada data multicollinear" |
| **Acceptance criteria** | RMSE improvement ≥10% |
| **Rejection criteria** | RMSE drop |
| **Sumber** | Wold et al., Chemometrics and Intelligent Laboratory Systems 58(2), 109–130 (2001)  |

---

### 20.3 Sliced Inverse Regression (SIR)

| Field | Detail |
|-------|--------|
| **Definisi** | Metode sufficient dimension reduction (SDR) pertama yang menggunakan **inverse regression** E(X\|Y) untuk menemukan central subspace  |
| **Problem yang didukung** | High-dimensional regression/classification |
| **Kapan digunakan** | Butuh low-dimensional subspace; linear conditional mean |
| **Kapan tidak digunakan** | Nonlinear conditional mean; dataset kecil |
| **Asumsi** | Linear conditional mean; constant variance |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_slices`, `n_components` |
| **Kisaran parameter awal** | n_slices: 5–20 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Slices salah; linearity condition gagal |
| **Teknik pembanding** | SAVE, PHD, MAVE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `dr` (R); `sliced` (Python)  |
| **Contoh hypothesis** | "SIR akan menemukan 2-dimensional subspace yang mencapai R² ≥0.80" |
| **Acceptance criteria** | R² ≥0.75 |
| **Rejection criteria** | R² <0.50 |
| **Sumber** | Li, JASA 86(414), 316–342 (1991)  |

---

### 20.4 SAVE (Sliced Average Variance Estimation)

| Field | Detail |
|-------|--------|
| **Definisi** | SDR method yang menggunakan **first two moments** dari inverse conditional expectation dan dapat menangkap struktur symmetric/nonlinear yang terlewat SIR  |
| **Problem yang didukung** | High-dimensional regression |
| **Kapan digunakan** | SIR gagal karena symmetric/nonlinear patterns |
| **Kapan tidak digunakan** | Linear relationship |
| **Asumsi** | Linearity and constant variance conditions |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_slices`, `n_components` |
| **Kisaran parameter awal** | n_slices: 5–20 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Conditions gagal |
| **Teknik pembanding** | SIR, PHD |
| **Ensemble compatibility** | Universal |
| **Exact package** | `dr` (R); `SAVE` (R)  |
| **Contoh hypothesis** | "SAVE akan menangkap curvature structure yang terlewat SIR" |
| **Acceptance criteria** | R² improvement ≥5% vs SIR |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Cook & Weisberg, JASA 86(414), 328–332 (1991)  |

---

### 20.5 MAVE (Minimum Average Variance Estimation)

| Field | Detail |
|-------|--------|
| **Definisi** | Adaptive semiparametric method untuk dimension reduction yang bekerja pada setting general dengan restrictions lebih sedikit pada covariate distribution  |
| **Problem yang didukung** | Nonlinear dimension reduction |
| **Kapan digunakan** | Nonlinear regression; ingin SDR yang fleksibel |
| **Kapan tidak digunakan** | Linear relationship |
| **Asumsi** | Semiparametric model; local smoothing |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `bandwidth` |
| **Kisaran parameter awal** | n_components: 1–5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Bandwidth salah |
| **Teknik pembanding** | SIR, SAVE, PHD |
| **Ensemble compatibility** | Universal |
| **Exact package** | `MAVE` (R)  |
| **Contoh hypothesis** | "MAVE akan mencapai R² ≥10% lebih tinggi dari SIR pada nonlinear regression" |
| **Acceptance criteria** | R² improvement ≥10% |
| **Rejection criteria** | R² drop |
| **Sumber** | Xia et al., JRSS-B 64(3), 363–410 (2002)  |

---

### 20.6 Principal Hessian Directions (PHD)

| Field | Detail |
|-------|--------|
| **Definisi** | SDR method yang mencari arah penting berdasarkan **curvature** dari regression surface (Hessian), bukan first-order association  |
| **Problem yang didukung** | Nonlinear regression; symmetric patterns |
| **Kapan digunakan** | Hubungan symmetric/nonlinear (Y=X²+ε) |
| **Kapan tidak digunakan** | Linear relationship |
| **Asumsi** | Stein's lemma; curvature informatif |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components` |
| **Kisaran parameter awal** | n_components: 1–3 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Curvature lemah |
| **Teknik pembanding** | SIR, SAVE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `dr` (R)  |
| **Contoh hypothesis** | "PHD akan mendeteksi curvature Y=X² yang terlewat SIR" |
| **Acceptance criteria** | Curvature direction terdeteksi |
| **Rejection criteria** | Tidak terdeteksi |
| **Sumber** | Li, JASA 87(420), 1025–1039 (1992)  |

---

### 20.7 Kernel Dimension Reduction (KDR)

| Field | Detail |
|-------|--------|
| **Definisi** | Nonlinear supervised dimension reduction yang mencari effective subspace sehingga Y ⊥ X \| B^TX, diukur melalui covariance operators di RKHS  |
| **Problem yang didukung** | Nonlinear SDR |
| **Kapan digunakan** | Nonlinear/kernelized supervised SDR; tidak butuh parametric conditional distribution |
| **Kapan tidak digunakan** | Dataset sangat besar (O(n³)) |
| **Asumsi** | RKHS; conditional independence |
| **Data preprocessing** | Scaling; kernel computation |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `kernel_width` |
| **Kisaran parameter awal** | n_components: 1–5 |
| **CPU/GPU/RAM** | O(n³) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Kernel width salah |
| **Teknik pembanding** | SIR, MAVE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `KDR` (R); custom  |
| **Contoh hypothesis** | "KDR akan mencapai R² ≥10% lebih tinggi dari SIR pada nonlinear SDR" |
| **Acceptance criteria** | R² improvement ≥10% |
| **Rejection criteria** | R² drop |
| **Sumber** | Fukumizu et al., Annals of Statistics 37(4), 1871–1905 (2009)  |

---

## 21. MANIFOLD / SPECTRAL FEATURES

### 21.1 Diffusion Maps

| Field | Detail |
|-------|--------|
| **Definisi** | Metode manifold learning yang menggunakan **diffusion process** pada neighborhood graph dan eigenfunctions sebagai coordinates, menggambarkan intrinsic multiscale geometry  |
| **Problem yang didukung** | Nonlinear dimensionality reduction; manifold features |
| **Kapan digunakan** | High-dimensional nonlinear manifold; local connectivity meaningful |
| **Kapan tidak digunakan** | Data linear |
| **Asumsi** | Manifold structure; diffusion distance informatif |
| **Data preprocessing** | Scaling; graph construction |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `k` (neighbors), `t` (diffusion time) |
| **Kisaran parameter awal** | n_components: 5–20; k: 5–15 |
| **CPU/GPU/RAM** | O(n³) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Graph construction salah |
| **Teknik pembanding** | Laplacian Eigenmaps, UMAP |
| **Ensemble compatibility** | Universal |
| **Exact package** | `diffusionMap` (R); `pydiffmap` (Python)  |
| **Contoh hypothesis** | "Diffusion maps akan menemukan manifold structure yang terlewat PCA" |
| **Acceptance criteria** | Manifold structure terdeteksi |
| **Rejection criteria** | Struktur tidak terdeteksi |
| **Sumber** | Coifman & Lafon, Applied and Computational Harmonic Analysis 21(1), 5–30 (2006)  |

---

### 21.2 Laplacian Eigenmaps

| Field | Detail |
|-------|--------|
| **Definisi** | Manifold learning yang menggunakan **graph Laplacian eigenvectors** sebagai low-dimensional representation untuk mempertahankan local neighborhood geometry  |
| **Problem yang didukung** | Nonlinear dimensionality reduction |
| **Kapan digunakan** | High-dimensional; local neighborhood meaningful |
| **Kapan tidak digunakan** | Data linear |
| **Asumsi** | Manifold structure; local neighborhoods |
| **Data preprocessing** | Scaling; graph construction |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `k` (neighbors) |
| **Kisaran parameter awal** | n_components: 5–20; k: 5–15 |
| **CPU/GPU/RAM** | O(n³) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Graph construction salah |
| **Teknik pembanding** | Diffusion Maps, Isomap |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.manifold.SpectralEmbedding`  |
| **Contoh hypothesis** | "Laplacian Eigenmaps akan mencapai clustering quality ≥10% lebih tinggi dari PCA" |
| **Acceptance criteria** | Clustering improvement ≥10% |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Belkin & Niyogi, Neural Computation 15(6), 1373–1396 (2003)  |

---

## 22. ADVANCED LATENT FEATURE EXTRACTION

### 22.1 Sparse PCA

| Field | Detail |
|-------|--------|
| **Definisi** | PCA dengan **sparse loadings** menggunakan Lasso/Elastic-Net constraints sehingga principal components lebih interpretable  |
| **Problem yang didukung** | Interpretable dimensionality reduction |
| **Kapan digunakan** | PCA components sulit diinterpretasi; ingin sparse loadings |
| **Kapan tidak digunakan** | Interpretability tidak penting |
| **Asumsi** | Sparse loading structure |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `alpha` (sparsity) |
| **Kisaran parameter awal** | n_components: 5–20; alpha: 0.01–0.1 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Alpha salah |
| **Teknik pembanding** | PCA, ICA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.decomposition.SparsePCA`  |
| **Contoh hypothesis** | "Sparse PCA akan menghasilkan komponen dengan ≤10 fitur per komponen" |
| **Acceptance criteria** | Sparsity tercapai |
| **Rejection criteria** | Tidak sparse |
| **Sumber** | Zou et al., JCGS 15(2), 265–286 (2006)  |

---

### 22.2 ICA (Independent Component Analysis)

| Field | Detail |
|-------|--------|
| **Definisi** | Mencari **statistically independent components** dari multivariate data non-Gaussian  |
| **Problem yang didukung** | Source separation; feature extraction |
| **Kapan digunakan** | Observasi mixture dari latent sources |
| **Kapan tidak digunakan** | Gaussian data |
| **Asumsi** | Non-Gaussian sources; statistical independence |
| **Data preprocessing** | Centering; whitening |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `algorithm` (FastICA) |
| **Kisaran parameter awal** | n_components: 5–20 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Gaussian data |
| **Teknik pembanding** | PCA, NMF |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.decomposition.FastICA`  |
| **Contoh hypothesis** | "ICA akan memisahkan latent signals yang terlewat PCA" |
| **Acceptance criteria** | Source separation quality |
| **Rejection criteria** | Tidak terpisah |
| **Sumber** | Hyvärinen & Oja, Neural Networks 13(4-5), 411–430 (2000)  |

---

### 22.3 NMF (Non-Negative Matrix Factorization)

| Field | Detail |
|-------|--------|
| **Definisi** | Faktorisasi X ≈ WH dengan W,H ≥0, menghasilkan **parts-based representation** yang interpretable  |
| **Problem yang didukung** | Counts; frequency; purchase matrices; TF-IDF |
| **Kapan digunakan** | Nonnegative data; parts-based representation |
| **Kapan tidak digunakan** | Data dengan negative values |
| **Asumsi** | X ≥0; parts-based structure |
| **Data preprocessing** | Nonnegative scaling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `init` |
| **Kisaran parameter awal** | n_components: 5–20 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | n_components salah |
| **Teknik pembanding** | PCA, ICA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.decomposition.NMF`  |
| **Contoh hypothesis** | "NMF akan menghasilkan latent features yang lebih interpretable dari PCA pada count data" |
| **Acceptance criteria** | Interpretability improvement |
| **Rejection criteria** | Tidak interpretable |
| **Sumber** | Lee & Seung, Nature 401, 788–791 (1999)  |

---

### 22.4 Random Projection

| Field | Detail |
|-------|--------|
| **Definisi** | Dimensionality reduction menggunakan **random projection matrix** R, Z=XR, berdasarkan Johnson-Lindenstrauss lemma yang menjamin pairwise distances approximately preserved  |
| **Problem yang didukung** | Ultra-high-dimensional data |
| **Kapan digunakan** | p sangat besar; PCA terlalu mahal; interpretability tidak penting |
| **Kapan tidak digunakan** | Interpretability penting; p kecil |
| **Asumsi** | Johnson-Lindenstrauss lemma |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components` |
| **Kisaran parameter awal** | n_components: 100–1000 |
| **CPU/GPU/RAM** | Sangat ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | n_components terlalu kecil |
| **Teknik pembanding** | PCA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.random_projection.GaussianRandomProjection`  |
| **Contoh hypothesis** | "Random projection akan mencapai akurasi setara PCA dengan speedup ≥10× pada p=100.000" |
| **Acceptance criteria** | Akurasi setara; speedup ≥10× |
| **Rejection criteria** | Akurasi drop >3% |
| **Sumber** | Johnson & Lindenstrauss (1984)  |

---

## 23. ADVANCED TIME-SERIES FEATURES

### 23.1 ROCKET / MiniRocket

| Field | Detail |
|-------|--------|
| **Definisi** | Transformasi time series menggunakan **ribuan random convolutional kernels** dan mengambil summary features (max, proportion positive), menghasilkan ~20k fitur untuk linear classifier  |
| **Problem yang didukung** | Time series classification |
| **Kapan digunakan** | Time series data; butuh state-of-the-art accuracy |
| **Kapan tidak digunakan** | Data non-sequential |
| **Asumsi** | Random convolutional features informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Accuracy, F1 |
| **Hyperparameter penting** | `num_kernels`, `kernel_lengths` |
| **Kisaran parameter awal** | num_kernels: 10.000–20.000 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; fitur dari training fold |
| **Failure modes** | num_kernels terlalu kecil |
| **Teknik pembanding** | tsfresh, catch22 |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sktime` (ROCKET); `pycatch22`  |
| **Contoh hypothesis** | "ROCKET akan mencapai akurasi setara deep learning dengan training 100× lebih cepat" |
| **Acceptance criteria** | Akurasi setara; speedup ≥50× |
| **Rejection criteria** | Akurasi drop >5% |
| **Sumber** | Dempster et al., KDD 2021  |

---

### 23.2 catch22

| Field | Detail |
|-------|--------|
| **Definisi** | 22 **canonical time-series characteristics** yang dipilih dari 4.791 filtered hctsa features, mencakup autocorrelation, distribution properties, successive differences, outlier behavior  |
| **Problem yang didukung** | Time series classification/regression |
| **Kapan digunakan** | Butuh compact TS features; 1000× pengurangan computation |
| **Kapan tidak digunakan** | Butuh broad scientific feature discovery (gunakan hctsa) |
| **Asumsi** | 22 features representatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Features tidak cukup |
| **Teknik pembanding** | tsfresh, hctsa |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pycatch22` (Python); `catch22` (R)  |
| **Contoh hypothesis** | "catch22 akan mencapai akurasi ≥95% dari full hctsa dengan 1000× lebih cepat" |
| **Acceptance criteria** | Akurasi ≥95% dari hctsa |
| **Rejection criteria** | Akurasi <90% |
| **Sumber** | Lubba et al., DMKD 33(6), 1821–1852 (2019)  |

---

### 23.3 hctsa

| Field | Detail |
|-------|--------|
| **Definisi** | Framework massive feature extraction dengan **>7.700 time-series features** dari literatur ilmiah  |
| **Problem yang didukung** | Time series phenotyping; scientific discovery |
| **Kapan digunakan** | Butuh broad scientific feature discovery |
| **Kapan tidak digunakan** | Butuh compact features (gunakan catch22) |
| **Asumsi** | Massive feature pool informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `feature_set` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Sedang |
| **Failure modes** | Feature explosion |
| **Teknik pembanding** | catch22, tsfresh |
| **Ensemble compatibility** | Universal |
| **Exact package** | `hctsa` (MATLAB/Python)  |
| **Contoh hypothesis** | "hctsa akan menemukan fitur yang tidak terdeteksi oleh catch22" |
| **Acceptance criteria** | Additional predictive features |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Fulcher & Jones, Cell Systems 5(5), 527–531 (2017)  |

---

### 23.4 Shapelet Features

| Field | Detail |
|-------|--------|
| **Definisi** | Subsequence time series yang **discriminative** terhadap target; engineered feature = distance/similarity terhadap shapelet  |
| **Problem yang didukung** | Time series classification |
| **Kapan digunakan** | Butuh interpretable discriminative subsequences |
| **Kapan tidak digunakan** | Data non-sequential |
| **Asumsi** | Discriminative subsequences exist |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Accuracy, F1 |
| **Hyperparameter penting** | `n_shapelets`, `shapelet_length` |
| **Kisaran parameter awal** | n_shapelets: 10–50 |
| **CPU/GPU/RAM** | O(n²) untuk shapelet search |
| **Risiko leakage** | Sedang |
| **Failure modes** | Shapelet tidak informatif |
| **Teknik pembanding** | ROCKET, Matrix Profile |
| **Ensemble compatibility** | Universal |
| **Exact package** | `tslearn` (Python); `shapelets` (R)  |
| **Contoh hypothesis** | "Shapelet features akan mencapai akurasi ≥0.85 pada time series classification" |
| **Acceptance criteria** | Akurasi ≥0.80 |
| **Rejection criteria** | Akurasi <0.70 |
| **Sumber** | Ye & Keogh, KDD 2009  |

---

### 23.5 Matrix Profile Features

| Field | Detail |
|-------|--------|
| **Definisi** | Data structure yang menghitung **nearest-neighbor distance** untuk setiap subsequence, memungkinkan motif, discord, dan novelty discovery  |
| **Problem yang didukung** | Time series motif/discord discovery; anomaly detection |
| **Kapan digunakan** | Sensor/fraud/activity sequences |
| **Kapan tidak digunakan** | Data non-sequential |
| **Asumsi** | Motif/discord patterns informatif |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `subsequence_length` |
| **Kisaran parameter awal** | subsequence_length: 10–100 |
| **CPU/GPU/RAM** | O(n²) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Subsequence length salah |
| **Teknik pembanding** | Shapelets, ROCKET |
| **Ensemble compatibility** | Universal |
| **Exact package** | `stumpy` (Python)  |
| **Contoh hypothesis** | "Matrix profile features akan mendeteksi anomaly yang terlewat oleh statistical features" |
| **Acceptance criteria** | Anomaly detection improvement ≥10% |
| **Rejection criteria** | Tidak ada improvement |
| **Sumber** | Yeh et al., ICDM 2016  |

---

### 23.6 Wavelet Scattering

| Field | Detail |
|-------|--------|
| **Definisi** | Hierarchical wavelet transform dengan modulus dan aggregation yang menghasilkan **translation-invariant, multiscale** representation stabil terhadap small deformations  |
| **Problem yang didukung** | Signal/time-series classification |
| **Kapan digunakan** | Sensor, ECG, accelerometer, vibration, audio-like signals |
| **Kapan tidak digunakan** | Data non-sequential |
| **Asumsi** | Multiscale structure |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_scales`, `n_orientations` |
| **Kisaran parameter awal** | n_scales: 3–5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Scales salah |
| **Teknik pembanding** | Fourier, Wavelet coefficients |
| **Ensemble compatibility** | Universal |
| **Exact package** | `kymatio` (Python)  |
| **Contoh hypothesis** | "Wavelet scattering akan mencapai akurasi ≥5% lebih tinggi dari Fourier features pada sensor data" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Mallat, Communications on Pure and Applied Mathematics 65(10), 1331–1398 (2012)  |

---

## 24. MODEL-DERIVED SUPERVISED FEATURES

### 24.1 GBDT Leaf-Index Features

| Field | Detail |
|-------|--------|
| **Definisi** | Menggunakan GBDT untuk mentransformasi fitur menjadi **leaf indices** sebagai categorical/sparse features, merepresentasikan nonlinear feature interactions yang dipelajari tree  |
| **Problem yang didukung** | Click prediction; classification |
| **Kapan digunakan** | Butuh nonlinear feature crosses; GBDT+LR hybrid |
| **Kapan tidak digunakan** | Dataset kecil; tree sudah cukup |
| **Asumsi** | Tree leaves merepresentasikan interactions |
| **Data preprocessing** | Standar; GBDT training |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_estimators`, `max_depth` |
| **Kisaran parameter awal** | n_estimators: 100–500; max_depth: 3–7 |
| **CPU/GPU/RAM** | GBDT training cost |
| **Risiko leakage** | **Tinggi**; leaf features harus OOF |
| **Failure modes** | Leakage jika tidak OOF |
| **Teknik pembanding** | Polynomial features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `lightgbm` (leaf indices); custom  |
| **Contoh hypothesis** | "GBDT+LR akan mencapai AUC ≥0.02 lebih tinggi dari LR-only pada click prediction" |
| **Acceptance criteria** | AUC improvement ≥0.02 |
| **Rejection criteria** | AUC drop |
| **Sumber** | He et al., Facebook (2014)  |

---

## 25. GRAPH & SPATIAL FEATURE LEARNING

### 25.1 node2vec / DeepWalk

| Field | Detail |
|-------|--------|
| **Definisi** | **DeepWalk**: random walks sebagai "sentences" untuk belajar latent representation node. **node2vec**: biased random walk dengan parameter p dan q untuk menyeimbangkan exploration  |
| **Problem yang didukung** | Graph-based prediction; social network; fraud detection |
| **Kapan digunakan** | Data dengan relasi graph; butuh learned node embeddings |
| **Kapan tidak digunakan** | Tidak ada struktur graph |
| **Asumsi** | Random walk neighborhoods informatif |
| **Data preprocessing** | Graph construction |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `walk_length`, `n_walks`, `p`, `q` |
| **Kisaran parameter awal** | walk_length: 40–80; n_walks: 10; p=q=1 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; graph dari training data |
| **Failure modes** | Graph terlalu sparse/dense |
| **Teknik pembanding** | PageRank, metapath2vec |
| **Ensemble compatibility** | Universal |
| **Exact package** | `node2vec` (Python); `DeepWalk` (Python)  |
| **Contoh hypothesis** | "node2vec akan meningkatkan AUC fraud detection ≥5% vs hand-crafted graph features" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Perozzi et al., KDD 2014; Grover & Leskovec, KDD 2016  |

---

### 25.2 metapath2vec

| Field | Detail |
|-------|--------|
| **Definisi** | Embedding untuk **heterogeneous network** menggunakan meta-path guided random walks sebelum skip-gram embedding  |
| **Problem yang didukung** | Heterogeneous relational data |
| **Kapan digunakan** | Multiple node/edge types; meta-paths known |
| **Kapan tidak digunakan** | Homogeneous graph |
| **Asumsi** | Meta-paths menangkap semantic relationships |
| **Data preprocessing** | Meta-path definition |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `meta_paths`, `walk_length` |
| **Kisaran parameter awal** | walk_length: 100 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Meta-path salah |
| **Teknik pembanding** | node2vec, graph2vec |
| **Ensemble compatibility** | Universal |
| **Exact package** | `torch_geometric.nn.models.MetaPath2Vec`  |
| **Contoh hypothesis** | "metapath2vec akan mencapai akurasi ≥5% lebih tinggi dari node2vec pada heterogeneous network" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Dong et al., KDD 2017  |

---

### 25.3 graph2vec

| Field | Detail |
|-------|--------|
| **Definisi** | Unsupervised fixed-length distributed representation untuk **arbitrary-sized graphs**  |
| **Problem yang didukung** | Graph-level prediction |
| **Kapan digunakan** | Unit prediksi adalah whole graph (molecule, patient interaction) |
| **Kapan tidak digunakan** | Node-level prediction |
| **Asumsi** | Graph structure informatif |
| **Data preprocessing** | Graph construction |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_dimensions`, `n_epochs` |
| **Kisaran parameter awal** | n_dimensions: 64–128 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Graph terlalu besar |
| **Teknik pembanding** | Node2vec |
| **Ensemble compatibility** | Universal |
| **Exact package** | `graph2vec` (Python)  |
| **Contoh hypothesis** | "graph2vec akan mencapai akurasi ≥0.80 pada graph classification" |
| **Acceptance criteria** | Akurasi ≥0.75 |
| **Rejection criteria** | Akurasi <0.60 |
| **Sumber** | Narayanan et al., arXiv:1707.05005 (2017)  |

---

### 25.4 Moran Eigenvector Maps (MEM)

| Field | Detail |
|-------|--------|
| **Definisi** | Menghasilkan **orthogonal spatial descriptors** dari spatial-distance/connectivity matrices; setiap eigenvector merepresentasikan spatial pattern/scale tertentu  |
| **Problem yang didukung** | Spatial prediction |
| **Kapan digunakan** | Data dengan latitude/longitude; spatial autocorrelation |
| **Kapan tidak digunakan** | Data non-spatial |
| **Asumsi** | Spatial autocorrelation signifikan |
| **Data preprocessing** | Spatial weight matrix |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_eigenvectors` |
| **Kisaran parameter awal** | n_eigenvectors: 10–50 |
| **CPU/GPU/RAM** | O(n³) |
| **Risiko leakage** | Sedang |
| **Failure modes** | Spatial matrix salah |
| **Teknik pembanding** | Haversine, geohash |
| **Ensemble compatibility** | Universal |
| **Exact package** | `adespatial` (R); `pMEM`  |
| **Contoh hypothesis** | "MEM akan meningkatkan R² ≥10% vs raw lat/lon pada spatial prediction" |
| **Acceptance criteria** | R² improvement ≥10% |
| **Rejection criteria** | R² drop |
| **Sumber** | Dray et al., Ecology 87(10), 2614–2625 (2006); Guénard et al., MEE (2024)  |

---

## 26. SYMBOLIC & ROBUST FEATURE LEARNING

### 26.1 Symbolic Regression (PySR)

| Field | Detail |
|-------|--------|
| **Definisi** | Mencari **mathematical expressions** dari data menggunakan evolutionary search, menghasilkan fitur engineered yang interpretable  |
| **Problem yang didukung** | Scientific discovery; tabular regression/classification |
| **Kapan digunakan** | Butuh interpretable feature expressions; domain knowledge |
| **Kapan tidak digunakan** | Dataset sangat besar (computational cost) |
| **Asumsi** | Expressions dapat ditemukan; overfitting dikontrol |
| **Data preprocessing** | Standar |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_iterations`, `max_depth`, `binary_operators` |
| **Kisaran parameter awal** | n_iterations: 100; max_depth: 5 |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | **Tinggi**; symbolic search harus di dalam training CV |
| **Failure modes** | Overfitting expressions |
| **Teknik pembanding** | AutoFeat, OpenFE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `PySR` (Python)  |
| **Contoh hypothesis** | "PySR akan menemukan expression yang meningkatkan R² ≥5% vs linear regression" |
| **Acceptance criteria** | R² improvement ≥5% |
| **Rejection criteria** | R² drop; overfitting |
| **Sumber** | Cranmer, arXiv:2305.01582 (2023)  |

---

### 26.2 Anchor Regression

| Field | Detail |
|-------|--------|
| **Definisi** | Robust regression yang mengoptimalkan predictor dengan penalty yang membuat residual **insensitive terhadap anchor-driven distribution changes**  |
| **Problem yang didukung** | Robust prediction under distribution shift |
| **Kapan digunakan** | Multiple environments; expected production shift; linear regime |
| **Kapan tidak digunakan** | Single environment; non-linear |
| **Asumsi** | Anchor variable tersedia; linear/approximately linear |
| **Data preprocessing** | Standar |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `gamma` (anchor penalty) |
| **Kisaran parameter awal** | gamma: 0–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | gamma salah |
| **Teknik pembanding** | Invariant Causal Prediction |
| **Ensemble compatibility** | Universal |
| **Exact package** | `anchor` (R)  |
| **Contoh hypothesis** | "Anchor regression akan mencapai worst-environment R² ≥10% lebih tinggi dari OLS" |
| **Acceptance criteria** | Worst-environment improvement ≥10% |
| **Rejection criteria** | Improvement <5% |
| **Sumber** | Rothenhäusler et al., JRSS-B 83(2), 215–246 (2021)  |

---

### 26.3 Invariant Causal Prediction (ICP)

| Field | Detail |
|-------|--------|
| **Definisi** | Mencari predictors yang hubungan terhadap target-nya **stabil across environments/interventions**  |
| **Problem yang didukung** | Robust feature selection under distribution shift |
| **Kapan digunakan** | Multiple environments/domains/time periods; distribution shift concern |
| **Kapan tidak digunakan** | Single environment |
| **Asumsi** | Environments known; invariant relationship |
| **Data preprocessing** | Standard; environment labeling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alpha` (significance) |
| **Kisaran parameter awal** | alpha: 0.05 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Environments tidak representatif |
| **Teknik pembanding** | Anchor Regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ICP` (R)  |
| **Contoh hypothesis** | "ICP akan memilih fitur yang robust across environments" |
| **Acceptance criteria** | Stable feature set |
| **Rejection criteria** | Fitur tidak stabil |
| **Sumber** | Peters et al., JRSS-B 78(5), 947–1012 (2016)  |

---

### 26.4 Markov Blanket Feature Selection

| Field | Detail |
|-------|--------|
| **Definisi** | Mencari **Markov Blanket** dari target Y (parents, children, parents of children) sebagai compact predictive subset  |
| **Problem yang didukung** | Causal feature selection |
| **Kapan digunakan** | Causal discovery; robust feature selection |
| **Kapan tidak digunakan** | Tidak ada causal structure |
| **Asumsi** | Bayesian network; conditional independence tests |
| **Data preprocessing** | Discretization untuk CI tests |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alpha` (CI test) |
| **Kisaran parameter awal** | alpha: 0.01–0.05 |
| **CPU/GPU/RAM** | O(n × p²) |
| **Risiko leakage** | Sedang |
| **Failure modes** | CI test error |
| **Teknik pembanding** | ICP, Lasso |
| **Ensemble compatibility** | Universal |
| **Exact package** | `bnlearn` (R); `pcalg` (R)  |
| **Contoh hypothesis** | "Markov blanket selection akan memilih subset fitur yang lebih compact dari Lasso dengan performa setara" |
| **Acceptance criteria** | Feature count lebih kecil; performa setara |
| **Rejection criteria** | Performa drop |
| **Sumber** | Yu et al., Information Sciences 589, 849–877 (2022)  |

---

### 26.5 CCA / Sparse CCA

| Field | Detail |
|-------|--------|
| **Definisi** | Mencari projections u=Xa dan v=Zb yang memaksimalkan correlation Corr(u,v), menghasilkan **shared latent features** antar modalities/views ; **Sparse CCA** menambahkan sparsity  |
| **Problem yang didukung** | Multi-view feature fusion |
| **Kapan digunakan** | Dua feature blocks; multi-modal data |
| **Kapan tidak digunakan** | Single view |
| **Asumsi** | Shared information antar views |
| **Data preprocessing** | Scaling per view |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_components`, `penalty` (sparse) |
| **Kisaran parameter awal** | n_components: 1–5 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Views tidak berkorelasi |
| **Teknik pembanding** | PLS |
| **Ensemble compatibility** | Universal |
| **Exact package** | `cca-zoo` (Python); `PMA` (R)  |
| **Contoh hypothesis** | "Sparse CCA akan menemukan shared latent features yang meningkatkan akurasi ≥5% vs single view" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Witten et al., Biostatistics 10(3), 515–534 (2009)  |

---

## Ringkasan Prioritas Tambahan

| Prioritas | Teknik | Keluarga |
|-----------|--------|----------|
| 1 | Stability Selection | Advanced Statistical FS |
| 2 | Model-X Knockoffs | Advanced Statistical FS |
| 3 | mRMR | Advanced Statistical FS |
| 4 | ReliefF / RReliefF | Advanced Statistical FS |
| 5 | SIS / ISIS | Advanced Statistical FS |
| 6 | Spline Basis + MARS | Nonlinear Basis |
| 7 | VIME + SCARF | Self-Supervised |
| 8 | ROCKET / MiniRocket | Time-Series |
| 9 | node2vec / DeepWalk | Graph |
| 10 | Invariant Causal Prediction | Robust/Causal |
| 11 | Group/Sparse Group Lasso | Structured Selection |
| 12 | Random Fourier Features / Nyström | Kernel Approximation |
| 13 | PLE / Periodic Embeddings | Numerical Embeddings |
| 14 | Supervised PCA / PLS / SIR / SAVE | Supervised DR |
| 15 | HSIC-Lasso / FOCI / CMIM | Information-Theoretic FS |
| 16 | Diffusion Maps / Laplacian Eigenmaps | Manifold |
| 17 | Sparse PCA / ICA / NMF / Random Projection | Latent Extraction |
| 18 | GBDT Leaf-Index | Model-Derived |
| 19 | Matrix Profile / Shapelets / Wavelet Scattering | Time-Series |
| 20 | Anchor Regression / Markov Blanket / CCA | Robust/Causal/Multi-View |

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*