# Domain Target Decomposition — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk 5 pola dekomposisi target pada dataset tabular. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

1. [Frequency-Severity Decomposition](#1-frequency-severity-decomposition)
2. [Zero-Inflated / Hurdle Models](#2-zero-inflated--hurdle-models)
3. [Hierarchical Target Decomposition](#3-hierarchical-target-decomposition)
4. [Exposure Normalization](#4-exposure-normalization)
5. [Survival / Time-to-Event Framing](#5-survival--time-to-event-framing)

---

## 1. FREQUENCY-SEVERITY DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Pola dekomposisi di mana target akhir (pure premium = total loss cost per policy) didekomposisi menjadi dua sub-model independen: model frequency (prediksi jumlah klaim) dan model severity (prediksi rata-rata biaya per klaim), yang hasilnya dikalikan: **Ŝ = N̂ × Ŷ** |
| **Problem yang didukung** | Regresi, Insurance pricing, Pure premium prediction, Risk modeling |
| **Kapan digunakan** | Target = total cost dengan banyak zeros; klaim terjadi secara terpisah; distribusi target right-skewed dengan point mass di nol; butuh interpretability frequency vs severity |
| **Kapan tidak digunakan** | Target kontinu tanpa struktur count × severity; dataset sangat kecil (tidak cukup untuk dua model); target sudah terdistribusi normal |
| **Asumsi** | Frequency dan severity independen kondisional terhadap fitur; jumlah klaim mengikuti distribusi count (Poisson/NB); severity positif dan right-skewed (Gamma/Lognormal) |
| **Data preprocessing** | Feature engineering standar; exposure column diperlukan untuk frequency model; tidak ada scaling untuk GBM, scaling untuk GLM; one-hot encoding untuk kategorik |
| **Dataset scale** | Menengah–Besar (10K–10M+ baris); minimal 1000 klaim positif untuk severity model |
| **Metric compatibility** | MSE, MAE, Poisson deviance (frequency), Gamma deviance (severity), Tweedie deviance (combined), Pearson correlation, Lift (decile) |
| **Hyperparameter penting** | `freq_model_type` (Poisson GLM/GBM), `sev_model_type` (Gamma GLM/GBM), `alpha` (regularization), `n_estimators`, `learning_rate`, `max_depth` |
| **Kisaran parameter awal** | Poisson GLM: alpha=1e-2; GBM: n_estimators=100–500, lr=0.01–0.1, max_depth=3–7; Gamma GLM: alpha=1e-2 |
| **CPU/GPU/RAM** | CPU-bound; RAM ~2–3× dataset; training 2× model (frequency + severity); parallelizable |
| **Risiko leakage** | Sedang; severity model hanya dilatih pada klaim positif — pastikan split konsisten; exposure harus di-handle dengan benar |
| **Failure modes** | Severity model tidak cukup data (klaim jarang); dependency antara frequency dan severity diabaikan; calibration error pada interval; extrapolation buruk untuk severity tinggi |
| **Teknik pembanding** | Direct Tweedie modeling (single model), XGBoost direct total loss |
| **Ensemble compatibility** | Universal; dapat digabung dengan stacking (frequency dan severity sebagai base models); conformal prediction untuk interval |
| **Exact package** | `sklearn.linear_model.PoissonRegressor`; `sklearn.linear_model.GammaRegressor`; `sklearn.linear_model.TweedieRegressor`; `xgboost.XGBRegressor`; `lightgbm.LGBMRegressor`; `catboost.CatBoostRegressor` |
| **Contoh hypothesis** | "Frequency-severity decomposition dengan XGBoost (freq) + Gamma GLM (sev) akan mencapai Poisson deviance ≥10% lebih rendah dari single Tweedie model pada dataset French Motor Third-Party Liability" |
| **Acceptance criteria** | Poisson deviance freq model < 0.5; Gamma deviance sev model < 1.0; combined Tweedie deviance improvement ≥5% vs direct model |
| **Rejection criteria** | Severity model tidak konvergen; combined deviance drop > 5% vs direct; frequency model AUC < 0.6 |
| **Sumber** | Quan et al., Prob. Eng. Inf. Sci. 37(2) (2023); Nguyen, Mathematics 14(10), 1640 (2026); scikit-learn Tweedie example; burning-cost.github.io (2026) |

### Sub-varian: Direct Tweedie Modeling

| Field | Detail |
|-------|--------|
| **Definisi** | Model tunggal yang langsung memprediksi pure premium menggunakan distribusi Tweedie (Compound Poisson-Gamma). Tweedie variance power ρ ∈ (1,2) secara natural menghasilkan point mass di nol dan ekor kontinu positif |
| **Problem yang didukung** | Regresi, Insurance pricing |
| **Kapan digunakan** | Frequency dan severity tidak dapat dipisahkan dengan baik; butuh single model; data exposure tersedia |
| **Kapan tidak digunakan** | Butuh interpretability frequency vs severity terpisah |
| **Hyperparameter penting** | `power` (Tweedie variance power), `alpha` (regularization) |
| **Kisaran parameter awal** | power: 1.2–1.8 (grid: 1.2, 1.5, 1.7, 1.8); alpha: 1e-4–1e-2 |
| **Exact package** | `sklearn.linear_model.TweedieRegressor`; `glum`; `h2o` (tweedie_power) |
| **Contoh hypothesis** | "Tweedie model dengan power=1.5 akan mencapai Tweedie deviance setara frequency-severity decomposition dengan satu model" |
| **Sumber** | scikit-learn Tweedie example; MDPI Mathematics (2026); Dunn & Smyth (2008) |

---

## 2. ZERO-INFLATED / HURDLE MODELS

### 2.1 Hurdle Model (Two-Part Model)

| Field | Detail |
|-------|--------|
| **Definisi** | Model dua tahap di mana semua zero berasal dari satu proses "non-participation". Tahap 1: binary classification untuk prediksi apakah count > 0. Tahap 2: model count untuk prediksi nilai positif (conditional on count > 0) |
| **Problem yang didukung** | Klasifikasi biner + Regresi count, Zero-inflated data, Insurance claims, Road crash frequency |
| **Kapan digunakan** | >30% zero dan ada "never claimers" sejati; zero berasal dari satu proses; data count dengan excess zeros |
| **Kapan tidak digunakan** | Zero berasal dari dua sumber berbeda (gunakan ZIP); data kontinu tanpa zero |
| **Asumsi** | Semua zero dari proses non-participation yang sama; proses partisipasi dan count independen |
| **Data preprocessing** | Standar; class imbalance handling untuk stage 1 (custom loss, class weights); SHAP untuk interpretability |
| **Dataset scale** | Menengah–Besar (10K–1M+) |
| **Metric compatibility** | AUC (stage 1), Poisson deviance (stage 2), MAE, RMSE |
| **Hyperparameter penting** | `classifier_type` (stage 1), `regressor_type` (stage 2), `class_weight`, `n_estimators`, `learning_rate` |
| **Kisaran parameter awal** | CatBoost: iterations=500–1000, lr=0.03–0.1, depth=4–8; class_weight='balanced' |
| **CPU/GPU/RAM** | CPU + GPU (CatBoost); RAM ~2–3× dataset; training 2× model |
| **Risiko leakage** | Sedang; stage 1 dan stage 2 harus dilatih pada split yang sama; pastikan tidak ada leakage dari stage 2 ke stage 1 |
| **Failure modes** | Class imbalance ekstrem di stage 1; stage 2 tidak cukup data positif; korelasi antar stage diabaikan |
| **Teknik pembanding** | ZIP Model, Tweedie, Frequency-Severity |
| **Ensemble compatibility** | Universal; dapat digabung dengan stacking |
| **Exact package** | `catboost.CatBoostClassifier` + `CatBoostRegressor`; `pscl::zeroinfl` (R); `statsmodels` |
| **Contoh hypothesis** | "Hurdle model dengan CatBoost (2-stage) akan mencapai akurasi klasifikasi ≥5% lebih tinggi dari single Poisson model pada data road crash frequency dengan >60% zero" |
| **Acceptance criteria** | Stage 1 AUC ≥0.75; stage 2 Poisson deviance improvement ≥5%; overall MAE improvement ≥10% |
| **Rejection criteria** | Stage 1 AUC <0.65; stage 2 tidak konvergen; overall MAE drop |
| **Sumber** | Ben Khedher & Yun, Appl. Sci. 14(23), 10790 (2024) |

### 2.2 Zero-Inflated Model (ZIP/ZINB)

| Field | Detail |
|-------|--------|
| **Definisi** | Model yang mengasumsikan zero dapat berasal dari dua sumber: **structural zeros** (customer yang tidak mungkin positif) dan **sampling zeros** (customer yang bisa positif tapi tidak kali ini). ZIP menggabungkan binary inflation model dengan Poisson count model |
| **Problem yang didukung** | Regresi count, Zero-inflated data, Healthcare admission, Insurance claims |
| **Kapan digunakan** | Zero berasal dari dua proses berbeda; butuh membedakan structural vs sampling zeros; excess zero signifikan |
| **Kapan tidak digunakan** | Zero dari satu proses (gunakan Hurdle); data tanpa zero |
| **Asumsi** | Dua proses terpisah: structural zeros (binary) dan sampling zeros (Poisson) |
| **Data preprocessing** | Standar; scaling untuk NN-based |
| **Dataset scale** | Kecil–Menengah (1K–100K) |
| **Metric compatibility** | Poisson deviance, AIC/BIC, MAE, RMSE |
| **Hyperparameter penting** | `inflation_model` (logistic), `count_model` (Poisson), `max_iter` |
| **Kisaran parameter awal** | max_iter: 100–500; regularization: 1e-4–1e-2 |
| **CPU/GPU/RAM** | CPU-bound (GLM); GPU untuk NN extension |
| **Risiko leakage** | Sedang; kedua komponen harus di-fit pada training fold |
| **Failure modes** | Identifiability issues (kedua komponen mirip); convergence problems |
| **Teknik pembanding** | Hurdle Model, Negative Binomial, Tweedie |
| **Exact package** | `statsmodels.discrete.count_model.ZeroInflatedPoisson`; `pscl::zeroinfl` (R); `pscl` R package |
| **Contoh hypothesis** | "ZIP model akan mencapai AIC ≥10% lebih rendah dari Poisson GLM pada data admission counts dengan excess zero" |
| **Acceptance criteria** | AIC improvement ≥10%; kedua komponen konvergen |
| **Rejection criteria** | Tidak konvergen; AIC tidak improvement |
| **Sumber** | Jose et al., Ann. Actuarial Sci. 18(3), 644–674 (2024) |

### 2.3 ZIPNN / ZIPCANN (Neural Network Extension)

| Field | Detail |
|-------|--------|
| **Definisi** | **ZIPNN**: Zero-Inflated Poisson Neural Network — menggabungkan pendekatan zero-inflated Poisson dengan arsitektur neural network. **ZIPCANN**: Zero-Inflated Combined Actuarial Neural Network — menggabungkan ZIPNN dengan pendekatan CANN (skip connections, one-hot encoding) |
| **Problem yang didukung** | Regresi count dengan excess zero, Healthcare admission, Insurance |
| **Kapan digunakan** | Hubungan non-linear antara fitur dan count; butuh interpretability (LocalGLMnet); excess zero signifikan |
| **Kapan tidak digunakan** | Dataset kecil (<5000); butuh model sederhana |
| **Asumsi** | Neural network dapat menangkap non-linearitas; ZIP structure valid |
| **Data preprocessing** | One-hot encoding, min-max scaler |
| **Dataset scale** | Menengah–Besar (10K–1M) |
| **Metric compatibility** | Poisson deviance, MAE, RMSE, AUC |
| **Hyperparameter penting** | `n_layers`, `n_neurons` (20,15,10,2), `activation`, `learning_rate`, `dropout` |
| **Kisaran parameter awal** | n_neurons: (20, 15, 10, 2); lr: 0.001–0.01; dropout: 0.1–0.3 |
| **CPU/GPU/RAM** | GPU recommended; RAM moderate |
| **Risiko leakage** | Rendah; NN dilatih end-to-end |
| **Failure modes** | Overfitting pada dataset kecil; interpretability terbatas tanpa LocalGLMnet |
| **Teknik pembanding** | ZIP GLM, Hurdle CatBoost |
| **Exact package** | Custom implementation (TensorFlow/PyTorch); `LocalGLMnet` untuk interpretability |
| **Contoh hypothesis** | "ZIPCANN dengan arsitektur (20,15,10,2) akan mencapai Poisson deviance ≥5% lebih rendah dari ZIP GLM pada data admission counts" |
| **Acceptance criteria** | Deviance improvement ≥5%; konvergensi stabil |
| **Rejection criteria** | Deviance drop; overfitting terdeteksi |
| **Sumber** | Jose et al., Ann. Actuarial Sci. 18(3), 644–674 (2024) |

---

## 3. HIERARCHICAL TARGET DECOMPOSITION

### 3.1 Hierarchical RF-XGBoost

| Field | Detail |
|-------|--------|
| **Definisi** | Arsitektur dua lapis di mana Random Forest di Layer 1 menghasilkan residual clustering features, kemudian XGBoost di Layer 2 memanfaatkan fitur residual tersebut untuk menghasilkan prediksi akhir. Prediksi final menggabungkan hasil kedua layer |
| **Problem yang didukung** | Regresi, Time series forecasting, Multi-level prediction |
| **Kapan digunakan** | Target memiliki hierarki (total → kategori → sub); hubungan non-linear antar level; butuh ensemble multi-layer |
| **Kapan tidak digunakan** | Target flat tanpa hierarki; dataset sangat kecil |
| **Asumsi** | Residual dari Layer 1 mengandung informasi yang berguna untuk Layer 2; clustering residual dapat menangkap pola tersembunyi |
| **Data preprocessing** | Standar; target hierarchy harus didefinisikan dengan jelas |
| **Dataset scale** | Menengah–Besar (10K–1M) |
| **Metric compatibility** | MSE, MAE, MAPE, R² |
| **Hyperparameter penting** | `rf_n_estimators`, `rf_max_depth`, `xgb_n_estimators`, `xgb_max_depth`, `residual_clustering_method` |
| **Kisaran parameter awal** | RF: n_estimators=100–300, max_depth=5–15; XGBoost: n_estimators=100–500, lr=0.01–0.1, max_depth=3–7 |
| **CPU/GPU/RAM** | CPU-bound; training 2× model; RAM ~2–3× dataset |
| **Risiko leakage** | Sedang; Layer 1 harus di-train pada training fold; residual dari Layer 1 tidak boleh mengandung test data |
| **Failure modes** | Residual clustering tidak informatif; Layer 2 overfitting pada residual; korelasi antar level diabaikan |
| **Teknik pembanding** | Multi-Task Learning, Stacking |
| **Ensemble compatibility** | Universal; dapat digabung dengan stacking |
| **Exact package** | `sklearn.ensemble.RandomForestRegressor`; `xgboost.XGBRegressor` |
| **Contoh hypothesis** | "Hierarchical RF-XGBoost akan mencapai MAPE ≥10% lebih rendah dari XGBoost standalone pada dataset dengan hierarki target" |
| **Acceptance criteria** | MAPE improvement ≥10%; MAPE < 15% |
| **Rejection criteria** | MAPE drop; runtime >3× XGBoost standalone |
| **Sumber** | Pratama et al. (2024); AIMS Press (2025) |

### 3.2 Multi-Task Learning (MTL)

| Field | Detail |
|-------|--------|
| **Definisi** | Shared backbone (Transformer/TabNet) dengan task-specific heads untuk setiap level target. Output layer atas bisa dijadikan fitur tambahan untuk layer bawah, atau model dioptimasi dengan weighted multi-task loss |
| **Problem yang didukung** | Multi-target regression, Hierarchical prediction, Tabular |
| **Kapan digunakan** | Multiple target terkait; butuh shared representation; transfer learning antar target |
| **Kapan tidak digunakan** | Single target; target tidak terkait |
| **Asumsi** | Task saling terkait dan dapat saling memperkuat; shared representation bermanfaat |
| **Data preprocessing** | Standar; normalisasi target per task |
| **Dataset scale** | Menengah–Besar (10K–1M) |
| **Metric compatibility** | MSE, MAE per task; weighted average |
| **Hyperparameter penting** | `backbone_type` (TabNet/Transformer), `n_shared_layers`, `n_task_layers`, `loss_weights`, `task_losses` |
| **Kisaran parameter awal** | n_shared_layers: 2–4; n_task_layers: 1–3; loss_weights: uniform atau uncertainty-based |
| **CPU/GPU/RAM** | GPU recommended; RAM moderate–tinggi |
| **Risiko leakage** | Rendah; MTL dilatih end-to-end |
| **Failure modes** | Negative transfer (task saling mengganggu); loss imbalance antar task; overfitting pada task dengan data sedikit |
| **Teknik pembanding** | Separate models per target, Hierarchical RF-XGBoost |
| **Exact package** | PyTorch/TensorFlow custom; `MultiTab-Net`; `TabNetMultiTaskClassifier` |
| **Contoh hypothesis** | "MTL dengan TabNet backbone akan mencapai MSE ≥8% lebih rendah dari separate models per target pada dataset dengan 3 target terkait" |
| **Acceptance criteria** | MSE improvement ≥8% untuk mayoritas task |
| **Rejection criteria** | Negative transfer terdeteksi; MSE drop >2% untuk task manapun |
| **Sumber** | MultiTab (2024); Frontiers (2024); MDPI (2025) |

---

## 4. EXPOSURE NORMALIZATION

| Field | Detail |
|-------|--------|
| **Definisi** | Teknik yang menangani observasi dengan durasi/volume berbeda-beda, sehingga raw count tidak sebanding antar baris. Dalam formalisasi Poisson claims frequency, setiap policy i diobservasi sebagai tuple (xᵢ, cᵢ, vᵢ, yᵢ) di mana vᵢ adalah exposure. Model menargetkan E[Yᵢ \| xᵢ, vᵢ] = vᵢ · λ(xᵢ), sehingga yang dipelajari adalah rate λ per unit exposure |
| **Problem yang didukung** | Regresi count, Insurance claims frequency, Rate modeling |
| **Kapan digunakan** | Durasi observasi berbeda-beda antar baris; raw count tidak sebanding; Poisson/NB distribution |
| **Kapan tidak digunakan** | Semua observasi memiliki exposure sama; data kontinu tanpa count |
| **Asumsi** | Expected count scales linearly dengan exposure; log link function |
| **Data preprocessing** | Exposure column harus tersedia; log(exposure) sebagai offset atau exposure sebagai sample_weight |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Poisson deviance, MAE, RMSE, Rate-based metrics |
| **Hyperparameter penting** | `exposure_column`, `offset_type` (log_offset vs sample_weight), `model_type` |
| **Kisaran parameter awal** | N/A (tergantung model) |
| **CPU/GPU/RAM** | Sama dengan model base |
| **Risiko leakage** | Rendah; exposure bukan target |
| **Failure modes** | Exposure = 0 (log(0) error); exposure tidak proporsional dengan count; overdispersion tidak ditangani |
| **Teknik pembanding** | Frequency-Severity, Tweedie |
| **Ensemble compatibility** | Universal; kompatibel dengan semua model count |
| **Exact package** | `sklearn.linear_model.PoissonRegressor` (sample_weight); `statsmodels` (offset); `xgboost` (base_margin/offset); `lightgbm` (init_score); `h2o` (offset_column) |
| **Contoh hypothesis** | "Poisson regression dengan log(exposure) sebagai offset akan mencapai Poisson deviance ≥15% lebih rendah dari Poisson regression tanpa offset pada data claims frequency" |
| **Acceptance criteria** | Poisson deviance improvement ≥15%; convergence stabil |
| **Rejection criteria** | Deviance drop; exposure tidak signifikan |
| **Sumber** | scikit-learn Tweedie example; XGBoost documentation; dergipark.org.tr |

### Implementasi per Framework

| Framework | Cara Implementasi | Catatan |
|-----------|-------------------|---------|
| **scikit-learn PoissonRegressor** | `sample_weight=Exposure` | Model `y = ClaimNb / Exposure` dengan exposure sebagai weight |
| **statsmodels GLM** | `offset=np.log(Exposure)` | Model `Count ~ x + offset(log(Exposure))` |
| **XGBoost** | `base_margin=np.log(Exposure)` | Objective Poisson dengan base_margin sebagai offset |
| **LightGBM** | `init_score=np.log(Exposure)` | Sama dengan XGBoost |
| **H2O** | `offset_column="Exposure"` | Built-in offset support |

---

## 5. SURVIVAL / TIME-TO-EVENT FRAMING

### 5.1 XGBoost-Surv (AFT & Cox)

| Field | Detail |
|-------|--------|
| **Definisi** | Modifikasi objective function XGBoost untuk mendukung survival analysis. **AFT (Accelerated Failure Time)**: ln(Y) = T(x) + σZ, di mana T(x) adalah output tree ensemble dan label diekspresikan dalam interval [lower, upper] untuk menangani data tersensor. **Cox**: Proportional hazards model dengan XGBoost sebagai linear predictor |
| **Problem yang didukung** | Survival analysis, Time-to-event prediction, Churn prediction, Credit risk |
| **Kapan digunakan** | Target = waktu sampai event; ada data tersensor (tidak semua event terobservasi); butuh prediksi hazard/risk |
| **Kapan tidak digunakan** | Tidak ada data tersensor; target kontinu biasa |
| **Asumsi** | **AFT**: distribusi log-time dapat dimodelkan dengan tree ensemble. **Cox**: proportional hazards assumption |
| **Data preprocessing** | Time dan status columns; interval labels untuk AFT ([lower, upper]); uncensored = [a, a] |
| **Dataset scale** | Menengah–Besar (10K–1M) |
| **Metric compatibility** | C-index (concordance), Integrated Brier Score (IBS), L1/L2 loss, AUC |
| **Hyperparameter penting** | `objective` ('survival:aft' atau 'survival:cox'), `aft_loss_distribution` (normal/logistic/extreme), `aft_loss_distribution_scale`, `n_estimators`, `learning_rate`, `max_depth` |
| **Kisaran parameter awal** | n_estimators: 100–500; lr: 0.01–0.1; max_depth: 3–7; aft_loss_distribution='normal'; scale=1.0 |
| **CPU/GPU/RAM** | CPU + GPU (xgboost); RAM ~2–3× dataset |
| **Risiko leakage** | **Tinggi**; time-based split wajib; pastikan tidak ada future data dalam training |
| **Failure modes** | Censoring rate terlalu tinggi (>90%); AFT distribution salah; proportional hazards violation (Cox) |
| **Teknik pembanding** | CoxBoost, Random Survival Forest (RSF), DeepSurv, DeepHit |
| **Ensemble compatibility** | Universal; dapat digabung dengan stacking (risk scores sebagai base models) |
| **Exact package** | `xgboost` (objective='survival:aft'/'survival:cox'); `mlsurvlrnrs::LearnerSurvXgboostAft` (R); `mlr3proba` (R); `scikit-survival` (Python); `lifelines` (Python) |
| **Contoh hypothesis** | "XGBoost-AFT dengan normal distribution akan mencapai C-index ≥0.02 lebih tinggi dari CoxPH pada data high-dimensional survival" |
| **Acceptance criteria** | C-index improvement ≥0.02; IBS improvement ≥5% |
| **Rejection criteria** | C-index drop; konvergensi gagal |
| **Sumber** | Barnwal et al., arXiv (2022); R `fastml` package (2026); R `mlsurvlrnrs` (2025); De Bin et al. (2024) |

### 5.2 Benchmark Metode Survival

| Metode | Definisi | Kelebihan | Kekurangan |
|--------|----------|-----------|------------|
| **CoxBoost** | Likelihood-based boosting untuk Cox model | Performa terkuat di antara boosted models; interpretable | Mengasumsikan proportional hazards |
| **XGBoost-AFT** | AFT model dengan XGBoost | Tidak memerlukan proportional hazards; fleksibel | Butuh pilih distribusi AFT |
| **XGBoost-Cox** | Cox model dengan XGBoost linear predictor | Interpretable; well-established | Proportional hazards assumption |
| **Random Survival Forest** | Random Forest untuk survival | Robust; non-parametric | Kurang efisien untuk high-dimensional |
| **DeepSurv** | Deep learning untuk Cox | Menangkap non-linearitas kompleks | Butuh data besar; black-box |

**Temuan benchmark:**
- CoxBoost adalah performa terkuat di antara model boosted
- XGBoost dengan linear model-based boosting mengungguli versi tree-based dalam hampir semua kasus pada data high-dimensional
- SHAP digunakan untuk menjelaskan prediksi individual dan hubungan non-linear antara prediktor dan outcome

---

## Ringkasan Pemilihan Pola

| Kondisi Data | Pola yang Tepat | Alasan |
|--------------|----------------|--------|
| Target = total cost, banyak zeros, klaim terpisah | **Frequency-Severity** | Interpretability; dua proses berbeda |
| Target = total cost, ingin single model | **Direct Tweedie** | Satu model; Compound Poisson-Gamma natural |
| >>30% zero, ada "never claimers" sejati | **Zero-Inflated / Hurdle** | Membedakan structural vs sampling zeros |
| Target punya hierarki (total → kategori → sub) | **Hierarchical MTL** | Shared representation; transfer antar level |
| Durasi observasi berbeda-beda antar baris | **Exposure Normalization** | Rate modeling; linear scaling |
| Target = waktu sampai event, ada data tersensor | **Survival / Time-to-Event** | Censoring handling; hazard modeling |

---

## Decision Matrix untuk AI Agent

| Sinyal Data | Pola yang Direkomendasikan | Prioritas |
|-------------|---------------------------|-----------|
| Target: total cost; >50% zero; ada kolom "jumlah klaim" | Frequency-Severity | Tinggi |
| Target: total cost; ingin single model; power parameter tersedia | Direct Tweedie | Sedang |
| Target: count; >60% zero; ada dua proses zero | Zero-Inflated / Hurdle | Tinggi |
| Target: multiple level (total, kategori, sub) | Hierarchical MTL | Sedang |
| Ada kolom exposure/durasi yang bervariasi | Exposure Normalization | **Wajib** |
| Target: time-to-event; ada status tersensor | Survival / Time-to-Event | Tinggi |
| Target: time-to-event; proportional hazards valid | XGBoost-Cox | Sedang |
| Target: time-to-event; proportional hazards tidak valid | XGBoost-AFT | Tinggi |

---

## Referensi

1. Quan, Z., Wang, Z., Gan, G., & Valdez, E. A. (2023). On hybrid tree-based methods for short-term insurance claims. *Probability in the Engineering and Informational Sciences*, 37(2), 597-620.
2. Nguyen, N. (2026). Machine Learning and Frequency-Severity Decomposition for Insurance Pricing. *Mathematics*, 14(10), 1640.
3. Ben Khedher, M. B., & Yun, D. (2024). An Interpretable Machine Learning-Based Hurdle Model for Zero-Inflated Road Crash Frequency Data Analysis. *Applied Sciences*, 14(23), 10790.
4. Jose, A., Macdonald, A. S., Tzougas, G., & Streftaris, G. (2024). Interpretable zero-inflated neural network models for predicting admission counts. *Annals of Actuarial Science*, 18(3), 644-674.
5. Barnwal, A., Cho, H., & Hocking, T. (2022). Survival Regression with Accelerated Failure Time Model in XGBoost. *arXiv:2203.07091*.
6. scikit-learn developers. (2025). Tweedie regression on insurance claims. scikit-learn examples.
7. burning-cost.github.io. (2026). Conformal Prediction for Insurance Python: A Frequency-Severity Tutorial.
8. R `fastml` package. (2026). xgboost_survival.R. RDRR.io.
9. R `mlsurvlrnrs` package. (2025). LearnerSurvXgboostAft. RDRR.io.
10. MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data. (2024). arXiv.
11. Frontiers. (2024). Cross-validated tree-based models for multi-target learning.
12. H2O documentation. tweedie_power parameter.

---

# Domain Target Decomposition — Dokumentasi Teknis Tier Berat (Edisi Lengkap)

> **Versi:** 2.0 | **Tier:** Berat (Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini merupakan perluasan dari versi sebelumnya dengan tambahan **12+ family dekomposisi target** yang mencakup dependent frequency-severity, semicontinuous two-part, bounded/inflated proportion, censored targets, competing risks, recurrent events, multi-state survival, joint longitudinal-survival, hierarchical reconciliation, multi-target chains, compositional targets, ordinal threshold decomposition, cure models, frailty, interval-censored survival, semi-competing risks, Heckman sample selection, body-tail EVT splicing, reporting-delay/IBNR, multivariate zero-inflated targets, latent-class mixture regression, dan zero-and-one inflated counts.

---

## Daftar Isi

1. Frequency-Severity Decomposition
2. Zero-Inflated / Hurdle Models
3. Hierarchical Target Decomposition
4. Exposure Normalization
5. Survival / Time-to-Event Framing
6. **Dependent Frequency-Severity**
7. **Generic Semicontinuous Two-Part Model**
8. **Zero/One-Inflated Proportion Decomposition**
9. **Censored / Truncated Target Regression**
10. **Competing Risks**
11. **Recurrent-Event Decomposition**
12. **Multi-State Survival**
13. **Joint Longitudinal-Survival**
14. **Hierarchical Forecast Reconciliation**
15. **Multi-Target Chains**
16. **Compositional Target Modeling**
17. **Ordinal Threshold Decomposition**
18. **Mixture Cure / Non-Mixture Cure Models**
19. **Frailty / Shared Frailty Survival**
20. **Interval-Censored Survival**
21. **Semi-Competing Risks**
22. **Heckman / Sample-Selection Decomposition**
23. **Body-Tail / EVT Splicing**
24. **Reporting-Delay / IBNR Decomposition**
25. **Multivariate Zero-Inflated Targets**
26. **Latent-Class / Mixture Regression**
27. **Zero-and-One Inflated Counts**

---

## 6. DEPENDENT FREQUENCY-SEVERITY

### 6.1 Count-as-Covariate Conditional Severity

| Field | Detail |
|-------|--------|
| **Definisi** | Frequency-severity decomposition di mana **count (N) dijadikan covariate** pada severity model, sehingga dependence antara frequency dan severity dimodelkan secara eksplisit |
| **Problem yang didukung** | Insurance pricing; Risk modeling; Claims data |
| **Kapan digunakan** | Frequency dan severity tidak independen; count tersedia saat prediksi severity; dependence signifikan |
| **Kapan tidak digunakan** | Frequency dan severity independen (standard FS cukup) |
| **Asumsi** | Severity conditional on N dan X; dependence via N |
| **Data preprocessing** | Frequency model dulu, prediksi N, masukkan sebagai fitur severity |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Poisson deviance (freq), Gamma deviance (sev), combined Tweedie deviance |
| **Hyperparameter penting** | `freq_model`, `sev_model`, `include_N` |
| **Kisaran parameter awal** | N as linear/ spline covariate |
| **CPU/GPU/RAM** | 2× model cost |
| **Risiko leakage** | Sedang; N harus dari training fold |
| **Failure modes** | N leakage dari test; overfitting pada N |
| **Teknik pembanding** | Copula FS, Sarmanov FS, Longitudinal FS |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels`; `sklearn`; custom |
| **Contoh hypothesis** | "Count-as-covariate akan mencapai Tweedie deviance ≥5% lebih rendah dari standard FS pada data dengan dependence" |
| **Acceptance criteria** | Deviance improvement ≥5% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Garrido et al., Insurance: Mathematics and Economics (2016) |

### 6.2 Copula Frequency-Severity

| Field | Detail |
|-------|--------|
| **Definisi** | Frequency dan severity dimodelkan dengan **marginal distributions** masing-masing, lalu dependence dimodelkan via **copula** |
| **Problem yang didukung** | Insurance; Dependent FS |
| **Kapan digunakan** | Dependence non-linear; marginal distributions well-defined |
| **Kapan tidak digunakan** | Dependence linear sederhana |
| **Asumsi** | Marginal distributions correct; copula family sesuai |
| **Data preprocessing** | Fit marginal freq dan sev |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Tweedie deviance; copula fit statistics |
| **Hyperparameter penting** | `copula_family` (Gaussian, Clayton, Gumbel, Frank) |
| **Kisaran parameter awal** | Auto-select via AIC |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Copula misspecification |
| **Teknik pembanding** | Sarmanov, Count-as-covariate |
| **Ensemble compatibility** | Universal |
| **Exact package** | `copula` (R); `vinecopulib` |
| **Contoh hypothesis** | "Copula FS akan mencapai Tweedie deviance ≥10% lebih rendah dari standard FS pada data dengan tail dependence" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Shi et al., Insurance: Mathematics and Economics (2015) |

### 6.3 Sarmanov Frequency-Severity

| Field | Detail |
|-------|--------|
| **Definisi** | Dependence dimodelkan dengan **Sarmanov distribution** yang menghubungkan frequency dan average severity |
| **Problem yang didukung** | Insurance; Dependent FS |
| **Kapan digunakan** | Dependence moderate; butuh closed-form |
| **Kapan tidak digunakan** | Dependence ekstrem |
| **Asumsi** | Sarmanov family sesuai |
| **Dataset scale** | Menengah |
| **Hyperparameter penting** | `dependence_parameter` |
| **Exact package** | Custom; `Sarmanov` (R) |
| **Sumber** | Yang et al., Insurance: Mathematics and Economics (2021) |

### 6.4 Longitudinal Dependent FS

| Field | Detail |
|-------|--------|
| **Definisi** | FS untuk data longitudinal dengan dependence N_t ↔ S_t dan serial dependence N_t ↔ N_{t-1}, S_t ↔ S_{t-1} |
| **Problem yang didukung** | Insurance longitudinal; Panel data |
| **Kapan digunakan** | Repeated observations per entity; time dependence |
| **Kapan tidak digunakan** | Single period |
| **Asumsi** | Serial dependence; entity random effects |
| **Dataset scale** | Menengah–Besar |
| **Exact package** | Custom; `glmmTMB` (R) |
| **Sumber** | Lee & Shi, Insurance: Mathematics and Economics (2019) |

---

## 7. GENERIC SEMICONTINUOUS TWO-PART MODEL

| Field | Detail |
|-------|--------|
| **Definisi** | Two-part model untuk target **semicontinuous**: zero mass + positive continuous (bukan count). Stage 1: binary (0 vs >0). Stage 2: continuous positive distribution (Gamma/Lognormal/Generalized Gamma) |
| **Problem yang didukung** | Medical expenditure; Customer spending; Donation; Utilization |
| **Kapan digunakan** | Target = 0 atau positif kontinu; >30% zero; bukan count |
| **Kapan tidak digunakan** | Target count (gunakan Hurdle count); target kontinu tanpa zero |
| **Asumsi** | Dua proses; positive part continuous |
| **Data preprocessing** | Stage 1 binary; Stage 2 positive only |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | AUC (stage 1); MAE, RMSE (stage 2); combined metrics |
| **Hyperparameter penting** | `classifier_type`, `regressor_type`, `class_weight` |
| **Kisaran parameter awal** | CatBoost + Gamma GLM |
| **CPU/GPU/RAM** | 2× model cost |
| **Risiko leakage** | Sedang; split konsisten |
| **Failure modes** | Stage 2 tidak cukup data; dependence antar stage |
| **Teknik pembanding** | Tweedie, Hurdle count |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn`; `statsmodels`; `catboost` |
| **Contoh hypothesis** | "Logistic + Gamma akan mencapai MAE ≥10% lebih rendah dari Tweedie pada medical expenditure" |
| **Acceptance criteria** | MAE improvement ≥10% |
| **Rejection criteria** | MAE drop |
| **Sumber** | Belotti et al., Stata Journal (2015); PubMed (2018) |

**Sub-varian:**

| Varian | Stage 1 | Stage 2 |
|--------|---------|---------|
| **Logistic + Gamma** | Logistic | Gamma GLM |
| **Logistic + Lognormal** | Logistic | Lognormal |
| **Logistic + Generalized Gamma** | Logistic | Gen. Gamma |
| **Marginalized Two-Part** | Logistic | Parameterisasi langsung marginal mean |

---

## 8. ZERO/ONE-INFLATED PROPORTION DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Target proporsi Y ∈ [0,1] dengan mass di 0 dan/atau 1. Tiga mekanisme: Y=0, 0<Y<1, Y=1. Dekomposisi: Stage 1 boundary state, Stage 2 Beta regression untuk interior |
| **Problem yang didukung** | Market share; Utilization; Completion ratio; Percentage sold |
| **Kapan digunakan** | Target [0,1]; excess zeros dan/atau ones; proporsi |
| **Kapan tidak digunakan** | Target binary; target kontinu tanpa boundary |
| **Asumsi** | Tiga proses; Beta untuk interior |
| **Data preprocessing** | Boundary indicator; interior proportion |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Beta deviance; boundary accuracy |
| **Hyperparameter penting** | `boundary_model`, `interior_model` |
| **Kisaran parameter awal** | Logistic boundary + Beta interior |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Boundary model salah |
| **Teknik pembanding** | Beta regression, Fractional regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `betareg` (R); `zoib` (R); custom |
| **Contoh hypothesis** | "Zero-one inflated beta akan mencapai deviance ≥15% lebih rendah dari Beta regression pada data dengan boundary mass" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Ospina & Ferrari, Communications in Statistics (2012); Taylor & Francis (2017) |

---

## 9. CENSORED / TRUNCATED TARGET REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Target numerik yang hanya diketahui ≤, ≥, atau dalam interval. Contoh: income ≥100 juta dicatat sebagai 100; lab < detection limit |
| **Problem yang didukung** | Income; Detection limit; Top-coded data |
| **Kapan digunakan** | Target censored/truncated; values like "<x", ">x", capped |
| **Kapan tidak digunakan** | Target fully observed |
| **Asumsi** | Censoring mechanism; distributional assumption (Tobit) atau robust (CLAD) |
| **Data preprocessing** | Censoring indicator; lower/upper bounds |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Censored likelihood; MAE |
| **Hyperparameter penting** | `censoring_type` (left/right/interval); `distribution` |
| **Kisaran parameter awal** | Tobit; CLAD untuk robustness |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Ordinary regression pada censored data → bias |
| **Teknik pembanding** | Interval regression; Survival |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels`; `Tobit` (R); `censReg` (R) |
| **Contoh hypothesis** | "Tobit akan mencapai MAE ≥10% lebih rendah dari OLS pada data income top-coded" |
| **Acceptance criteria** | MAE improvement ≥10% |
| **Rejection criteria** | MAE drop |
| **Sumber** | Tobin (1958); Powell (1984) |

**Taxonomy:**

| Tipe | Deskripsi |
|------|-----------|
| **Left censored** | Y < L |
| **Right censored** | Y > U |
| **Interval censored** | L < Y < U |
| **Truncated** | Data di luar range tidak ada |
| **Top-coded** | Nilai tinggi dicap |
| **Bottom-coded** | Nilai rendah dicap |

---

## 10. COMPETING RISKS

| Field | Detail |
|-------|--------|
| **Definisi** | Survival dengan multiple mutually exclusive event types. Begitu satu event terjadi, event lain tidak lagi dapat terjadi dengan cara biasa |
| **Problem yang didukung** | Churn vs account closed vs death; Cancer death vs non-cancer death |
| **Kapan digunakan** | Multiple event types; exclusive occurrence; cause-specific prediction |
| **Kapan tidak digunakan** | Single event type |
| **Asumsi** | Events mutually exclusive; cause-specific hazard atau subdistribution hazard |
| **Data preprocessing** | Event type column; time; censoring |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Cause-specific C-index; time-dependent AUC; cause-specific Brier; CIF calibration |
| **Hyperparameter penting** | `model_type` (cause-specific vs Fine-Gray); `event_type` |
| **Kisaran parameter awal** | Fine-Gray untuk CIF |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Tinggi; event-specific risk |
| **Failure modes** | Mengabaikan competing events → bias |
| **Teknik pembanding** | Standard survival; Multi-state |
| **Ensemble compatibility** | Universal |
| **Exact package** | `cmprsk` (R); `riskRegression` (R); `mstate` (R) |
| **Contoh hypothesis** | "Fine-Gray akan mencapai cause-specific C-index ≥0.70 pada data transplant 3 event types" |
| **Acceptance criteria** | CIF calibration baik; C-index improvement ≥0.05 |
| **Rejection criteria** | Mengabaikan competing events |
| **Sumber** | Putter et al., Statistics in Medicine (2007); Fine & Gray (1999) |

**Family:**

| Metode | Deskripsi |
|--------|-----------|
| **Cause-Specific Cox** | Hazard untuk event tertentu |
| **Fine-Gray** | Subdistribution hazard untuk CIF |
| **Aalen–Johansen** | Non-parametric CIF estimator |
| **Competing-Risk RSF** | Random Survival Forest untuk competing risks |
| **Competing-Risk Boosting** | Gradient boosting untuk competing risks |

---

## 11. RECURRENT-EVENT DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Event yang sama dapat terjadi berkali-kali pada entity yang sama. Time-to-first-event analysis membuang informasi event berikutnya |
| **Problem yang didukung** | Recurrent hospitalizations; Recurrent infections; Machine failures |
| **Kapan digunakan** | Entity repeated; multiple event times per entity; event dapat recur |
| **Kapan tidak digunakan** | Single event per entity |
| **Asumsi** | Recurrent events dapat dimodelkan dengan Andersen-Gill, PWP, atau WLW |
| **Data preprocessing** | Multiple event times per entity; gap time vs calendar time |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Recurrent-event likelihood; event-rate metrics |
| **Hyperparameter penting** | `model_type` (Andersen-Gill/PWP/WLW); `time_scale` |
| **Kisaran parameter awal** | Andersen-Gill untuk counting process |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Tinggi; within-entity correlation |
| **Failure modes** | Hanya event pertama digunakan |
| **Teknik pembanding** | Standard survival; Multi-state |
| **Ensemble compatibility** | Universal |
| **Exact package** | `survival` (R); `reReg` (R); `mets` (R) |
| **Contoh hypothesis** | "Andersen-Gill akan mencapai event-rate accuracy ≥10% lebih baik dari time-to-first-event" |
| **Acceptance criteria** | Event-rate improvement ≥10% |
| **Rejection criteria** | Hanya event pertama |
| **Sumber** | Andersen & Gill (1982); PubMed (2024) |

**Family:**

| Metode | Deskripsi |
|--------|-----------|
| **Andersen–Gill** | Counting process; events independent |
| **PWP Total Time** | Ordered events; total time scale |
| **PWP Gap Time** | Ordered events; gap time scale |
| **WLW** | Marginal models; robust SE |
| **Frailty Models** | Random effect per entity |

---

## 12. MULTI-STATE SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Entity berpindah antar banyak state (Healthy → Mild → Severe → Hospitalized → Recovered → Relapse → Death). Target: P(S(t)=j\|X) atau transition hazards λ_ij(t) |
| **Problem yang didukung** | Disease progression; Multi-stage credit risk |
| **Kapan digunakan** | Multiple intermediate states; transitions antar states |
| **Kapan tidak digunakan** | Single event; competing events tanpa intermediate states |
| **Asumsi** | Markov property (atau semi-Markov); transition intensities |
| **Data preprocessing** | State sequence per subject; transition times |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Transition probabilities; state occupancy |
| **Hyperparameter penting** | `transition_matrix`; `time_scale` |
| **Kisaran parameter awal** | Markov multi-state model |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Tinggi; state sequence dari past |
| **Failure modes** | Markov assumption violation |
| **Teknik pembanding** | Competing risks; Recurrent events |
| **Ensemble compatibility** | Universal |
| **Exact package** | `mstate` (R); `msm` (R); `flexsurv` (R) |
| **Contoh hypothesis** | "Multi-state model akan mencapai state occupancy accuracy ≥0.70" |
| **Acceptance criteria** | State occupancy accuracy ≥0.65 |
| **Rejection criteria** | Markov assumption gagal |
| **Sumber** | Putter et al., Statistics in Medicine (2007); PubMed (2008) |

**Structure:**

```text
12. MULTI-STATE TARGET
├── Markov Multi-State
├── Semi-Markov
├── Illness–Death
└── Competing Risks as special case
```

---

## 13. JOINT LONGITUDINAL-SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Joint model untuk repeated measurements + time-to-event. Longitudinal submodel + survival submodel + shared latent/random effect |
| **Problem yang didukung** | Dynamic risk prediction; Biomarker trajectory + event time |
| **Kapan digunakan** | Repeated measurement + time-to-event; latest trajectory informs risk |
| **Kapan tidak digunakan** | Single measurement; no event time |
| **Asumsi** | Shared random effects; longitudinal trajectory informatif |
| **Data preprocessing** | Longitudinal measurements; event time; censoring |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Joint likelihood; dynamic AUC; Brier |
| **Hyperparameter penting** | `longitudinal_model`, `survival_model`, `association_structure` |
| **Kisaran parameter awal** | Linear mixed-effects + Cox |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Tinggi; time-based split |
| **Failure modes** | Model misspecification |
| **Teknik pembanding** | Landmarking; Time-dependent Cox |
| **Ensemble compatibility** | Universal |
| **Exact package** | `JM` (R); `JMbayes` (R); `joineRML` (R) |
| **Contoh hypothesis** | "Joint model akan mencapai dynamic AUC ≥0.75 pada data biomarker + survival" |
| **Acceptance criteria** | Dynamic AUC ≥0.70 |
| **Rejection criteria** | Dynamic AUC <0.60 |
| **Sumber** | Rizopoulos (2012); PubMed (2011, 2018) |

---

## 14. HIERARCHICAL FORECAST RECONCILIATION

| Field | Detail |
|-------|--------|
| **Definisi** | Memastikan forecasts pada hierarchy memenuhi aggregation constraint: Ŷ_total = Σ Ŷ_j. Bottom-Up, Top-Down, Middle-Out, OLS, WLS, MinT |
| **Problem yang didukung** | Hierarchical forecasting; Retail sales; Energy demand |
| **Kapan digunakan** | Target hierarchy dengan aggregation constraint |
| **Kapan tidak digunakan** | Flat target |
| **Asumsi** | Hierarchy known; linear constraints |
| **Data preprocessing** | Hierarchy structure |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Coherent forecast metrics |
| **Hyperparameter penting** | `reconciliation_method` (OLS/WLS/MinT) |
| **Kisaran parameter awal** | MinT (minimize trace) |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Incoherent forecasts |
| **Teknik pembanding** | Independent per level |
| **Ensemble compatibility** | Universal |
| **Exact package** | `hts` (R); `fable` (R); `HierarchicalForecast` (Python) |
| **Contoh hypothesis** | "MinT akan mencapai RMSE ≥10% lebih rendah dari bottom-up pada hierarchical sales" |
| **Acceptance criteria** | Coherence tercapai; RMSE improvement ≥5% |
| **Rejection criteria** | RMSE drop |
| **Sumber** | Hyndman et al. (2011); ScienceDirect (2024) |

**Family:**

| Metode | Deskripsi |
|--------|-----------|
| **Bottom-Up** | Forecast level bawah, aggregate ke atas |
| **Top-Down** | Forecast total, disaggregate |
| **Middle-Out** | Forecast level menengah |
| **OLS** | Ordinary least squares reconciliation |
| **WLS** | Weighted least squares |
| **MinT** | Minimize trace of forecast-error covariance |
| **Probabilistic** | Probabilistic reconciliation |
| **Cross-temporal** | Cross-temporal reconciliation |

---

## 15. MULTI-TARGET CHAINS

| Field | Detail |
|-------|--------|
| **Definisi** | Regressor Chain: X → Y1; X + Ŷ1 → Y2; X + Ŷ1 + Ŷ2 → Y3. Eksploitasi dependency antar target |
| **Problem yang didukung** | Multi-target regression; Dependent targets |
| **Kapan digunakan** | Multiple target terkait; dependency dapat dieksploitasi |
| **Kapan tidak digunakan** | Target independen |
| **Asumsi** | Target dependency |
| **Data preprocessing** | Standard; target normalization |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | MSE, MAE per target |
| **Hyperparameter penting** | `order` (chain order); `base_estimator` |
| **Kisaran parameter awal** | Random chain order |
| **CPU/GPU/RAM** | M × model cost |
| **Risiko leakage** | Sedang; chain order dari training |
| **Failure modes** | Error propagation |
| **Teknik pembanding** | Independent MultiOutput; Multi-Task |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.multioutput.RegressorChain` |
| **Contoh hypothesis** | "Regressor Chain akan mencapai MSE ≥5% lebih rendah dari independent MultiOutput" |
| **Acceptance criteria** | MSE improvement ≥5% |
| **Rejection criteria** | MSE drop |
| **Sumber** | Read et al. (2011); ScienceDirect (2018) |

---

## 16. COMPOSITIONAL TARGET MODELING

| Field | Detail |
|-------|--------|
| **Definisi** | Target vector proportions yang totalnya = 1. Hidup pada simplex. Dirichlet Regression, Logistic-Normal, ALR/CLR/ILR transformation |
| **Problem yang didukung** | Budget allocation; Portfolio; Time allocation |
| **Kapan digunakan** | Multiple proportions sum to 1 |
| **Kapan tidak digunakan** | Independent outputs |
| **Asumsi** | Simplex geometry |
| **Data preprocessing** | Log-ratio transformation |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Aitchison distance; Dirichlet deviance |
| **Hyperparameter penting** | `transformation` (ALR/CLR/ILR) |
| **Kisaran parameter awal** | ILR + multivariate regression |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Sum constraint diabaikan |
| **Teknik pembanding** | Multi-output regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `compositions` (R); `DirichletReg` (R) |
| **Contoh hypothesis** | "ILR + regression akan mencapai Aitchison distance ≥20% lebih rendah dari ordinary multi-output" |
| **Acceptance criteria** | Aitchison improvement ≥15% |
| **Rejection criteria** | Sum constraint diabaikan |
| **Sumber** | Aitchison (1986); Taylor & Francis (2022) |

---

## 17. ORDINAL THRESHOLD DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Ordered class → cumulative binary thresholds: P(Y≤k\|X), k=1,...,K-1. Mengubah ordinal multiclass menjadi set binary problems yang monotonic |
| **Problem yang didukung** | Rating prediction; Disease staging; Severity |
| **Kapan digunakan** | Target ordinal; cumulative structure |
| **Kapan tidak digunakan** | Nominal multiclass |
| **Asumsi** | Ordered categories; monotonic thresholds |
| **Data preprocessing** | Ordinal encoding |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | QWK; MAE ordinal; MZE |
| **Hyperparameter penting** | `link` (logit/probit); `threshold_structure` |
| **Kisaran parameter awal** | Proportional odds |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Monotonicity violation |
| **Teknik pembanding** | Multiclass; Ordinal regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ordinal` (R); `mord` (Python) |
| **Contoh hypothesis** | "Cumulative link akan mencapai QWK ≥0.80 pada data staging" |
| **Acceptance criteria** | QWK ≥0.75; monotonic thresholds |
| **Rejection criteria** | QWK <0.60 |
| **Sumber** | McCullagh (1980); Frank & Hall (2001) |

**Sub-varian:**

| Varian | Deskripsi |
|--------|-----------|
| **Cumulative Link / Proportional Odds** | Common slope |
| **K−1 Binary Threshold** | Binary reduction |
| **Partial Proportional Odds** | Slope bervariasi untuk sebagian covariates |

---

## 18. MIXTURE CURE / NON-MIXTURE CURE MODELS

| Field | Detail |
|-------|--------|
| **Definisi** | Survival di mana sebagian populasi **tidak pernah mengalami event** (cured). Mixture Cure: P(T>t) = π(X) + [1-π(X)]S_u(t\|X). Non-mixture: promotion-time |
| **Problem yang didukung** | Oncology; Churn; Credit risk |
| **Kapan digunakan** | Long follow-up; survival curve plateaus substantially above 0; domain allows "never event" |
| **Kapan tidak digunakan** | Semua subject susceptible |
| **Asumsi** | Cure fraction; uncured survival |
| **Data preprocessing** | Long follow-up |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Cure rate; C-index; Brier |
| **Hyperparameter penting** | `cure_model` (logistic); `survival_model` |
| **Kisaran parameter awal** | Logistic + Weibull |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Cure fraction misspecified |
| **Teknik pembanding** | Standard survival |
| **Ensemble compatibility** | Universal |
| **Exact package** | `smcure` (R); `cureph` (R) |
| **Contoh hypothesis** | "Mixture cure akan mencapai C-index ≥0.75 pada data oncology dengan plateau" |
| **Acceptance criteria** | C-index ≥0.70; cure fraction estimable |
| **Rejection criteria** | C-index <0.60 |
| **Sumber** | PubMed (2021) |

---

## 19. FRAILTY / SHARED FRAILTY SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Frailty model menambahkan latent random effect: h_i(t\|X_i,Z_i) = Z_i h_0(t) exp(X_i^T β). Shared frailty untuk clustered data |
| **Problem yang didukung** | Clustered survival; Recurrent events; Unobserved heterogeneity |
| **Kapan digunakan** | Clustered data; unobserved heterogeneity; dependent survival |
| **Kapan tidak digunakan** | Independent subjects |
| **Asumsi** | Frailty distribution (Gamma, Lognormal) |
| **Data preprocessing** | Cluster ID |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | C-index; frailty variance |
| **Hyperparameter penting** | `frailty_distribution`; `cluster_column` |
| **Kisaran parameter awal** | Gamma frailty |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; cluster split |
| **Failure modes** | Frailty distribution misspecified |
| **Teknik pembanding** | Standard survival |
| **Ensemble compatibility** | Universal |
| **Exact package** | `frailtypack` (R); `coxme` (R) |
| **Contoh hypothesis** | "Shared frailty akan mencapai C-index ≥0.75 pada data multi-hospital" |
| **Acceptance criteria** | C-index ≥0.70; frailty variance estimable |
| **Rejection criteria** | C-index <0.60 |
| **Sumber** | PubMed (2020) |

---

## 20. INTERVAL-CENSORED SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Event time diketahui terjadi dalam interval (L, R]. Contoh: test negatif hari 100, positif hari 130 → T ∈ (100, 130] |
| **Problem yang didukung** | Disease onset; Equipment failure |
| **Kapan digunakan** | Event time interval-censored |
| **Kapan tidak digunakan** | Right-censoring standar |
| **Asumsi** | Interval-censoring dapat dimodelkan dengan PH, AFT, PO, GOR |
| **Data preprocessing** | Lower bound; upper bound |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Interval-censored likelihood |
| **Hyperparameter penting** | `model_type` (PH/AFT/PO/GOR) |
| **Kisaran parameter awal** | PH untuk interpretability |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Tinggi |
| **Failure modes** | Interval terlalu lebar |
| **Teknik pembanding** | Standard survival |
| **Ensemble compatibility** | Universal |
| **Exact package** | `icenReg` (R); `interval` (R) |
| **Contoh hypothesis** | "AFT akan mencapai likelihood ≥10% lebih baik dari midpoint imputation" |
| **Acceptance criteria** | Likelihood improvement ≥10% |
| **Rejection criteria** | Midpoint imputation |
| **Sumber** | Annual Reviews (2026) |

---

## 21. SEMI-COMPETING RISKS

| Field | Detail |
|-------|--------|
| **Definisi** | Nonterminal event (readmission) bisa dicegah oleh terminal event (death). Illness-death structure |
| **Problem yang didukung** | Hospital readmission; Clinical trials |
| **Kapan digunakan** | Nonterminal event + terminal event |
| **Kapan tidak digunakan** | Mutually exclusive competing events |
| **Asumsi** | Illness-death structure |
| **Data preprocessing** | Event times; event types |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | C-index; transition probabilities |
| **Hyperparameter penting** | `model_type` |
| **Kisaran parameter awal** | Illness-death model |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Tinggi |
| **Failure modes** | Mengabaikan semi-competing structure |
| **Teknik pembanding** | Competing risks; Multi-state |
| **Ensemble compatibility** | Universal |
| **Exact package** | `mstate` (R); `SemiCompRisks` (R) |
| **Contoh hypothesis** | "Illness-death model akan mencapai C-index ≥0.75 pada data readmission" |
| **Acceptance criteria** | C-index ≥0.70 |
| **Rejection criteria** | C-index <0.60 |
| **Sumber** | PubMed (2016, 2026) |

---

## 22. HECKMAN / SAMPLE-SELECTION DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Target hanya terlihat pada subset non-random. Selection model + outcome model. Heckman correction |
| **Problem yang didukung** | Salary (jika employed); Satisfaction (jika respond); Business data |
| **Kapan digunakan** | Target missing karena selection; selection potentially related to outcome |
| **Kapan tidak digunakan** | Missing at random |
| **Asumsi** | Selection equation; exclusion restriction |
| **Data preprocessing** | Selection indicator |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Outcome model metrics; selection diagnostics |
| **Hyperparameter penting** | `selection_model`, `outcome_model` |
| **Kisaran parameter awal** | Probit selection + OLS outcome |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Tinggi; selection mechanism |
| **Failure modes** | Exclusion restriction tidak valid |
| **Teknik pembanding** | Imputation; Heckman |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sampleSelection` (R); `statsmodels` |
| **Contoh hypothesis** | "Heckman akan mencapai MAE ≥10% lebih rendah dari OLS pada data salary dengan selection" |
| **Acceptance criteria** | MAE improvement ≥10%; selection coefficient signifikan |
| **Rejection criteria** | MAE drop |
| **Sumber** | Heckman (1979); JSTOR |

---

## 23. BODY-TAIL / EVT SPLICING

| Field | Detail |
|-------|--------|
| **Definisi** | Splicing distribution: body distribution untuk Y ≤ u, tail model (GPD) untuk Y > u |
| **Problem yang didukung** | High-severity insurance; Transaction amounts; Operational risk |
| **Kapan digunakan** | Ordinary losses + catastrophic losses |
| **Kapan tidak digunakan** | Light-tailed |
| **Asumsi** | Threshold u; GPD untuk excess |
| **Data preprocessing** | Threshold selection (mean residual life) |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Tail index; VaR; ES |
| **Hyperparameter penting** | `threshold`; `body_distribution`; `tail_distribution` |
| **Kisaran parameter awal** | Gamma body + GPD tail |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Threshold salah |
| **Teknik pembanding** | Single distribution |
| **Ensemble compatibility** | Universal |
| **Exact package** | `evir` (R); `extRemes` (R); `pyextremes` |
| **Contoh hypothesis** | "Splicing akan mencapai VaR accuracy ≥15% lebih baik dari single distribution" |
| **Acceptance criteria** | VaR improvement ≥10% |
| **Rejection criteria** | VaR drop |
| **Sumber** | ScienceDirect (2016); Cambridge (2003) |

---

## 24. REPORTING-DELAY / IBNR DECOMPOSITION

| Field | Detail |
|-------|--------|
| **Definisi** | Occurrence model + Reporting-delay model + Settlement/payment model = IBNR / future liability |
| **Problem yang didukung** | Insurance reserving; IBNR |
| **Kapan digunakan** | Claims occurred but not reported |
| **Kapan tidak digunakan** | Reported claims only |
| **Asumsi** | Delay distribution; settlement distribution |
| **Data preprocessing** | Occurrence time; reporting time; settlement time |
| **Dataset scale** | Besar |
| **Metric compatibility** | Reserving metrics; IBNR accuracy |
| **Hyperparameter penting** | `delay_model`, `settlement_model` |
| **Kisaran parameter awal** | Chain-ladder; GLM |
| **CPU/GPU/RAM** | Sedang–Berat |
| **Risiko leakage** | Tinggi; temporal structure |
| **Failure modes** | Delay distribution misspecified |
| **Teknik pembanding** | Direct regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `ChainLadder` (Python); `apc` (R) |
| **Contoh hypothesis** | "IBNR decomposition akan mencapai reserving accuracy ≥10% lebih baik dari direct regression" |
| **Acceptance criteria** | Accuracy improvement ≥10% |
| **Rejection criteria** | Accuracy drop |
| **Sumber** | ScienceDirect (2015) |

---

## 25. MULTIVARIATE ZERO-INFLATED TARGETS

| Field | Detail |
|-------|--------|
| **Definisi** | Multiple correlated zero-heavy count targets. Multivariate ZIP, zero-adjusted models, zero-and-one-inflated multivariate |
| **Problem yang didukung** | Healthcare utilization; Multiple count outcomes |
| **Kapan digunakan** | Multiple correlated count targets; many joint/marginal zeros |
| **Kapan tidak digunakan** | Single target |
| **Asumsi** | Multivariate count distribution |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Multivariate count deviance |
| **Hyperparameter penting** | `model_family` |
| **Kisaran parameter awal** | Multivariate ZIP |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Dependence diabaikan |
| **Teknik pembanding** | Independent ZIP |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `mzip` (R) |
| **Contoh hypothesis** | "Multivariate ZIP akan mencapai deviance ≥10% lebih rendah dari independent ZIP" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | ScienceDirect (2014); PubMed (2022) |

---

## 26. LATENT-CLASS / MIXTURE REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | P(Y\|X) = Σ π_k(X) P_k(Y\|X). Decomposition via latent class membership. Mixture likelihood/EM |
| **Problem yang didukung** | Hidden regimes; Heterogeneous subpopulations |
| **Kapan digunakan** | Residual multimodality; strong hidden regimes; global model fails |
| **Kapan tidak digunakan** | Homogeneous population |
| **Asumsi** | K latent classes; mixture components |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Mixture likelihood; BIC |
| **Hyperparameter penting** | `n_components`; `covariance_type` |
| **Kisaran parameter awal** | K=2–5 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Label switching; local optima |
| **Teknik pembanding** | Clustering + regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `flexmix` (R); `sklearn.mixture.GaussianMixture` |
| **Contoh hypothesis** | "Mixture regression akan mencapai BIC ≥10% lebih rendah dari single regression" |
| **Acceptance criteria** | BIC improvement ≥10% |
| **Rejection criteria** | BIC drop |
| **Sumber** | ScienceDirect (2026); Leisch (2004) |

---

## 27. ZERO-AND-ONE INFLATED COUNTS

| Field | Detail |
|-------|--------|
| **Definisi** | Count distribution dengan excess 0 DAN excess 1. Zero-and-One Inflated Poisson (ZOIP) |
| **Problem yang didukung** | Doctor visits; Minor incidents |
| **Kapan digunakan** | Observed P(Y=0) >> expected DAN P(Y=1) >> expected |
| **Kapan tidak digunakan** | Hanya excess zero |
| **Asumsi** | Excess zero dan one |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah |
| **Metric compatibility** | ZOIP likelihood |
| **Hyperparameter penting** | `inflation_0`, `inflation_1` |
| **Kisaran parameter awal** | ZOIP |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Inflation misspecified |
| **Teknik pembanding** | ZIP, Hurdle |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; `zoip` (R) |
| **Contoh hypothesis** | "ZOIP akan mencapai AIC ≥10% lebih rendah dari ZIP pada data dengan excess one" |
| **Acceptance criteria** | AIC improvement ≥10% |
| **Rejection criteria** | AIC drop |
| **Sumber** | ScienceDirect (2019) |

---

## Ringkasan Pemilihan Pola

| Kondisi Data | Pola yang Tepat | Prioritas |
|--------------|----------------|-----------|
| Total cost, banyak zeros, klaim terpisah | Frequency-Severity | Tinggi |
| Total cost, single model | Direct Tweedie | Sedang |
| >30% zero, "never claimers" | Zero-Inflated / Hurdle | Tinggi |
| Target hierarki (total → kategori → sub) | Hierarchical MTL / Reconciliation | Sedang |
| Durasi observasi berbeda | Exposure Normalization | **Wajib** |
| Time-to-event, censored | Survival / Time-to-Event | Tinggi |
| Frequency-severity dependent | Dependent FS | Tinggi |
| Semicontinuous (0 + positive continuous) | Semicontinuous Two-Part | Tinggi |
| Proporsi [0,1] dengan boundary mass | Zero/One-Inflated Proportion | Sedang |
| Target censored/truncated | Censored Regression | Tinggi |
| Multiple exclusive events | Competing Risks | Tinggi |
| Recurrent events | Recurrent-Event | Tinggi |
| Multiple states | Multi-State Survival | Tinggi |
| Longitudinal + survival | Joint Longitudinal-Survival | Sedang |
| Hierarchy dengan aggregation constraint | Hierarchical Reconciliation | Tinggi |
| Multiple dependent targets | Multi-Target Chains | Sedang |
| Proportions sum to 1 | Compositional Target | Tinggi |
| Ordered classes | Ordinal Threshold | Sedang |
| Survival curve plateaus | Cure Models | Tinggi |
| Clustered survival | Frailty | Tinggi |
| Interval-censored | Interval-Censored Survival | Tinggi |
| Nonterminal + terminal event | Semi-Competing Risks | Tinggi |
| Target missing due to selection | Heckman | Tinggi |
| Body + catastrophic tail | Body-Tail Splicing | Tinggi |
| Occurrence + reporting delay | IBNR | Tinggi |
| Multiple correlated zero-heavy | Multivariate ZI | Sedang |
| Hidden regimes | Latent-Class Mixture | Sedang |
| Excess 0 and 1 | Zero-and-One Inflated | Niche |

---

## Referensi Utama

1. Quan et al., Prob. Eng. Inf. Sci. 37(2) (2023)
2. Nguyen, Mathematics 14(10), 1640 (2026)
3. Ben Khedher & Yun, Appl. Sci. 14(23), 10790 (2024)
4. Jose et al., Ann. Actuarial Sci. 18(3), 644–674 (2024)
5. Barnwal et al., arXiv:2203.07091 (2022)
6. Garrido et al., Insurance: Mathematics and Economics (2016)
7. Shi et al., Insurance: Mathematics and Economics (2015)
8. Lee & Shi, Insurance: Mathematics and Economics (2019)
9. Yang et al., Insurance: Mathematics and Economics (2021)
10. Belotti et al., Stata Journal (2015)
11. Ospina & Ferrari, Communications in Statistics (2012)
12. Tobin (1958); Powell (1984)
13. Putter et al., Statistics in Medicine (2007)
14. Fine & Gray (1999)
15. Andersen & Gill (1982)
16. Rizopoulos (2012)
17. Hyndman et al. (2011)
18. Read et al. (2011)
19. Aitchison (1986)
20. McCullagh (1980)
21. Frank & Hall (2001)
22. PubMed (2021) — Cure Models
23. PubMed (2020) — Frailty
24. Annual Reviews (2026) — Interval-Censored
25. PubMed (2016, 2026) — Semi-Competing Risks
26. Heckman (1979)
27. ScienceDirect (2016) — Splicing
28. Cambridge (2003) — EVT
29. ScienceDirect (2015) — IBNR
30. ScienceDirect (2014) — Multivariate ZI
31. ScienceDirect (2026) — Mixture Regression
32. ScienceDirect (2019) — Zero-and-One Inflated

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*