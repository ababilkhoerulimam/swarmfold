# Problem Formulation Agent — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk formulasi problem pada data tabular. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

**Keluarga 1 — Predictive Supervised**
1. Binary Classification
2. Multiclass Classification
3. Ordinal Regression / Classification
4. Multi-label Classification
5. Regression (Conditional Mean)
6. Quantile Regression
7. Count Regression
8. Zero-Inflated / Hurdle
9. Multi-Output Regression
10. Distributional Regression

**Keluarga 2 — Ranking & Retrieval**
11. Learning-to-Rank

**Keluarga 3 — Time-to-Event**
12. Standard Survival
13. Competing Risks
14. Recurrent Events
15. Multi-State Survival
16. Interval-Censored Survival

**Keluarga 4 — Rare / Unknown Classes**
17. Anomaly Detection
18. One-Class / Novelty Detection
19. Open-Set Recognition
20. Selective Prediction / Abstention

**Keluarga 5 — Weak / Partial Supervision**
21. Positive–Unlabeled (PU) Learning
22. Semi-Supervised Learning
23. Weak / Noisy Supervision
24. Multiple-Instance Learning
25. Partial-Label Learning
26. Complementary-Label Learning

**Keluarga 6 — Causal / Decision**
27. Uplift / Heterogeneous Treatment Effect
28. Contextual Bandit / Policy Learning
29. Cost-Sensitive Decision

**Keluarga 7 — Specialized Target Structures**
30. Fractional Regression
31. Beta Regression
32. Compositional Regression
33. Tweedie / Semi-Continuous Regression
34. Hierarchical Classification
35. Label Distribution Learning
36. Set-Valued Classification
37. Structured Prediction
38. Monotonic Prediction

---

## 1. BINARY CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi satu label dengan dua kelas nominal mutually exclusive. Output dapat berupa hard label, probability, atau risk score |
| **Problem yang didukung** | Klasifikasi biner: fraud detection, churn prediction, disease diagnosis |
| **Kapan digunakan** | Target = 2 kelas nominal; setiap row punya tepat satu label; tidak ada urutan natural |
| **Kapan tidak digunakan** | >2 kelas (multiclass); kelas terurut (ordinal); multiple labels (multi-label) |
| **Asumsi** | Kelas mutually exclusive; i.i.d. samples (kecuali ada struktur dependensi) |
| **Data preprocessing** | Standar; scaling untuk model distance/gradient; encoding untuk kategorikal |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC, Log Loss, Brier Score |
| **Hyperparameter penting** | `threshold`, `class_weight`, `calibration_method` |
| **Kisaran parameter awal** | threshold: 0.5 (atau di-tune pada validation); class_weight: 'balanced' untuk imbalanced |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; threshold selection harus pada validation set terpisah |
| **Failure modes** | Accuracy misleading untuk imbalanced; threshold tidak di-tune; calibration buruk |
| **Teknik pembanding** | Multiclass, ordinal, anomaly detection |
| **Ensemble compatibility** | Universal; soft voting, stacking |
| **Exact package** | `sklearn.linear_model.LogisticRegression`; `sklearn.ensemble.RandomForestClassifier`; `xgboost.XGBClassifier`; `lightgbm.LGBMClassifier`; `catboost.CatBoostClassifier` |
| **Contoh hypothesis** | "Binary classification dengan threshold tuning akan mencapai F1 ≥0.80 pada data fraud dengan 2% positive rate" |
| **Acceptance criteria** | AUC ≥0.80; F1 ≥0.75 pada threshold optimal; calibration error <0.05 |
| **Rejection criteria** | AUC <0.70; threshold default menghasilkan recall <0.5 |
| **Sumber** | Fawcett (2006) An Introduction to ROC Analysis; scikit-learn docs |

**Sub-varian: Probabilistic Classification**

| Field | Detail |
|-------|--------|
| **Definisi** | Binary classification yang membutuhkan probability output terkalibrasi; metric Log Loss / Brier Score |
| **Kapan digunakan** | Metric = Log Loss / Brier Score; butuh probability untuk decision cost-sensitive |
| **Asumsi** | Proper scoring rules: prediksi probability jujur optimal |
| **Hyperparameter penting** | `calibration_method` (sigmoid/isotonic); `cv` untuk calibration |
| **Exact package** | `sklearn.calibration.CalibratedClassifierCV` |
| **Contoh hypothesis** | "Calibrated classifier akan mencapai Brier Score ≥10% lebih baik dari uncalibrated pada data dengan 100K samples" |
| **Sumber** | Gneiting & Raftery, JASA 102(477), 359–378 (2007) |

---

## 2. MULTICLASS CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi satu label dari >2 kelas nominal yang mutually exclusive dan unordered |
| **Problem yang didukung** | Klasifikasi multiclass: species classification, digit recognition, topic classification |
| **Kapan digunakan** | Target kategorikal dengan >2 kelas; tidak ada urutan natural; setiap row punya tepat satu label |
| **Kapan tidak digunakan** | Kelas terurut (ordinal); multiple labels (multi-label); hierarki (hierarchical) |
| **Asumsi** | Kelas mutually exclusive; i.i.d. |
| **Data preprocessing** | Standar; encoding target dengan LabelEncoder atau OrdinalEncoder |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Accuracy, Macro-F1, Micro-F1, Log Loss, Top-k Accuracy |
| **Hyperparameter penting** | `multi_class` (ovr/softmax); `class_weight` |
| **Kisaran parameter awal** | multi_class='multinomial' untuk softmax; 'ovr' untuk one-vs-rest |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; encoding target harus konsisten |
| **Failure modes** | Kelas minoritas diabaikan; macro-F1 misleading jika kelas tidak seimbang; ordinalitas hilang |
| **Teknik pembanding** | Ordinal regression, hierarchical classification |
| **Ensemble compatibility** | Universal; soft voting, stacking |
| **Exact package** | `sklearn.linear_model.LogisticRegression(multi_class='multinomial')`; `sklearn.ensemble.RandomForestClassifier`; `xgboost.XGBClassifier`; `lightgbm.LGBMClassifier` |
| **Contoh hypothesis** | "Multiclass logistic regression akan mencapai macro-F1 ≥0.85 pada dataset 10 kelas" |
| **Acceptance criteria** | Macro-F1 ≥0.80; per-class recall ≥0.70 |
| **Rejection criteria** | Macro-F1 <0.70; kelas minoritas recall <0.5 |
| **Sumber** | scikit-learn docs |

---

## 3. ORDINAL REGRESSION / CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Klasifikasi pada categorical scale dengan natural order between labels. Label adalah kategorikal tetapi memiliki urutan I < II < III < IV |
| **Problem yang didukung** | Rating prediction, disease staging, credit rating, severity classification |
| **Kapan digunakan** | Target kategorikal dengan urutan natural; ordinal-specific metrics (QWK, MAE over rank) |
| **Kapan tidak digunakan** | Kelas nominal tanpa urutan; target kontinu dengan jarak bermakna (regression) |
| **Asumsi** | Order antar kelas bermakna; informasi order meningkatkan performa dibanding nominal classification |
| **Data preprocessing** | Standar; target encoding harus mempertahankan urutan |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MAE ordinal, MZE, Weighted Kappa, Quadratic Weighted Kappa (QWK) |
| **Hyperparameter penting** | `ordinal_encoder`; `loss` (ordinal-specific) |
| **Kisaran parameter awal** | QWK sebagai primary metric |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; encoding ordinal harus konsisten |
| **Failure modes** | Menggunakan accuracy biasa → kehilangan informasi order; mengabaikan struktur ordinal |
| **Teknik pembanding** | Multiclass classification, regression |
| **Ensemble compatibility** | Universal; ordinal stacking |
| **Exact package** | `mord` (Python); `ordinal` (R); `sklearn.linear_model.LogisticRegression` (dengan ordinal encoding) |
| **Contoh hypothesis** | "Ordinal regression dengan QWK akan mencapai QWK ≥0.80 pada data staging, mengalahkan multiclass accuracy" |
| **Acceptance criteria** | QWK ≥0.75; MAE ordinal <1.0 |
| **Rejection criteria** | QWK <0.60; menggunakan accuracy sebagai metric utama |
| **Sumber** | Gutiérrez et al., IEEE TKDE 28(1), 127–146 (2016) |

---

## 4. MULTI-LABEL CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Setiap instance dapat memiliki beberapa label sekaligus secara simultan. Berbeda dari multiclass di mana hanya satu label yang benar |
| **Problem yang didukung** | Medical diagnosis (multiple diseases), image tagging, document classification |
| **Kapan digunakan** | Setiap row dapat memiliki >1 label benar; label tidak mutually exclusive |
| **Kapan tidak digunakan** | Satu label per row (multiclass); label terurut (ordinal) |
| **Asumsi** | Label dependencies mungkin ada; evaluasi label-wise vs exact-set berbeda secara fundamental |
| **Data preprocessing** | Multi-label binarization; label powerset (untuk dataset kecil) |
| **Dataset scale** | Menengah–Besar; label powerset tidak scalable untuk banyak label |
| **Metric compatibility** | Hamming Loss, Subset Accuracy, Jaccard, LRAP, Ranking Loss |
| **Hyperparameter penting** | `classifier_chain_order`; `label_powerset` |
| **Kisaran parameter awal** | Hamming Loss untuk label-wise; Subset Accuracy untuk exact-set |
| **CPU/GPU/RAM** | M × binary classifier cost (M = jumlah label) |
| **Risiko leakage** | Sedang; label dependencies harus ditangani dengan benar |
| **Failure modes** | Hamming Loss tinggi tapi Subset Accuracy nol; mengabaikan label dependencies |
| **Teknik pembanding** | Multiclass, multi-output classification |
| **Ensemble compatibility** | Universal; classifier chains, stacking |
| **Exact package** | `sklearn.multioutput.MultiOutputClassifier`; `skmultilearn` (Python); `mldr` (R) |
| **Contoh hypothesis** | "Classifier chain akan mencapai Hamming Loss ≤0.10 pada dataset dengan 10 label" |
| **Acceptance criteria** | Hamming Loss ≤0.15; LRAP ≥0.70 |
| **Rejection criteria** | Subset Accuracy = 0 pada semua fold |
| **Sumber** | Tsoumakas & Katakis, Multi-Label Classification: An Overview (2007); Dembczyński et al., Machine Learning 88(1-2), 5–45 (2012) |

---

## 5. REGRESSION (CONDITIONAL MEAN)

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi nilai kontinu dengan jarak bermakna. Dengan squared error loss, optimal predictor adalah conditional mean E[Y|X=x] |
| **Problem yang didukung** | House price prediction, sales forecasting, insurance severity |
| **Kapan digunakan** | Target kontinu; jarak antar nilai bermakna; metric MSE/RMSE |
| **Kapan tidak digunakan** | Target count (count regression); target proportion (fractional); target terurut kategorikal (ordinal) |
| **Asumsi** | Conditional mean adalah estimand yang tepat; homoskedastisitas (untuk OLS) |
| **Data preprocessing** | Scaling untuk model linear/NN; tree-based tidak butuh scaling |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | MSE, RMSE, MAE, R², MAPE |
| **Hyperparameter penting** | `loss` (squared_error); `regularization` |
| **Kisaran parameter awal** | MSE/RMSE untuk conditional mean; MAE untuk conditional median |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; target leakage detection penting |
| **Failure modes** | Heteroskedastisitas → OLS tidak efisien; outlier → MSE sensitif; non-linear relationship → underfitting |
| **Teknik pembanding** | Quantile regression, distributional regression, robust regression |
| **Ensemble compatibility** | Universal; averaging, stacking |
| **Exact package** | `sklearn.linear_model.LinearRegression`; `sklearn.ensemble.RandomForestRegressor`; `xgboost.XGBRegressor`; `lightgbm.LGBMRegressor`; `catboost.CatBoostRegressor` |
| **Contoh hypothesis** | "Gradient boosting akan mencapai RMSE ≥20% lebih rendah dari linear regression pada data house price" |
| **Acceptance criteria** | RMSE improvement ≥15% vs baseline; R² ≥0.80 |
| **Rejection criteria** | RMSE tidak improvement; residual tidak acak |
| **Sumber** | Gneiting, JASA 106(494), 746–762 (2011) |

---

## 6. QUANTILE REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi conditional quantile dari distribusi target, bukan hanya conditional mean. Pinball loss optimal untuk conditional α-quantile |
| **Problem yang didukung** | Risk forecasting (VaR), prediction intervals, uncertainty quantification |
| **Kapan digunakan** | Butuh prediction intervals; metric Pinball Loss; distribusi target heteroskedastik |
| **Kapan tidak digunakan** | Hanya butuh point prediction (conditional mean); metric MSE/RMSE |
| **Asumsi** | Conditional quantile dapat diestimasi; quantile regression robust terhadap outlier |
| **Data preprocessing** | Standar; scaling untuk model linear |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Pinball Loss, Quantile Loss |
| **Hyperparameter penting** | `alpha` (quantile level); `loss`='quantile' |
| **Kisaran parameter awal** | α = 0.50 (median); α = 0.90/0.95 untuk upper quantile |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Rendah |
| **Failure modes** | Quantile crossing; α salah; sample size tidak cukup untuk extreme quantile |
| **Teknik pembanding** | Distributional regression, conditional mean regression |
| **Ensemble compatibility** | Universal; quantile averaging |
| **Exact package** | `statsmodels.QuantReg`; `sklearn.ensemble.GradientBoostingRegressor(loss='quantile')`; `lightgbm.LGBMRegressor(objective='quantile')` |
| **Contoh hypothesis** | "Quantile regression α=0.90 akan mencapai Pinball Loss ≥15% lebih rendah dari linear quantile regression pada data heteroskedastik" |
| **Acceptance criteria** | Pinball Loss improvement ≥10%; tidak ada quantile crossing |
| **Rejection criteria** | Quantile crossing; sample size <100 per quantile |
| **Sumber** | Koenker & Bassett, Econometrica 46(1), 33–50 (1978) |

---

## 7. COUNT REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi count (0,1,2,...) dengan model yang sesuai: Poisson jika variance/mean ≈ 1, Negative Binomial jika overdispersion |
| **Problem yang didukung** | Insurance claims frequency, healthcare utilization, road crash frequency |
| **Kapan digunakan** | Target non-negative integer; count semantics; mean/variance structure |
| **Kapan tidak digunakan** | Target kontinu; target binary; count dengan excess zeros (gunakan zero-inflated) |
| **Asumsi** | Poisson: E[X]=Var[X]; NB: variance > mean; independent counts |
| **Data preprocessing** | Exposure/offset untuk rate modeling; standardisasi fitur |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Poisson deviance, NB deviance, log-likelihood |
| **Hyperparameter penting** | `objective` (poisson/nb); `alpha` (NB dispersion) |
| **Kisaran parameter awal** | Poisson jika variance/mean ≈ 1; NB jika variance/mean > 2 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; exposure harus di-handle dengan benar |
| **Failure modes** | Overdispersion diabaikan; Poisson menghasilkan SE terlalu kecil; zero-inflation tidak ditangani |
| **Teknik pembanding** | Zero-inflated, hurdle, Tweedie |
| **Ensemble compatibility** | Universal; count ensemble |
| **Exact package** | `sklearn.linear_model.PoissonRegressor`; `statsmodels.GLM`; `lightgbm.LGBMRegressor(objective='poisson')`; `xgboost.XGBRegressor(objective='count:poisson')` |
| **Contoh hypothesis** | "Negative Binomial regression akan mencapai Poisson deviance ≥10% lebih rendah dari Poisson regression pada data claims dengan overdispersion" |
| **Acceptance criteria** | Deviance improvement ≥5%; dispersion ratio ≈1 setelah modeling |
| **Rejection criteria** | Deviance tidak improvement; variance/mean masih >2 |
| **Sumber** | Mullahy, Journal of Econometrics 33(3), 341–365 (1986) |

---

## 8. ZERO-INFLATED / HURDLE

| Field | Detail |
|-------|--------|
| **Definisi** | Model count yang menangani excess zeros dari dua mekanisme. **Hurdle**: semua zero dari satu proses "non-participation". **Zero-inflated**: zero dari dua sumber (structural + sampling) |
| **Problem yang didukung** | Insurance claims, healthcare utilization, road crash frequency, e-commerce purchase |
| **Kapan digunakan** | >30% zero; ada "never claimers" sejati; dua proses zero |
| **Kapan tidak digunakan** | Zero dari satu proses (gunakan count regression biasa); tidak ada excess zero |
| **Asumsi** | Hurdle: zero dari satu proses; ZIP: dua proses zero |
| **Data preprocessing** | Standard; zero indicator untuk stage 1 |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Count likelihood/deviance + zero fit; binary metric (stage 1) + count/regression metric (stage 2) |
| **Hyperparameter penting** | `model_type` (hurdle/zip); `stage_1_model`; `stage_2_model` |
| **Kisaran parameter awal** | Hurdle: 2-stage CatBoost; ZIP: count model + inflation model |
| **CPU/GPU/RAM** | 2× model cost |
| **Risiko leakage** | Sedang; stage 1 dan 2 harus pada split yang sama |
| **Failure modes** | Identifiability issues; stage 2 tidak cukup data positif; zero ratio tinggi tidak otomatis zero-inflated |
| **Teknik pembanding** | Count regression, Tweedie |
| **Ensemble compatibility** | Universal; two-part ensemble |
| **Exact package** | `statsmodels.discrete.count_model.ZeroInflatedPoisson`; `pscl::zeroinfl` (R); `Rfast2::hp.reg` (R); custom 2-stage |
| **Contoh hypothesis** | "Hurdle model dengan CatBoost 2-stage akan mencapai Poisson deviance ≥10% lebih rendah dari single Poisson pada data dengan >60% zero" |
| **Acceptance criteria** | Stage 1 AUC ≥0.75; stage 2 deviance improvement ≥5% |
| **Rejection criteria** | Stage 1 AUC <0.65; stage 2 tidak konvergen |
| **Sumber** | Mullahy (1986); Jose et al., Annals of Actuarial Science 18(3), 644–674 (2024) |

---

## 9. MULTI-OUTPUT REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi beberapa numeric targets simultan per observation. Berbeda dari multi-label (categorical) dan multi-task (related learning tasks) |
| **Problem yang didukung** | Forecasting multiple related quantities: sales, profit, returns per week |
| **Kapan digunakan** | Multiple continuous outputs; dependencies antar outputs |
| **Kapan tidak digunakan** | Single output; outputs tidak related |
| **Asumsi** | Dependencies antar outputs dapat dieksploitasi; output correlations informatif |
| **Data preprocessing** | Standardisasi target per output |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Averaged RMSE/MAE; per-target scores |
| **Hyperparameter penting** | `model_type` (problem transformation vs algorithm adaptation); `output_correlation` |
| **Kisaran parameter awal** | MultiOutputRegressor dengan base regressor |
| **CPU/GPU/RAM** | M × regressor cost |
| **Risiko leakage** | Sedang; output correlations harus dari training |
| **Failure modes** | Output dependencies diabaikan; averaged metric menyembunyikan per-output performance |
| **Teknik pembanding** | Independent regression per output; multi-task learning |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.multioutput.MultiOutputRegressor`; `sklearn.multioutput.RegressorChain` |
| **Contoh hypothesis** | "Regressor chain akan mencapai RMSE ≥5% lebih rendah dari independent regressors pada data dengan correlated outputs" |
| **Acceptance criteria** | RMSE improvement ≥5%; per-output R² ≥0.70 |
| **Rejection criteria** | RMSE tidak improvement; output correlations ≈0 |
| **Sumber** | Borchani et al., WIREs Data Mining and Knowledge Discovery 5(5), 216–233 (2015) |

---

## 10. DISTRIBUTIONAL REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Memodelkan seluruh conditional distribution P(Y|X), termasuk location, scale, shape/tail characteristics sebagai fungsi covariates |
| **Problem yang didukung** | Probabilistic forecasting, risk modeling, uncertainty quantification |
| **Kapan digunakan** | Metric NLL, CRPS, distributional log score; butuh uncertainty |
| **Kapan tidak digunakan** | Hanya butuh point prediction; distributional assumptions tidak valid |
| **Asumsi** | Conditional distribution dapat dimodelkan dengan parametrik family |
| **Data preprocessing** | Scaling; distribusi target |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | NLL, CRPS, coverage + sharpness |
| **Hyperparameter penting** | `distribution_family`; `location_formula`; `scale_formula` |
| **Kisaran parameter awal** | GAMLSS dengan distribusi yang sesuai |
| **CPU/GPU/RAM** | CPU-bound; MLE optimization |
| **Risiko leakage** | Rendah |
| **Failure modes** | Distribution family salah; overfitting parameter; convergence issues |
| **Teknik pembanding** | Quantile regression, heteroskedastic regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `gamlss` (R); `statsmodels`; `scipy.stats` |
| **Contoh hypothesis** | "GAMLSS dengan distribusi Gamma akan mencapai CRPS ≥15% lebih rendah dari linear regression pada data insurance claims" |
| **Acceptance criteria** | CRPS improvement ≥10%; coverage ≈ nominal |
| **Rejection criteria** | CRPS tidak improvement; distribution tidak konvergen |
| **Sumber** | Rigby & Stasinopoulos, JRSS-C 54(4), 507–554 (2005); Stasinopoulos et al., Statistical Modelling 18(3-4), 248–273 (2018) |

---

## 11. LEARNING-TO-RANK

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi urutan items dalam grup (query/session), bukan nilai absolut. Training data = queries + associated documents + relevance labels |
| **Problem yang didukung** | Search ranking, recommendation, ads ranking, product ranking |
| **Kapan digunakan** | Candidates grouped; evaluasi berdasarkan ordering; metric NDCG, MAP, MRR |
| **Kapan tidak digunakan** | Tidak ada grouping; evaluasi berdasarkan nilai absolut |
| **Asumsi** | Relevance labels tersedia; ordering matters; query-level grouping |
| **Data preprocessing** | Query-level grouping; relevance labels; fitur per candidate |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | NDCG@K, MAP, MRR, Precision@K, Recall@K |
| **Hyperparameter penting** | `group_column`; `objective` (pairwise/listwise); `eval_metric` |
| **Kisaran parameter awal** | NDCG@10 untuk eval; LambdaMART untuk model |
| **CPU/GPU/RAM** | GPU recommended untuk large-scale |
| **Risiko leakage** | **Tinggi**; split harus query-level, bukan row-level |
| **Failure modes** | Query-level split salah; metric tidak query-aware; position bias diabaikan |
| **Teknik pembanding** | Regression (salah); multiclass (salah) |
| **Ensemble compatibility** | Universal; rank ensemble |
| **Exact package** | `lightgbm.LGBMRanker`; `xgboost.XGBRanker`; `catboost.CatBoostRanker`; `allrank` |
| **Contoh hypothesis** | "LambdaMART akan mencapai NDCG@10 ≥0.50 pada data search ranking" |
| **Acceptance criteria** | NDCG@10 improvement ≥10% vs baseline; query-level split valid |
| **Rejection criteria** | NDCG@10 <0.30; row-level split |
| **Sumber** | Liu, Learning to Rank for Information Retrieval (Springer, 2011); Järvelin & Kekäläinen, ACM TOIS 20(4), 422–446 (2002) |

---

## 12. STANDARD SURVIVAL (TIME-TO-EVENT)

| Field | Detail |
|-------|--------|
| **Definisi** | Prediksi waktu sampai event dengan data tersensor. event=0 berarti belum terobservasi sampai akhir, bukan "tidak akan terjadi" |
| **Problem yang didukung** | Churn prediction, credit risk, clinical survival, predictive maintenance |
| **Kapan digunakan** | Target = time-to-event; ada censoring; butuh risk ranking |
| **Kapan tidak digunakan** | Tidak ada censoring; target binary biasa |
| **Asumsi** | Censoring at random (CAR); independent censoring |
| **Data preprocessing** | Time column; event indicator; censoring-aware preprocessing |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | C-index (concordance), time-dependent AUC, Brier(t), Integrated Brier Score |
| **Hyperparameter penting** | `objective` (survival:cox/aft); `aft_loss_distribution` |
| **Kisaran parameter awal** | Cox untuk PH; AFT untuk non-PH |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; censoring distribution harus di-estimasi tanpa test fold |
| **Failure modes** | event=0 dianggap "tidak terjadi"; accuracy/F1 digunakan (salah); censoring diabaikan |
| **Teknik pembanding** | Binary classification (salah); competing risks |
| **Ensemble compatibility** | Universal; risk score ensemble |
| **Exact package** | `lifelines` (Python); `scikit-survival` (Python); `xgboost` (objective='survival:cox'/'survival:aft'); `mlr3proba` (R); `mstate` (R) |
| **Contoh hypothesis** | "XGBoost-AFT dengan normal distribution akan mencapai C-index ≥0.75 pada data churn dengan 40% censoring" |
| **Acceptance criteria** | C-index ≥0.70; IBS improvement ≥10% vs baseline |
| **Rejection criteria** | C-index <0.60; menggunakan accuracy sebagai metric |
| **Sumber** | Harrell et al., Statistics in Medicine 15(4), 361–387 (1996); Graf et al., Statistics in Medicine 18(19), 2529–2545 (1999) |

---

## 13. COMPETING RISKS

| Field | Detail |
|-------|--------|
| **Definisi** | Survival dengan multiple mutually exclusive event types. Begitu satu event terjadi, event lain tidak lagi dapat terjadi dalam cara yang sama |
| **Problem yang didukung** | Clinical trials dengan multiple causes of death; credit risk dengan default vs prepayment |
| **Kapan digunakan** | Multiple event types; exclusive event occurrence; cause-specific prediction |
| **Kapan tidak digunakan** | Single event type (standard survival); events tidak exclusive |
| **Asumsi** | Events mutually exclusive; cause-specific hazard atau subdistribution hazard |
| **Data preprocessing** | Event type column; time column; censoring indicator |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Cause-specific C-index, CIF-based metrics, Brier(t) |
| **Hyperparameter penting** | `model_type` (cause-specific vs Fine-Gray); `event_type` |
| **Kisaran parameter awal** | Cause-specific Cox atau Fine-Gray |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; event-specific risk harus di-estimasi dengan benar |
| **Failure modes** | Mengabaikan competing events; binary classification salah |
| **Teknik pembanding** | Standard survival, multi-state |
| **Ensemble compatibility** | Universal |
| **Exact package** | `cmprsk` (R); `riskRegression` (R); `mstate` (R); `lifelines` (Python) |
| **Contoh hypothesis** | "Fine-Gray model akan mencapai cause-specific C-index ≥0.70 pada data transplant dengan 3 event types" |
| **Acceptance criteria** | CIF calibration baik; C-index improvement ≥0.05 vs naive |
| **Rejection criteria** | Mengabaikan competing events; C-index <0.60 |
| **Sumber** | Fine & Gray, JASA 94(446), 496–509 (1999); Putter et al., Statistics in Medicine 26(11), 2389–2430 (2007) |

---

## 14. RECURRENT EVENTS

| Field | Detail |
|-------|--------|
| **Definisi** | Event dapat terjadi berkali-kali pada entity yang sama. Time-to-first-event analysis membuang informasi event berikutnya |
| **Problem yang didukung** | Recurrent hospitalizations, recurrent infections, machine failures |
| **Kapan digunakan** | Entity repeated; multiple event times per entity; event dapat recur |
| **Kapan tidak digunakan** | Single event per entity (standard survival) |
| **Asumsi** | Recurrent events dapat dimodelkan dengan Andersen-Gill, PWP, atau WLW |
| **Data preprocessing** | Multiple event times per entity; gap time vs calendar time |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Recurrent-event likelihood, event-rate metrics |
| **Hyperparameter penting** | `model_type` (Andersen-Gill/PWP/WLW); `time_scale` |
| **Kisaran parameter awal** | Andersen-Gill untuk counting process |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; multiple events per entity → group structure |
| **Failure modes** | Hanya event pertama digunakan; mengabaikan within-entity correlation |
| **Teknik pembanding** | Standard survival, multi-state |
| **Ensemble compatibility** | Universal |
| **Exact package** | `survival` (R); `reReg` (R); `mets` (R) |
| **Contoh hypothesis** | "Andersen-Gill model akan mencapai event-rate accuracy ≥10% lebih baik dari time-to-first-event pada data recurrent hospitalizations" |
| **Acceptance criteria** | Event-rate improvement ≥10%; within-entity correlation ditangani |
| **Rejection criteria** | Hanya event pertama; mengabaikan recurrence |
| **Sumber** | Andersen & Gill, Annals of Statistics 10(4), 1100–1120 (1982) |

---

## 15. MULTI-STATE SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Subject berpindah antar states (Healthy → Disease → Hospitalized → Recovered → Relapse → Death). Generalisasi dari competing risks |
| **Problem yang didukung** | Disease progression modeling, multi-stage credit risk |
| **Kapan digunakan** | Multiple intermediate states; transitions antar states; goal = P(S_t=j|X) |
| **Kapan tidak digunakan** | Single event (standard survival); competing events tanpa intermediate states |
| **Asumsi** | Markov property (atau semi-Markov); transition intensities dapat diestimasi |
| **Data preprocessing** | State sequence per subject; transition times |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Transition probabilities, state occupancy |
| **Hyperparameter penting** | `transition_matrix`; `time_scale` |
| **Kisaran parameter awal** | Markov multi-state model |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; state sequence harus dari past data |
| **Failure modes** | Markov assumption violation; state definition unclear |
| **Teknik pembanding** | Competing risks, recurrent events |
| **Ensemble compatibility** | Universal |
| **Exact package** | `mstate` (R); `msm` (R); `flexsurv` (R) |
| **Contoh hypothesis** | "Multi-state model akan mencapai state occupancy accuracy ≥0.70 pada data disease progression" |
| **Acceptance criteria** | Transition probabilities well-calibrated; state occupancy accuracy ≥0.65 |
| **Rejection criteria** | Markov assumption gagal; state definition tidak jelas |
| **Sumber** | Putter et al., Statistics in Medicine 26(11), 2389–2430 (2007) |

---

## 16. INTERVAL-CENSORED SURVIVAL

| Field | Detail |
|-------|--------|
| **Definisi** | Event time diketahui terjadi dalam interval (L, R], bukan exact time. Contoh: test negatif di hari 100, positif di hari 130 → event di (100, 130] |
| **Problem yang didukung** | Disease onset dengan periodic testing; equipment failure dengan inspection interval |
| **Kapan digunakan** | Event time interval-censored; last negative test + first positive test |
| **Kapan tidak digunakan** | Right-censoring standar (standard survival) |
| **Asumsi** | Interval-censoring dapat dimodelkan dengan PH, AFT, PO, atau GOR |
| **Data preprocessing** | Lower bound; upper bound; interval definition |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Interval-censored likelihood |
| **Hyperparameter penting** | `model_type` (PH/AFT/PO/GOR) |
| **Kisaran parameter awal** | PH untuk interpretability; AFT untuk robustness |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; interval harus dari past data |
| **Failure modes** | Interval terlalu lebar; model salah |
| **Teknik pembanding** | Standard survival (salah); competing risks |
| **Ensemble compatibility** | Universal |
| **Exact package** | `icenReg` (R); `interval` (R) |
| **Contoh hypothesis** | "AFT model untuk interval-censored akan mencapai likelihood ≥10% lebih baik dari midpoint imputation" |
| **Acceptance criteria** | Interval-censored likelihood improvement ≥10% |
| **Rejection criteria** | Midpoint imputation digunakan; interval diabaikan |
| **Sumber** | Gómez Melis et al., Annual Review of Statistics and Its Application 13, 321–346 (2026) |

---

## 17. ANOMALY DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Deteksi observasi yang menyimpang dari normalitas. Dapat supervised, semi-supervised, atau unsupervised tergantung label availability |
| **Problem yang didukung** | Fraud detection (dengan labeled anomalies), intrusion detection, quality control |
| **Kapan digunakan** | Anomaly labels tersedia atau sebagian tersedia; anomaly concept known |
| **Kapan tidak digunakan** | Fraud dengan fully labeled historical cases (gunakan imbalanced binary classification); tidak ada anomaly concept |
| **Asumsi** | Anomaly concept stable; representative labels tersedia (supervised) |
| **Data preprocessing** | Standard; anomaly-specific preprocessing |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | PR-AUC, Precision@K, Recall@K, ROC-AUC (bila label tersedia) |
| **Hyperparameter penting** | `contamination`; `n_estimators`; `threshold` |
| **Kisaran parameter awal** | Isolation Forest; LOF; One-Class SVM |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; anomaly labels harus dari past data |
| **Failure modes** | Rare class ≠ anomaly; unsupervised ketika supervised lebih tepat; concept drift tidak ditangani |
| **Teknik pembanding** | Imbalanced binary classification (jika labels tersedia); novelty detection |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.ensemble.IsolationForest`; `sklearn.neighbors.LocalOutlierFactor`; `pyod` (Python) |
| **Contoh hypothesis** | "Supervised anomaly detection akan mencapai PR-AUC ≥0.80 pada data fraud dengan 20K labeled anomalies" |
| **Acceptance criteria** | PR-AUC ≥0.75; Precision@K ≥0.70 |
| **Rejection criteria** | PR-AUC <0.50; unsupervised digunakan padahal labels tersedia |
| **Sumber** | Chandola et al., ACM Computing Surveys 41(3), Article 15 (2009) |

---

## 18. ONE-CLASS / NOVELTY DETECTION

| Field | Detail |
|-------|--------|
| **Definisi** | Train hanya dengan class normal; production bisa normal atau anomaly. Model belajar boundary dari satu known class |
| **Problem yang didukung** | Novelty detection, fault detection, one-class classification |
| **Kapan digunakan** | Hanya normal samples tersedia saat training; anomaly concept unknown |
| **Kapan tidak digunakan** | Positive dan unlabeled tersedia (gunakan PU learning); anomaly labels tersedia (gunakan supervised) |
| **Asumsi** | Normal class dapat dimodelkan; anomaly berbeda dari normal |
| **Data preprocessing** | Standard; scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Anomaly score metrics; ROC-AUC (jika label tersedia) |
| **Hyperparameter penting** | `nu` (One-Class SVM); `contamination` |
| **Kisaran parameter awal** | One-Class SVM; Isolation Forest |
| **CPU/GPU/RAM** | O(n²) untuk SVM |
| **Risiko leakage** | Sedang; hanya normal samples |
| **Failure modes** | Normal class tidak representatif; anomaly mirip normal |
| **Teknik pembanding** | PU learning; anomaly detection |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.svm.OneClassSVM`; `sklearn.ensemble.IsolationForest` |
| **Contoh hypothesis** | "One-Class SVM akan mencapai recall ≥0.70 pada anomaly detection dengan hanya normal training samples" |
| **Acceptance criteria** | Recall ≥0.65; false positive rate <10% |
| **Rejection criteria** | Recall <0.50; normal class tidak representatif |
| **Sumber** | Bekker & Davis, Machine Learning 109(4), 719–760 (2020) |

---

## 19. OPEN-SET RECOGNITION

| Field | Detail |
|-------|--------|
| **Definisi** | Classification di mana test samples dapat berasal dari classes yang tidak terlihat saat training. Model harus classify known classes + reject unknown classes |
| **Problem yang didukung** | Face recognition, species classification, intrusion detection |
| **Kapan digunakan** | Unknown classes mungkin muncul di production; perlu reject option |
| **Kapan tidak digunakan** | Test classes ⊂ train classes (standard classification) |
| **Asumsi** | Known classes dapat diklasifikasi; unknown classes dapat dideteksi |
| **Data preprocessing** | Standard; known classes training |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Known-class accuracy + unknown rejection |
| **Hyperparameter penting** | `threshold`; `model_type` (OpenMax, etc.) |
| **Kisaran parameter awal** | OpenMax atau prototype-based |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; unknown classes tidak boleh ada di training |
| **Failure modes** | Unknown class dianggap known; threshold salah |
| **Teknik pembanding** | Anomaly detection; standard classification |
| **Ensemble compatibility** | Universal |
| **Exact package** | `openmax`; custom implementation |
| **Contoh hypothesis** | "Open-set recognition akan mencapai known-class accuracy ≥0.90 + unknown rejection ≥0.80 pada data dengan 20% unknown classes" |
| **Acceptance criteria** | Known accuracy ≥0.85; unknown rejection ≥0.75 |
| **Rejection criteria** | Unknown rejection <0.50 |
| **Sumber** | Geng et al., IEEE TPAMI 43(10), 3614–3631 (2021) |

---

## 20. SELECTIVE PREDICTION / ABSTENTION

| Field | Detail |
|-------|--------|
| **Definisi** | Model diperbolehkan menolak prediction ketika confidence terlalu rendah. Trade-off antara prediction error dan rejection probability |
| **Problem yang didukung** | Medical diagnosis, autonomous driving, high-stakes decisions |
| **Kapan digunakan** | High cost of mistake; human review available; model permitted to abstain |
| **Kapan tidak digunakan** | Tidak ada human review; harus selalu predict |
| **Asumsi** | Confidence score berkorelasi dengan accuracy; reject option berguna |
| **Data preprocessing** | Standard; confidence estimation |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Coverage, selective risk, risk-coverage curve, rejection rate |
| **Hyperparameter penting** | `rejection_threshold`; `confidence_estimator` |
| **Kisaran parameter awal** | Threshold untuk coverage 0.80 |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; threshold harus di-tune pada validation |
| **Failure modes** | Confidence tidak calibrated; threshold salah |
| **Teknik pembanding** | Standard classification; conformal prediction |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation; `MAPIE` (Python) |
| **Contoh hypothesis** | "Selective prediction dengan coverage 0.80 akan mencapai selective risk ≤0.05 pada data medical diagnosis" |
| **Acceptance criteria** | Selective risk ≤0.05 pada coverage 0.80 |
| **Rejection criteria** | Selective risk >0.15; confidence tidak calibrated |
| **Sumber** | Chow, IEEE Transactions on Information Theory 16(1), 41–46 (1970) |

---

## 21. POSITIVE–UNLABELED (PU) LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Binary classifier dengan hanya positive labels dan unlabeled data. Unlabeled adalah campuran positive dan negative, bukan negative |
| **Problem yang didukung** | Fraud detection (confirmed fraud + unknown), disease gene identification |
| **Kapan digunakan** | Positive labels trustworthy; negative labels unavailable; unlabeled mixture |
| **Kapan tidak digunakan** | Negative labels tersedia (standard classification); one-class only |
| **Asumsi** | Unlabeled contains both positive dan negative; positive labels clean |
| **Data preprocessing** | Standard; positive-unlabeled split |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | PU-specific risk, corrected metrics |
| **Hyperparameter penting** | `prior` (class prior); `method` (two-step/biased SVM) |
| **Kisaran parameter awal** | Elkan-Noto atau biased SVM |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; unlabeled ≠ negative |
| **Failure modes** | Unlabeled dianggap negative; prior salah |
| **Teknik pembanding** | One-class; supervised classification (jika labels tersedia) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pulearn` (Python); `PUlearning` (R) |
| **Contoh hypothesis** | "PU learning akan mencapai recall ≥0.70 pada fraud detection dengan 20K confirmed fraud dan 2M unlabeled" |
| **Acceptance criteria** | Recall ≥0.65; PU risk improvement vs naive |
| **Rejection criteria** | Unlabeled dianggap negative; recall <0.50 |
| **Sumber** | Bekker & Davis, Machine Learning 109(4), 719–760 (2020) |

---

## 22. SEMI-SUPERVISED LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Sedikit labeled + banyak unlabeled. Unlabeled digunakan untuk memperbaiki decision boundary |
| **Problem yang didukung** | Text classification, image classification dengan limited labels |
| **Kapan digunakan** | Labeled data sedikit; unlabeled banyak; cluster assumption holds |
| **Kapan tidak digunakan** | Unlabeled tidak informatif; labeled cukup |
| **Asumsi** | Cluster assumption; manifold assumption |
| **Data preprocessing** | Standard; pseudo-labeling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Standard classification metrics |
| **Hyperparameter penting** | `confidence_threshold`; `method` (self-training/co-training) |
| **Kisaran parameter awal** | Confidence threshold 0.90 |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; pseudo-labels bisa memperkuat error |
| **Failure modes** | Pseudo-label noisy; confirmation bias |
| **Teknik pembanding** | PU learning; self-training |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.semi_supervised.LabelPropagation`; `semisupervised` (Python) |
| **Contoh hypothesis** | "Self-training dengan confidence threshold 0.90 akan mencapai accuracy ≥10% lebih tinggi dari supervised-only pada dataset dengan 5% labeled" |
| **Acceptance criteria** | Accuracy improvement ≥8% vs supervised-only |
| **Rejection criteria** | Accuracy drop; pseudo-label accuracy <70% |
| **Sumber** | Van Engelen & Hoos, Machine Learning 109(2), 373–440 (2020) |

---

## 23. WEAK / NOISY SUPERVISION

| Field | Detail |
|-------|--------|
| **Definisi** | Training labels noisy, heuristic, conflicting, atau dari multiple sources. Data programming menggabungkan noisy labeling functions menjadi probabilistic labels |
| **Problem yang didukung** | Training set creation, distant supervision, crowd-sourced labels |
| **Kapan digunakan** | Label quality rendah; multiple label sources; conflicting labels |
| **Kapan tidak digunakan** | Clean ground truth labels tersedia |
| **Asumsi** | Label noise dapat dimodelkan; labeling functions informatif |
| **Data preprocessing** | Labeling functions; generative model untuk denoising |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Robust loss; evaluation pada clean subset |
| **Hyperparameter penting** | `labeling_functions`; `denoising_model` |
| **Kisaran parameter awal** | Snorkel-style generative model |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; label noise dari multiple sources |
| **Failure modes** | Confirmation bias; noisy labels memperkuat error |
| **Teknik pembanding** | Clean labels (supervised); PU learning |
| **Ensemble compatibility** | Universal |
| **Exact package** | `snorkel` (Python); `cleanlab` (Python) |
| **Contoh hypothesis** | "Data programming dengan Snorkel akan mencapai accuracy ≥5% lebih tinggi dari majority vote labeling functions" |
| **Acceptance criteria** | Accuracy improvement ≥5%; label noise di-handle |
| **Rejection criteria** | Accuracy drop; noisy labels dominan |
| **Sumber** | Ratner et al., NeurIPS 2016; Data Programming: Creating Large Training Sets, Quickly |

---

## 24. MULTIPLE-INSTANCE LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Observations disusun dalam bags; label hanya di bag-level, bukan instance-level |
| **Problem yang didukung** | Patient (multiple measurements), customer (multiple transactions), company (multiple employees) |
| **Kapan digunakan** | Multiple instances per entity; label hanya di entity level |
| **Kapan tidak digunakan** | Label per instance; tidak ada bag structure |
| **Asumsi** | Bag label determined by instances; instance-level label unknown |
| **Data preprocessing** | Bag construction; bag-level aggregation |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Bag-level AUC/F1/accuracy |
| **Hyperparameter penting** | `aggregation_method` (max/mean/attention) |
| **Kisaran parameter awal** | Attention-based MIL |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; bag structure harus dijaga |
| **Failure modes** | Bag dipecah menjadi rows; instance-level label diasumsikan |
| **Teknik pembanding** | Standard classification; hierarchical |
| **Ensemble compatibility** | Universal |
| **Exact package** | `MIL` (Python); `milr` (R) |
| **Contoh hypothesis** | "Attention-based MIL akan mencapai bag-level AUC ≥0.80 pada data patient dengan multiple measurements" |
| **Acceptance criteria** | Bag-level AUC ≥0.75; bag structure preserved |
| **Rejection criteria** | Bag structure diabaikan; AUC <0.60 |
| **Sumber** | Carbonneau et al., Pattern Recognition 77, 329–353 (2018) |

---

## 25. PARTIAL-LABEL LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Setiap training example diberikan candidate set of labels, hanya satu yang benar. y_i ∈ S_i, |S_i| > 1 |
| **Problem yang didukung** | Image tagging dengan candidate labels, crowdsourced annotation |
| **Kapan digunakan** | Candidate label set tersedia; tepat satu label benar; tidak tahu yang mana |
| **Kapan tidak digunakan** | Multiple labels benar (multi-label); single label (standard classification) |
| **Asumsi** | Tepat satu label benar dalam candidate set; candidate set informatif |
| **Data preprocessing** | Candidate set encoding; partial-label risk |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Partial-label risk, accuracy |
| **Hyperparameter penting** | `loss` (partial-label loss); `candidate_set_size` |
| **Kisaran parameter awal** | Partial-label risk minimization |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; candidate set dari training |
| **Failure modes** | Candidate set dianggap multi-label; ambiguous |
| **Teknik pembanding** | Multi-label; standard classification |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation; `partial-label` (GitHub) |
| **Contoh hypothesis** | "Partial-label learning akan mencapai accuracy ≥5% lebih tinggi dari majority vote pada candidate sets" |
| **Acceptance criteria** | Accuracy improvement ≥5% vs majority vote |
| **Rejection criteria** | Accuracy drop; candidate set diabaikan |
| **Sumber** | Cour et al., JMLR 12, 1501–1536 (2011) |

---

## 26. COMPLEMENTARY-LABEL LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Label menyatakan class yang **bukan** jawaban. true class unknown; label = excluded class |
| **Problem yang didukung** | Annotation di mana hanya "not this" tersedia |
| **Kapan digunakan** | Complementary labels tersedia; true label unknown |
| **Kapan tidak digunakan** | True labels tersedia; partial-label (candidate set) |
| **Asumsi** | Complementary labels informatif; loss symmetric condition |
| **Data preprocessing** | Complementary label encoding |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Complementary risk |
| **Hyperparameter penting** | `loss` (symmetric); `complementary_ratio` |
| **Kisaran parameter awal** | Symmetric loss |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang |
| **Failure modes** | Complementary dianggap true label; symmetric condition gagal |
| **Teknik pembanding** | Partial-label; standard classification |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation |
| **Contoh hypothesis** | "Complementary-label learning akan mencapai accuracy ≥5% lebih tinggi dari naived approach pada data dengan complementary labels" |
| **Acceptance criteria** | Accuracy improvement ≥5% |
| **Rejection criteria** | Accuracy drop; complementary dianggap true |
| **Sumber** | Ishida et al., NeurIPS 2017 |

---

## 27. UPLIFT / HETEROGENEOUS TREATMENT EFFECT

| Field | Detail |
|-------|--------|
| **Definisi** | Estimate heterogeneous causal effect τ(x) = E[Y(1)−Y(0)|X=x]. Pertanyaannya bukan "siapa yang akan membeli?" tetapi "siapa yang akan membeli **karena** diberi treatment?" |
| **Problem yang didukung** | Marketing campaign targeting, personalized promotion, treatment selection |
| **Kapan digunakan** | Ada treatment/control groups; butuh estimate causal effect; metric Qini, AUUC, uplift@K |
| **Kapan tidak digunakan** | Tidak ada treatment assignment; hanya butuh prediction |
| **Asumsi** | Unconfoundedness; overlap; SUTVA |
| **Data preprocessing** | Treatment indicator; control group; causal preprocessing |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Qini, AUUC, uplift@K |
| **Hyperparameter penting** | `learner` (S/T/X-learner); `treatment_column` |
| **Kisaran parameter awal** | T-learner atau X-learner |
| **CPU/GPU/RAM** | 2× model cost |
| **Risiko leakage** | **Tinggi**; treatment assignment harus random atau confounders di-handle |
| **Failure modes** | Confounding diabaikan; treatment assignment tidak random; Qini misleading |
| **Teknik pembanding** | Standard classification; contextual bandit |
| **Ensemble compatibility** | Universal |
| **Exact package** | `causalml` (Python); `uplift` (R); `econml` (Python) |
| **Contoh hypothesis** | "X-learner akan mencapai Qini ≥0.05 pada data marketing campaign dengan treatment/control" |
| **Acceptance criteria** | Qini ≥0.04; AUUC improvement ≥10% vs baseline |
| **Rejection criteria** | Qini <0.02; confounding diabaikan |
| **Sumber** | Devriendt et al., Big Data 6(1), 36–56 (2018) |

---

## 28. CONTEXTUAL BANDIT / POLICY LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Sistem memilih action berdasarkan context untuk maximize future reward. Bukan prediction, tetapi sequential decision |
| **Problem yang didukung** | Personalized recommendation, ad selection, dynamic pricing |
| **Kapan digunakan** | Sequential action selection; context available; reward observed |
| **Kapan tidak digunakan** | Single prediction; no action selection |
| **Asumsi** | Contextual bandit framework; reward observed for chosen action |
| **Data preprocessing** | Context features; action set; reward |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | IPS/DR policy value, regret |
| **Hyperparameter penting** | `algorithm` (LinUCB/Thompson); `exploration_rate` |
| **Kisaran parameter awal** | LinUCB atau Thompson Sampling |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | **Tinggi**; reward hanya untuk chosen action (counterfactual) |
| **Failure modes** | Full feedback diasumsikan; exploration/exploitation imbalance |
| **Teknik pembanding** | Uplift modeling; reinforcement learning |
| **Ensemble compatibility** | Universal |
| **Exact package** | `contextualbandits` (Python); `vowpal wabbit` |
| **Contoh hypothesis** | "LinUCB akan mencapai regret ≥20% lebih rendah dari ε-greedy pada data news recommendation" |
| **Acceptance criteria** | Regret reduction ≥15%; policy value improvement |
| **Rejection criteria** | Regret tidak improvement; counterfactual diabaikan |
| **Sumber** | Li et al., WWW 2010 |

---

## 29. COST-SENSITIVE DECISION

| Field | Detail |
|-------|--------|
| **Definisi** | FP dan FN punya biaya berbeda. Decision optimal berdasarkan expected cost, bukan posterior probability terbesar |
| **Problem yang didukung** | Fraud detection (FN mahal), medical diagnosis (FN mahal), spam filtering (FP mahal) |
| **Kapan digunakan** | Misclassification costs berbeda; metric expected cost/utility |
| **Kapan tidak digunakan** | FP dan FN cost sama |
| **Asumsi** | Cost matrix diketahui; probability calibrated |
| **Data preprocessing** | Standard; cost matrix |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Expected cost, utility, net benefit |
| **Hyperparameter penting** | `cost_matrix`; `threshold` (dari cost) |
| **Kisaran parameter awal** | Threshold = cost_FP/(cost_FP+cost_FN) |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; cost matrix dari domain |
| **Failure modes** | Cost matrix salah; probability tidak calibrated; threshold default |
| **Teknik pembanding** | Standard classification |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn` dengan custom threshold; `costcla` (Python) |
| **Contoh hypothesis** | "Cost-sensitive threshold akan mencapai expected cost ≥30% lebih rendah dari threshold 0.5 pada data fraud dengan FN cost 10× FP cost" |
| **Acceptance criteria** | Expected cost reduction ≥25% vs threshold 0.5 |
| **Rejection criteria** | Cost matrix diabaikan; threshold 0.5 digunakan |
| **Sumber** | Elkan, IJCAI 2001 |

---

## 30. FRACTIONAL REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Target 0 ≤ Y ≤ 1 berupa proportion/share/rate. Ordinary linear regression dapat menghasilkan prediksi di luar [0,1] |
| **Problem yang didukung** | Conversion rate, market share, participation rate, utilization |
| **Kapan digunakan** | Target ∈ [0,1]; proportion semantics; exact zeros/ones mungkin |
| **Kapan tidak digunakan** | Target binary (classification); target kontinu di luar [0,1] |
| **Asumsi** | Conditional mean ∈ (0,1); quasi-likelihood |
| **Data preprocessing** | Tidak ada arbitrary transformation; langsung model conditional mean |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Fractional deviance |
| **Hyperparameter penting** | `link_function` (logit/probit); `model_type` |
| **Kisaran parameter awal** | Logit link; quasi-Binomial |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Ordinary regression digunakan; prediksi di luar [0,1] |
| **Teknik pembanding** | Beta regression; Tweedie |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels` dengan custom; `glm` (R) |
| **Contoh hypothesis** | "Fractional regression akan mencapai fractional deviance ≥15% lebih rendah dari linear regression pada data conversion rate" |
| **Acceptance criteria** | Fractional deviance improvement ≥10%; prediksi ∈ [0,1] |
| **Rejection criteria** | Prediksi di luar [0,1]; linear regression digunakan |
| **Sumber** | Papke & Wooldridge, Journal of Applied Econometrics 11(6), 619–632 (1996) |

---

## 31. BETA REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Regression untuk continuous proportions 0 < Y < 1 dengan Beta distribution indexed by mean dan dispersion parameters |
| **Problem yang didukung** | Percentage fat, humidity ratio, market fraction |
| **Kapan digunakan** | Target continuous proportion ∈ (0,1); tidak ada exact 0/1 |
| **Kapan tidak digunakan** | Exact 0/1 (gunakan fractional/zero-one inflated); target binary |
| **Asumsi** | Beta distribution untuk conditional distribution |
| **Data preprocessing** | Standard; tidak ada scaling target |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Beta log-likelihood/deviance |
| **Hyperparameter penting** | `link_function` (logit/probit); `dispersion_model` |
| **Kisaran parameter awal** | Logit link; beta regression |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Exact 0/1 ada; Beta distribution tidak sesuai |
| **Teknik pembanding** | Fractional regression; extended beta |
| **Ensemble compatibility** | Universal |
| **Exact package** | `betareg` (R) |
| **Contoh hypothesis** | "Beta regression akan mencapai beta deviance ≥10% lebih rendah dari linear regression pada data percentage fat" |
| **Acceptance criteria** | Beta deviance improvement ≥10% |
| **Rejection criteria** | Exact 0/1 ada; Beta distribution gagal |
| **Sumber** | Ferrari & Cribari-Neto, Journal of Applied Statistics 31(7), 799–815 (2004) |

---

## 32. COMPOSITIONAL REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Target vector of proportions yang jumlahnya = 1. Data hidup pada simplex, bukan Euclidean space |
| **Problem yang didukung** | Food spending allocation, portfolio allocation, time allocation |
| **Kapan digunakan** | Multiple continuous outputs; nonnegative; sum ≈1 |
| **Kapan tidak digunakan** | Outputs independent (multi-output regression); tidak ada sum constraint |
| **Asumsi** | Compositional data; log-ratio structure |
| **Data preprocessing** | Log-ratio transformation (ALR/CLR/ILR) |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Aitchison distance |
| **Hyperparameter penting** | `transformation` (ALR/CLR/ILR) |
| **Kisaran parameter awal** | ILR transformation; multivariate regression |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Sum constraint diabaikan; ordinary multi-output regression digunakan |
| **Teknik pembanding** | Multi-output regression; Beta regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `compositions` (R); `scikit-bio` (Python) |
| **Contoh hypothesis** | "Compositional regression dengan ILR akan mencapai Aitchison distance ≥20% lebih rendah dari ordinary multi-output regression" |
| **Acceptance criteria** | Aitchison distance improvement ≥15% |
| **Rejection criteria** | Sum constraint diabaikan |
| **Sumber** | Aitchison, The Statistical Analysis of Compositional Data (Chapman & Hall, 1986) |

---

## 33. TWEEDIE / SEMI-CONTINUOUS REGRESSION

| Field | Detail |
|-------|--------|
| **Definisi** | Outcome dengan probability mass di zero + continuous positive values. Tweedie compound Poisson-Gamma menghubungkan Poisson event counts dengan Gamma amounts |
| **Problem yang didukung** | Insurance claim amount, customer spend, medical cost |
| **Kapan digunakan** | Banyak zero + positive continuous; metric Tweedie deviance |
| **Kapan tidak digunakan** | Pure count (count regression); pure continuous tanpa zero |
| **Asumsi** | Compound Poisson-Gamma; 1 < p < 2 |
| **Data preprocessing** | Standard; exposure |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Tweedie deviance |
| **Hyperparameter penting** | `power` (Tweedie variance power); `alpha` |
| **Kisaran parameter awal** | power ∈ (1.2, 1.8) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | power salah; Tweedie distribution tidak sesuai |
| **Teknik pembanding** | Zero-inflated; frequency-severity |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.TweedieRegressor`; `glm` (R); `lightgbm.LGBMRegressor(objective='tweedie')` |
| **Contoh hypothesis** | "Tweedie regression dengan power=1.5 akan mencapai Tweedie deviance ≥10% lebih rendah dari frequency-severity decomposition" |
| **Acceptance criteria** | Tweedie deviance improvement ≥5% |
| **Rejection criteria** | Power tidak optimal; deviance tidak improvement |
| **Sumber** | Jørgensen & Paes de Souza, Scandinavian Actuarial Journal 1994(1), 69–89 (1994) |

---

## 34. HIERARCHICAL CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Class berada dalam tree/DAG. Laptop → Desktop lebih baik daripada Laptop → iPhone |
| **Problem yang didukung** | Product categorization, species classification, disease taxonomy |
| **Kapan digunakan** | Target taxonomy present; hierarchical relationships; metric hierarchical F1 |
| **Kapan tidak digunakan** | Flat classes (standard multiclass) |
| **Asumsi** | Taxonomy known; hierarchical relationships informatif |
| **Data preprocessing** | Taxonomy encoding; hierarchical features |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Hierarchical precision/recall/F1, path loss |
| **Hyperparameter penting** | `hierarchy`; `hierarchical_loss` |
| **Kisaran parameter awal** | Hierarchical loss dengan partial credit |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; taxonomy dari domain |
| **Failure modes** | Flat classification digunakan; hierarchy diabaikan |
| **Teknik pembanding** | Multiclass; multi-label |
| **Ensemble compatibility** | Universal |
| **Exact package** | `hiclass` (Python); custom |
| **Contoh hypothesis** | "Hierarchical classification akan mencapai hierarchical F1 ≥0.70, mengalahkan flat multiclass yang mengabaikan taxonomy" |
| **Acceptance criteria** | Hierarchical F1 improvement ≥10% vs flat |
| **Rejection criteria** | Hierarchy diabaikan; flat classification digunakan |
| **Sumber** | Silla & Freitas, Data Mining and Knowledge Discovery 22(1-2), 31–72 (2011) |

---

## 35. LABEL DISTRIBUTION LEARNING

| Field | Detail |
|-------|--------|
| **Definisi** | Target = distribution/degree-of-description atas labels. Bukan single-label atau multi-label, tetapi distribution |
| **Problem yang didukung** | Facial age estimation, emotion recognition, gene expression |
| **Kapan digunakan** | Multiple output values sum ≈1; degree of label description |
| **Kapan tidak digunakan** | Single label; multiple binary labels (multi-label) |
| **Asumsi** | Label distribution informatif; degree-of-description relevan |
| **Data preprocessing** | Label distribution encoding |
| **Dataset scale** | Menengah |
| **Metric compatibility** | KL, Chebyshev, Clark, cosine |
| **Hyperparameter penting** | `loss` (KL/Clark); `distribution_constraint` |
| **Kisaran parameter awal** | KL divergence sebagai loss |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Rendah |
| **Failure modes** | Dianggap multi-label; distribution diabaikan |
| **Teknik pembanding** | Multi-label; soft classification |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom implementation |
| **Contoh hypothesis** | "Label distribution learning akan mencapai KL divergence ≥15% lebih rendah dari multi-label classification pada data facial age estimation" |
| **Acceptance criteria** | KL improvement ≥10%; distribution well-calibrated |
| **Rejection criteria** | Multi-label digunakan; KL tidak improvement |
| **Sumber** | Geng, IEEE TKDE 28(7), 1734–1748 (2016) |

---

## 36. SET-VALUED CLASSIFICATION

| Field | Detail |
|-------|--------|
| **Definisi** | Output boleh beberapa candidate labels dengan trade-off antara ambiguity/set size dan reliability |
| **Problem yang didukung** | Diagnosis candidate set, conformal prediction |
| **Kapan digunakan** | Uncertainty tinggi; model boleh output set; coverage + set size metric |
| **Kapan tidak digunakan** | Harus single prediction; multi-label (semua benar) |
| **Asumsi** | Set-valued prediction berguna; coverage guarantee |
| **Data preprocessing** | Standard; conformal calibration |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Coverage, average set size, ambiguity |
| **Hyperparameter penting** | `alpha` (coverage level); `conformal_method` |
| **Kisaran parameter awal** | Conformal prediction dengan alpha=0.10 |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; calibration set terpisah |
| **Failure modes** | Set size terlalu besar; coverage tidak tercapai |
| **Teknik pembanding** | Multi-label; selective prediction |
| **Ensemble compatibility** | Universal |
| **Exact package** | `MAPIE` (Python); `crepes` (Python) |
| **Contoh hypothesis** | "Conformal prediction akan mencapai coverage 0.90 dengan set size ≤3 pada data medical diagnosis" |
| **Acceptance criteria** | Coverage ≥0.90; set size ≤3 |
| **Rejection criteria** | Coverage <0.80; set size >5 |
| **Sumber** | Chzhen et al., arXiv:2102.12318 (2021) |

---

## 37. STRUCTURED PREDICTION

| Field | Detail |
|-------|--------|
| **Definisi** | Output berupa sequence/tree/set/graph dengan constraint antar-output. Bukan ordinary multi-output |
| **Problem yang didukung** | Sequence labeling, entity matching, schedule assignment |
| **Kapan digunakan** | Output tidak dapat dipisahkan; constraint antar-output penting |
| **Kapan tidak digunakan** | Output independent (multi-output regression) |
| **Asumsi** | Structured output space; dependencies antar-output |
| **Data preprocessing** | Structured output encoding |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Task-specific structured loss |
| **Hyperparameter penting** | `structured_loss`; `constraint_type` |
| **Kisaran parameter awal** | Structured SVM |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Sedang; struktur output dari domain |
| **Failure modes** | Output independent diasumsikan; constraint diabaikan |
| **Teknik pembanding** | Multi-output regression; independent prediction |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pystruct` (Python); `structured` (R) |
| **Contoh hypothesis** | "Structured prediction akan mencapai structured loss ≥20% lebih rendah dari independent prediction pada data sequence labeling" |
| **Acceptance criteria** | Structured loss improvement ≥15% |
| **Rejection criteria** | Constraint diabaikan; independent prediction digunakan |
| **Sumber** | Tsochantaridis et al., JMLR 6, 1453–1484 (2005) |

---

## 38. MONOTONIC PREDICTION

| Field | Detail |
|-------|--------|
| **Definisi** | Constraint monotonic: X ↑ ⇒ Y tidak boleh ↓ untuk feature yang ditetapkan. Contoh credit risk: debt ↑ tidak boleh menyebabkan predicted risk ↓ |
| **Problem yang didukung** | Credit scoring, insurance pricing, medical risk |
| **Kapan digunakan** | Domain constraint monotonic; regulatory requirement |
| **Kapan tidak digunakan** | Tidak ada monotonic constraint |
| **Asumsi** | Monotonic relationship valid; constraint dari domain |
| **Data preprocessing** | Monotonic constraint specification |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Accuracy + monotonicity violation |
| **Hyperparameter penting** | `monotonic_constraints`; `model_type` (monotonic GBM) |
| **Kisaran parameter awal** | Monotonic constraint untuk debt ↑ |
| **CPU/GPU/RAM** | Model-dependent |
| **Risiko leakage** | Rendah |
| **Failure modes** | Constraint diabaikan; model violates monotonicity |
| **Teknik pembanding** | Standard regression; constrained optimization |
| **Ensemble compatibility** | Universal |
| **Exact package** | `lightgbm.LGBMRegressor(monotone_constraints=...)`; `xgboost.XGBRegressor(monotone_constraints=...)` |
| **Contoh hypothesis** | "Monotonic GBM akan mencapai accuracy setara dengan standard GBM + 0 monotonicity violations pada data credit risk" |
| **Acceptance criteria** | Monotonicity violations = 0; accuracy tidak drop >2% |
| **Rejection criteria** | Monotonicity violations >0; accuracy drop >5% |
| **Sumber** | Cano et al., Neurocomputing 354, 2–15 (2019) |

---

## Ringkasan Decision Hierarchy

```
START
 │
 ▼
Is target available?
 │
 ├── NO
 │    ├── goal = rare/deviant observations → UNSUPERVISED ANOMALY DETECTION
 │    └── otherwise → UNSUPERVISED / clustering
 │
 └── YES
      │
      ▼
Does each observation have multiple simultaneous labels?
      ├── YES → MULTI-LABEL
      │        ├── hierarchical? → HIERARCHICAL MULTI-LABEL
      │        ├── candidate set, one correct? → PARTIAL-LABEL
      │        ├── excluded class? → COMPLEMENTARY-LABEL
      │        └── distribution? → LABEL DISTRIBUTION LEARNING
      │
      └── NO
           │
           ▼
Is outcome time-to-event with censoring?
           ├── YES
           │    ├── competing events? → COMPETING RISKS
           │    ├── recurrent events? → RECURRENT EVENTS
           │    ├── intermediate states? → MULTI-STATE
           │    ├── interval-censored? → INTERVAL-CENSORED SURVIVAL
           │    └── otherwise → STANDARD SURVIVAL
           │
           └── NO
                │
                ▼
Are candidates grouped and evaluated by ordering?
                ├── YES → LEARNING-TO-RANK
                └── NO
                     │
                     ▼
Is target categorical?
                     ├── YES
                     │    ├── natural order? → ORDINAL
                     │    ├── hierarchy? → HIERARCHICAL
                     │    └── NO → CLASSIFICATION
                     │
                     └── NO
                          │
                          ▼
Is target non-negative integer?
                          ├── YES
                          │    ├── excess zeros? → ZERO-INFLATED / HURDLE
                          │    └── otherwise → COUNT REGRESSION
                          │
                          └── NO
                               │
                               ▼
Is target in [0,1]?
                               ├── YES
                               │    ├── proportion/share? → FRACTIONAL / BETA
                               │    ├── sum to 1? → COMPOSITIONAL
                               │    └── binary? → CLASSIFICATION
                               │
                               └── NO
                                    │
                                    ▼
Is target numeric?
                                    ├── YES
                                    │    ├── semi-continuous? → TWEEDIE
                                    │    ├── multiple outputs? → MULTI-OUTPUT REGRESSION
                                    │    ├── distributional? → DISTRIBUTIONAL REGRESSION
                                    │    ├── quantile? → QUANTILE REGRESSION
                                    │    └── otherwise → REGRESSION
                                    │
                                    └── NO → OTHER / STRUCTURED
```

---

## Referensi Utama

1. **Gutiérrez et al.** (2016). Ordinal Regression Methods: Survey and Experimental Study. *IEEE TKDE*, 28(1), 127–146. 
2. **Tsoumakas & Katakis** (2007). Multi-Label Classification: An Overview. *IJDM*, 3(3), 1–13. 
3. **Gneiting & Raftery** (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359–378. 
4. **Liu, T.-Y.** (2011). *Learning to Rank for Information Retrieval*. Springer. 
5. **Harrell et al.** (1996). Multivariable Prognostic Models. *Statistics in Medicine*, 15(4), 361–387. 
6. **Chandola et al.** (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), Article 15. 
7. **Mullahy** (1986). Specification and Testing of Some Modified Count Data Models. *Journal of Econometrics*, 33(3), 341–365. 
8. **Bekker & Davis** (2020). Learning from Positive and Unlabeled Data: A Survey. *Machine Learning*, 109(4), 719–760. 
9. **Devriendt et al.** (2018). A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling. *Big Data*, 6(1), 36–56. 
10. **Elkan** (2001). The Foundations of Cost-Sensitive Learning. *IJCAI 2001*. 
11. **Papke & Wooldridge** (1996). Econometric Methods for Fractional Response Variables. *Journal of Applied Econometrics*, 11(6), 619–632. 
12. **Ferrari & Cribari-Neto** (2004). Beta Regression for Modelling Rates and Proportions. *Journal of Applied Statistics*, 31(7), 799–815. 
13. **Aitchison** (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall. 
14. **Borchani et al.** (2015). A Survey on Multi-Output Regression. *WIREs DMKD*, 5(5), 216–233. 
15. **Caruana** (1997). Multitask Learning. *Machine Learning*, 28(1), 41–75. 
16. **Ratner et al.** (2016). Data Programming: Creating Large Training Sets, Quickly. *NeurIPS 2016*. 
17. **Carbonneau et al.** (2018). Multiple Instance Learning: A Survey of Problem Characteristics and Applications. *Pattern Recognition*, 77, 329–353. 
18. **Geng et al.** (2021). Recent Advances in Open Set Recognition: A Survey. *IEEE TPAMI*, 43(10), 3614–3631. 
19. **Chow** (1970). On Optimum Recognition Error and Reject Tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41–46. 
20. **Rigby & Stasinopoulos** (2005). Generalized Additive Models for Location, Scale and Shape. *JRSS-C*, 54(4), 507–554. 
21. **Fine & Gray** (1999). A Proportional Hazards Model for the Subdistribution of a Competing Risk. *JASA*, 94(446), 496–509. 
22. **Putter et al.** (2007). Tutorial in Biostatistics: Competing Risks and Multi-State Models. *Statistics in Medicine*, 26(11), 2389–2430. 
23. **Silla & Freitas** (2011). A Survey of Hierarchical Classification Across Different Application Domains. *DMKD*, 22(1-2), 31–72. 
24. **Cour et al.** (2011). Learning from Partial Labels. *JMLR*, 12, 1501–1536. 
25. **Ishida et al.** (2017). Learning from Complementary Labels. *NeurIPS 2017*. 
26. **Geng** (2016). Label Distribution Learning. *IEEE TKDE*, 28(7), 1734–1748. 
27. **Chzhen et al.** (2021). Set-Valued Classification — Overview via a Unified Framework. *arXiv:2102.12318*. 
28. **Tsochantaridis et al.** (2005). Large Margin Methods for Structured and Interdependent Output Variables. *JMLR*, 6, 1453–1484. 
29. **Cano et al.** (2019). Monotonic Classification: An Overview on Algorithms, Performance Measures and Data Sets. *Neurocomputing*, 354, 2–15. 
30. **Li et al.** (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *WWW 2010*. 
31. **Andersen & Gill** (1982). Cox's Regression Model for Counting Processes. *Annals of Statistics*, 10(4), 1100–1120. 
32. **Gómez Melis et al.** (2026). Regression Models with Interval-Censored Variables. *Annual Review of Statistics and Its Application*, 13, 321–346. 
33. **Jørgensen & Paes de Souza** (1994). Fitting Tweedie's Compound Poisson Model to Insurance Claims Data. *Scandinavian Actuarial Journal*, 1994(1), 69–89. 
34. **Gneiting** (2011). Making and Evaluating Point Forecasts. *JASA*, 106(494), 746–762. 
35. **Koenker & Bassett** (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50. 
36. **Fawcett** (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874. 

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*