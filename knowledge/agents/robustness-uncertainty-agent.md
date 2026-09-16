# Robustness & Uncertainty Agent — Dokumentasi Teknis Tier Berat

> **Versi:** 1.0 | **Tier:** Berat (Advanced / Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini menyediakan spesifikasi operasional lengkap untuk teknik-teknik robustness dan uncertainty pada data tabular. Setiap entri mencakup 20 field yang dirancang untuk konsumsi AI agents dalam pengambilan keputusan otomatis.

---

## Daftar Isi

**A. Stochastic Training Stability**
1. Multi-Seed Retraining
2. Fixed-Split Seed Test
3. Split Stability Test
4. Crossed Seed × Split Design
5. Paired Seed Comparison
6. Seed Win-Rate
7. Worst-Seed / Lower-Tail Performance
8. Failure-Rate Stability

**B. Split / Resampling Stability**
9. Fold SD
10. Fold Range & IQR
11. Fold-Wise Paired Delta
12. Repeated K-Fold Stability
13. Repeated Nested CV
14. Algorithmic Stability

**C. Variance Decomposition**
15. Seed Variance
16. Split Variance
17. Seed × Split Interaction
18. Mixed-Effects Decomposition

**D. Performance Uncertainty**
19. Naive Fold t-Interval (Avoid)
20. Nadeau–Bengio Corrected Resampled t-Test
21. Bayle et al. CV Confidence Interval
22. Nested-CV Confidence Interval (Bates et al.)
23. Independent Test-Set Bootstrap CI
24. Paired Bootstrap CI
25. Wilson / Binomial Interval
26. DeLong CI / Test
27. McNemar Test
28. Permutation Test
29. Bayesian Correlated t-Test

**E. Model Comparison Uncertainty**
30. ROPE (Region of Practical Equivalence)
31. Equivalence Testing (TOST)

**F. Data Perturbation**
32. Data Resampling Stress Test
33. Data-Size / Learning-Curve Stress Test
34. Label-Noise Stress Test
35. Numeric Feature Noise
36. Missingness Injection
37. Feature Dropout Stress Test
38. Categorical Corruption / Unseen-Level Stress
39. Outlier / Contamination Stress

**G. Distribution Robustness**
40. Covariate-Shift Stress
41. Label / Prevalence-Shift Stress
42. Temporal Robustness
43. Geographic / Domain Stress
44. External / OOD Robustness

**H. Subgroup Robustness**
45. Subgroup / Slice Robustness
46. Worst-Group Gap

**I. Pipeline Robustness**
47. Hyperparameter Perturbation Test
48. HPO Re-Run Stability
49. Preprocessing Perturbation
50. Feature-Selection Stability
51. Threshold Stability
52. Calibration Robustness

**J. Advanced Uncertainty**
53. Aleatoric vs Epistemic Uncertainty
54. Conformal Prediction / Risk Control

**K. Advanced Comparison & Resampling**
55. Multiple-Comparison Correction / Winner's Curse
56. Cluster / Block Bootstrap

**L. Adversarial Robustness**
57. Adversarial / Constraint-Aware Tabular Robustness

---

## A. STOCHASTIC TRAINING STABILITY

### 1. Multi-Seed Retraining

| Field | Detail |
|-------|--------|
| **Definisi** | Retraining model dengan data, split, hyperparameter, dan preprocessing identik, tetapi random seed berbeda. Mengisolasi stochasticity dari initialization, batch order, feature subsampling, bootstrap, dropout, dan optimizer |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Setelah model selection; untuk mengukur stochastic stability; sebelum klaim improvement |
| **Kapan tidak digunakan** | Model deterministik (OLS, linear regression tanpa stochastic optimizer) |
| **Asumsi** | Variasi seed mencerminkan variasi performa di deployment; model tidak extremely sensitive to seed |
| **Data preprocessing** | Fixed; sama untuk semua seed |
| **Dataset scale** | Kecil–Menengah; mahal untuk dataset besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_seeds` (5–10 untuk tree models; 10–30 untuk deep learning) |
| **Kisaran parameter awal** | n_seeds: 5–10 |
| **CPU/GPU/RAM** | n_seeds × training cost |
| **Risiko leakage** | Rendah; hanya randomness training |
| **Failure modes** | Seed terlalu sedikit → estimasi variance noisy; model sangat tidak stabil → butuh lebih banyak seed |
| **Teknik pembanding** | Fixed-split seed test, crossed seed × split |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual loop; `sklearn` `random_state`; `numpy.random` |
| **Contoh hypothesis** | "Model dengan 10 seeds akan menunjukkan seed SD <0.005, mengindikasikan training stabil" |
| **Acceptance criteria** | Seed SD <5% dari mean performance |
| **Rejection criteria** | Seed SD >10% dari mean; worst-seed catastrophic |
| **Sumber** | Mosbach et al., ICLR 2021; Nature Scientific Reports (2026) |

---

### 2. Fixed-Split Seed Test

| Field | Detail |
|-------|--------|
| **Definisi** | Mengisolasi training randomness dengan mempertahankan split fixed, hanya mengubah seed |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Ingin memisahkan variance dari training vs split; model stochastic |
| **Kapan tidak digunakan** | Model deterministik |
| **Asumsi** | Split fixed menghilangkan split variance |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_seeds` |
| **Kisaran parameter awal** | n_seeds: 10 |
| **CPU/GPU/RAM** | n_seeds × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Split fixed mungkin tidak representatif |
| **Teknik pembanding** | Split stability test |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual loop |
| **Sumber** | Mosbach et al., ICLR 2021 |

---

### 3. Split Stability Test

| Field | Detail |
|-------|--------|
| **Definisi** | Mengukur variance dari dataset partitioning dengan mengubah split seed sementara model seed dikontrol |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Ingin tahu apakah hasil bagus karena validation set mudah |
| **Kapan tidak digunakan** | Data sangat besar (split variance kecil) |
| **Asumsi** | Split seed mencerminkan variasi partitioning |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_split_seeds` |
| **Kisaran parameter awal** | 5–10 split seeds |
| **CPU/GPU/RAM** | n_split_seeds × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Split seed terlalu sedikit |
| **Teknik pembanding** | Crossed seed × split |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual loop |
| **Sumber** | Bouckaert & Frank (2004) |

---

### 4. Crossed Seed × Split Design

| Field | Detail |
|-------|--------|
| **Definisi** | Desain eksperimen faktorial dengan seed dan split sebagai dua faktor crossed, memungkinkan estimasi variance dari masing-masing sumber dan interaksi |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Butuh dekomposisi variance lengkap; klaim improvement robust |
| **Kapan tidak digunakan** | Computational budget terbatas |
| **Asumsi** | Faktor seed dan split independent |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_seeds` × `n_splits` |
| **Kisaran parameter awal** | 5 seeds × 5 splits = 25 runs |
| **CPU/GPU/RAM** | 25× training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Mahal; interaction effect sulit diinterpretasi |
| **Teknik pembanding** | Separate seed/split tests |
| **Ensemble compatibility** | Universal |
| **Exact package** | Custom; mixed-effects model |
| **Contoh hypothesis** | "Crossed design akan menunjukkan split variance > seed variance pada dataset kecil" |
| **Acceptance criteria** | Variance decomposition dilakukan; interaction effect dilaporkan |
| **Rejection criteria** | Hanya satu faktor diuji |
| **Sumber** | Design of experiments principles |

---

### 5. Paired Seed Comparison

| Field | Detail |
|-------|--------|
| **Definisi** | Membandingkan model A dan B menggunakan seed yang identik, bukan independent samples. Menganalisis Δ_s = Score_{B,s} − Score_{A,s} |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Compare baseline vs experiment; kontrol nuisance variability |
| **Kapan tidak digunakan** | Model tidak dapat dijalankan dengan seed yang sama |
| **Asumsi** | Paired design membuang nuisance variability |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_seeds` |
| **Kisaran parameter awal** | 10 seeds |
| **CPU/GPU/RAM** | 2 × n_seeds × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Tidak paired → variance lebih besar; kesimpulan lemah |
| **Teknik pembanding** | Independent comparison (avoid) |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual; paired t-test; Bayesian correlated t-test |
| **Contoh hypothesis** | "Paired seed comparison akan menunjukkan Δ konsisten positif dengan win rate ≥80%" |
| **Acceptance criteria** | Win rate ≥80%; Δ sign consistency |
| **Rejection criteria** | Win rate ≈50%; Δ tidak konsisten |
| **Sumber** | Best practices |

---

### 6. Seed Win-Rate

| Field | Detail |
|-------|--------|
| **Definisi** | Proporsi seed di mana model B outperform model A: WinRate = #(Δ_s > 0)/S |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Setelah paired seed comparison |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Win rate mencerminkan konsistensi improvement |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan (post-hoc) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Win rate tinggi tapi mean kecil; win rate rendah tapi mean besar (driven by outlier) |
| **Teknik pembanding** | Mean Δ only |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual computation |
| **Contoh hypothesis** | "Win rate ≥90% mengindikasikan improvement yang highly consistent" |
| **Acceptance criteria** | Win rate ≥80% |
| **Rejection criteria** | Win rate <60% |
| **Sumber** | Best practices; engineering heuristic |

---

### 7. Worst-Seed / Lower-Tail Performance

| Field | Detail |
|-------|--------|
| **Definisi** | Melaporkan distribusi lengkap performa antar seed: min, P05, P10, median, P90, max |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Mean saja tidak cukup; risiko catastrophic seed |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Lower-tail mencerminkan risiko deployment |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Distribusi non-Gaussian; outlier |
| **Teknik pembanding** | Mean only |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy percentile |
| **Contoh hypothesis** | "Model B mean lebih tinggi tapi P05 lebih rendah dari A, mengindikasikan risiko lebih besar" |
| **Acceptance criteria** | Lower-tail dilaporkan |
| **Rejection criteria** | Hanya mean dilaporkan |
| **Sumber** | Nature Scientific Reports (2026) |

---

### 8. Failure-Rate Stability

| Field | Detail |
|-------|--------|
| **Definisi** | Proporsi run di mana model gagal: score < threshold, diverged, NaN, convergence failed |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Model stochastic dengan risiko failure |
| **Kapan tidak digunakan** | Model sangat stabil |
| **Asumsi** | Failure criterion didefinisikan sebelum eksperimen |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Failure rate |
| **Hyperparameter penting** | `failure_threshold` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Failure criterion terlalu longgar/ketat |
| **Teknik pembanding** | Mean performance only |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Model dengan failure rate 15% berisiko operasional lebih tinggi dari model dengan failure rate 0% meskipun mean lebih tinggi" |
| **Acceptance criteria** | Failure rate <5% |
| **Rejection criteria** | Failure rate >10% |
| **Sumber** | Best practices |

---

## B. SPLIT / RESAMPLING STABILITY

### 9. Fold SD

| Field | Detail |
|-------|--------|
| **Definisi** | Standard deviation dari fold scores: SD_fold = SD(M_1,...,M_K). Hanya descriptive heterogeneity indicator, bukan SE |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Descriptive stability indicator |
| **Kapan tidak digunakan** | Untuk CI — folds correlated |
| **Asumsi** | Fold SD bukan estimator variance independent |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Menggunakan fold SD/sqrt(K) sebagai SE → underestimate |
| **Teknik pembanding** | Corrected CI methods |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy std |
| **Contoh hypothesis** | "Fold SD >0.02 mengindikasikan heterogenitas performa antar fold" |
| **Acceptance criteria** | Fold SD dilaporkan sebagai descriptive only |
| **Rejection criteria** | Fold SD digunakan sebagai SE |
| **Sumber** | Bengio & Grandvalet (2004) |

---

### 10. Fold Range & IQR

| Field | Detail |
|-------|--------|
| **Definisi** | Min fold, max fold, range, IQR dari fold scores |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Distribusi fold tidak normal; hidden subgroup detection |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Range/IQR robust terhadap outlier |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Range misleading jika K kecil |
| **Teknik pembanding** | SD |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy |
| **Contoh hypothesis** | "Range fold yang besar (misal .75–.91) mengindikasikan hidden subgroup" |
| **Acceptance criteria** | Range dilaporkan |
| **Rejection criteria** | Hanya mean dilaporkan |
| **Sumber** | Best practices |

---

### 11. Fold-Wise Paired Delta

| Field | Detail |
|-------|--------|
| **Definisi** | Untuk setiap fold k, hitung Δ_k = M_{B,k} − M_{A,k}. Analisis distribusi Δ |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Compare model A vs B dengan folds identik |
| **Kapan tidak digunakan** | Folds berbeda |
| **Asumsi** | Paired folds membuang nuisance variability |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Fold sangat sedikit → estimasi noisy |
| **Teknik pembanding** | Separate fold SD |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Fold-wise delta konsisten positif di 5/5 fold mengindikasikan improvement robust" |
| **Acceptance criteria** | Fold win rate ≥80% |
| **Rejection criteria** | Fold win rate <60% |
| **Sumber** | Best practices |

---

### 12. Repeated K-Fold Stability

| Field | Detail |
|-------|--------|
| **Definisi** | K-fold diulang R kali dengan randomization berbeda. Mengurangi ketergantungan pada satu partition |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Ingin distribution estimate dari CV; dataset kecil–menengah |
| **Kapan tidak digunakan** | Dataset sangat besar (single KFold cukup) |
| **Asumsi** | Repeated randomization mengurangi variance |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_splits`, `n_repeats` |
| **Kisaran parameter awal** | n_splits=5, n_repeats=10 |
| **CPU/GPU/RAM** | R × KFold cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Menganggap 50 fold scores sebagai 50 independent observations |
| **Teknik pembanding** | Single KFold |
| **Ensemble compatibility** | Universal |
| **Exact package** | `RepeatedKFold`, `RepeatedStratifiedKFold` |
| **Contoh hypothesis** | "Repeated K-Fold 5×10 akan mengurangi variance CV estimate ≥50% vs single KFold" |
| **Acceptance criteria** | Variance reduction ≥40% |
| **Rejection criteria** | Variance tidak berkurang |
| **Sumber** | scikit-learn docs |

---

### 13. Repeated Nested CV

| Field | Detail |
|-------|--------|
| **Definisi** | Nested CV yang diulang beberapa kali untuk stabilitas estimasi model selection dan performance |
| **Problem yang didukung** | Hyperparameter search besar |
| **Kapan digunakan** | Final evidence; dataset kecil–menengah; compute tersedia |
| **Kapan tidak digunakan** | Budget terbatas |
| **Asumsi** | Repeated nested CV mengurangi selection bias |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `outer_repeats`, `inner_repeats` |
| **Kisaran parameter awal** | Outer 5 × Inner 5; repeats=2–3 |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Rendah |
| **Failure modes** | Sangat mahal; diminishing returns |
| **Teknik pembanding** | Single nested CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn` nested CV + loop |
| **Contoh hypothesis** | "Repeated nested CV akan menunjukkan selection stability ≥80% antar repeats" |
| **Acceptance criteria** | Selection stability ≥75% |
| **Rejection criteria** | Selection stability <50% |
| **Sumber** | Cawley & Talbot (2010) |

---

### 14. Algorithmic Stability

| Field | Detail |
|-------|--------|
| **Definisi** | Mengukur perubahan prediksi/loss ketika dataset diubah sedikit (leave-one-sample perturbation). Fondasi algorithmic stability vs generalization |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Analisis fundamental stability; dataset kecil |
| **Kapan tidak digunakan** | Dataset besar |
| **Asumsi** | Algoritma stabil → generalization baik |
| **Dataset scale** | Kecil |
| **Metric compatibility** | Score change, prediction change, feature importance change |
| **Hyperparameter penting** | `perturbation_fraction` |
| **Kisaran parameter awal** | 1 sample; 5% sample |
| **CPU/GPU/RAM** | N × training cost (leave-one-out) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Mahal; model tidak stabil |
| **Teknik pembanding** | Bousquet-Elisseeff bounds |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Model stabil akan menunjukkan prediction change <1% ketika 1 sample dihapus" |
| **Acceptance criteria** | Prediction change <5% |
| **Rejection criteria** | Prediction change >10% |
| **Sumber** | Bousquet & Elisseeff, JMLR 2, 499–526 (2002) |

---

## C. VARIANCE DECOMPOSITION

### 15. Seed Variance

| Field | Detail |
|-------|--------|
| **Definisi** | Variance dari performance akibat random seed |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Setelah multi-seed retraining |
| **Kapan tidak digunakan** | Model deterministik |
| **Asumsi** | Seed variance dapat diestimasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Variance |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Seed variance kecil ≠ model robust |
| **Teknik pembanding** | Split variance |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy var |
| **Contoh hypothesis** | "Seed variance > split variance mengindikasikan training randomness dominan" |
| **Acceptance criteria** | Variance decomposition dilakukan |
| **Rejection criteria** | Hanya total variance dilaporkan |
| **Sumber** | Design of experiments |

---

### 16. Split Variance

| Field | Detail |
|-------|--------|
| **Definisi** | Variance dari performance akibat random split |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Setelah split stability test |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Split variance dapat diestimasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Variance |
| **Hyperparameter penting** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Split variance kecil ≠ model robust |
| **Teknik pembanding** | Seed variance |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy var |
| **Sumber** | Design of experiments |

---

### 17. Seed × Split Interaction

| Field | Detail |
|-------|--------|
| **Definisi** | Interaksi antara seed dan split dalam crossed design |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Crossed seed × split design |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Interaction dapat diestimasi |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Variance |
| **Hyperparameter penting** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Interpretasi sulit |
| **Teknik pembanding** | Main effects only |
| **Ensemble compatibility** | Universal |
| **Exact package** | statsmodels mixed-effects |
| **Sumber** | Design of experiments |

---

### 18. Mixed-Effects Decomposition

| Field | Detail |
|-------|--------|
| **Definisi** | Model mixed-effects untuk dekomposisi variance: Score_{ij} = μ + α_i^split + β_j^seed + ε_{ij} |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Crossed design; variance components |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Random effects |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Variance components |
| **Hyperparameter penting** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Model misspecification |
| **Teknik pembanding** | ANOVA |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels` MixedLM; `lme4` (R) |
| **Contoh hypothesis** | "Mixed-effects akan menunjukkan split variance 3× lebih besar dari seed variance" |
| **Acceptance criteria** | Variance components signifikan |
| **Rejection criteria** | Tidak konvergen |
| **Sumber** | Design of experiments; mixed-effects literature |

---

## D. PERFORMANCE UNCERTAINTY

### 19. Naive Fold t-Interval (Avoid)

| Field | Detail |
|-------|--------|
| **Definisi** | CI naive: \bar M ± t_{K-1,0.975} × s/√K. Menganggap folds independent — salah karena overlapping training sets |
| **Problem yang didukung** | N/A (avoid) |
| **Kapan digunakan** | Tidak digunakan |
| **Kapan tidak digunakan** | Selalu |
| **Asumsi** | Folds independent (invalid) |
| **Dataset scale** | N/A |
| **Metric compatibility** | N/A |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N/A |
| **Risiko leakage** | N/A |
| **Failure modes** | Coverage jauh di bawah nominal |
| **Teknik pembanding** | Corrected methods |
| **Ensemble compatibility** | N/A |
| **Exact package** | Avoid |
| **Contoh hypothesis** | N/A |
| **Acceptance criteria** | N/A |
| **Rejection criteria** | Menggunakan naive fold t-interval |
| **Sumber** | Bates et al., JASA 119(546), 1434–1445 (2024) |

---

### 20. Nadeau–Bengio Corrected Resampled t-Test

| Field | Detail |
|-------|--------|
| **Definisi** | Koreksi standard error untuk memperhitungkan correlation akibat reused training data: t = \bar d / √(s_d² (1/n + n_test/n_train)) |
| **Problem yang didukung** | Model comparison dengan repeated CV |
| **Kapan digunakan** | Compare model A vs B; repeated CV; same folds |
| **Kapan tidak digunakan** | Fold benar-benar independent |
| **Asumsi** | Correlation heuristic valid |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_train`, `n_test`, `n` (folds) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Correlation lebih kompleks dari heuristic |
| **Teknik pembanding** | Naive t-test (avoid) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `correctR` (R); `correctipy` (Python); Weka `PairedCorrectedTTester` |
| **Contoh hypothesis** | "Corrected t-test akan mengurangi false positive rate dari 20% menjadi <5% pada repeated CV" |
| **Acceptance criteria** | p-value lebih konservatif; false positive rate <5% |
| **Rejection criteria** | Menggunakan naive t-test |
| **Sumber** | Nadeau & Bengio, Machine Learning 52, 239–281 (2003) |

---

### 21. Bayle et al. CV Confidence Interval

| Field | Detail |
|-------|--------|
| **Definisi** | CI asymptotically exact untuk k-fold test error berdasarkan CLT untuk CV dan consistent variance estimator di bawah weak stability conditions |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Butuh valid CI untuk CV error; stability conditions terpenuhi |
| **Kapan tidak digunakan** | Stability conditions gagal |
| **Asumsi** | Decomposable loss; stability conditions |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan (post-hoc) |
| **Risiko leakage** | Rendah |
| **Failure modes** | Stability conditions tidak terpenuhi |
| **Teknik pembanding** | Naive CI |
| **Ensemble compatibility** | Universal |
| **Exact package** | `cvci` (R); custom |
| **Contoh hypothesis** | "Bayle CI akan mencapai coverage 95% vs naive CI yang hanya 80%" |
| **Acceptance criteria** | Coverage mendekati nominal |
| **Rejection criteria** | Coverage <90% |
| **Sumber** | Bayle et al., NeurIPS 2020 |

---

### 22. Nested-CV Confidence Interval (Bates et al.)

| Field | Detail |
|-------|--------|
| **Definisi** | Nested CV scheme untuk estimasi variance CV lebih akurat; mengoreksi bias akibat correlation folds |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | High-confidence final model assessment; small/medium data |
| **Kapan tidak digunakan** | Budget terbatas |
| **Asumsi** | Nested scheme valid |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `outer_folds`, `inner_folds` |
| **Kisaran parameter awal** | Outer 5 × Inner 5 |
| **CPU/GPU/RAM** | Berat |
| **Risiko leakage** | Rendah |
| **Failure modes** | Mahal |
| **Teknik pembanding** | Standard CV CI |
| **Ensemble compatibility** | Universal |
| **Exact package** | `mlr3inferr` (R); custom |
| **Contoh hypothesis** | "Nested-CV CI akan mencapai coverage mendekati nominal di mana standard CV CI gagal" |
| **Acceptance criteria** | Coverage ≥90% |
| **Rejection criteria** | Coverage <80% |
| **Sumber** | Bates et al., JASA 119(546), 1434–1445 (2024) |

---

### 23. Independent Test-Set Bootstrap CI

| Field | Detail |
|-------|--------|
| **Definisi** | Bootstrap test observations untuk CI metric. Resample test rows dengan replacement, hitung metric, repeat B kali |
| **Problem yang didukung** | Semua problem supervised dengan locked test set |
| **Kapan digunakan** | Independent test set tersedia |
| **Kapan tidak digunakan** | Tidak ada test set |
| **Asumsi** | Test observations exchangeable |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `B` (1000–10000) |
| **Kisaran parameter awal** | B=1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Test set tidak representatif |
| **Teknik pembanding** | Analytic CI |
| **Ensemble compatibility** | Universal |
| **Exact package** | `scipy.stats.bootstrap`; `boot` (R) |
| **Contoh hypothesis** | "Bootstrap CI akan memberikan interval [0.82, 0.88] untuk AUC test" |
| **Acceptance criteria** | CI stabil |
| **Rejection criteria** | CI sangat lebar |
| **Sumber** | Efron & Tibshirani (1993) |

---

### 24. Paired Bootstrap CI

| Field | Detail |
|-------|--------|
| **Definisi** | Bootstrap paired test rows untuk CI improvement Δ = B − A |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Compare A vs B pada test set yang sama |
| **Kapan tidak digunakan** | Independent samples |
| **Asumsi** | Paired exchangeability |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `B` |
| **Kisaran parameter awal** | B=1000 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | CI terlalu lebar jika Δ kecil |
| **Teknik pembanding** | Separate bootstrap |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual; `scipy.stats.bootstrap` |
| **Contoh hypothesis** | "Paired bootstrap akan menunjukkan 95% CI untuk Δ AUC = [+0.003, +0.013]" |
| **Acceptance criteria** | CI tidak mencakup 0 |
| **Rejection criteria** | CI sangat lebar; mencakup 0 |
| **Sumber** | Efron & Tibshirani (1993) |

---

### 25. Wilson / Binomial Interval

| Field | Detail |
|-------|--------|
| **Definisi** | CI untuk accuracy sebagai binomial proportion. Wilson interval lebih baik dari Wald untuk sample kecil atau p dekat 0/1 |
| **Problem yang didukung** | Classification dengan accuracy |
| **Kapan digunakan** | Test set independent; metric = accuracy |
| **Kapan tidak digunakan** | Metric kompleks (AUC, F1) |
| **Asumsi** | Binomial proportion |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Accuracy |
| **Hyperparameter penting** | `confidence_level` |
| **Kisaran parameter awal** | 95% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Wald interval gagal untuk p dekat 0/1 |
| **Teknik pembanding** | Wald interval |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels.stats.proportion.proportion_confint` |
| **Contoh hypothesis** | "Wilson interval akan memberikan coverage mendekati nominal untuk accuracy=0.95 dengan n=100" |
| **Acceptance criteria** | Coverage ≥90% |
| **Rejection criteria** | Coverage <80% |
| **Sumber** | Wilson (1927) |

---

### 26. DeLong CI / Test

| Field | Detail |
|-------|--------|
| **Definisi** | Nonparametric covariance estimation untuk correlated ROC areas menggunakan generalized U-statistics |
| **Problem yang didukung** | ROC-AUC comparison |
| **Kapan digunakan** | Compare AUC model A vs B pada test set yang sama |
| **Kapan tidak digunakan** | Metric bukan AUC |
| **Asumsi** | Paired AUC |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | ROC-AUC |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | AUC tidak normally distributed |
| **Teknik pembanding** | Bootstrap |
| **Ensemble compatibility** | Universal |
| **Exact package** | `pROC` (R); `scikit-learn` custom |
| **Contoh hypothesis** | "DeLong test akan menunjukkan AUC_A vs AUC_B berbeda signifikan (p<0.05)" |
| **Acceptance criteria** | p-value valid |
| **Rejection criteria** | Menggunakan independent comparison |
| **Sumber** | DeLong et al., Biometrics 44(3), 837–845 (1988) |

---

### 27. McNemar Test

| Field | Detail |
|-------|--------|
| **Definisi** | Test untuk dua classifiers dengan paired binary decisions pada test observations. Fokus pada discordant pairs |
| **Problem yang didukung** | Hard-label classification comparison |
| **Kapan digunakan** | Binary/multiclass correctness; same test set |
| **Kapan tidak digunakan** | ROC-AUC, RMSE, calibration |
| **Asumsi** | Paired binary decisions |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Accuracy |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Tidak cocok untuk metric kontinu |
| **Teknik pembanding** | DeLong (AUC) |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels.stats.contingency_tables.mcnemar` |
| **Contoh hypothesis** | "McNemar akan menunjukkan model B lebih akurat dari A pada discordant pairs (p<0.05)" |
| **Acceptance criteria** | p-value valid |
| **Rejection criteria** | Digunakan untuk AUC/RMSE |
| **Sumber** | McNemar (1947) |

---

### 28. Permutation Test

| Field | Detail |
|-------|--------|
| **Definisi** | Randomly swap paired predictions/losses untuk empirical null distribution Δ |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Analytic variance sulit; metric flexible |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Exchangeability under null |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_permutations` |
| **Kisaran parameter awal** | 1000–10000 |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | Rendah |
| **Failure modes** | Computational cost |
| **Teknik pembanding** | Analytic test |
| **Ensemble compatibility** | Universal |
| **Exact package** | `scipy.stats.permutation_test` |
| **Contoh hypothesis** | "Permutation test akan menunjukkan p=0.02 untuk Δ AUC" |
| **Acceptance criteria** | p-value stabil |
| **Rejection criteria** | n_permutations terlalu kecil |
| **Sumber** | Good (2005) |

---

### 29. Bayesian Correlated t-Test

| Field | Detail |
|-------|--------|
| **Definisi** | Bayesian version dari correlated t-test untuk CV comparisons. Output posterior P(B>A), P(B≈A), P(B<A) |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Compare multiple classifiers; ROPE; practical equivalence |
| **Kapan tidak digunakan** | Frequentist inference required |
| **Asumsi** | Student-t posterior; correlation factor ρ |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `rope`; `rho` |
| **Kisaran parameter awal** | rope=[-0.01, 0.01]; rho=n_test/n_total |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Prior sensitivity; rho heuristic |
| **Teknik pembanding** | Frequentist correlated t-test |
| **Ensemble compatibility** | Universal |
| **Exact package** | `scmamp::bCorrelatedTtest` (R); `bayesiantests` (Python) |
| **Contoh hypothesis** | "Bayesian correlated t-test akan menunjukkan P(B>A)=0.94, P(B≈A)=0.05, P(B<A)=0.01" |
| **Acceptance criteria** | Posterior probabilities stabil |
| **Rejection criteria** | rho salah |
| **Sumber** | Benavoli et al., JMLR 18, 1–36 (2017) |

---

## E. MODEL COMPARISON UNCERTAINTY

### 30. ROPE (Region of Practical Equivalence)

| Field | Detail |
|-------|--------|
| **Definisi** | Region di mana perbedaan dianggap praktis tidak penting. Memisahkan statistical difference dari practical difference |
| **Problem yang didukung** | Model comparison |
| **Kapan digunakan** | Butuh definisi fragile vs robust; practical significance |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | ROPE domain-specific |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `rope_width` |
| **Kisaran parameter awal** | AUC: 0.005; RMSE: -1% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | ROPE terlalu lebar/sempit |
| **Teknik pembanding** | p-value only |
| **Ensemble compatibility** | Universal |
| **Exact package** | Bayesian correlated t-test |
| **Contoh hypothesis** | "Δ AUC = +0.003 dengan ROPE [-0.005, +0.005] → practically equivalent" |
| **Acceptance criteria** | ROPE didefinisikan sebelum eksperimen |
| **Rejection criteria** | Hanya p-value dilaporkan |
| **Sumber** | Benavoli et al. (2017) |

---

### 31. Equivalence Testing (TOST)

| Field | Detail |
|-------|--------|
| **Definisi** | Two One-Sided Tests untuk menguji apakah effect berada dalam equivalence bounds. Menggantikan misinterpretasi p>0.05 sebagai "no difference" |
| **Problem yang didukung** | Model comparison; non-inferiority |
| **Kapan digunakan** | Ingin menunjukkan model "praktis sama"; operational metric lebih baik |
| **Kapan tidak digunakan** | Ingin menunjukkan superiority |
| **Asumsi** | Equivalence bounds didefinisikan sebelum eksperimen |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `equivalence_bounds` |
| **Kisaran parameter awal** | Standardized effect size |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Bounds salah |
| **Teknik pembanding** | ROPE; p-value only |
| **Ensemble compatibility** | Universal |
| **Exact package** | `TOSTER` (R); `equivalence` (Python) |
| **Contoh hypothesis** | "TOST akan menunjukkan model B non-inferior terhadap A dengan bounds ±0.01" |
| **Acceptance criteria** | p_TOST <0.05 |
| **Rejection criteria** | Menggunakan p>0.05 sebagai "equivalent" |
| **Sumber** | Lakens, SPPS 8(4), 355–362 (2017) |

---

## F. DATA PERTURBATION

### 32. Data Resampling Stress Test

| Field | Detail |
|-------|--------|
| **Definisi** | Retrain pada bootstrap/subsample datasets dan lihat apakah improvement tetap positif |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Robustness stress |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Resampling menangkap variasi data |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_bootstrap`, `subsample_fraction` |
| **Kisaran parameter awal** | n_bootstrap=50–100; fraction=0.7–0.9 |
| **CPU/GPU/RAM** | n_bootstrap × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Improvement hilang pada subsample |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Δ tetap positif pada 80% bootstrap samples" |
| **Acceptance criteria** | Stress survival ≥80% |
| **Rejection criteria** | Stress survival <50% |
| **Sumber** | Best practices |

---

### 33. Data-Size / Learning-Curve Stress Test

| Field | Detail |
|-------|--------|
| **Definisi** | Train dengan 20%, 40%, 60%, 80%, 100% data dan lihat Δ pada setiap ukuran |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Robustness terhadap data size |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Learning curve informatif |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `fractions` |
| **Kisaran parameter awal** | [0.2, 0.4, 0.6, 0.8, 1.0] |
| **CPU/GPU/RAM** | Σ fractions × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Improvement hanya pada 100% |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.learning_curve` |
| **Contoh hypothesis** | "Δ positif konsisten di semua fractions" |
| **Acceptance criteria** | Δ positif pada ≥80% fractions |
| **Rejection criteria** | Δ hilang pada 80% |
| **Sumber** | Best practices |

---

### 34. Label-Noise Stress Test

| Field | Detail |
|-------|--------|
| **Definisi** | Inject controlled label corruption (0%, 1%, 2.5%, 5%, 10%) dan lihat degradation curve |
| **Problem yang didukung** | Classification, Regression |
| **Kapan digunakan** | Robustness terhadap label noise |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Noise dapat diinjeksi secara controlled |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `noise_rates` |
| **Kisaran parameter awal** | [0, 0.01, 0.025, 0.05, 0.10] |
| **CPU/GPU/RAM** | Σ noise rates × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Performance drop signifikan |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Model B tetap outperform A pada 5% noise" |
| **Acceptance criteria** | Δ positif pada 5% noise |
| **Rejection criteria** | Δ hilang pada 1% noise |
| **Sumber** | Frénay & Verleysen, IEEE TNNLS 25(5), 845–869 (2014) |

---

### 35. Numeric Feature Noise

| Field | Detail |
|-------|--------|
| **Definisi** | Tambahkan perturbation ε ~ N(0, σ²) pada numeric features |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Measurement error simulation |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Gaussian noise |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `noise_scale` |
| **Kisaran parameter awal** | 1%, 5%, 10% dari std |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Performance drop |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | numpy random |
| **Contoh hypothesis** | "Model robust akan menunjukkan Δ <5% pada 5% feature noise" |
| **Acceptance criteria** | Degradation <10% |
| **Rejection criteria** | Degradation >20% |
| **Sumber** | Best practices |

---

### 36. Missingness Injection

| Field | Detail |
|-------|--------|
| **Definisi** | Inject missing values (1%, 5%, 10%, 20%) dengan skenario MCAR, group-dependent, feature-dependent |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Robustness terhadap missing values |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Missingness mechanism controlled |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `missing_rates`, `mechanism` |
| **Kisaran parameter awal** | [0.01, 0.05, 0.10, 0.20] |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Imputation strategy gagal |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual; `sklearn.impute` |
| **Contoh hypothesis** | "Model B lebih robust terhadap missingness 10% dari A" |
| **Acceptance criteria** | Δ positif pada 10% missing |
| **Rejection criteria** | Δ hilang pada 5% missing |
| **Sumber** | Best practices |

---

### 37. Feature Dropout Stress Test

| Field | Detail |
|-------|--------|
| **Definisi** | Drop kolom secara sengaja: top feature, top 3, random 10% |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Robustness terhadap feature availability |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Feature dropout simulasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_features_to_drop` |
| **Kisaran parameter awal** | 1, 3, 10% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Model bergantung pada satu feature |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Model robust akan menunjukkan Δ <10% setelah top feature di-drop" |
| **Acceptance criteria** | Degradation <15% |
| **Rejection criteria** | Degradation >30% |
| **Sumber** | TabFSBench (2025) |

---

### 38. Categorical Corruption / Unseen-Level Stress

| Field | Detail |
|-------|--------|
| **Definisi** | Simulasikan unknown category, rare-category surge, category remapping, new merchant/product ID |
| **Problem yang didukung** | Semua problem dengan categorical features |
| **Kapan digunakan** | Robustness terhadap unseen categories |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Corruption simulasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `corruption_rate` |
| **Kisaran parameter awal** | 5%, 10% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Encoder gagal |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Target encoding lebih robust terhadap unseen levels dari one-hot" |
| **Acceptance criteria** | Degradation <15% |
| **Rejection criteria** | Degradation >30% |
| **Sumber** | Best practices |

---

### 39. Outlier / Contamination Stress

| Field | Detail |
|-------|--------|
| **Definisi** | Inject extreme values, heavy-tail perturbations, contaminated rows |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Robustness terhadap outlier |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Contamination controlled |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `epsilon` (contamination rate) |
| **Kisaran parameter awal** | 1%, 5% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Model sensitive |
| **Teknik pembanding** | RobustScaler vs StandardScaler |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "RobustScaler lebih robust dari StandardScaler pada 5% contamination" |
| **Acceptance criteria** | Degradation <10% |
| **Rejection criteria** | Degradation >25% |
| **Sumber** | Best practices |

---

## G. DISTRIBUTION ROBUSTNESS

### 40. Covariate-Shift Stress

| Field | Detail |
|-------|--------|
| **Definisi** | Ubah P(X) tanpa mengubah target mechanism. Evaluate Gap_shift = Score_ID − Score_OOD |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Production shift; distribution robustness |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Shift dapat disimulasikan |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `shift_magnitude` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | ID performance tidak menjamin OOD |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual; `TableShift` benchmark |
| **Contoh hypothesis** | "Gap_shift <5% mengindikasikan robustness terhadap covariate shift" |
| **Acceptance criteria** | Gap <10% |
| **Rejection criteria** | Gap >20% |
| **Sumber** | Gardner et al., NeurIPS 2023 (TableShift) |

---

### 41. Label / Prevalence-Shift Stress

| Field | Detail |
|-------|--------|
| **Definisi** | Ubah prevalence target (train 5%, production 1%) dan lihat operating metrics |
| **Problem yang didukung** | Classification |
| **Kapan digunakan** | Prevalence shift |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Prevalence shift controlled |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Precision, PPV, calibration, threshold utility |
| **Hyperparameter penting** | `prevalence_shift` |
| **Kisaran parameter awal** | 5% → 1% |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | AUC stabil tapi precision drop |
| **Teknik pembanding** | AUC only |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "AUC stabil tapi precision drop dari 0.60 menjadi 0.30 pada prevalence shift" |
| **Acceptance criteria** | Operating metrics dilaporkan |
| **Rejection criteria** | Hanya AUC dilaporkan |
| **Sumber** | Best practices |

---

### 42. Temporal Robustness

| Field | Detail |
|-------|--------|
| **Definisi** | Pisahkan performance per periode waktu: Jan, Feb, Mar... atau 2024, 2025, 2026 |
| **Problem yang didukung** | Time series; temporal tabular |
| **Kapan digunakan** | Business tabular; concept drift |
| **Kapan tidak digunakan** | Data tanpa time |
| **Asumsi** | Temporal degradation dapat diukur |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `time_periods` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Trend menurun |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Temporal slope negatif mengindikasikan degradation" |
| **Acceptance criteria** | Worst period dilaporkan |
| **Rejection criteria** | Hanya global metric |
| **Sumber** | Best practices |

---

### 43. Geographic / Domain Stress

| Field | Detail |
|-------|--------|
| **Definisi** | Train Java, test Sumatra, Kalimantan; atau train store A/B/C, test store D |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Domain generalization |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Domain shift |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `domain_column` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Domain performance drop |
| **Teknik pembanding** | Random CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.model_selection.GroupKFold` |
| **Contoh hypothesis** | "Leave-one-domain-out akan menunjukkan AUC drop ≥15%" |
| **Acceptance criteria** | Domain gap dilaporkan |
| **Rejection criteria** | Hanya random CV |
| **Sumber** | Roberts et al., Ecography 40(8), 913–929 (2017) |

---

### 44. External / OOD Robustness

| Field | Detail |
|-------|--------|
| **Definisi** | Final model evaluated pada completely external dataset |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | External validation; OOD |
| **Kapan tidak digunakan** | Tidak ada external data |
| **Asumsi** | External dataset representatif |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | **Kritis**; external data tidak boleh untuk tuning |
| **Failure modes** | Performance drop signifikan |
| **Teknik pembanding** | Internal CV |
| **Ensemble compatibility** | Universal |
| **Exact package** | N/A (protocol) |
| **Contoh hypothesis** | "External validation akan menunjukkan AUC drop <10%" |
| **Acceptance criteria** | AUC drop <15% |
| **Rejection criteria** | AUC drop >30% |
| **Sumber** | Nature npj Digital Medicine (2021) |

---

## H. SUBGROUP ROBUSTNESS

### 45. Subgroup / Slice Robustness

| Field | Detail |
|-------|--------|
| **Definisi** | Hitung metric M_g untuk setiap meaningful group (age, region, product category, customer type) |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Fairness; worst-group performance |
| **Kapan tidak digunakan** | Tidak ada subgroup |
| **Asumsi** | Subgroup meaningful |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `group_columns` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Average performance menyembunyikan subgroup failure |
| **Teknik pembanding** | Global metric only |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual; `Fairlearn` |
| **Contoh hypothesis** | "Worst-group AUC = 0.68 vs overall AUC = 0.91 → large gap" |
| **Acceptance criteria** | Worst-group dilaporkan |
| **Rejection criteria** | Hanya global metric |
| **Sumber** | Sagawa et al., ICLR 2020 |

---

### 46. Worst-Group Gap

| Field | Detail |
|-------|--------|
| **Definisi** | RobustnessGap = M_overall − M_worst_group |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Fairness assessment |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Gap meaningful |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | N/A |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Gap besar → warning |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Gap >0.15 mengindikasikan subgroup robustness warning" |
| **Acceptance criteria** | Gap <0.10 |
| **Rejection criteria** | Gap >0.20 |
| **Sumber** | Sagawa et al., ICLR 2020 |

---

## I. PIPELINE ROBUSTNESS

### 47. Hyperparameter Perturbation Test

| Field | Detail |
|-------|--------|
| **Definisi** | Setelah HPO, uji neighborhood: LR × {0.5, 0.8, 1.0, 1.25, 2}, depth ±1/2, regularization ±25% |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Fragile hyperparameter optimum detection |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Broad plateau desirable |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `perturbation_factors` |
| **Kisaran parameter awal** | {0.5, 0.8, 1.0, 1.25, 2} |
| **CPU/GPU/RAM** | 5–10× training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Best hanya pada satu titik tipis |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Performance plateau dalam ±20% LR mengindikasikan robust optimum" |
| **Acceptance criteria** | Degradation <5% dalam neighborhood |
| **Rejection criteria** | Degradation >20% |
| **Sumber** | Best practices |

---

### 48. HPO Re-Run Stability

| Field | Detail |
|-------|--------|
| **Definisi** | Run optimization dengan HPO seed berbeda; cek best score, best family, hyperparameter stability |
| **Problem yang didukung** | Hyperparameter search |
| **Kapan digunakan** | Bayesian/Optuna/random search stochastic |
| **Kapan tidak digunakan** | Grid search deterministic |
| **Asumsi** | HPO stability |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Best score stability |
| **Hyperparameter penting** | `hpo_seeds` |
| **Kisaran parameter awal** | 3–5 |
| **CPU/GPU/RAM** | 3–5× HPO cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Setiap run menemukan konfigurasi berbeda |
| **Teknik pembanding** | Single HPO run |
| **Ensemble compatibility** | Universal |
| **Exact package** | Optuna; `hyperopt` |
| **Contoh hypothesis** | "Best score stability <2% antar HPO seeds" |
| **Acceptance criteria** | Selection stability ≥75% |
| **Rejection criteria** | Selection stability <50% |
| **Sumber** | Best practices |

---

### 49. Preprocessing Perturbation

| Field | Detail |
|-------|--------|
| **Definisi** | Coba plausible alternatives: median vs iterative imputer, StandardScaler vs RobustScaler, winsorization on/off |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Fragility terhadap preprocessing choice |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Preprocessing alternatives plausible |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `alternatives` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | K × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Improvement hilang karena minor preprocessing |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.pipeline` |
| **Contoh hypothesis** | "Δ tetap positif di 80% preprocessing alternatives" |
| **Acceptance criteria** | Stress survival ≥80% |
| **Rejection criteria** | Stress survival <50% |
| **Sumber** | Best practices |

---

### 50. Feature-Selection Stability

| Field | Detail |
|-------|--------|
| **Definisi** | SelectionFrequency_j = # runs feature j selected / # runs |
| **Problem yang didukung** | Feature selection |
| **Kapan digunakan** | Sparse/wrapper/Lasso/Boruta pipelines |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Stable selection desirable |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Selection frequency |
| **Hyperparameter penting** | `n_runs` |
| **Kisaran parameter awal** | 10–50 |
| **CPU/GPU/RAM** | n_runs × training cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | Selection tidak stabil |
| **Teknik pembanding** | Single selection |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "Feature A = 100%, B = 95%, C = 12% → A/B stable, C unstable" |
| **Acceptance criteria** | Top features ≥90% frequency |
| **Rejection criteria** | Top features <50% |
| **Sumber** | Best practices |

---

### 51. Threshold Stability

| Field | Detail |
|-------|--------|
| **Definisi** | Uji t ∈ {0.30, 0.35, ..., 0.70}; check plateau around optimal threshold |
| **Problem yang didukung** | Classification dengan F1/cost-sensitive |
| **Kapan digunakan** | Threshold selection |
| **Kapan tidak digunakan** | N/A |
| **Asumsi** | Plateau desirable |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | F1, cost |
| **Hyperparameter penting** | `threshold_range` |
| **Kisaran parameter awal** | 0.30–0.70 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Best hanya pada t=0.473 |
| **Teknik pembanding** | N/A |
| **Ensemble compatibility** | Universal |
| **Exact package** | Manual |
| **Contoh hypothesis** | "F1 plateau dalam ±0.10 threshold mengindikasikan robust operating point" |
| **Acceptance criteria** | Plateau width ≥0.10 |
| **Rejection criteria** | Single-point optimum |
| **Sumber** | Best practices |

---

### 52. Calibration Robustness

| Field | Detail |
|-------|--------|
| **Definisi** | Stress Brier, log-loss, ECE, calibration slope/intercept pada seed, fold, time, domain, subgroup |
| **Problem yang didukung** | Probabilistic classification |
| **Kapan digunakan** | Butuh calibrated probability |
| **Kapan tidak digunakan** | Hard-label only |
| **Asumsi** | Calibration dapat diukur |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Brier, log-loss, ECE |
| **Hyperparameter penting** | `calibration_method` |
| **Kisaran parameter awal** | sigmoid/isotonic |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; calibration set terpisah |
| **Failure modes** | Discrimination stabil tapi probabilities rusak |
| **Teknik pembanding** | ROC-AUC only |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.calibration` |
| **Contoh hypothesis** | "ECE tetap <0.05 di semua subgroup" |
| **Acceptance criteria** | ECE <0.10 di semua subgroup |
| **Rejection criteria** | ECE >0.20 |
| **Sumber** | Gneiting & Raftery (2007) |

---

## J. ADVANCED UNCERTAINTY

### 53. Aleatoric vs Epistemic Uncertainty

| Field | Detail |
|-------|--------|
| **Definisi** | Aleatoric: irreducible noise from data generation. Epistemic: model uncertainty from lack of knowledge/data |
| **Problem yang didukung** | Semua problem supervised |
| **Kapan digunakan** | Uncertainty decomposition; safety-critical |
| **Kapan tidak digunakan** | Point prediction only |
| **Asumsi** | Kedua jenis uncertainty dapat diestimasi |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Predictive uncertainty |
| **Hyperparameter penting** | `method` (deep ensembles, MC dropout) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Method-dependent |
| **Risiko leakage** | Rendah |
| **Failure modes** | Confounding aleatoric dan epistemic |
| **Teknik pembanding** | Total uncertainty only |
| **Ensemble compatibility** | Deep Ensembles; MC Dropout |
| **Exact package** | PyTorch/TensorFlow; `torch.optim.swa_utils` |
| **Contoh hypothesis** | "Epistemic uncertainty tinggi pada OOD samples; aleatoric tinggi pada noisy region" |
| **Acceptance criteria** | Decomposition meaningful |
| **Rejection criteria** | Hanya total uncertainty |
| **Sumber** | Hüllermeier & Waegeman, Machine Learning 110(3), 457–506 (2021); IEEE TPAMI (2025) |

---

### 54. Conformal Prediction / Risk Control

| Field | Detail |
|-------|--------|
| **Definisi** | Distribution-free prediction intervals/sets dengan finite-sample coverage guarantee P(Y_new ∈ C(X_new)) ≥ 1−α. Conformal risk control extends ke monotone losses |
| **Problem yang didukung** | Regression intervals; classification sets |
| **Kapan digunakan** | Butuh prediction uncertainty; distribution-free guarantee |
| **Kapan tidak digunakan** | Exchangeability tidak terpenuhi (untuk guarantee) |
| **Asumsi** | Exchangeability (untuk guarantee) |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Coverage, set size |
| **Hyperparameter penting** | `alpha` (1−coverage) |
| **Kisaran parameter awal** | α=0.10 (90% coverage) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; calibration set terpisah |
| **Failure modes** | Exchangeability violation |
| **Teknik pembanding** | Quantile regression |
| **Ensemble compatibility** | Universal |
| **Exact package** | `MAPIE` (Python); `crepes` (Python); `predictset` (R) |
| **Contoh hypothesis** | "Conformal prediction akan mencapai coverage 0.90 dengan set size ≤3" |
| **Acceptance criteria** | Coverage ≥0.88 |
| **Rejection criteria** | Coverage <0.80 |
| **Sumber** | Angelopoulos et al., ICLR 2024 (Conformal Risk Control); Vovk et al. (2005) |

---

## K. ADVANCED COMPARISON & RESAMPLING

### 55. Multiple-Comparison Correction / Winner's Curse

| Field | Detail |
|-------|--------|
| **Definisi** | Best observed score ≠ unbiased estimate. Semakin banyak experiments, semakin tinggi kemungkinan satu configuration terlihat bagus karena noise |
| **Problem yang didukung** | Hyperparameter search; model selection |
| **Kapan digunakan** | Banyak experiments/tested models |
| **Kapan tidak digunakan** | Single a priori hypothesis |
| **Asumsi** | Winner's curse dapat diukur |
| **Dataset scale** | Semua skala |
| **Metric compatibility** | Expected best score vs unbiased estimate |
| **Hyperparameter penting** | `n_candidates` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | N/A |
| **Risiko leakage** | **Tinggi**; selection leakage |
| **Failure modes** | Overestimated best score |
| **Teknik pembanding** | Nested CV; held-out test |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels` multipletests; Bonferroni, Holm, FDR |
| **Contoh hypothesis** | "Best observed AUC = 0.913 tetapi unbiased estimate = 0.890 setelah correction" |
| **Acceptance criteria** | Winner's curse dilaporkan |
| **Rejection criteria** | Hanya best score dilaporkan |
| **Sumber** | Forstmeier & Schielzeth, Behav Ecol Sociobiol 65(1), 47–55 (2011); Cawley & Talbot (2010) |

---

### 56. Cluster / Block Bootstrap

| Field | Detail |
|-------|--------|
| **Definisi** | Bootstrap dengan cluster/block sebagai unit resampling, bukan row. Untuk data clustered atau time series |
| **Problem yang didukung** | Clustered data; Time series |
| **Kapan digunakan** | Customer repeated; patient repeated; temporal dependence |
| **Kapan tidak digunakan** | Data i.i.d. |
| **Asumsi** | Cluster/block exchangeable |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `resampling_unit` (row/group/temporal_block) |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Block size salah |
| **Teknik pembanding** | Row bootstrap |
| **Ensemble compatibility** | Universal |
| **Exact package** | `clusterSEs` (R); `boot` (R); manual |
| **Contoh hypothesis** | "Cluster bootstrap akan memberikan CI lebih lebar dari row bootstrap pada data customer repeated" |
| **Acceptance criteria** | Coverage mendekati nominal |
| **Rejection criteria** | Row bootstrap digunakan untuk clustered data |
| **Sumber** | MacKinnon, Cluster-Robust Inference: A Guide (2023) |

---

## L. ADVERSARIAL ROBUSTNESS

### 57. Adversarial / Constraint-Aware Tabular Robustness

| Field | Detail |
|-------|--------|
| **Definisi** | Stress-test dengan perturbation yang secara sengaja mencari failure model, memperhatikan constraint tabular: feature bounds, immutable features, categorical validity, cross-feature constraints |
| **Problem yang didukung** | Semua problem tabular; security-critical |
| **Kapan digunakan** | Adversarial robustness assessment |
| **Kapan tidak digunakan** | Non-adversarial environment |
| **Asumsi** | Perturbation harus valid secara domain |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Adversarial accuracy, robustness gap |
| **Hyperparameter penting** | `immutable_features`; `constraints`; `attack_type` |
| **Kisaran parameter awal** | FGSM, PGD, C&W |
| **CPU/GPU/RAM** | Attack-dependent |
| **Risiko leakage** | Rendah |
| **Failure modes** | Tidak realistis perturbation; constraints diabaikan |
| **Teknik pembanding** | Natural perturbation stress |
| **Ensemble compatibility** | Universal |
| **Exact package** | `TabAttackBench`; `ART`; custom |
| **Contoh hypothesis** | "Model robust akan menunjukkan adversarial accuracy ≥0.70 dengan constraint-aware attacks" |
| **Acceptance criteria** | Adversarial accuracy ≥0.65 |
| **Rejection criteria** | Adversarial accuracy <0.50 |
| **Sumber** | He et al., Expert Systems with Applications 301, 130491 (2025) |

---

## Robustness Verdict Framework

### Decision Logic

```
ROBUST
  median Δ > ε
  AND lower CI/HDI approximately > 0
  AND seed win rate ≥ 90%
  AND fold/repeat win rate ≥ 80%
  AND no catastrophic seed failures
  AND stress survival ≥ 80%
  AND worst important subgroup does not materially regress

BORDERLINE
  median Δ > 0
  BUT CI crosses 0
  OR win rate 60–90%
  OR one important stress condition reverses effect
  → promising_but_uncertain

FRAGILE
  mean positive but median ≈ 0
  win rate ≈ 50%
  gain driven by 1–2 seeds
  CI heavily overlaps zero
  improvement < ROPE / practical margin
  worst subgroup regresses materially
  performance reverses under plausible shift
  small hyperparameter perturbation removes gain
```

### Signal-to-Noise Ratio (Diagnostic)

$$
SNR_\Delta = \frac{|\bar\Delta|}{SD(\Delta)}
$$

Contoh: Δ = +0.010 ± .002 → SNR = 5 (credible); Δ = +0.010 ± .020 → SNR = 0.5 (not credible).

### Robustness Profile Output Schema

```yaml
robustness:
  nominal:
    mean_delta: 0.0081
    median_delta: 0.0078

  seeds:
    n: 10
    win_rate: 0.90
    delta_sd: 0.0021
    worst_delta: -0.0005

  folds:
    repeats: 5
    fold_win_rate: 0.84

  uncertainty:
    method: bayesian_correlated_t
    rope: [-0.003, 0.003]
    p_better: 0.94
    p_equivalent: 0.05
    p_worse: 0.01

  stress:
    missingness_5pct:
      delta: 0.007
    feature_noise:
      delta: 0.006
    temporal_ood:
      delta: 0.004
    worst_group:
      delta: -0.002

  verdict:
    nominal: robust
    subgroup: warning
    overall: conditionally_robust
```

---

## Referensi Utama

1. **Nadeau, C., & Bengio, Y.** (2003). Inference for the Generalization Error. *Machine Learning*, 52, 239–281.
2. **Bengio, Y., & Grandvalet, Y.** (2004). No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *JMLR*, 5, 1089–1105.
3. **Bouckaert, R. R., & Frank, E.** (2004). Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms. *PAKDD 2004*.
4. **Benavoli, A., et al.** (2017). Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. *JMLR*, 18, 1–36.
5. **Bayle, P., et al.** (2020). Cross-validation Confidence Intervals for Test Error. *NeurIPS 2020*.
6. **Bates, S., Hastie, T., & Tibshirani, R.** (2024). Cross-validation: What Does It Estimate and How Well Does It Do It? *JASA*, 119(546), 1434–1445.
7. **Bousquet, O., & Elisseeff, A.** (2002). Stability and Generalization. *JMLR*, 2, 499–526.
8. **Sagawa, S., et al.** (2020). Distributionally Robust Neural Networks. *ICLR 2020*.
9. **Gardner, J., Popovic, Z., & Schmidt, L.** (2023). Benchmarking Distribution Shift in Tabular Data with TableShift. *NeurIPS 2023*.
10. **Lakens, D.** (2017). Equivalence Tests: A Practical Primer. *SPPS*, 8(4), 355–362.
11. **MacKinnon, J. G.** (2023). Cluster-Robust Inference: A Guide. *Queen's University*.
12. **Hüllermeier, E., & Waegeman, W.** (2021). Aleatoric and Epistemic Uncertainty in Machine Learning. *Machine Learning*, 110(3), 457–506.
13. **Angelopoulos, A. N., et al.** (2024). Conformal Risk Control. *ICLR 2024*.
14. **Forstmeier, W., & Schielzeth, H.** (2011). Cryptic Multiple Hypotheses Testing in Linear Models. *Behav Ecol Sociobiol*, 65(1), 47–55.
15. **He, Z., et al.** (2025). TabAttackBench. *Expert Systems with Applications*, 301, 130491.
16. **DeLong, E. R., et al.** (1988). Comparing the Areas Under Two or More Correlated ROC Curves. *Biometrics*, 44(3), 837–845.
17. **Mosbach, M., et al.** (2021). On the Stability of Fine-Tuning BERT. *ICLR 2021*.
18. **Demšar, J.** (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR*, 7, 1–30.
19. **Cawley, G. C., & Talbot, N. L. C.** (2010). On Over-fitting in Model Selection. *JMLR*, 11, 2079–2107.
20. **Frénay, B., & Verleysen, M.** (2014). Classification in the Presence of Label Noise: A Survey. *IEEE TNNLS*, 25(5), 845–869.

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*