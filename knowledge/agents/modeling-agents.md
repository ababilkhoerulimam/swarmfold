---
title: SwarmFold ML Method Knowledge Base
schema_version: 1.0.0
language: en-id
retrieval_scope: role_scoped
last_reviewed: 2026-09-15
---

# SwarmFold ML Method Knowledge Base

Dokumen ini adalah seed knowledge untuk retrieval agent SwarmFold. Entri bukan perintah
eksekusi. Model hanya boleh masuk experiment queue setelah task, metric, validation,
package availability, resource budget, dan competition rules dikonfirmasi.

## Evidence labels

- `V`: verified against primary paper or official documentation.
- `P`: provisional; technically established but implementation/version details require checking.
- `U`: unverified seed; preserve for later research but never present as executable fact.

## Entry field mapping

Setiap baris memuat seluruh kontrak berikut dalam bentuk padat:

- `Fit`: definisi, supported problem, kapan digunakan/tidak, asumsi, preprocessing, scale, metric.
- `Tune/compute`: hyperparameter penting, initial range, CPU/GPU/RAM profile.
- `Risk`: leakage risks dan common failure modes.
- `Compare/ensemble`: teknik pembanding dan ensemble compatibility.
- `Implementation`: exact package dan estimator jika terverifikasi.
- `Experiment gate`: contoh hypothesis serta acceptance/rejection criteria.
- `Evidence`: source ID, status, dan tanggal review.

Nilai `n/a` berarti field tidak berlaku, bukan belum diperiksa. `Unverified implementation`
berarti agent dilarang memasukkan metode ke execution queue.

## 1. Classification

| Technique | Fit | Tune/compute | Risk | Compare/ensemble | Implementation | Experiment gate | Evidence |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear probabilistic classifier for binary/multiclass targets. Use as interpretable baseline on small-large sparse or tabular data; avoid when unengineered nonlinear interactions dominate. Scale numeric values; encode categories fold-safely. Metrics: ROC-AUC, log-loss, accuracy/F1. | `C` 0.01-100 log scale, penalty L1/L2/elastic-net, class weight. CPU low, RAM low-medium for wide one-hot data. | Leakage from target encoding or preprocessing before folds; separation, multicollinearity, poor nonlinear fit. | Compare dummy/tree baselines. Strong diverse member for voting, blending, stacking meta-learner. | `sklearn.linear_model.LogisticRegression` | Hypothesis: regularized linear effects provide a stable score floor. Accept when CV beats trivial baseline with stable folds; reject as champion when nonlinear models show robust gain. | SK, V, reviewed 2026-09-15 |
| Linear Discriminant Analysis | Generative linear classifier assuming class-conditional Gaussian features and shared covariance. Use on small numeric data; avoid heavily non-Gaussian, sparse, or singular high-dimensional data without shrinkage. | Solver `svd/lsqr/eigen`, shrinkage `auto` or 0-1. CPU/RAM low-medium; covariance cost grows with features. | Leakage from global scaling; unstable covariance and violated distribution assumptions. | Compare logistic/QDA. Blend when errors differ. | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | Accept if repeated CV is stable and calibration useful; reject if covariance warnings or consistent loss to logistic. | SK, V, reviewed 2026-09-15 |
| Quadratic Discriminant Analysis | Gaussian classifier with class-specific covariance and nonlinear boundaries. Use when classes plausibly have different covariance and samples per class are sufficient; avoid high dimension with few rows. | `reg_param` 0-1, `tol`. CPU medium, RAM medium-high. | Singular covariance, severe overfit; all transforms must be fold-local. | Compare LDA/logistic/tree models; useful only as low-correlation member. | `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` | Accept only with stable gain across folds and no covariance failure; reject on high variance. | SK, V, reviewed 2026-09-15 |
| Perceptron | Online linear margin classifier without calibrated probabilities. Use as very fast separability baseline; avoid probability metrics unless calibrated. Scale inputs. | `alpha` 1e-6-1e-2, penalty, epochs, class weight. CPU/RAM low. | Order sensitivity and convergence instability; metric mismatch for ROC/log-loss. | Compare SGD/logistic; bag or average seeds only if justified. | `sklearn.linear_model.Perceptron` | Accept as compute baseline when it matches slower linear models; reject for required probability output without calibration. | SK, V, reviewed 2026-09-15 |
| Gaussian Naive Bayes | Gaussian independent-feature probabilistic classifier. Use for small continuous data and fast baseline; avoid strongly dependent/non-Gaussian features. | `var_smoothing` 1e-12-1e-6. CPU/RAM very low. | Independence violations, probability overconfidence, global preprocessing leakage. | Compare logistic/LDA; diversity member when errors differ. | `sklearn.naive_bayes.GaussianNB` | Accept if calibrated CV is competitive; reject on poor log-loss or unstable probabilities. | SK, V, reviewed 2026-09-15 |
| Multinomial Naive Bayes | Count/non-negative feature classifier for text and frequencies. Use sparse counts/TF-IDF; do not use negative scaled features. | `alpha` 1e-3-10, class prior. CPU/RAM low. | Negative inputs, vocabulary leakage, uncalibrated confidence. | Compare linear SVM/logistic; useful sparse-model blend member. | `sklearn.naive_bayes.MultinomialNB` | Accept if fold-local vectorization beats dummy and linear baseline cost-adjusted; reject on invalid input domain. | SK, V, reviewed 2026-09-15 |
| Bernoulli Naive Bayes | Bernoulli event classifier for binary feature occurrence. Use binary text/event matrices; avoid continuous magnitude data unless binarization is meaningful. | `alpha` 1e-3-10, `binarize`. CPU/RAM low. | Arbitrary thresholding and vocabulary leakage. | Compare MultinomialNB/logistic. | `sklearn.naive_bayes.BernoulliNB` | Accept when occurrence matters more than frequency and CV gain is stable; otherwise reject. | SK, V, reviewed 2026-09-15 |
| K-Nearest Neighbors Classifier | Non-parametric local-distance classifier. Use small-medium low-dimensional scaled data; avoid very large, sparse, or high-dimensional data. | `n_neighbors` 3-100, distance metric, weights. Fit cheap; inference and RAM high. | Distance domination, curse of dimensionality, preprocessing leakage. | Compare linear/SVM/tree; can add local-boundary diversity. | `sklearn.neighbors.KNeighborsClassifier` | Accept only when validation gain survives neighbor/metric sensitivity and inference budget; reject on slow prediction or unstable neighborhoods. | SK, V, reviewed 2026-09-15 |
| Support Vector Classifier | Maximum-margin linear/kernel classifier. Use small-medium scaled data; RBF for nonlinear boundaries. Avoid very large row counts without approximation. | `C` 1e-3-1e3, `gamma` 1e-5-10, kernel. CPU/RAM medium-very high; probability calibration adds cost. | Scaling errors, quadratic runtime, leakage in calibration. | Compare logistic/RBF alternatives; blend only with full aligned OOF probabilities. | `sklearn.svm.SVC` or `LinearSVC` | Accept if fixed-fold gain justifies runtime and probability needs; reject on scaling sensitivity or impractical fit. | SK, V, reviewed 2026-09-15 |
| Decision Tree Classifier | Single recursive partition tree. Use as interpretable nonlinear baseline; avoid unrestricted depth as final model. | Depth 2-20, min leaf 1-5% rows, criterion. CPU/RAM low-medium. | Severe variance/overfit; target-derived features. | Baseline for Random Forest, Extra Trees, bagging, boosting. | `sklearn.tree.DecisionTreeClassifier` | Accept as diagnostic baseline; promote only if pruned tree is robust and complexity is required. | SK, V, reviewed 2026-09-15 |
| Random Forest Classifier | Bagged randomized decision trees. Use robust nonlinear baseline on mixed encoded tabular data; avoid extreme sparse width or strict latency. | Trees 200-1500, max features `sqrt`/0.3-1.0, depth, min leaf, class weight. CPU parallel, RAM high. | Group/time leakage, biased impurity importance, memory blowup. | Compare Extra Trees/GBM; strong bagging and stacking member. | `sklearn.ensemble.RandomForestClassifier` | Accept if OOF gain and seed stability beat a single tree; reject if dominated at high compute with correlated errors. | SK, V, reviewed 2026-09-15 |
| Extra Trees Classifier | Highly randomized tree ensemble reducing variance through random thresholds. Use for nonlinear diversity; avoid when random splits create systematic underfit. | Trees 200-1500, max features, depth, min leaf. CPU parallel, RAM high. | Same leakage risks as forests; probability granularity and underfit. | Compare Random Forest/GBM; valuable low-correlation blend member. | `sklearn.ensemble.ExtraTreesClassifier` | Accept when OOF diversity improves a blend or standalone score; reject if correlation is high and score inferior. | SK, V, reviewed 2026-09-15 |
| Bagging Classifier | Bootstrap aggregation of a configurable base estimator. Use to reduce variance of unstable learners; avoid bagging stable high-bias learners without evidence. | Estimators 50-1000, samples/features 0.5-1.0, bootstrap. CPU/RAM medium-high. | Row/group resampling can violate split structure; correlated members yield no gain. | Compare Random Forest and single base estimator; itself an ensemble member. | `sklearn.ensemble.BaggingClassifier` | Accept when identical-fold score or stability beats base estimator; reject negligible gain per compute. | SK, V, reviewed 2026-09-15 |
| AdaBoost Classifier | Sequentially reweights difficult observations using weak learners. Use clean small-medium tabular data; avoid heavy label noise/outliers. | Estimators 50-1000, learning rate 0.01-1, base-tree depth 1-3. CPU medium. | Chases mislabeled rows, unstable probabilities, leakage from precomputed features. | Compare gradient boosting; diversify tree ensembles if OOF correlation permits. | `sklearn.ensemble.AdaBoostClassifier` | Accept when gain is consistent and hard-row analysis is credible; reject if noisy cohorts drive score. | SK, V, reviewed 2026-09-15 |
| Histogram Gradient Boosting Classifier | Histogram-binned gradient-boosted trees. Use medium-large numeric/encoded tabular data; avoid expecting semantic categorical handling without explicit configuration. | Iterations 100-1500, learning rate 0.02-0.15, depth 3-12, leaves, L2, early stopping. CPU medium, RAM medium. | Generic naming hides estimator/config; ordinal encoding can imply false order; split leakage. | Compare LightGBM/XGBoost/CatBoost; strong blend/stack member. | `sklearn.ensemble.HistGradientBoostingClassifier` | Accept when OOF gain survives depth/rate ablation; reject if gain depends on unsafe encoding. | SK, V, reviewed 2026-09-15 |
| XGBoost Classifier | Regularized gradient-boosted decision trees. Use medium-large tabular data and sparse matrices; avoid unbounded tuning. | Trees 300-5000, depth 3-10, rate 0.01-0.2, subsample/columns 0.5-1, min child, L1/L2, early stopping. CPU/GPU medium-high. | Leakage, overfit from depth/trials, categorical serialization/version pitfalls. | Compare LightGBM/CatBoost; core weighted-blend and stacking member. | `xgboost.XGBClassifier` | Accept robust fixed-fold gain with saved best iteration; reject if only one fold/leaderboard improves. | XGB, V, reviewed 2026-09-15 |
| LightGBM Classifier | Leaf-wise histogram gradient boosting optimized for speed. Use large tabular and categorical/integer-coded data with careful constraints; avoid unconstrained leaves on small data. | Leaves 15-255, depth, min child 20-1000, rate 0.01-0.15, feature/bagging fractions, L1/L2. CPU/GPU efficient. | Leaf-wise overfit, category coding errors, early-stopping leakage. | Compare XGBoost/CatBoost; high-value ensemble member. | `lightgbm.LGBMClassifier` | Accept when fold-stable gain and runtime improve portfolio frontier; reject high variance or leakage-sensitive gain. | LGB, V, reviewed 2026-09-15 |
| CatBoost Classifier | Ordered boosting with native categorical feature processing. Use mixed tabular data with categorical columns; do not one-hot encode by default. | Iterations 300-5000, depth 4-10, rate 0.02-0.15, L2 1-20, random strength, bagging temperature, early stopping. CPU/GPU medium-high. | Incorrect `cat_features`, time/order leakage, target-statistic misuse outside CatBoost. | Compare LightGBM/XGBoost/HGB; often a strong standalone and diverse blend member. | `catboost.CatBoostClassifier` | Accept if native categorical pipeline gives robust OOF gain; reject if gain disappears under valid split/order controls. | CAT, V, reviewed 2026-09-15 |
| Multi-Layer Perceptron Classifier | Fully connected neural classifier. Use sufficiently large normalized data or learned embeddings; avoid as automatic tabular default on small data. | Layers 1-5, width 64-1024, dropout 0-0.5, weight decay 1e-7-1e-2, learning rate 1e-5-1e-2. GPU preferred; RAM/VRAM medium-high. | Scaling, seed variance, overfit, early-stopping leakage. | Compare GBM and linear baselines; use as diversity member only with OOF evidence. | `sklearn.neural_network.MLPClassifier` for basic CPU; framework-specific for deep variants. | Accept if multi-seed OOF gain/diversity justifies cost; reject unstable or dominated results. | SK, V, reviewed 2026-09-15 |
| TabNet | Sequential attention architecture for tabular feature selection. Use as research candidate on medium-large tabular data; avoid assuming attention equals reliable interpretability. | Decision/attention dims, steps, sparsity, batch size, learning rate. GPU preferred, VRAM medium-high. | Seed sensitivity, preprocessing and benchmark-selection bias. | Compare strong tuned GBMs/MLP; possible neural diversity member. | `pytorch_tabnet.tab_model.TabNetClassifier` | Accept only on repeated-fold gain or useful ensemble diversity; reject default-only comparison. | TABNET, P, reviewed 2026-09-15 |
| FT-Transformer | Transformer over embedded tabular features. Use medium-large data with categorical/numeric interactions; avoid low-data/high-cost settings without evidence. | Layers 2-6, dim 64-512, heads, dropout, weight decay, learning rate. GPU preferred. | Category leakage, rare-level handling, seed variance. | Compare tuned GBM/MLP; ensemble for representation diversity. | Common implementation: `rtdl_revisiting_models` or compatible PyTorch implementation; verify API. | Accept multi-seed gain/diversity; reject if slower and no robust improvement. | FTTR, P, reviewed 2026-09-15 |
| Tabular ResNet | Residual MLP architecture for tabular data. Use as strong neural baseline when data volume supports it; avoid replacing tree baselines by assumption. | Width/depth, hidden factor, dropout, normalization, LR/weight decay. GPU preferred. | Seed variance and preprocessing sensitivity. | Compare MLP/FT-Transformer/GBMs; potential stack member. | PyTorch implementation from RTDL-style libraries; exact package must be pinned. | Accept only with repeated-seed evidence; reject unstable marginal gains. | RTDL, P, reviewed 2026-09-15 |
| TabPFN 2.5 | Prior-data fitted tabular foundation model for classification/regression within supported row/feature limits. Use small-medium tabular data; avoid datasets beyond documented context limits without subsampling/distillation evidence. | Ensemble/config count, preprocessing and device settings; typically GPU/high RAM. | Benchmark transfer failure, context limits, hidden preprocessing assumptions. | Compare tuned GBM and AutoML; ensemble only with OOF predictions. | Official TabPFN package/API must be version-pinned. | Accept when local fixed-fold performance and runtime beat tuned baselines; reject claims based only on reported benchmark win rates. | TPFN25, V paper claim, reviewed 2026-09-15 |
| TabICLv2 | In-context-learning tabular foundation model reported on tabular benchmarks. Use only after package, weights, task limits, and license are verified. | Model-specific; GPU expected. | Unverified local compatibility and benchmark transfer. | Compare TabPFN/tuned GBMs. | Unverified implementation in this catalog. | Research-only until reproducible local adapter and OOF evaluation exist. | USER-SEED, U, reviewed 2026-09-15 |
| LimiX | Tabular foundation model candidate. | Unknown until primary artifact review. | Name/version ambiguity and unavailable adapter. | Compare other verified TFMs only after validation. | Unverified implementation. | Never enqueue while status is U. | USER-SEED, U, reviewed 2026-09-15 |
| Mitra | Transformer TFM trained with mixed synthetic priors for classification/regression. Use as research candidate within published limits after code/weights verification. | Model/version-specific; GPU expected. | New-model instability, benchmark transfer, version confusion with Mitra-v2. | Compare TabPFN/TabICL/tuned GBM. | Primary repository/API must be pinned before execution. | Accept only via local OOF evidence; reject benchmark-only inference. | MITRA, V paper existence, reviewed 2026-09-15 |
| TabPFN-Wide | Wide-feature TabPFN variant named in seed list. | Unknown. | Identity, source, and implementation unverified. | n/a until verified. | Unverified implementation. | Never enqueue while status is U. | USER-SEED, U, reviewed 2026-09-15 |
| Orion-MSP | Tabular foundation model name from seed list. | Unknown. | Identity and evidence unverified. | n/a until verified. | Unverified implementation. | Never enqueue while status is U. | USER-SEED, U, reviewed 2026-09-15 |
| Orion-Bix | Tabular foundation model name from seed list. | Unknown. | Identity and evidence unverified. | n/a until verified. | Unverified implementation. | Never enqueue while status is U. | USER-SEED, U, reviewed 2026-09-15 |

## 2. Regression

| Technique | Fit | Tune/compute | Risk | Compare/ensemble | Implementation | Experiment gate | Evidence |
|---|---|---|---|---|---|---|---|
| Ordinary Least Squares / Linear Regression | Conditional-mean linear regression. Use interpretable baseline with numeric/fold-safe encoded features; avoid when nonlinearities, heteroscedasticity, or extreme collinearity dominate. Metrics: RMSE/MAE/R2. | Fit intercept; numerical conditioning. CPU/RAM low, but wide dense matrices can be costly. | Leakage from preprocessing/feature selection; outlier sensitivity and extrapolation error. | Compare mean baseline, Ridge, Huber, trees; stacking meta-learner. | `sklearn.linear_model.LinearRegression` or `statsmodels.api.OLS` | Accept if CV beats mean/median baseline stably; reject implausible coefficients or unstable folds. | SK/STAT, V, reviewed 2026-09-15 |
| Ridge Regression | L2-regularized linear regression. Use correlated/wide features; scale numeric inputs. | `alpha` 1e-6-1e6 log scale. CPU/RAM low-medium. | Global scaling leakage; excessive shrinkage underfit. | Compare OLS/Lasso/Elastic Net; reliable blend/meta member. | `sklearn.linear_model.Ridge` | Accept stable CV gain over OLS; reject flat or degraded response across alpha. | SK, V, reviewed 2026-09-15 |
| Lasso Regression | L1-regularized sparse linear regression. Use high-dimensional settings where sparse selection is plausible; scale inputs. | `alpha` 1e-7-1e2, iterations/tolerance. CPU medium. | Unstable feature selection under correlation; selection outside folds leaks. | Compare Ridge/Elastic Net; use selected set only through pipeline. | `sklearn.linear_model.Lasso` | Accept if score and selected-set stability pass; reject unstable support or predictive loss. | SK, V, reviewed 2026-09-15 |
| Elastic Net | Combined L1/L2 regularization for correlated sparse effects. | `alpha` 1e-7-1e2, `l1_ratio` 0.05-0.95. CPU medium. | Same fold leakage and scaling risks; convergence. | Compare Ridge/Lasso. Useful stable linear ensemble member. | `sklearn.linear_model.ElasticNet` | Accept when joint search beats both endpoints robustly; reject no benefit over simpler regularizer. | SK, V, reviewed 2026-09-15 |
| Least-Angle Regression | Efficient path-following linear method for high-dimensional sparse regression. Use when rows exceed or are comparable to features; avoid noisy collinear instability. | Number of nonzero coefficients; CPU low-medium. | Selection instability and leakage. | Compare Lasso/Elastic Net. | `sklearn.linear_model.Lars` or `LassoLars` | Accept only with stable support and CV; otherwise reject. | SK, V, reviewed 2026-09-15 |
| Polynomial Regression | Linear estimator over polynomial/interaction expansion. Use low-dimensional smooth nonlinear relationships; avoid high-dimensional combinatorial explosion. | Degree 2-4, interaction-only, regularization. CPU/RAM can become very high. | Expansion before split, extrapolation, collinearity. | Compare splines/MARS/GBMs; combine with Ridge. | `sklearn.preprocessing.PolynomialFeatures` + linear estimator | Accept through ablation and stable CV; reject memory explosion or fragile gain. | SK, V, reviewed 2026-09-15 |
| Stepwise Regression | Iterative add/drop feature selection by criterion. Use only for controlled low-dimensional explanatory analysis; avoid as default predictive search. | Entry/removal threshold or AIC/BIC. CPU medium. | Multiple testing, unstable selection, severe validation leakage if not nested. | Compare L1/RFE and full model. Rarely an ensemble target. | No single canonical sklearn estimator; implementation must be declared. | Accept only under nested validation and stable selection; otherwise reject. | STAT, P, reviewed 2026-09-15 |
| Quantile Regression | Estimates conditional quantiles rather than mean. Use asymmetric loss, intervals, or heteroscedastic targets; do not compare directly under incompatible mean metrics. | Quantile 0.01-0.99, regularization. CPU medium-high. | Quantile crossing, metric mismatch, leakage. | Compare median/GB quantile; ensemble quantiles only coherently. | `sklearn.linear_model.QuantileRegressor` or statsmodels quantile regression | Accept on official pinball/quantile metric and coverage diagnostics; reject metric mismatch. | SK/STAT, V, reviewed 2026-09-15 |
| RANSAC Regression | Robust estimator fitted from repeated inlier subsets. Use data with gross outliers and a meaningful residual threshold; avoid when outliers are the signal. | Residual threshold, min samples, trials. CPU medium-high. | Arbitrary inlier definition, unstable random subsets. | Compare Huber/quantile/tree models. | `sklearn.linear_model.RANSACRegressor` | Accept if robust CV and inlier diagnostics improve; reject unstable consensus. | SK, V, reviewed 2026-09-15 |
| Huber Regression | Linear robust regression with quadratic-to-linear loss. Use moderate contamination; scale features. | `epsilon` 1.1-2.0, `alpha` 1e-6-10. CPU medium. | Underfit legitimate extremes, scaling leakage. | Compare OLS/RANSAC/quantile. | `sklearn.linear_model.HuberRegressor` | Accept if error-tail robustness improves without central degradation; reject otherwise. | SK, V, reviewed 2026-09-15 |
| Support Vector Regression | Epsilon-insensitive kernel/linear regression. Use small-medium scaled datasets; avoid large row counts with nonlinear kernel. | `C` 1e-3-1e3, epsilon 1e-4-1, gamma. CPU/RAM high for RBF. | Scaling and quadratic complexity; leakage. | Compare Ridge/kernel approximations/GBM. | `sklearn.svm.SVR` or `LinearSVR` | Accept if fixed-fold gain justifies inference/fit cost; reject impractical scaling. | SK, V, reviewed 2026-09-15 |
| MARS | Adaptive piecewise linear spline regression. Use nonlinear low-medium-dimensional relationships; avoid huge sparse feature spaces. | Max terms, interactions, pruning penalty. CPU medium-high. | Knot overfit and package inconsistency. | Compare polynomial splines/GBM/EBM. | Common package `pyearth.Earth`; package maintenance must be verified. | Research-only until adapter/package status verified; then require CV ablation. | USER-SEED, P, reviewed 2026-09-15 |
| Poisson Regression | Generalized linear model for non-negative counts, usually with log link and optional exposure offset. Avoid continuous signed targets or severe overdispersion without alternatives. | Alpha/penalty, exposure/offset. CPU low-medium. | Target/exposure leakage, zeros/overdispersion, wrong likelihood. | Compare mean count, Negative Binomial, Tweedie, GB Poisson. | `sklearn.linear_model.PoissonRegressor` or statsmodels GLM | Accept on suitable deviance/official metric and residual checks; reject distribution mismatch. | SK/STAT, V, reviewed 2026-09-15 |
| Random Forest Regressor | Bagged randomized regression trees for nonlinear interactions. | Trees 200-1500, depth, min leaf, max features. CPU/RAM high. | Poor extrapolation, memory, group/time leakage. | Compare Extra Trees/GBMs; blend for diversity. | `sklearn.ensemble.RandomForestRegressor` | Accept robust CV gain and diversity; reject compute-dominated correlated result. | SK, V, reviewed 2026-09-15 |
| XGBoost Regressor | Regularized boosted trees for tabular regression. | Trees, depth, rate, sampling, L1/L2, objective, early stopping. CPU/GPU medium-high. | Overfit, metric/objective mismatch, leakage. | Compare LightGBM/CatBoost; core blend member. | `xgboost.XGBRegressor` | Accept robust same-fold gain; reject leaderboard-only or unstable improvement. | XGB, V, reviewed 2026-09-15 |
| LightGBM Regressor | Leaf-wise histogram boosting for efficient large tabular regression. | Leaves, min child, rate, sampling, L1/L2. CPU/GPU efficient. | Small-data overfit and category coding errors. | Compare XGB/CatBoost/HGB; blend/stack. | `lightgbm.LGBMRegressor` | Accept stable frontier gain; reject fold-sensitive leaf complexity. | LGB, V, reviewed 2026-09-15 |
| CatBoost Regressor | Ordered boosted trees with native categorical handling. | Iterations, depth, rate, L2, loss, early stopping. CPU/GPU medium-high. | Category declaration/order leakage and expensive tuning. | Compare LGB/XGB; strong heterogeneous blend member. | `catboost.CatBoostRegressor` | Accept native-category OOF gain; reject unsafe preprocessing or unstable gain. | CAT, V, reviewed 2026-09-15 |
| Explainable Boosting Machine | Interpretable generalized additive boosted model with optional pair interactions. Use when shape functions matter; avoid assuming interpretability removes validation needs. | Bins, rounds, interactions, learning rate. CPU/RAM medium-high. | Interaction overfit and misleading post-hoc interpretation. | Compare GAM/GBM; possible interpretable blend member. | `interpret.glassbox.ExplainableBoostingRegressor` | Accept competitive CV plus stable shape diagnostics; reject if complexity adds no score or clarity. | EBM, P, reviewed 2026-09-15 |
| IGANN | Interpretable generalized additive neural network candidate. | Model-specific; CPU/GPU varies. | Package, implementation, and interpretability claims require verification. | Compare EBM/GAM/MLP. | Unverified adapter. | Do not enqueue until primary source and package are pinned. | USER-SEED, U, reviewed 2026-09-15 |
| GOSDT | Globally optimized sparse decision-tree approach. Use small-medium datasets when sparse tree optimality/interpretability matters; avoid large unrestricted search. | Depth/leaves, regularization, time limit. CPU/RAM potentially very high. | Solver timeout, discretization and validation leakage. | Compare CART/EBM. | Package/API must be verified and pinned. | Accept if solver completes and CV/complexity frontier improves; reject timeout or no advantage. | USER-SEED, P, reviewed 2026-09-15 |
| TabPFN 2.5 Regression | TFM regression mode within published context limits. | Device/configuration specific; GPU/high RAM likely. | Context limits and benchmark transfer. | Compare tuned GBM/AutoGluon. | Official version-pinned TabPFN API. | Accept only from local fixed-fold evidence. | TPFN25, V paper claim, reviewed 2026-09-15 |
| TabICLv2 Regression | TFM regression candidate from seed list. | Unknown. | Source/API/task limits unverified. | Compare verified TFMs and tuned GBM after validation. | Unverified implementation. | Never enqueue while status is U. | USER-SEED, U, reviewed 2026-09-15 |

# SwarmFold ML Method Knowledge Base — Expanded Documentation

> **Versi:** 2.0 | **Tier:** Berat (Research-Grade) | **Bahasa:** Indonesia
>
> Dokumen ini merupakan perluasan dari SwarmFold ML Method Knowledge Base dengan tambahan **probabilistic regression, kernel methods, GLM families, deep tabular architectures, imbalance-specialized ensembles, online/streaming methods, sparse interaction models, dan tabular foundation models terbaru**. Setiap entri mengikuti format 20-field untuk konsumsi AI agents.

---

## Daftar Isi (Tambahan)

1. [Probabilistic Regression](#1-probabilistic-regression)
2. [Kernel Methods](#2-kernel-methods)
3. [GLM Families (Tweedie, Gamma, Negative Binomial, Zero-Inflated)](#3-glm-families)
4. [Deep Tabular Architectures](#4-deep-tabular-architectures)
5. [Imbalance-Specialized Ensembles](#5-imbalance-specialized-ensembles)
6. [Rule-Based / Interpretable Ensembles](#6-rule-based--interpretable-ensembles)
7. [Bayesian / Distributional Trees](#7-bayesian--distributional-trees)
8. [Online / Streaming Methods](#8-online--streaming-methods)
9. [Sparse Interaction / CTR Models](#9-sparse-interaction--ctr-models)
10. [Tabular Foundation Models (Update)](#10-tabular-foundation-models-update)

---

## 1. PROBABILISTIC REGRESSION

### 1.1 NGBoost (Natural Gradient Boosting)

| Field | Detail |
|-------|--------|
| **Definisi** | Gradient boosting untuk **probabilistic prediction** yang memprediksi parameter distribusi kondisional (mean, variance) secara simultan menggunakan natural gradient. Alih-alih hanya point estimate, NGBoost memodelkan seluruh distribusi P(Y\|X)  |
| **Problem yang didukung** | Regression probabilistik, Uncertainty quantification, Prediction intervals |
| **Kapan digunakan** | Heteroscedastic regression; butuh prediction intervals; uncertainty-sensitive applications; metric NLL/CRPS |
| **Kapan tidak digunakan** | Hanya butuh point prediction; dataset sangat kecil; distribusi tidak dapat dimodelkan secara parametrik |
| **Asumsi** | Distribusi kondisional dapat dimodelkan dengan parametrik family; natural gradient mempercepat konvergensi |
| **Data preprocessing** | Standar; scaling untuk model linear |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | NLL, CRPS, RMSE (secondary) |
| **Hyperparameter penting** | `n_estimators`, `learning_rate`, `minibatch_frac`, `Distribution` (Normal, LogNormal, Exponential) |
| **Kisaran parameter awal** | n_estimators: 100–500; lr: 0.01–0.1; Distribution: Normal |
| **CPU/GPU/RAM** | CPU-bound; RAM moderate |
| **Risiko leakage** | Rendah; training/validation split standar |
| **Failure modes** | Distribusi salah → calibration buruk; natural gradient tidak stabil untuk distribusi kompleks |
| **Teknik pembanding** | GPR, BART, Quantile Regression Forest |
| **Ensemble compatibility** | Universal; probabilistic ensemble member |
| **Exact package** | `ngboost.NGBRegressor`, `ngboost.NGBClassifier`  |
| **Contoh hypothesis** | "NGBoost dengan distribusi Normal akan mencapai CRPS ≥15% lebih rendah dari linear regression pada data heteroskedastik" |
| **Acceptance criteria** | CRPS improvement ≥10%; calibration coverage ≈ nominal |
| **Rejection criteria** | CRPS tidak improvement; distribusi tidak konvergen |
| **Sumber** | Duan et al., ICML 2020  |

---

### 1.2 Gaussian Process Regression (GPR)

| Field | Detail |
|-------|--------|
| **Definisi** | Bayesian nonparametric regression yang menghasilkan mean μ(x) dan variance σ(x) untuk setiap prediction, dengan posterior predictive distribution  |
| **Problem yang didukung** | Regression dengan uncertainty; small-data; smooth function approximation |
| **Kapan digunakan** | Dataset kecil; numerical features; uncertainty valuable; smooth function plausible |
| **Kapan tidak digunakan** | Dataset besar (>10.000 rows); high-dimensional; tidak butuh uncertainty |
| **Asumsi** | Function smoothness; kernel yang sesuai; Gaussian noise |
| **Data preprocessing** | Scaling; kernel selection |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | MSE, NLL, CRPS |
| **Hyperparameter penting** | `kernel` (RBF, Matern, dll.), `alpha` (noise), `optimizer`, `n_restarts_optimizer` |
| **Kisaran parameter awal** | kernel=RBF; alpha=1e-10; n_restarts=3–10 |
| **CPU/GPU/RAM** | O(n³) untuk training; O(n²) untuk prediction |
| **Risiko leakage** | Sedang; scaling harus fold-local |
| **Failure modes** | Kernel misspecification; computational cost O(n³); numerical instability |
| **Teknik pembanding** | NGBoost, BART, Kernel Ridge |
| **Ensemble compatibility** | Universal; dapat dikombinasi dengan stacking |
| **Exact package** | `sklearn.gaussian_process.GaussianProcessRegressor`; `GaussianProcessClassifier`  |
| **Contoh hypothesis** | "GPR dengan kernel Matern akan mencapai NLL ≥10% lebih rendah dari NGBoost pada dataset small-data" |
| **Acceptance criteria** | NLL improvement ≥10%; calibration baik |
| **Rejection criteria** | NLL tidak improvement; training >1 jam |
| **Sumber** | Rasmussen & Williams (2006); scikit-learn docs  |

---

### 1.3 Bayesian Ridge Regression

| Field | Detail |
|-------|--------|
| **Definisi** | Probabilistic linear regression dengan prior pada koefisien; menghasilkan posterior distribution untuk coefficients dan predictions  |
| **Problem yang didukung** | Regression dengan uncertainty; correlated features |
| **Kapan digunakan** | Butuh probabilistic interpretation; n relatif kecil; multicollinearity |
| **Kapan tidak digunakan** | Nonlinear relationships; dataset sangat besar |
| **Asumsi** | Linear relationship; Gaussian noise; prior pada coefficients |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | MSE, NLL, R² |
| **Hyperparameter penting** | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` (Gamma priors) |
| **Kisaran parameter awal** | Default priors (1e-6) |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Nonlinearity tidak tertangkap; prior terlalu kuat |
| **Teknik pembanding** | OLS, Ridge, ARD |
| **Ensemble compatibility** | Universal; meta-learner |
| **Exact package** | `sklearn.linear_model.BayesianRidge`  |
| **Contoh hypothesis** | "Bayesian Ridge akan mencapai NLL ≥10% lebih rendah dari OLS pada data multicollinear" |
| **Acceptance criteria** | NLL improvement ≥10%; coefficient stability |
| **Rejection criteria** | NLL tidak improvement |
| **Sumber** | Tipping (2001); scikit-learn docs |

---

### 1.4 ARD Regression (Automatic Relevance Determination)

| Field | Detail |
|-------|--------|
| **Definisi** | Bayesian sparse linear regression yang memberikan prior berbeda untuk setiap koefisien, menghasilkan sparsity otomatis  |
| **Problem yang didukung** | Regression dengan feature selection; high-dimensional |
| **Kapan digunakan** | High-dimensional; butuh sparsity; Bayesian interpretation |
| **Kapan tidak digunakan** | Semua fitur relevan; dataset sangat besar |
| **Asumsi** | Sparsity pada coefficients; Gaussian noise |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Menengah |
| **Metric compatibility** | MSE, NLL, R² |
| **Hyperparameter penting** | `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` |
| **Kisaran parameter awal** | Default priors |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Terlalu sparse → underfitting |
| **Teknik pembanding** | Lasso, Bayesian Ridge |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.ARDRegression`  |
| **Contoh hypothesis** | "ARD akan menghasilkan sparse solution yang mencapai MSE setara Ridge dengan 50% fitur" |
| **Acceptance criteria** | MSE setara; sparsity tercapai |
| **Rejection criteria** | MSE drop |
| **Sumber** | MacKay (1992); scikit-learn docs |

---

## 2. KERNEL METHODS

### 2.1 Kernel Ridge Regression (KRR)

| Field | Detail |
|-------|--------|
| **Definisi** | Ridge regression dengan kernel trick untuk nonlinear regression; closed-form solution  |
| **Problem yang didukung** | Nonlinear regression; small-medium data |
| **Kapan digunakan** | Nonlinear relationships; medium-sized data (<1000 samples); butuh closed-form solution |
| **Kapan tidak digunakan** | Dataset sangat besar (O(n³)); sparse solution dibutuhkan |
| **Asumsi** | Squared-error loss; kernel yang sesuai |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | MSE, RMSE, R² |
| **Hyperparameter penting** | `alpha` (regularization), `kernel`, `gamma` |
| **Kisaran parameter awal** | alpha: 1e-3–1e2; kernel='rbf' |
| **CPU/GPU/RAM** | O(n³) training; non-sparse prediction |
| **Risiko leakage** | Sedang |
| **Failure modes** | Non-sparse solution → slow prediction; alpha salah |
| **Teknik pembanding** | SVR, GPR, Random Fourier Features |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.kernel_ridge.KernelRidge`  |
| **Contoh hypothesis** | "KRR akan mencapai RMSE ≥10% lebih rendah dari Ridge pada data nonlinear" |
| **Acceptance criteria** | RMSE improvement ≥10% |
| **Rejection criteria** | RMSE drop; prediction time >10× |
| **Sumber** | scikit-learn docs; Murphy (2012)  |

---

## 3. GLM FAMILIES

### 3.1 Tweedie Regression

| Field | Detail |
|-------|--------|
| **Definisi** | GLM dengan distribusi Tweedie (Compound Poisson-Gamma) untuk target non-negatif dengan point mass di nol dan ekor kontinu positif  |
| **Problem yang didukung** | Insurance claim amount; customer spending; medical cost; semi-continuous revenue |
| **Kapan digunakan** | Many zeros + positive continuous tail; metric Tweedie deviance |
| **Kapan tidak digunakan** | Pure count (gunakan Poisson/NB); pure continuous tanpa zero |
| **Asumsi** | Compound Poisson-Gamma; 1 < p < 2 |
| **Data preprocessing** | Standard; exposure offset |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Tweedie deviance |
| **Hyperparameter penting** | `power` (Tweedie variance power), `alpha` |
| **Kisaran parameter awal** | power ∈ (1.2, 1.8); alpha: 1e-4–1e-2 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang; exposure handling |
| **Failure modes** | Power salah → distribution mismatch |
| **Teknik pembanding** | Frequency-Severity, Zero-Inflated |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.TweedieRegressor`  |
| **Contoh hypothesis** | "Tweedie dengan power=1.5 akan mencapai Tweedie deviance ≥10% lebih rendah dari linear regression" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Jørgensen & Paes de Souza (1994); scikit-learn docs  |

---

### 3.2 Gamma Regression

| Field | Detail |
|-------|--------|
| **Definisi** | GLM untuk target strictly positive dan right-skewed  |
| **Problem yang didukung** | Positive continuous skewed target |
| **Kapan digunakan** | Target >0; right-skewed; heteroskedasticity |
| **Kapan tidak digunakan** | Target dengan zero; symmetric |
| **Asumsi** | Gamma distribution; log link |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Gamma deviance |
| **Hyperparameter penting** | `alpha`, `link` |
| **Kisaran parameter awal** | alpha: 1e-4–1e-2 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Rendah |
| **Failure modes** | Distribution mismatch |
| **Teknik pembanding** | Tweedie, LogNormal |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.GammaRegressor`  |
| **Contoh hypothesis** | "Gamma regression akan mencapai deviance ≥10% lebih rendah dari log-transformed OLS" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Dunn & Smyth (2018); scikit-learn docs |

---

### 3.3 Negative Binomial GLM

| Field | Detail |
|-------|--------|
| **Definisi** | GLM untuk count data dengan overdispersion (variance > mean)  |
| **Problem yang didukung** | Overdispersed count data |
| **Kapan digunakan** | Count data; variance/mean > 1; Poisson overdispersion |
| **Kapan tidak digunakan** | Poisson dispersion (variance/mean ≈ 1) |
| **Asumsi** | Negative Binomial distribution; gamma-distributed latent variable |
| **Data preprocessing** | Exposure offset |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | NB deviance, log-likelihood |
| **Hyperparameter penting** | `alpha` (dispersion), `fit_intercept` |
| **Kisaran parameter awal** | alpha: 0.1–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Dispersion salah |
| **Teknik pembanding** | Poisson, Zero-Inflated NB |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels.GLM(family=NegativeBinomial)`; `lightgbm.LGBMRegressor(objective='poisson')` (proxy) |
| **Contoh hypothesis** | "NB GLM akan mencapai deviance ≥10% lebih rendah dari Poisson pada data overdispersed" |
| **Acceptance criteria** | Deviance improvement ≥10% |
| **Rejection criteria** | Deviance drop |
| **Sumber** | Hilbe (2011) |

---

### 3.4 Zero-Inflated Poisson / Negative Binomial

| Field | Detail |
|-------|--------|
| **Definisi** | Model count yang menangani excess zeros dari dua mekanisme: structural zeros dan sampling zeros |
| **Problem yang didukung** | Excess-zero count; insurance claims; healthcare utilization |
| **Kapan digunakan** | >30% zero; dua proses zero |
| **Kapan tidak digunakan** | Zero dari satu proses (gunakan Hurdle) |
| **Asumsi** | Dua proses; distribusi count |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah |
| **Metric compatibility** | ZIP/ZINB log-likelihood |
| **Hyperparameter penting** | `inflation_model`, `count_model` |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Identifiability |
| **Teknik pembanding** | Hurdle, Poisson |
| **Ensemble compatibility** | Universal |
| **Exact package** | `statsmodels.ZeroInflatedPoisson`; `pscl::zeroinfl` (R) |
| **Contoh hypothesis** | "ZINB akan mencapai AIC ≥10% lebih rendah dari Poisson pada data excess-zero" |
| **Acceptance criteria** | AIC improvement ≥10% |
| **Rejection criteria** | Tidak konvergen |
| **Sumber** | Cameron & Trivedi (2013) |

---

## 4. DEEP TABULAR ARCHITECTURES

### 4.1 TabM (Parameter-Efficient Ensembling)

| Field | Detail |
|-------|--------|
| **Definisi** | Model MLP sederhana yang meniru **ensemble MLP** dengan shared parameters dan multiple predictions per object; parameter-efficient ensembling  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Butuh neural tabular baseline yang kuat; resource-efficient |
| **Kapan tidak digunakan** | Dataset sangat kecil; tree-based sudah cukup |
| **Asumsi** | Shared parameters + rank-1 perturbation cukup untuk ensemble-like diversity |
| **Data preprocessing** | Numerical embeddings (PLE); scaling |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `k` (ensemble size), `n_blocks`, `d_block`, `dropout` |
| **Kisaran parameter awal** | k: 4–32; n_blocks: 2–4 |
| **CPU/GPU/RAM** | GPU preferred; memory-efficient |
| **Risiko leakage** | Rendah |
| **Failure modes** | k terlalu kecil → kurang diversity |
| **Teknik pembanding** | MLP, ResNet, FT-Transformer |
| **Ensemble compatibility** | Single model representing ensemble |
| **Exact package** | `tabm` (PyPI)  |
| **Contoh hypothesis** | "TabM dengan k=8 akan mencapai akurasi setara deep ensemble dengan training 5× lebih cepat" |
| **Acceptance criteria** | Akurasi setara; speedup ≥3× |
| **Rejection criteria** | Akurasi drop >2% |
| **Sumber** | Gorishniy et al., ICLR 2025  |

---

### 4.2 TabR (Retrieval-Augmented Tabular DL)

| Field | Detail |
|-------|--------|
| **Definisi** | Feed-forward network dengan **kNN-like retrieval component** di tengah: query row → learn representation → retrieve similar training rows → features + labels neighbors → prediction  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Tabular DL; ingin memanfaatkan retrieval |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Neighbor retrieval informatif |
| **Data preprocessing** | Numerical embeddings |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `d_main`, `d_multiplier`, `n_blocks`, `context_size` |
| **Kisaran parameter awal** | context_size: 10–100 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang; retrieval harus dari training fold |
| **Failure modes** | Context size salah |
| **Teknik pembanding** | TabM, ModernNCA, FT-Transformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `tabr` (GitHub)  |
| **Contoh hypothesis** | "TabR akan mencapai akurasi ≥3% lebih tinggi dari MLP pada data tabular" |
| **Acceptance criteria** | Akurasi improvement ≥3% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Gorishniy et al., ICLR 2024  |

---

### 4.3 ModernNCA

| Field | Detail |
|-------|--------|
| **Definisi** | Revitalisasi Neighbourhood Components Analysis (NCA) dengan teknik deep learning modern; differentiable KNN  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Tabular DL; competitive dengan CatBoost |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Learned representation untuk semantic similarity |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `d_block`, `n_blocks`, `dropout`, `lr` |
| **Kisaran parameter awal** | d_block: 64–256 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang |
| **Failure modes** | Representation collapse |
| **Teknik pembanding** | KNN, TabR, CatBoost |
| **Ensemble compatibility** | Universal |
| **Exact package** | `talent` (PyPI); ModernNCA (GitHub)  |
| **Contoh hypothesis** | "ModernNCA akan competitive dengan CatBoost pada 300 datasets" |
| **Acceptance criteria** | Akurasi ≥ CatBoost - 2% |
| **Rejection criteria** | Akurasi < CatBoost - 5% |
| **Sumber** | Ye et al., ICLR 2025  |

---

### 4.4 NODE (Neural Oblivious Decision Ensembles)

| Field | Detail |
|-------|--------|
| **Definisi** | Deep learning architecture yang menggeneralisasi **ensembles of oblivious decision trees** dengan differentiable, end-to-end gradient learning  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Tabular DL; differentiable tree family |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Oblivious trees dapat didiferensiasi |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `n_trees`, `depth`, `lr` |
| **Kisaran parameter awal** | n_layers: 1–4; n_trees: 128–2048 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Overfitting pada dataset kecil |
| **Teknik pembanding** | MLP, FT-Transformer, GBDT |
| **Ensemble compatibility** | Universal |
| **Exact package** | `node` (PyTorch)  |
| **Contoh hypothesis** | "NODE akan mencapai akurasi setara GBDT pada data tabular" |
| **Acceptance criteria** | Akurasi ≥ GBDT - 2% |
| **Rejection criteria** | Akurasi < GBDT - 5% |
| **Sumber** | Popov et al., ICLR 2020  |

---

### 4.5 TabTransformer

| Field | Detail |
|-------|--------|
| **Definisi** | Transformer yang mengubah categorical embeddings menjadi **contextual embeddings**; robust terhadap missing/noisy features  |
| **Problem yang didukung** | Tabular classification |
| **Kapan digunakan** | Mixed categorical/numerical; semi-supervised pretraining |
| **Kapan tidak digunakan** | Hanya numerical; dataset kecil |
| **Asumsi** | Categorical contextual embeddings informatif |
| **Data preprocessing** | Categorical embeddings |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `d_model`, `n_heads`, `dropout` |
| **Kisaran parameter awal** | n_layers: 2–6; d_model: 64–256 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Numerical features diabaikan |
| **Teknik pembanding** | FT-Transformer, SAINT |
| **Ensemble compatibility** | Universal |
| **Exact package** | `tabular-transformers` (PyPI)  |
| **Contoh hypothesis** | "TabTransformer akan mencapai akurasi ≥3% lebih tinggi dari MLP pada data categorical-heavy" |
| **Acceptance criteria** | Akurasi improvement ≥3% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Huang et al., arXiv:2012.06678  |

---

### 4.6 SAINT (Self-Attention and Intersample Attention Transformer)

| Field | Detail |
|-------|--------|
| **Definisi** | Attention over both **rows and columns** + contrastive self-supervised pretraining untuk labels scarce  |
| **Problem yang didukung** | Tabular classification |
| **Kapan digunakan** | Labels scarce; unlabeled rows available |
| **Kapan tidak digunakan** | Dataset sangat kecil; labels cukup |
| **Asumsi** | Row + column attention informatif |
| **Data preprocessing** | Enhanced embedding |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `d_model`, `n_heads`, `pretraining` |
| **Kisaran parameter awal** | n_layers: 2–6 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Pretraining tidak konvergen |
| **Teknik pembanding** | TabTransformer, VIME |
| **Ensemble compatibility** | Universal |
| **Exact package** | `saint` (GitHub)  |
| **Contoh hypothesis** | "SAINT dengan contrastive pretraining akan mencapai akurasi ≥5% lebih tinggi dari supervised-only pada 10% labeled" |
| **Acceptance criteria** | Akurasi improvement ≥5% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Somepalli et al., arXiv:2106.01342  |

---

### 4.7 DANets (Deep Abstract Networks)

| Field | Detail |
|-------|--------|
| **Definisi** | Neural network dengan **Abstract Layers** yang mengelompokkan correlated features dan membentuk higher-level abstractions  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Correlated features; feature abstraction |
| **Kapan tidak digunakan** | Fitur independen |
| **Asumsi** | Feature grouping informatif |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `n_abst_layers`, `d_block` |
| **Kisaran parameter awal** | n_layers: 2–6 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Abstraction tidak informatif |
| **Teknik pembanding** | FT-Transformer, MLP |
| **Ensemble compatibility** | Universal |
| **Exact package** | `DANet` (GitHub)  |
| **Contoh hypothesis** | "DANets akan mencapai akurasi ≥2% lebih tinggi dari MLP pada data dengan correlated features" |
| **Acceptance criteria** | Akurasi improvement ≥2% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Chen et al., AAAI 2022  |

---

### 4.8 Trompt

| Field | Detail |
|-------|--------|
| **Definisi** | Prompt-inspired tabular DL yang memisahkan **table-level intrinsic knowledge** + **sample-specific prompts**  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Tabular DL; prompt-based approach |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Intrinsic + sample-specific information dapat dipisahkan |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `n_prompts`, `d_model` |
| **Kisaran parameter awal** | n_layers: 2–6 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Prompt collapse |
| **Teknik pembanding** | FT-Transformer, TabTransformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `Trompt` (GitHub)  |
| **Contoh hypothesis** | "Trompt akan competitive dengan GBDT pada data tabular" |
| **Acceptance criteria** | Akurasi ≥ GBDT - 2% |
| **Rejection criteria** | Akurasi < GBDT - 5% |
| **Sumber** | Chen et al., ICML 2023  |

---

### 4.9 GRANDE (Gradient-Based Decision Tree Ensembles)

| Field | Detail |
|-------|--------|
| **Definisi** | End-to-end gradient descent untuk **hard, axis-aligned decision tree ensembles** menggunakan dense representation dan straight-through operator  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Tabular DL; hard tree ensemble |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Hard splits dapat dioptimasi dengan gradient |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_trees`, `depth`, `lr`, `temperature` |
| **Kisaran parameter awal** | n_trees: 32–256; depth: 4–8 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Gradient tidak stabil |
| **Teknik pembanding** | GBDT, NODE |
| **Ensemble compatibility** | Universal |
| **Exact package** | `GRANDE` (PyPI)  |
| **Contoh hypothesis** | "GRANDE akan mencapai akurasi setara GBDT dengan interpretability tree" |
| **Acceptance criteria** | Akurasi ≥ GBDT - 2% |
| **Rejection criteria** | Akurasi < GBDT - 5% |
| **Sumber** | Marton et al., ICLR 2024  |

---

## 5. IMBALANCE-SPECIALIZED ENSEMBLES

### 5.1 Balanced Random Forest

| Field | Detail |
|-------|--------|
| **Definisi** | Random Forest dengan **balanced bootstrap sampling**: setiap tree mengambil bootstrap sample dari minority class dan sample dengan replacement sejumlah sama dari majority class  |
| **Problem yang didukung** | Imbalanced classification |
| **Kapan digunakan** | Severe imbalance; tree model competitive |
| **Kapan tidak digunakan** | Data seimbang |
| **Asumsi** | Balancing sampling meningkatkan minority recall |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | F1-macro, AUC, Recall |
| **Hyperparameter penting** | `n_estimators`, `sampling_strategy`, `max_depth` |
| **Kisaran parameter awal** | n_estimators: 100–500; sampling_strategy='auto' |
| **CPU/GPU/RAM** | CPU parallel; RAM high |
| **Risiko leakage** | Sedang; balancing harus fold-local |
| **Failure modes** | Majority class underfitting |
| **Teknik pembanding** | EasyEnsemble, RUSBoost |
| **Ensemble compatibility** | Universal |
| **Exact package** | `imblearn.ensemble.BalancedRandomForestClassifier`  |
| **Contoh hypothesis** | "Balanced RF akan mencapai F1-macro ≥10% lebih tinggi dari RF pada data 1:100 imbalanced" |
| **Acceptance criteria** | F1-macro improvement ≥10% |
| **Rejection criteria** | F1-macro drop |
| **Sumber** | Chen et al. (2004); imbalanced-learn docs  |

---

### 5.2 EasyEnsemble

| Field | Detail |
|-------|--------|
| **Definisi** | Ensemble of AdaBoost learners yang dilatih pada **different balanced bootstrap samples**  |
| **Problem yang didukung** | Imbalanced classification |
| **Kapan digunakan** | Severe imbalance; majority class tidak ingin dibuang |
| **Kapan tidak digunakan** | Data seimbang |
| **Asumsi** | Multiple balanced subsets menangkap majority diversity |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | F1-macro, AUC |
| **Hyperparameter penting** | `n_estimators`, `sampling_strategy` |
| **Kisaran parameter awal** | n_estimators: 10–50 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Subset terlalu kecil |
| **Teknik pembanding** | Balanced RF, RUSBoost |
| **Ensemble compatibility** | Universal |
| **Exact package** | `imblearn.ensemble.EasyEnsembleClassifier`  |
| **Contoh hypothesis** | "EasyEnsemble akan mencapai recall ≥15% lebih tinggi dari RF pada data 1:100" |
| **Acceptance criteria** | Recall improvement ≥15% |
| **Rejection criteria** | Recall drop |
| **Sumber** | Liu et al. (2009); imbalanced-learn docs  |

---

### 5.3 RUSBoost

| Field | Detail |
|-------|--------|
| **Definisi** | Random under-sampling integrated in the learning of **AdaBoost**  |
| **Problem yang didukung** | Imbalanced classification |
| **Kapan digunakan** | Imbalance + boosting |
| **Kapan tidak digunakan** | Data seimbang |
| **Asumsi** | Undersampling + boosting efektif |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | F1-macro, AUC |
| **Hyperparameter penting** | `n_estimators`, `learning_rate`, `sampling_strategy` |
| **Kisaran parameter awal** | n_estimators: 50–200 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Underfitting |
| **Teknik pembanding** | EasyEnsemble, Balanced RF |
| **Ensemble compatibility** | Universal |
| **Exact package** | `imblearn.ensemble.RUSBoostClassifier`  |
| **Contoh hypothesis** | "RUSBoost akan mencapai AUC ≥5% lebih tinggi dari AdaBoost pada data imbalanced" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Seiffert et al. (2010); imbalanced-learn docs  |

---

### 5.4 Balanced Bagging

| Field | Detail |
|-------|--------|
| **Definisi** | Bagging dengan additional balancing (random under-sampling)  |
| **Problem yang didukung** | Imbalanced classification |
| **Kapan digunakan** | Imbalance + bagging |
| **Kapan tidak digunakan** | Data seimbang |
| **Asumsi** | Balanced subsets meningkatkan minority performance |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | F1-macro, AUC |
| **Hyperparameter penting** | `n_estimators`, `sampling_strategy` |
| **Kisaran parameter awal** | n_estimators: 10–50 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Underfitting |
| **Teknik pembanding** | EasyEnsemble, Balanced RF |
| **Ensemble compatibility** | Universal |
| **Exact package** | `imblearn.ensemble.BalancedBaggingClassifier`  |
| **Sumber** | imbalanced-learn docs |

---

## 6. RULE-BASED / INTERPRETABLE ENSEMBLES

### 6.1 RuleFit

| Field | Detail |
|-------|--------|
| **Definisi** | Kombinasi tree ensemble (untuk extract rules) + sparse linear model (Lasso) untuk prediction; f(x) = β₀ + Σβⱼxⱼ + Σαₖrₖ(x)  |
| **Problem yang didukung** | Interpretable nonlinear classification/regression |
| **Kapan digunakan** | Butuh interpretable nonlinear model; global/local interpretation |
| **Kapan tidak digunakan** | Black-box acceptable; dataset sangat besar |
| **Asumsi** | Rules dari tree ensemble informatif; sparse linear combination |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `tree_size`, `max_rules`, `memory_par`, `lam` |
| **Kisaran parameter awal** | tree_size: 2–10; max_rules: 500–2000 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Rule explosion; overfitting |
| **Teknik pembanding** | EBM, GAM, GOSDT |
| **Ensemble compatibility** | Universal |
| **Exact package** | `rulefit` (PyPI)  |
| **Contoh hypothesis** | "RuleFit akan mencapai akurasi setara GBDT dengan interpretable rules" |
| **Acceptance criteria** | Akurasi ≥ GBDT - 3% |
| **Rejection criteria** | Akurasi < GBDT - 5% |
| **Sumber** | Friedman & Popescu, Annals of Applied Statistics 2(3), 916–954 (2008)  |

---

### 6.2 Rotation Forest

| Field | Detail |
|-------|--------|
| **Definisi** | Ensemble yang menerapkan PCA pada random feature subsets untuk rotate feature space, lalu train tree pada rotated space  |
| **Problem yang didukung** | Classification |
| **Kapan digunakan** | Medium-dimensional numeric; trees; diversity needed |
| **Kapan tidak digunakan** | Interpretability penting |
| **Asumsi** | PCA rotation meningkatkan diversity |
| **Data preprocessing** | Scaling untuk PCA |
| **Dataset scale** | Menengah |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `K` (feature subsets), `L` (trees) |
| **Kisaran parameter awal** | K: 3–10; L: 10–50 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang; PCA per fold |
| **Failure modes** | K terlalu besar |
| **Teknik pembanding** | RF, Extra Trees |
| **Ensemble compatibility** | Universal |
| **Exact package** | `aeon.classification.RotationForest`  |
| **Contoh hypothesis** | "Rotation Forest akan mencapai akurasi ≥3% lebih tinggi dari RF pada data correlated features" |
| **Acceptance criteria** | Akurasi improvement ≥3% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Rodriguez et al., IEEE TPAMI 28(10), 1619–1630 (2006)  |

---

## 7. BAYESIAN / DISTRIBUTIONAL TREES

### 7.1 BART (Bayesian Additive Regression Trees)

| Field | Detail |
|-------|--------|
| **Definisi** | Bayesian sum-of-trees dengan prior dan posterior distribution; memberikan full posterior inference, point estimates, dan interval estimates  |
| **Problem yang didukung** | Regression dengan uncertainty; interpretable nonlinear |
| **Kapan digunakan** | Uncertainty-sensitive regression; small-medium data |
| **Kapan tidak digunakan** | Dataset sangat besar (MCMC mahal) |
| **Asumsi** | Additive trees; Bayesian prior |
| **Data preprocessing** | Standard |
| **Dataset scale** | Kecil–Menengah |
| **Metric compatibility** | MSE, NLL, CRPS |
| **Hyperparameter penting** | `n_trees`, `n_samples`, `n_burn`, `alpha`, `beta` |
| **Kisaran parameter awal** | n_trees: 50–200; n_samples: 1000–5000 |
| **CPU/GPU/RAM** | CPU-bound; MCMC cost |
| **Risiko leakage** | Rendah |
| **Failure modes** | MCMC tidak konvergen |
| **Teknik pembanding** | NGBoost, GPR, RF |
| **Ensemble compatibility** | Universal; probabilistic member |
| **Exact package** | `pymc_bart.BART`  |
| **Contoh hypothesis** | "BART akan mencapai NLL ≥10% lebih rendah dari RF pada data nonlinear" |
| **Acceptance criteria** | NLL improvement ≥10%; calibration baik |
| **Rejection criteria** | NLL drop; MCMC tidak konvergen |
| **Sumber** | Chipman et al. (2010); PyMC-BART docs  |

---

### 7.2 Quantile Regression Forest

| Field | Detail |
|-------|--------|
| **Definisi** | Random Forest yang mengestimasi **conditional quantiles** (bukan hanya mean); generalisasi RF untuk full conditional distribution  |
| **Problem yang didukung** | Regression dengan prediction intervals; heteroscedastic |
| **Kapan digunakan** | Butuh prediction intervals; metric pinball |
| **Kapan tidak digunakan** | Hanya point prediction |
| **Asumsi** | Random forest dapat mengestimasi conditional distribution |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Pinball loss, coverage |
| **Hyperparameter penting** | `n_estimators`, `quantiles` |
| **Kisaran parameter awal** | n_estimators: 100–500; quantiles: [0.1, 0.5, 0.9] |
| **CPU/GPU/RAM** | CPU parallel |
| **Risiko leakage** | Rendah |
| **Failure modes** | Quantile crossing |
| **Teknik pembanding** | NGBoost, GPR |
| **Ensemble compatibility** | Universal |
| **Exact package** | `quantile_forest.RandomForestQuantileRegressor`  |
| **Contoh hypothesis** | "QRF akan mencapai pinball loss ≥10% lebih rendah dari linear quantile regression" |
| **Acceptance criteria** | Pinball loss improvement ≥10% |
| **Rejection criteria** | Pinball loss drop |
| **Sumber** | Meinshausen, JMLR 7, 983–999 (2006)  |

---

## 8. ONLINE / STREAMING METHODS

### 8.1 SGDClassifier / SGDRegressor

| Field | Detail |
|-------|--------|
| **Definisi** | Linear learners dengan SGD yang mendukung `partial_fit` untuk minibatch/online/out-of-core learning  |
| **Problem yang didukung** | Large-scale sparse; streaming; limited RAM |
| **Kapan digunakan** | 10M+ rows; high-dimensional sparse; text/one-hot; limited RAM |
| **Kapan tidak digunakan** | Dataset kecil (gunakan LogisticRegression) |
| **Asumsi** | Linear relationship; streaming data |
| **Data preprocessing** | Scaling; sparse matrix |
| **Dataset scale** | Besar–Sangat Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `loss`, `penalty`, `alpha`, `learning_rate` |
| **Kisaran parameter awal** | loss='log_loss'; alpha: 1e-4–1e-2 |
| **CPU/GPU/RAM** | Ringan; memory-efficient |
| **Risiko leakage** | Sedang; streaming order |
| **Failure modes** | Learning rate salah |
| **Teknik pembanding** | LogisticRegression, Passive-Aggressive |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.SGDClassifier/SGDRegressor`  |
| **Contoh hypothesis** | "SGDClassifier akan mencapai akurasi setara LogisticRegression dengan memory 10× lebih kecil" |
| **Acceptance criteria** | Akurasi setara; memory ≤20% |
| **Rejection criteria** | Akurasi drop >3% |
| **Sumber** | scikit-learn docs  |

---

### 8.2 Passive-Aggressive (Deprecated → SGDClassifier)

| Field | Detail |
|-------|--------|
| **Definisi** | Online margin learner dengan worst-case cumulative-loss bounds; **deprecated** sejak sklearn 1.8, gunakan SGDClassifier dengan `learning_rate='pa1'` atau `'pa2'`  |
| **Problem yang didukung** | Online classification/regression |
| **Kapan digunakan** | Streaming; margin-based |
| **Kapan tidak digunakan** | Batch learning |
| **Asumsi** | Online updates |
| **Data preprocessing** | Scaling |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `C`, `learning_rate='pa1'/'pa2'` |
| **Kisaran parameter awal** | C: 0.01–10 |
| **CPU/GPU/RAM** | Ringan |
| **Risiko leakage** | Sedang |
| **Failure modes** | Deprecated API |
| **Teknik pembanding** | SGDClassifier, Perceptron |
| **Ensemble compatibility** | Universal |
| **Exact package** | `sklearn.linear_model.SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1')`  |
| **Contoh hypothesis** | "SGDClassifier dengan learning_rate='pa1' akan mencapai akurasi setara PassiveAggressiveClassifier" |
| **Acceptance criteria** | Akurasi setara |
| **Rejection criteria** | Menggunakan estimator deprecated |
| **Sumber** | Crammer et al., JMLR 7, 551–585 (2006); scikit-learn docs  |

---

### 8.3 Hoeffding Tree

| Field | Detail |
|-------|--------|
| **Definisi** | Incremental decision tree untuk streaming classification menggunakan Hoeffding bound untuk split decisions  |
| **Problem yang didukung** | Streaming classification |
| **Kapan digunakan** | Row arrives → predict → label arrives → update tree |
| **Kapan tidak digunakan** | Batch learning |
| **Asumsi** | Streaming data; statistical guarantee |
| **Data preprocessing** | Standard |
| **Dataset scale** | Streaming (unbounded) |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `grace_period`, `delta` (Hoeffding bound), `tie_threshold` |
| **Kisaran parameter awal** | grace_period: 50–200; delta: 1e-5–1e-3 |
| **CPU/GPU/RAM** | Ringan; incremental |
| **Risiko leakage** | Sedang; streaming order |
| **Failure modes** | Concept drift |
| **Teknik pembanding** | Adaptive Random Forest |
| **Ensemble compatibility** | Universal |
| **Exact package** | `river.tree.HoeffdingTreeClassifier`  |
| **Contoh hypothesis** | "Hoeffding Tree akan mencapai akurasi ≥0.80 pada streaming data" |
| **Acceptance criteria** | Akurasi ≥0.75 |
| **Rejection criteria** | Akurasi <0.60 |
| **Sumber** | Domingos & Hulten (2000); River docs  |

---

### 8.4 Adaptive Random Forest (ARF)

| Field | Detail |
|-------|--------|
| **Definisi** | Online forest dengan **drift detection per tree** dan selective resets  |
| **Problem yang didukung** | Streaming classification/regression dengan concept drift |
| **Kapan digunakan** | Fraud stream; ad click; sensor; concept drift |
| **Kapan tidak digunakan** | Static data |
| **Asumsi** | Drift dapat dideteksi dan di-adapt |
| **Data preprocessing** | Standard |
| **Dataset scale** | Streaming |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_models`, `drift_detector` (ADWIN), `warning_detector` |
| **Kisaran parameter awal** | n_models: 10–50 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Drift tidak terdeteksi |
| **Teknik pembanding** | Hoeffding Tree, Mondrian Forest |
| **Ensemble compatibility** | Universal |
| **Exact package** | `river.forest.ARFClassifier/ARFRegressor`  |
| **Contoh hypothesis** | "ARF akan mencapai akurasi ≥10% lebih tinggi dari static RF pada data dengan concept drift" |
| **Acceptance criteria** | Akurasi improvement ≥10% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Gomes et al. (2017); River docs  |

---

## 9. SPARSE INTERACTION / CTR MODELS

### 9.1 Factorization Machines (FM)

| Field | Detail |
|-------|--------|
| **Definisi** | Model yang memodelkan pairwise interactions via **factorized parameters**: Σ_{i<j} ⟨vᵢ,vⱼ⟩ xᵢxⱼ, memungkinkan estimasi interactions pada sparse data  |
| **Problem yang didukung** | CTR prediction; recommender systems; sparse categorical |
| **Kapan digunakan** | Huge categorical; sparse one-hot; user-item |
| **Kapan tidak digunakan** | Dense numerical |
| **Asumsi** | Interactions dapat difaktorkan |
| **Data preprocessing** | One-hot encoding |
| **Dataset scale** | Besar–Sangat Besar |
| **Metric compatibility** | AUC, log-loss |
| **Hyperparameter penting** | `n_factors`, `n_epochs`, `lr`, `reg` |
| **Kisaran parameter awal** | n_factors: 10–100 |
| **CPU/GPU/RAM** | Ringan–Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | n_factors salah |
| **Teknik pembanding** | Linear, Polynomial, FFM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `river.facto.FMClassifier`; `fastFM` (PyPI)  |
| **Contoh hypothesis** | "FM akan mencapai AUC ≥5% lebih tinggi dari linear model pada sparse CTR data" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Rendle (2010)  |

---

### 9.2 Field-aware Factorization Machine (FFM)

| Field | Detail |
|-------|--------|
| **Definisi** | FM yang membuat latent vector bergantung pada **field lawan**  |
| **Problem yang didukung** | CTR prediction; large sparse |
| **Kapan digunakan** | Multiple fields; interactions field-dependent |
| **Kapan tidak digunakan** | Single field |
| **Asumsi** | Field-aware interactions |
| **Data preprocessing** | Field definition |
| **Dataset scale** | Besar |
| **Metric compatibility** | AUC, log-loss |
| **Hyperparameter penting** | `n_factors`, `n_epochs`, `lr` |
| **Kisaran parameter awal** | n_factors: 4–20 |
| **CPU/GPU/RAM** | Sedang |
| **Risiko leakage** | Sedang |
| **Failure modes** | Field definition salah |
| **Teknik pembanding** | FM, DeepFM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `river.facto.FFMClassifier`  |
| **Contoh hypothesis** | "FFM akan mencapai AUC ≥3% lebih tinggi dari FM pada data multi-field" |
| **Acceptance criteria** | AUC improvement ≥3% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Juan et al., RecSys 2016  |

---

### 9.3 DeepFM

| Field | Detail |
|-------|--------|
| **Definisi** | FM untuk low-order interactions + DNN untuk high-order interactions dalam satu architecture  |
| **Problem yang didukung** | CTR prediction |
| **Kapan digunakan** | Sparse categorical; CTR; recommender |
| **Kapan tidak digunakan** | Dense numerical; general tabular |
| **Asumsi** | Low-order + high-order interactions |
| **Data preprocessing** | Embeddings |
| **Dataset scale** | Besar |
| **Metric compatibility** | AUC, log-loss |
| **Hyperparameter penting** | `embedding_dim`, `dnn_layers`, `lr` |
| **Kisaran parameter awal** | embedding_dim: 10–50 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang |
| **Failure modes** | Overfitting |
| **Teknik pembanding** | FM, DCN, AutoInt |
| **Ensemble compatibility** | Universal |
| **Exact package** | `deepctr-torch` (PyPI)  |
| **Contoh hypothesis** | "DeepFM akan mencapai AUC ≥5% lebih tinggi dari FM pada CTR data" |
| **Acceptance criteria** | AUC improvement ≥5% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Guo et al., IJCAI 2017  |

---

### 9.4 Deep & Cross Network (DCN)

| Field | Detail |
|-------|--------|
| **Definisi** | Cross network yang secara eksplisit menerapkan feature crossing di setiap layer + DNN  |
| **Problem yang didukung** | CTR prediction |
| **Kapan digunakan** | Ad click; feature interactions |
| **Kapan tidak digunakan** | Dense numerical |
| **Asumsi** | Explicit feature crossing |
| **Data preprocessing** | Embeddings |
| **Dataset scale** | Besar |
| **Metric compatibility** | AUC, log-loss |
| **Hyperparameter penting** | `n_cross_layers`, `dnn_layers`, `lr` |
| **Kisaran parameter awal** | n_cross_layers: 2–4 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang |
| **Failure modes** | Overfitting |
| **Teknik pembanding** | DeepFM, AutoInt |
| **Ensemble compatibility** | Universal |
| **Exact package** | `deepctr-torch` (PyPI)  |
| **Contoh hypothesis** | "DCN akan mencapai log-loss ≥3% lebih rendah dari DNN pada CTR data" |
| **Acceptance criteria** | Log-loss improvement ≥3% |
| **Rejection criteria** | Log-loss drop |
| **Sumber** | Wang et al., ADKDD 2017  |

---

### 9.5 AutoInt

| Field | Detail |
|-------|--------|
| **Definisi** | Multi-head self-attention untuk **automatic feature interaction learning**  |
| **Problem yang didukung** | CTR prediction |
| **Kapan digunakan** | High-order interactions; CTR |
| **Kapan tidak digunakan** | Dense numerical |
| **Asumsi** | Self-attention menangkap interactions |
| **Data preprocessing** | Embeddings |
| **Dataset scale** | Besar |
| **Metric compatibility** | AUC, log-loss |
| **Hyperparameter penting** | `n_layers`, `n_heads`, `embedding_dim` |
| **Kisaran parameter awal** | n_layers: 2–4 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Sedang |
| **Failure modes** | Overfitting |
| **Teknik pembanding** | DeepFM, DCN |
| **Ensemble compatibility** | Universal |
| **Exact package** | `deepctr-torch` (PyPI)  |
| **Contoh hypothesis** | "AutoInt akan mencapai AUC ≥3% lebih tinggi dari DeepFM pada CTR data" |
| **Acceptance criteria** | AUC improvement ≥3% |
| **Rejection criteria** | AUC drop |
| **Sumber** | Song et al., AAAI 2019  |

---

### 9.6 T2G-Former

| Field | Detail |
|-------|--------|
| **Definisi** | Transformer yang mengorganisasi tabular features sebagai **relation graph** sebelum heterogeneous interactions diproses  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Feature relation graph informatif |
| **Kapan tidak digunakan** | Fitur independen |
| **Asumsi** | Relation graph dapat diestimasi |
| **Data preprocessing** | Standard |
| **Dataset scale** | Menengah–Besar |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_layers`, `d_model`, `graph_estimator` |
| **Kisaran parameter awal** | n_layers: 2–6 |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Graph estimation salah |
| **Teknik pembanding** | FT-Transformer |
| **Ensemble compatibility** | Universal |
| **Exact package** | `t2g-former` (GitHub)  |
| **Contoh hypothesis** | "T2G-Former akan mencapai akurasi ≥3% lebih tinggi dari FT-Transformer" |
| **Acceptance criteria** | Akurasi improvement ≥3% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Yan et al., AAAI 2023  |

---

## 10. TABULAR FOUNDATION MODELS (UPDATE)

### 10.1 TabPFN-2.5

| Field | Detail |
|-------|--------|
| **Definisi** | Generasi terbaru TabPFN untuk datasets hingga 50.000 data points dan 2.000 features; prior-data fitted tabular foundation model  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Small-medium tabular; no tuning; minimal preprocessing |
| **Kapan tidak digunakan** | >50K rows; >2K features |
| **Asumsi** | In-context learning; prior transfer |
| **Data preprocessing** | Minimal |
| **Dataset scale** | Small–Medium (≤50K rows, ≤2K features) |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | `n_estimators` (ensemble count), device |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | GPU/high RAM |
| **Risiko leakage** | Rendah |
| **Failure modes** | Context limits |
| **Teknik pembanding** | AutoGluon, tuned GBM |
| **Ensemble compatibility** | Universal |
| **Exact package** | `tabpfn` (PyPI)  |
| **Contoh hypothesis** | "TabPFN-2.5 akan mencapai akurasi ≥2% lebih tinggi dari tuned XGBoost pada datasets ≤10K rows" |
| **Acceptance criteria** | Akurasi improvement ≥2% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Grinsztajn et al., arXiv:2511.08667  |

---

### 10.2 TabICLv2

| Field | Detail |
|-------|--------|
| **Definisi** | Tabular foundation model untuk regression dan classification; scalable ke million-scale datasets under 50GB GPU memory  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Large-scale tabular; no tuning |
| **Kapan tidak digunakan** | Dataset kecil (TabPFN lebih efisien) |
| **Asumsi** | In-context learning; scalable pretraining |
| **Data preprocessing** | Minimal |
| **Dataset scale** | Large (million-scale) |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Device, batch size |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | GPU; 50GB memory |
| **Risiko leakage** | Rendah |
| **Failure modes** | GPU memory |
| **Teknik pembanding** | TabPFN-2.5 |
| **Ensemble compatibility** | Universal |
| **Exact package** | `tabicl` (PyPI)  |
| **Contoh hypothesis** | "TabICLv2 akan mencapai akurasi setara RealTabPFN-2.5 dengan speedup ≥2×" |
| **Acceptance criteria** | Akurasi setara; speedup ≥2× |
| **Rejection criteria** | Akurasi drop >2% |
| **Sumber** | arXiv:2602.11139; ICML 2026  |

---

### 10.3 TabDPT (Tabular Data Prior Transformer)

| Field | Detail |
|-------|--------|
| **Definisi** | Tabular foundation model yang dilatih pada **real data** (bukan hanya synthetic prior); in-context learning dengan retrieval  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Butuh foundation model dengan real-data pretraining |
| **Kapan tidak digunakan** | Dataset sangat kecil |
| **Asumsi** | Real-data pretraining meningkatkan generalisasi |
| **Data preprocessing** | Minimal |
| **Dataset scale** | Small–Large |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Device, context size |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Context limits |
| **Teknik pembanding** | TabPFN, TabICL |
| **Ensemble compatibility** | Universal |
| **Exact package** | `TabDPT-inference` (GitHub)  |
| **Contoh hypothesis** | "TabDPT akan mencapai akurasi ≥2% lebih tinggi dari TabPFN pada regression benchmarks" |
| **Acceptance criteria** | Akurasi improvement ≥2% |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | Ma et al., NeurIPS 2025  |

---

### 10.4 TabH2O

| Field | Detail |
|-------|--------|
| **Definisi** | Unified foundation model untuk classification dan regression dalam satu forward pass via in-context learning; builds on TabICL  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Unified prediction; enterprise tabular |
| **Kapan tidak digunakan** | Dataset sangat besar (>500K) |
| **Asumsi** | Unified architecture efektif |
| **Data preprocessing** | Minimal |
| **Dataset scale** | Small–Medium |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Device |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | GPU/high RAM |
| **Risiko leakage** | Rendah |
| **Failure modes** | Implementation unverified |
| **Teknik pembanding** | TabPFN, TabICL |
| **Ensemble compatibility** | Universal |
| **Exact package** | H2O.ai API  |
| **Contoh hypothesis** | "TabH2O akan mencapai akurasi setara model terpisah untuk classification dan regression" |
| **Acceptance criteria** | Akurasi setara |
| **Rejection criteria** | Akurasi drop |
| **Sumber** | H2O.ai (2026); arXiv:2605.18383 |

---

### 10.5 Xiaomi-TabLDM

| Field | Detail |
|-------|--------|
| **Definisi** | Tabular foundation model yang dilatih pada **synthetic SCM data**; in-context learning untuk classification dan regression tanpa task-specific fine-tuning  |
| **Problem yang didukung** | Tabular classification/regression |
| **Kapan digunakan** | Research candidate; strong regression focus |
| **Kapan tidak digunakan** | Production (sangat baru) |
| **Asumsi** | SCM pretraining efektif |
| **Data preprocessing** | Minimal |
| **Dataset scale** | Small–Medium |
| **Metric compatibility** | Semua metric |
| **Hyperparameter penting** | Device |
| **Kisaran parameter awal** | N/A |
| **CPU/GPU/RAM** | GPU recommended |
| **Risiko leakage** | Rendah |
| **Failure modes** | Reproducibility unverified |
| **Teknik pembanding** | TabPFN, TabICL |
| **Ensemble compatibility** | Universal |
| **Exact package** | `xiaomi-tabldm` (GitHub)  |
| **Contoh hypothesis** | "Xiaomi-TabLDM akan mencapai regression performance competitive dengan TabPFN" |
| **Acceptance criteria** | Performance setara |
| **Rejection criteria** | Performance drop |
| **Sumber** | arXiv:2609.03880 (2026)  |

---

## Ringkasan Prioritas Tambahan

| Prioritas | Teknik | Keluarga | Status |
|-----------|--------|----------|--------|
| 1 | NGBoost | Probabilistic Regression | V |
| 2 | GPR / GPC | Kernel Methods | V |
| 3 | Kernel Ridge | Kernel Methods | V |
| 4 | Tweedie / Gamma / NB GLM | GLM Families | V |
| 5 | TabM | Deep Tabular | V + impl |
| 6 | TabR | Deep Tabular | V |
| 7 | ModernNCA | Deep Tabular | V |
| 8 | NODE | Deep Tabular | V |
| 9 | TabTransformer | Deep Tabular | V |
| 10 | SAINT | Deep Tabular | V |
| 11 | DANets | Deep Tabular | V |
| 12 | Trompt | Deep Tabular | V |
| 13 | GRANDE | Deep Tabular | V paper |
| 14 | Balanced RF / EasyEnsemble / RUSBoost | Imbalance | V |
| 15 | RuleFit | Interpretable | V |
| 16 | Rotation Forest | Interpretable | V |
| 17 | BART | Bayesian Trees | V |
| 18 | Quantile RF | Distributional Trees | V |
| 19 | SGD / Passive-Aggressive | Online | V |
| 20 | Hoeffding Tree / ARF | Online | V |
| 21 | FM / FFM / DeepFM / DCN / AutoInt | Sparse Interaction | V |
| 22 | T2G-Former | Deep Tabular | V |
| 23 | TabPFN-2.5 | TFM | V |
| 24 | TabICLv2 | TFM | V paper |
| 25 | TabDPT | TFM | V |
| 26 | TabH2O | TFM | P |
| 27 | Xiaomi-TabLDM | TFM | V paper / P impl |

---

*Dokumen ini disusun berdasarkan sumber akademik dan dokumentasi resmi per 2026. Setiap klaim dapat diverifikasi melalui referensi yang tercantum.*
