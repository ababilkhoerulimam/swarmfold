# SwarmFold Capability Matrix

This matrix defines the operational status of all modeling, feature, validation, task, and metric components in SwarmFold across 6 lifecycle dimensions:
- **Declared**: Defined in contract schemas and architecture specifications.
- **Installed**: Underlying libraries and dependencies are present in the environment.
- **Executable**: Runtime executor or adapter is wired, tested, and active.
- **Scheduled**: Candidate factory and planners can assign work to this component.
- **Tested**: Verified with automated hermetic unit and integration test fixtures.
- **UI Exposed**: Accessible, configurable, or visualized in the web dashboard.

## Model Families

| Component | Status | Declared | Installed | Executable | Scheduled | Tested | UI Exposed | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `logistic_regression` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Regularized linear classifier/regressor |
| `random_forest` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Ensemble of bagging decision trees |
| `hist_gradient_boosting` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Histogram-based gradient boosted trees |
| `extra_trees` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Extremely randomized ensemble trees |
| `sgd_log_loss` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Online linear model with SGD optimization |
| `bagging` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Intra-lane bootstrap aggregating ensemble |
| `soft_voting` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Probability-weighted voting ensemble |
| `stacking` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Meta-learner stacked generalization |
| `lightgbm` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fast, distributed gradient boosting framework |
| `xgboost` | **DECLARED (uninstalled dependency)** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | Scalable, regularized gradient boosted trees |
| `catboost` | **DECLARED (uninstalled dependency)** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | Gradient boosting with categorical feature support |

## Feature Engineering Strategies

| Component | Status | Declared | Installed | Executable | Scheduled | Tested | UI Exposed | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `raw` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Pass raw features without transformation |
| `scaled_numeric` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Standard scaling for numeric columns |
| `robust_scaled` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Robust quantile-based feature scaling |
| `categorical_target` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Target-aware categorical mean encoding |
| `date_expanded` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Datetime cyclical and calendar expansion |
| `interaction_pairs` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Pairwise numeric interaction features |
| `power_transformed` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Yeo-Johnson power transformation |
| `pca_projection` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Principal component dimensionality projection |
| `text_tfidf` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fold-safe n-gram sublinear TF-IDF encoder |
| `polynomial_features` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Polynomial expansion with degrees 2-3 |

## Validation Splitters

| Component | Status | Declared | Installed | Executable | Scheduled | Tested | UI Exposed | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `stratified_kfold` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stratified K-fold cross-validation |
| `kfold` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Standard K-fold cross-validation |
| `group_kfold` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Entity/group-aware split partitioning |
| `stratified_group_kfold` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Stratified group K-fold partitioning |
| `temporal_split` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Time-ordered forward split partitioning |
| `rolling_origin` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Expanding rolling window time series split |

## Task Adapters

| Component | Status | Declared | Installed | Executable | Scheduled | Tested | UI Exposed | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `binary_classification` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Two-class probability prediction |
| `multiclass_classification` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Multi-category probability distribution |
| `regression` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Continuous numeric target prediction |
| `multilabel_classification` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Multi-label binary indicator matrix |
| `time_series_forecasting` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Sequential multi-horizon forecasting |

## Evaluation Metrics

| Component | Status | Declared | Installed | Executable | Scheduled | Tested | UI Exposed | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `accuracy` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fraction of correct classification predictions |
| `roc_auc` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Area under receiver operating characteristic curve |
| `log_loss` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Cross-entropy loss over predicted probabilities |
| `macro_f1` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Unweighted mean F1 score across classes |
| `multilabel_f1` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Macro-averaged F1 score over multiple labels |
| `rmse` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Root mean squared error for regression |
| `mae` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Mean absolute error for regression |
| `r2` | **OPERATIONAL** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Coefficient of determination for regression |
