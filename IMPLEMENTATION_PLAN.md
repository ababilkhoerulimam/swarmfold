# SwarmFold Implementation Plan

Status: reviewed in progress; Phases 0-2 operational, Phases 3-8 partial  
Last reviewed and runtime-verified: 2026-09-16  
Scope: correctness fixes, integration of existing AutoML modules, and the next maturity increment

Detailed closure plan for adaptive HPO, production ensembles, the eight remaining blockers, and
the portfolio-level Agent F stage:
[`docs/HPO_ENSEMBLE_AGENT_F_IMPLEMENTATION_PLAN.md`](docs/HPO_ENSEMBLE_AGENT_F_IMPLEMENTATION_PLAN.md)

## Implementation progress

- Phases 0-2 are operational in the default runtime and covered by integration tests.
- Phase 3 is partially operational: deterministic recommendations now flow into frozen runtime
  contracts, but the confirmation UI still needs to display the locally computed recommendation
  rather than relying only on the profiler response.
- Phase 4 is partially operational: fold-safe transforms run, but ablation scheduling and transform
  fingerprint caching are not wired into the default graph.
- Phase 5 is partially operational: the unified ensemble engine exists, while the default portfolio
  path still exercises only a subset of its methods and reconstruction must be proven against a real
  multi-lane package.
- Phase 6 is partially operational: bounded training repairs and checkpoints run, while provider
  retries remain correctly owned by a separate provider-operation layer that is not implemented yet.
- Phase 7 is partially operational: Parquet and text pipelines work; multilabel and forecasting are
  declared and component-tested but intentionally blocked at dashboard confirmation.
- Phase 8 is partially operational: package portability and capability truthfulness are verified,
  while documentation convergence and the stale Graphify index remain open.
- Metric semantics are centralized and used for candidate, lane, and cross-lane selection.
- Worse ensembles are rejected without replacing the validated champion submission.
- Binary classification, multiclass classification, and regression execute through the real graph.
- Ordinary tests are isolated from local planner credentials and network access.
- Release gates pass: `uv run pytest -q`, `uv run ruff check .`, and `git diff --check`.
- Candidate generation now uses an executable, installed, task-compatible, lane-scoped factory.
- Available boosting adapters are scheduled through the real graph, including LightGBM.
- Adaptive scheduling reorders remaining candidates after each observed result.
- K-fold trials can terminate as `PRUNED` with fold scores, benchmark, and reason preserved.
- Iteration reflection is persisted, packaged, and can reprioritize the next iteration.
- Knowledge runtime is deterministic, byte-stable, and repository-local (`knowledge/agents/*.md`).
- 468 unique canonical cards compiled with zero title collisions; the corpus audit records remaining
  field-completeness gaps instead of claiming 100% completeness.
- Semantic deduplication merged duplicate clusters with aliases and secondary roles preserved.
- Formal corpus audit published (`knowledge/CORPUS_AUDIT.md` and `corpus_audit.json`).
- Typed `TechniquePlan` contract enforces execution capability gating before trial allocation.
- Role-scoped hybrid retrieval with reranker protocol and provenance tracking integrated.
- AutoValidationSplitter extracts dataset traits (timestamps, entity groups, class imbalance, rare classes, duplicate rows).
- 7 concrete validation splitters (StratifiedKFold, RepeatedStratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold, ExpandingTimeSeriesSplitter, PurgedTimeSeriesSplitter with embargo).
- Validation recommendation engine with confidence, evidence, risks, and contract freeze.
- Original row IDs preserved end-to-end across CSV and Parquet OOF predictions.
- Leakage audits enforce zero group overlap and temporal causality in real graph runs.
- FeaturePipelineExecutor with typed FeatureRegistry cataloging fold-safe transformers.
- TargetEncoderFoldSafe isolates training targets using inner CV with smoothed m-estimates (zero validation target leakage).
- FrequencyEncoder, DateDecomposer, CyclicTimeEncoder (harmonic projections), QuantileBinner, InteractionPolynomial.
- Robust categorical casting, zero-feature fallback, pipeline serialization to `feature_pipeline.joblib`, and test inference replay parity.
- FeatureAblationReport computes feature count, memory, training time, and metric score deltas.
- Shared `PredictionRegistry` with row alignment verification, SHA-256 prediction hashing, Pearson correlation matrix, and collinearity diversity filtering.
- `UnifiedEnsembleEngine` evaluating equal-weight average, champion/challenger pair blends, rank averaging, SLSQP constrained weighted blending, Caruana forward selection, stacking meta-learner, and majority voting.
- Strict metric-aware adoption gate: worse ensembles are rejected without replacing champion submission for both higher-is-better and lower-is-better metrics.
- Deterministic submission reconstruction (`reconstruct_ensemble_submission`) verifying exact parity with packaged submission artifacts from `ensemble_spec.json`.
- Structured failure taxonomy (`FailureKind`, `FailureRecord`, `classify_failure`) classifying OOM, invalid parameters, numerical divergence, shape/dtype mismatch, timeouts, missing dependencies, and verification errors.
- Bounded `SelfHealingDebugger` with deterministic repair actions (OOM tree depth/estimator/thread reduction, parameter sanitization, representation fallback), bounded retries (`max_retries = 2`), and lineage preservation (`original_experiment_id`, `repair_attempt`).
- `CheckpointManager` for fold-level persistence and resumption, ensuring restarted runs skip completed folds.
- `ResourcePolicy` with visible budgets (`max_training_seconds_per_fold`, `max_total_seconds`, `max_memory_mb`) and `CancellationToken` cooperative cancellation checks.
- Integration into `RunResult` and candidate execution node with sub-millisecond `perf_counter` precision.
- Next implementation slice: close the remaining Phase 3-8 runtime gaps before adding more task
  categories.


## 1. Objective

Move SwarmFold from an evidence-first multi-lane prototype into an integrated personal AutoML
system where every advertised capability changes the real execution path, produces verifiable
artifacts, and can be reconstructed from the final package.

The implementation must preserve these principles:

- The LLM proposes and explains; deterministic code validates and executes.
- Every preprocessing operation is fitted inside the permitted fold boundary.
- An ensemble is expected and actively searched, but it is not selected when it is worse than the
  best validated member.
- Historical knowledge changes experiment priority, never the frozen validation contract.
- Declared, installed, executable, scheduled, and verified are separate capability states.
- A UI option must not be exposed unless the corresponding runtime path is tested.

## 2. Current Truth

The following capabilities are **operational** in the default execution path:

- Five configurable AI lanes (AI-A through AI-E) with parallel portfolio execution.
- `CandidateFactory` drives all lane candidate generation, backed by `model_capability_registry`.
  XGBoost, CatBoost, and LightGBM candidates are generated by AI-B when their adapters are installed
  (`uv sync --extra boosting` or `uv sync --extra lightgbm`).
- `HPOController` (adaptive sampling + fold-level median pruning) is active for lane AI-B on every
  run. Parameters are suggested from a persistent SQLite study and updated after each fold.
- `AdaptiveScheduler` reorders the remaining queue after every completed experiment, prioritising
  models with positive marginal contribution.
- `IterationReflector` and `ReflectionMemoryStore` run after the portfolio and write per-iteration
  reflection reports that are forwarded as prior knowledge to the next iteration.
- `TechniqueRetriever` performs role-scoped RAG retrieval from the local knowledge corpus (468
  canonical technique cards) before each `scout_models` call, and `build_technique_plan` maps the
  selected cards to typed runtime references.
- All validation splitters (stratified K-fold, group K-fold, stratified group K-fold,
  expanding walk-forward, purged time series with embargo) are registered and wired into the
  experiment runner. Group isolation and temporal causality are verified before training.
- `SelfHealingDebugger` repairs OOM, invalid parameter, numerical instability, and data-shape
  failures with bounded deterministic actions. Provider/network failures now receive bounded
  exponential backoff retries (max 3, up to 30 s) before the experiment is marked FAILED.
- `UnifiedEnsembleEngine` runs Caruana, SLSQP blending, weighted averaging, and cross-fitted
  stacking for binary classification, multiclass classification, multilabel classification,
  regression, and time-series forecasting. Champion Gate rejects any ensemble that does not
  strictly improve over the best single model.
- `FeaturePipelineExecutor` applies fold-safe transformers from `FeatureRegistry` for numeric,
  categorical, and text strategies. The feature pipeline is serialised for test-time inference.
- Binary, multiclass, and regression end-to-end paths are tested and executable through the default
  dashboard workflow. Multilabel and time-series forecasting have typed adapters but are blocked
  at the dashboard confirmation gate pending complete end-to-end fixtures.
- `full-package.zip` includes contracts, data fingerprints, OOF predictions, experiment ledger,
  retrieval provenance, ensemble reconstruction metadata, and a consolidated knowledge report.

Remaining open work:

- Multilabel and time-series forecasting end-to-end execution paths and dashboard enablement.
- Graphify graph index predates recent runtime changes and needs regeneration.
- README, PROJECT_SPEC, and the UI capability panels must be reconciled to describe the same
  default architecture as the runtime.

## 3. Delivery Rules

Each phase must ship as an independently testable change. A phase is not complete when its classes
exist; it is complete only when the default dashboard workflow invokes them and their evidence is
included in `full-package.zip`.

For every new capability:

1. Add or update its typed contract.
2. Register its runtime implementation.
3. Make planning select it under a reproducible condition.
4. Execute it through the real graph.
5. Verify its metric and artifacts independently.
6. Display its decision and status in the UI.
7. Include its provenance in the experiment ledger and final knowledge report.

## 4. Phase 0: Correctness and Release Gate

### Goal

Remove cases where SwarmFold can report or package the wrong winner, use the wrong metric semantics,
or expose a task that cannot run.

### Work

- Introduce a central `MetricSpec` registry containing:
  - metric name and version;
  - higher-is-better direction;
  - prediction type: label, probability, score, or continuous;
  - valid task types;
  - range and clipping rules;
  - deterministic scorer and verifier.
- Replace every direct `max(score)` and `score_a > score_b` comparison with metric-aware helpers.
- Change champion selection so all successful single and ensemble candidates compete under the same
  metric direction.
- Change cross-lane ensemble adoption so:
  - the best single model remains the fallback;
  - an ensemble is adopted only after its OOF score passes the configured improvement and stability
    gate;
  - a rejected ensemble cannot overwrite the champion submission.
- Make `freeze_contracts` derive `task_type` and metric from the confirmed competition profile.
- Complete executable paths for the first supported task set:
  - binary classification;
  - multiclass classification;
  - regression.
- Add task-aware split behavior:
  - stratification only for classification;
  - K-fold for ordinary regression;
  - no binary positive-class assumptions for multiclass or regression.
- Add classifier and regressor implementations for every scheduled sklearn family.
- Add multiclass probability matrices and regression predictions to artifact schemas.
- Disable unsupported task and metric combinations at confirmation time with an actionable reason.
- Make tests hermetic:
  - force mock planner by default in tests;
  - mark live provider tests separately;
  - prevent `.env` from causing network calls during ordinary tests.
- Fix all Ruff failures and make lint a required quality gate.

### Primary files

- `src/swarmfold/contracts.py`
- `src/swarmfold/schemas.py`
- `src/swarmfold/nodes.py`
- `src/swarmfold/roles.py`
- `src/swarmfold/quality.py`
- `src/swarmfold/portfolio.py`
- `tests/`

### Acceptance criteria

- A deliberately worse ensemble is rejected and the best single submission is preserved.
- Binary, multiclass, and regression smoke tests run end to end through the real graph.
- Accuracy, ROC-AUC, log-loss, macro F1, RMSE, MAE, and R2 use correct direction and prediction
  shapes.
- `uv run pytest -q` passes without OmniRouter or internet access.
- `uv run ruff check .` passes.

## 5. Phase 1: Integrate Existing AutoML Components

### Goal

Turn the model adapters, scheduler, pruner, and reflection modules that already exist into the
default experiment loop.

### Work

- Replace hardcoded lane arrays with a typed `CandidateFactory` backed by the capability registry.
- Make candidate generation filter on all five capability states:
  - declared;
  - installed;
  - executable;
  - applicable to the current contract;
  - scheduled for this lane.
- Schedule LightGBM, XGBoost, and CatBoost when installed and compatible.
- Preserve distinct lane behavior:
  - AI A: conservative baselines and calibration;
  - AI B: boosting and bounded HPO;
  - AI C: feature-family ablations;
  - AI D: alternative geometry and diversity;
  - AI E: robustness, seed stability, and ensemble stress tests.
- Wire `AdaptiveScheduler` into the queue after each completed experiment.
- Wire `TrialPruner` into fold-level evaluation and record every prune decision.
- Ensure a failed or infeasible proposal is replaced rather than silently reducing the requested
  experiment count.
- Run `IterationReflector` after lane completion and after portfolio ensemble assessment.
- Store reflection output in:
  - lane findings;
  - experiment ledger;
  - `experiment-knowledge.md`;
  - the next iteration's prior-knowledge input.
- Keep historical memory advisory. It may reorder or reject duplicate hypotheses but may not reuse a
  champion blindly.

### Acceptance criteria

- A normal five-lane run contains experiments from all applicable installed model families.
- Queue order changes in a deterministic test after new evidence is recorded.
- A pruned trial appears in the ledger with fold evidence and a reason.
- Reflection from iteration N changes the plan of iteration N+1 without changing its validation
  contract.
- The requested experiment budget is either executed or explicitly accounted for by terminal
  failure, prune, duplicate, or infeasibility records.

## 6. Phase 2: Knowledge Runtime and Typed Technique Plans

### Goal

Move RAG from prompt decoration into a constrained planning input that maps knowledge to executable
components.

### Work

- Make corpus compilation deterministic and repository-local.
- Remove the hardcoded dependency on `C:/Users/Hi/Downloads/agents knowledge` from production code.
- Deduplicate cards using normalized title, aliases, category, implementation, and content hash.
- Merge complementary duplicates while preserving all source provenance.
- Validate field completeness and publish a corpus audit with:
  - total cards;
  - missing critical fields;
  - duplicate clusters;
  - executable coverage;
  - stale review dates.
- Add hybrid retrieval:
  - deterministic metadata filtering;
  - lexical ranking;
  - optional embeddings and reranking behind an interface.
- Create a typed `TechniquePlan` containing:
  - task and metric assumptions;
  - validation strategy;
  - feature steps;
  - model family and search space;
  - ensemble compatibility;
  - resource estimate;
  - leakage rules;
  - acceptance and rejection gates;
  - source card IDs.
- Reject any plan step without an exact runtime registry mapping.
- Inject role-scoped knowledge into the relevant agents rather than giving every agent the entire
  corpus.
- Record retrieval scores, selected cards, rejected cards, and mapping decisions in the package.

### Acceptance criteria

- Recompiling the same source documents produces byte-stable canonical cards.
- No same-category semantic duplicate remains without an explicit alias relationship.
- Every executable plan step resolves to a tested registry entry.
- Changing dataset traits changes retrieved techniques and the structured plan in a deterministic
  fixture.
- Removing a runtime adapter makes its technique ineligible for execution without hiding it from
  research results.

## 7. Phase 3: AutoValidationSplitter [PARTIAL]

### Goal

Select a defensible validation strategy from confirmed dataset structure instead of assuming random
holdout or ordinary K-fold.

### Work

- Add dataset traits for:
  - timestamp and ordering columns;
  - entity and group identifiers;
  - repeated observations;
  - class imbalance;
  - rare classes;
  - train-test temporal overlap;
  - duplicate and near-duplicate rows.
- Implement splitter registry entries for:
  - stratified K-fold;
  - repeated stratified K-fold;
  - K-fold;
  - GroupKFold;
  - StratifiedGroupKFold;
  - expanding walk-forward;
  - purged temporal split with embargo.
- Generate a validation recommendation with confidence, evidence, risks, and unresolved questions.
- Add the recommended splitter to the human confirmation gate.
- Freeze group, time, purge, embargo, fold count, seed, and row identity into the validation
  contract.
- Store original row IDs. Never reset validation row identity to positional IDs.
- Add leakage tests for target encoding, selection, calibration, resampling, and early stopping.

### Acceptance criteria

- Grouped fixtures never place one entity in both train and validation for a fold.
- Temporal fixtures never train on future observations.
- Purge and embargo boundaries are reproducible from the frozen contract.
- Every saved OOF row maps back to exactly one original training row.
- The metric verifier can recompute the score solely from the contract and prediction artifact.

## 8. Phase 4: FeaturePipelineExecutor [PARTIAL]

### Goal

Translate approved feature technique plans into fold-safe executable pipelines.

### Work

- Build a typed feature registry rather than executing arbitrary LLM-generated code.
- Start with a high-value, testable set instead of attempting all corpus techniques immediately:
  - missing indicators;
  - robust and standard scaling;
  - Yeo-Johnson power transform;
  - low-cardinality one-hot encoding;
  - ordinal encoding;
  - hashing for high-cardinality categories;
  - OOF target encoding;
  - frequency encoding;
  - quantile binning;
  - date decomposition;
  - cyclic time encoding;
  - polynomial and interaction terms;
  - univariate feature selection;
  - model-based feature selection;
  - optional dimensionality reduction.
- Define preconditions, supported dtypes, output sparsity, memory estimates, and leakage boundaries for
  each transformer.
- Fit every learned transformer inside each fold.
- Add feature-family ablations so improvements are attributable to controlled changes.
- Cache fold-safe deterministic transforms by data, split, and transform fingerprint.
- Record generated feature count, memory growth, training-time delta, and score delta.

### Acceptance criteria

- Every technique card selected for execution maps to one registry implementation and version.
- OOF target encoding has no access to its validation labels.
- Feature pipelines can be serialized and replayed for test inference.
- An ablation report identifies which feature family produced the observed change.
- Unsafe or infeasible expansions are rejected before allocation.

## 9. Phase 5: Unified OOF Ensemble Factory [PARTIAL]

### Goal

Use one metric-aware ensemble system for intra-lane and cross-lane candidates, with a single adoption
gate and reconstructable artifacts.

### Work

- Create a shared prediction registry keyed by experiment, split, fold, row, target, and prediction
  hash.
- Unify the currently separate quality assessment and package-time ensemble implementations.
- Support:
  - equal averaging;
  - rank averaging where valid;
  - constrained weighted blending;
  - Caruana greedy selection;
  - seed and fold averaging;
  - probability calibration;
  - regularized stacking with OOF meta-features;
  - majority voting only for compatible label metrics.
- Select members using score, fold stability, prediction correlation, error overlap, compute cost,
  and marginal contribution.
- Compare ensemble gain against a configurable minimum improvement and uncertainty threshold.
- Fit final members on the full training data after selection, then create test predictions using the
  exact frozen member and preprocessing specifications.
- Serialize ensemble weights or meta-model and include reconstruction metadata in the package.

### Acceptance criteria

- The ensemble meta-learner never sees in-sample base predictions.
- Lower-is-better metrics work correctly.
- A worse ensemble is rejected in both lane and portfolio selection.
- The final submission can be regenerated from packaged member artifacts and ensemble metadata.
- Ensemble assessment and packaged submission always refer to the same method and weights.

## 10. Phase 6: Self-Healing and Resource-Aware Execution [PARTIAL]

### Goal

Recover from predictable runtime failures without hiding them or entering unlimited retry loops.

### Work

- Introduce a failure taxonomy:
  - dependency unavailable;
  - invalid parameter combination;
  - out of memory;
  - timeout or stalled worker;
  - numerical instability;
  - incompatible data shape or dtype;
  - provider or network failure;
  - artifact verification failure.
- Add a bounded `SelfHealingDebugger` policy with deterministic repair actions:
  - reduce batch size, estimators, leaves, depth, or parallelism after OOM;
  - sanitize invalid parameter combinations;
  - switch sparse or dense representation when supported;
  - fall back to CPU only when the plan permits it;
  - retry provider operations separately from deterministic training.
- Never apply an unregistered code patch or arbitrary generated code during a run.
- Add experiment and fold checkpoints so completed work survives restart.
- Replace a hidden maximum training timeout with visible resource policies, cancellation controls,
  and optional user-defined soft limits.
- Record the original failure, repair, retry count, and final outcome.

### Acceptance criteria

- Every retry has a bounded count and a machine-readable reason.
- OOM fixtures reduce resource demand and either recover or terminate cleanly.
- A resumed run does not repeat completed folds.
- A repaired experiment remains distinguishable from the original proposal in lineage.
- No failure is converted into success without verifier approval.

## 11. Phase 7: Data and Task Expansion [PARTIAL]

### Goal

Expand beyond ordinary tabular CSV only after the common contracts and validation engine are stable.

### Implemented Capabilities

1. **Parquet and Tabular Ingestion**:
   - Transparent `.parquet` / `.pq` and `.csv` / `.tsv` data loading in `data.py` (`load_training_dataset`, `inspect_training_dataset`, `infer_task_type_dataset`).
   - Transparent feature and test loading in `competition_bundle.py` (`load_test_features`, `_read_dataset`).
   - End-to-end dashboard run upload path support for Parquet files via `coordinator_api.py` and `dashboard.py`.

2. **Modality Profiling & Strict Blocker Defense**:
   - Modality detection for audio, video, raw image, and unsupported binary formats in `intake.py`.
   - Unsupported files automatically set `execution_ready=False` and populate `profile.blockers`.
   - Dual-layer confirmation gate in `dashboard.py`: blocks unconfirmed runs with unaddressed questions/blockers and directly enforces modality bans during confirmation.

3. **Fold-Safe Text Feature Extraction**:
   - `TextTfidfEncoder(BaseEstimator, TransformerMixin)` in `features/transformers.py` with n-gram sublinear TF-IDF.
   - Registered under the `"text"` family (`"text_tfidf"`) in `FeatureRegistry`.
   - Integrated into `FeaturePipelineExecutor`, fitted strictly on training fold data with missing column imputation.

4. **Formal TaskAdapter Protocol & Registry**:
   - Runtime protocol `TaskAdapter` in `tasks/base.py`.
   - Implementations: `BinaryClassificationAdapter`, `MulticlassClassificationAdapter`, `RegressionAdapter`, `MultilabelClassificationAdapter`, `TimeSeriesForecastingAdapter`.
   - Central registry `TaskAdapterRegistry` with `get_task_adapter()` and `get_task_registry()`.
   - Updated metric registry (`multilabel_f1` metric with macro F1 aggregation) and target class validation in `contracts.py`.

### Primary files

- `src/swarmfold/data.py`
- `src/swarmfold/competition_bundle.py`
- `src/swarmfold/intake.py`
- `src/swarmfold/dashboard.py`
- `src/swarmfold/features/transformers.py`
- `src/swarmfold/features/registry.py`
- `src/swarmfold/features/executor.py`
- `src/swarmfold/tasks/` (`base.py`, `binary.py`, `multiclass.py`, `regression.py`, `multilabel.py`, `timeseries.py`, `registry.py`)
- `src/swarmfold/schemas.py`
- `src/swarmfold/metrics.py`
- `src/swarmfold/contracts.py`
- `tests/test_phase7_data_task_expansion.py`

### Acceptance criteria

- [x] Unsupported modalities are identified during profiling and blocked before training.
- [ ] Each enabled task has at least one sanity baseline, one strong baseline, one compatible ensemble
  strategy, and an end-to-end fixture. Multilabel and forecasting currently have adapters only and
  are not enabled for execution.
- [x] File type support is tested through the dashboard upload path, not only through isolated loaders.

## 12. Phase 8: Product, Package, and Documentation Closure [PARTIAL]

### Goal

Make system state, evidence, and limitations understandable from the UI and one downloadable
package.

### Work

- [x] Add UI panels for:
  - confirmed problem and validation contract;
  - enabled lanes and budgets;
  - retrieved technique evidence;
  - planned and completed experiment DAG;
  - fold metrics and resource use;
  - prune, failure, and repair events;
  - prediction diversity;
  - ensemble adoption decision;
  - reflection and next-iteration recommendations.
- [x] Keep one primary `Download Full Package` action.
- [x] Include in `full-package.zip`:
  - original competition bundle;
  - contracts and data fingerprints;
  - all selected experiment notebooks and artifacts;
  - experiment ledger and findings;
  - retrieval and literature evidence;
  - validation and leakage audit;
  - ensemble reconstruction metadata;
  - final notebook and submission;
  - one consolidated `experiment-knowledge.md` for the next iteration;
  - manifest with hashes and environment lock information.
- [x] Reconcile README and PROJECT_SPEC around one primary five-AI local topology. Keep distributed
  three-AI plus three-human behavior explicitly labeled as compatibility mode.
- [ ] Update Graphify after architecture documentation and runtime converge. The current graph index
  predates the reviewed runtime changes.
- [x] Add a generated capability matrix showing declared, installed, executable, scheduled, tested, and
  UI-exposed status.

### Acceptance criteria

- [ ] A fresh reader can identify what ran, why it ran, what failed, and why the final model won using
  only the package.
- [ ] README, project specification, UI, capability registry, and runtime describe the same default
  architecture.
- [x] Package verification succeeds on a fresh environment without absolute local paths.
- [x] No secret, provider credential, or external competition credential is included.

## 13. Proposed Pull Request Sequence

Implement the plan in reviewable slices:

1. `fix/metric-task-contracts`
2. `fix/champion-ensemble-gates`
3. `test/hermetic-suite-and-ruff`
4. `feat/registry-backed-candidate-factory`
5. `feat/adaptive-scheduler-pruner-integration`
6. `feat/reflection-memory-loop`
7. `feat/canonical-knowledge-and-technique-plan`
8. `feat/automatic-validation-splitters`
9. `feat/fold-safe-feature-executor`
10. `feat/unified-oof-ensemble-factory`
11. `feat/self-healing-execution`
12. `feat/task-and-data-adapters`
13. `docs/ui-package-capability-closure`

Do not combine all phases into one large merge. Phase 0 must land before feature or task expansion,
and every later pull request must keep the full test and artifact verification gates green.

## 14. Global Definition of Done

SwarmFold reaches the next maturity level when all of the following are true:

- The default dashboard invokes every capability it advertises.
- At least one run demonstrates adaptive scheduling, pruning, reflection, technique retrieval,
  automatic validation, feature ablation, model diversity, and ensemble selection together.
- All recorded metrics are reproducible from immutable prediction artifacts.
- The final winner is selected with metric-aware, leakage-safe evidence.
- Failed, pruned, repaired, and rejected experiments remain visible and reusable as knowledge.
- The next iteration consumes the prior package and produces a measurably different evidence-based
  plan.
- Unit, integration, end-to-end, lint, package verification, and offline test suites pass.
- Documentation and generated architecture outputs match the runtime implementation.
