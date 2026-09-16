# SwarmFold HPO, Ensemble, and Agent F Implementation Plan

Status: approved design, implementation pending  
Owner: SwarmFold runtime and modeling platform  
Last updated: 2026-09-16  
Scope: close the eight known blockers, add an adaptive HPO controller, and introduce Agent F
as the portfolio-level ensemble specialist

## 1. Outcome

This plan upgrades SwarmFold from five independent modeling lanes with a limited portfolio
ensemble into a traceable AutoML system with:

- five parallel base-model lanes, AI-A through AI-E;
- a shared registry containing every eligible experiment prediction, not only one winner per lane;
- adaptive, resumable HPO with fold-aware pruning;
- Agent F as a sequential portfolio-level ensemble stage;
- leakage-safe Caruana selection and cross-fitted stacking;
- fold-safe feature pass-through for the stacking meta-model;
- reproducible stacking artifacts and submission reconstruction;
- threshold optimization only for compatible label metrics;
- one Champion Gate that compares all compatible single and ensemble candidates fairly; and
- runtime, UI, package, documentation, and capability claims that describe the same behavior.

The implementation is complete only when these capabilities execute through the production
portfolio path and their evidence is included in the downloadable package.

## 2. Decisions locked before implementation

### 2.1 Agent F is not a sixth training lane

AI-A through AI-E remain independent base-model lanes. Agent F starts only after all enabled lanes
reach a terminal state and the shared prediction registry is frozen for the session.

Agent F is represented internally as an `ensemble` stage, not as lane ID `ai-f`. This avoids
polluting lane schemas, lane budgets, scheduling rules, and regular expressions with a component
that has different inputs and lifecycle semantics.

### 2.2 Ensemble use is preferred, not forced

Agent F is enabled by default, but it may return `SKIPPED` or `REJECTED`. No ensemble replaces the
best single model unless the Champion Gate proves an improvement under the same validation
contract. A rejected ensemble remains documented as an experiment result.

### 2.3 All comparisons use one validation contract

Base models, HPO trials, Caruana candidates, and stacking candidates must share:

- identical training row IDs;
- identical fold assignments;
- identical metric definition and optimization direction;
- identical target encoding and class order;
- identical group or temporal constraints; and
- a recorded dataset, split, preprocessing, and prediction fingerprint.

An ensemble with misaligned rows, folds, targets, or class columns is ineligible rather than
silently repaired.

### 2.4 Stacking activation is intentionally staged

Initial portfolio wiring may expose stacking diagnostics, but stacking is not adoption-eligible
until both conditions are met:

1. its meta-level score is produced from cross-fitted predictions; and
2. its final submission can be reconstructed from serialized artifacts within tolerance.

### 2.5 HPO and Agent F have separate budgets

The base-model experiment budget controls trials in AI-A through AI-E. Agent F has a separate
ensemble search budget. Selecting 20 experiments for two lanes means up to 40 base experiments,
followed by an independently bounded ensemble search over eligible predictions.

There is no hidden 120-second lane timeout. Limits are explicit configuration for trial count,
resource use, cancellation, and optional user-defined wall-clock budget.

### 2.6 Deterministic code owns scoring and adoption

The LLM can propose hypotheses, model families, search-space emphasis, and explanations. Python
code owns fold construction, training, prediction alignment, metric computation, pruning,
reconstruction verification, and Champion Gate decisions.

## 3. Target runtime architecture

```text
Competition bundle
        |
Problem understanding and human confirmation
        |
Frozen validation and submission contracts
        |
 +------+------+------+------+------+
 | AI-A | AI-B | AI-C | AI-D | AI-E |  parallel base-model lanes
 +------+------+------+------+------+
        |
All successful experiment OOF and test predictions
        |
Shared Prediction Registry barrier
        |
Agent F: audit, curate, blend, Caruana, stack, threshold, verify
        |
Champion Gate across best single model and eligible ensembles
        |
Final model decision, submission, notebook, ledger, findings, and full package
```

The registry barrier must retain every eligible experiment. It must not collapse each lane to one
candidate before Agent F, because diversity across 5 lanes by 20 experiments may provide up to 100
potential members before deduplication and curation.

## 4. Delivery phases

Phases are ordered by dependency. Later phases must not be marked complete when an earlier safety
gate is still open.

### Phase 0: Freeze contracts and expose validation truth

Goal: establish the immutable contracts used by HPO and Agent F while closing blocker 1.

Main changes:

- Add typed `ValidationContract`, `PredictionArtifactRef`, `EnsembleStageConfig`, and
  `ChampionGatePolicy` schemas.
- Include dataset fingerprint, row-ID contract, split fingerprint, fold assignments, metric
  direction, class order, and submission contract in the frozen run context.
- Display the locally computed validation recommendation in the human confirmation UI.
- Require the user-visible profile and deterministic runtime validation recommendation to agree,
  or show the exact conflict before training.
- Add package schema versioning so new Agent F artifacts do not break older packages.

Primary files:

- `src/swarmfold/schemas.py`
- `src/swarmfold/coordinator_models.py`
- `src/swarmfold/dashboard.py`
- `src/swarmfold/web/dashboard.html`
- `src/swarmfold/artifacts.py`

Tests:

- schema validation and backward-compatibility tests;
- validation recommendation UI/API integration test;
- dataset and split fingerprint determinism test; and
- conflict test where profiler text disagrees with the deterministic contract.

Acceptance criteria:

- the confirmation page shows the exact split strategy and reason used by training;
- all later experiment records reference one frozen validation contract ID; and
- incompatible prediction artifacts cannot enter the ensemble registry.

### Phase 1: Register all experiment predictions centrally

Goal: give Agent F access to the full experiment portfolio instead of only lane champions.

Main changes:

- Extend the experiment result contract to store OOF and test prediction artifact references.
- Register every successful, validation-compatible experiment in `PredictionRegistry`.
- Store lane, trial, model family, parameters, feature recipe, fold scores, row IDs, class order,
  prediction shape, artifact hash, and resource measurements.
- Deduplicate exact prediction hashes while preserving provenance links to all originating trials.
- Freeze the registry only after all enabled lanes are terminal.
- Make partial lane failure non-fatal when at least one valid champion exists.

Primary files:

- `src/swarmfold/ensemble/prediction_registry.py`
- `src/swarmfold/portfolio.py`
- `src/swarmfold/graph.py`
- `src/swarmfold/artifacts.py`

Tests:

- 5 lanes by multiple experiments produces the expected registry size;
- duplicate predictions are represented once with complete provenance;
- shuffled row IDs, changed class order, or mismatched folds are rejected; and
- deterministic registry order is preserved across identical seeded runs.

Acceptance criteria:

- Agent F can query every eligible prediction from a completed run;
- no member is accepted without aligned OOF targets and row IDs; and
- the package ledger maps each registry member back to its experiment.

### Phase 2: Implement the adaptive HPO controller

Goal: replace fixed boosting grids with a resumable, metric-aware TPE search while preserving
strict fold comparability.

Main changes:

- Add an optional HPO dependency group for Optuna.
- Create `HPOConfig`, `HPOStudyRecord`, `HPOTrialRecord`, and typed conditional search spaces.
- Add an `HPOController` responsible for study creation, seeded sampling, trial lifecycle,
  persistence, resumption, and conversion into executable candidate specifications.
- Define task-aware search spaces for LightGBM, XGBoost, and CatBoost using only parameters
  supported by the installed adapters.
- Reuse the frozen validation folds for every trial.
- Report fold results to the controller after each completed fold.
- Integrate `TrialPruner` with a median-at-same-step rule after a configurable number of startup
  trials. Fold 1 may prune only when enough comparable Fold 1 observations exist.
- Keep cross-trial pruning separate from estimator-native early stopping.
- Handle maximize and minimize metrics without sign ambiguity.
- Persist studies and trials locally, including pruned trials and reasons.
- Emit HPO lineage to the experiment ledger, package, and dashboard.

Primary files:

- `pyproject.toml`
- `src/swarmfold/optimization/controller.py`
- `src/swarmfold/optimization/search_spaces.py`
- `src/swarmfold/optimization/pruner.py`
- `src/swarmfold/optimization/scheduler.py`
- `src/swarmfold/portfolio.py`

Tests:

- deterministic suggestions with a fixed sampler seed;
- conditional parameter validity for all three boosting families;
- maximize and minimize metric pruning;
- no pruning before the startup evidence threshold;
- pruning after Fold 1 when the same-step median is available;
- interruption and resume without duplicate trial numbers; and
- a bounded integration study that improves or matches the fixed baseline on synthetic data.

Acceptance criteria:

- AI-B can run adaptive HPO through the production portfolio;
- every trial is reproducible from persisted configuration and seed;
- a pruned trial records completed fold scores and its comparison benchmark; and
- HPO failure falls back to a valid scheduled candidate without corrupting the run.

### Phase 3: Add Agent F in audit-only mode

Goal: introduce the Agent F lifecycle, knowledge, UI, and artifacts before granting adoption
authority.

Agent F responsibilities:

1. audit eligibility and prediction alignment;
2. remove duplicate prediction vectors;
3. measure prediction correlation, error overlap, and lane or family diversity;
4. curate a computationally reasonable member pool;
5. propose compatible ensemble trials;
6. execute deterministic ensemble code;
7. verify stability and reconstruction evidence; and
8. submit eligible candidates to the Champion Gate.

Main changes:

- Add an `EnsembleStage` orchestrator invoked after the registry barrier.
- Add Agent F role guidance and role-scoped retrieval using the existing
  `oof_ensemble_scientist` knowledge domain.
- Keep external status separate from base lane IDs.
- Add Agent F UI card with status, member count, excluded members, methods attempted, best result,
  and adoption decision.
- In audit-only mode, generate diagnostics without changing the champion.

Suggested states:

- `DISABLED`
- `WAITING_FOR_MEMBERS`
- `AUDITING`
- `SEARCHING`
- `VERIFYING`
- `ADOPTED`
- `REJECTED`
- `SKIPPED`
- `FAILED`

Primary files:

- `src/swarmfold/ensemble/stage.py`
- `src/swarmfold/roles.py`
- `src/swarmfold/portfolio.py`
- `src/swarmfold/dashboard.py`
- `src/swarmfold/web/dashboard.html`

Tests:

- stage starts only after all enabled lanes are terminal;
- disabled, skipped, partial-lane, rejected, and failed states are visible and non-ambiguous;
- one lane with two diverse predictions can still be audited;
- fewer than two eligible predictions produces `SKIPPED`; and
- audit-only mode cannot mutate the champion.

Acceptance criteria:

- Agent F appears as a separate stage in API, UI, ledger, and package;
- base-lane counts and lane schemas remain AI-A through AI-E only; and
- Agent F produces a reproducible member audit before ensemble search begins.

### Phase 4: Activate Caruana and wire stacking into the portfolio

Goal: make existing ensemble implementations reachable from the real portfolio without yet
allowing unsafe stacking adoption.

Main changes:

- Enable Caruana greedy forward selection with replacement in `UnifiedEnsembleEngine`.
- Let Caruana select 3, 4, 5, or more selections while deriving proportional weights from member
  selection counts.
- Feed Caruana and stacking from the curated shared registry, not lane-only winners.
- Add method compatibility checks before scheduling.
- Replace silent method failures with structured diagnostics containing method, reason, traceback
  summary, and affected member IDs.
- Mark Caruana adoption-eligible after normal OOF and reconstruction checks.
- Mark stacking diagnostic-only with `adoption_eligible=false` until Phases 5 and 6 pass.

Primary files:

- `src/swarmfold/ensemble/engine.py`
- `src/swarmfold/ensemble/caruana.py`
- `src/swarmfold/ensemble/stacker.py`
- `src/swarmfold/portfolio.py`

Tests:

- selection with replacement yields normalized proportional weights;
- selected ensemble may contain 3, 4, or 5 unique members;
- lower-is-better metrics choose the correct candidate;
- a worse Caruana result cannot replace the champion;
- stacking results are visible but cannot be adopted; and
- method errors are persisted rather than swallowed.

Acceptance criteria:

- production portfolio runs Caruana from the full prediction registry;
- Caruana submission reconstruction matches engine output within tolerance; and
- stacking cannot pass the Champion Gate yet.

### Phase 5: Cross-fit the meta-model and add fold-safe feature pass-through

Goal: remove meta-level in-sample optimism before making stacking eligible.

Main changes:

- Build the level-2 training matrix exclusively from aligned base OOF predictions.
- Generate meta OOF predictions by fitting the meta-model on meta-train folds and predicting the
  held-out meta fold.
- Respect the original contract type when constructing meta folds, including stratified, group,
  and time-aware splitting.
- Compute the stacking validation score only from the concatenated meta OOF predictions.
- After evaluation, refit the final meta-model on the complete meta-training matrix for test
  inference.
- Add optional pass-through matrices with explicit train and test row alignment.
- Permit only fold-safe features or unsupervised transformations fitted inside the relevant fold.
- Freeze pass-through column names, order, dtypes, transformer hashes, and missing-value behavior.
- Start with regularized logistic regression for classification and Ridge for regression, then add
  other meta-models only through typed configuration.

Primary files:

- `src/swarmfold/ensemble/stacker.py`
- `src/swarmfold/ensemble/engine.py`
- `src/swarmfold/features/executor.py`
- `src/swarmfold/schemas.py`

Tests:

- leakage sentinel proves held-out targets cannot affect their meta prediction;
- cross-fitted score differs from an intentionally in-sample score on overfit-prone data;
- group boundaries and temporal order are preserved;
- pass-through row and schema mismatch fails closed;
- classification probability shapes and class order remain stable; and
- deterministic seeds produce stable meta OOF predictions.

Acceptance criteria:

- every stacking score presented to the gate is cross-fitted;
- pass-through features carry fold-fit evidence and an immutable schema; and
- in-sample meta predictions are never used as validation evidence.

### Phase 6: Serialize and reconstruct stacking artifacts

Goal: prove that the adopted stacking result can be reproduced outside the live run.

Required artifacts:

- `ensemble/ensemble-spec.json`
- `ensemble/member-manifest.json`
- `ensemble/meta-folds.json`
- `ensemble/meta-oof-predictions.parquet`
- `ensemble/meta-model.joblib`
- `ensemble/pass-through-schema.json`, when enabled
- `ensemble/threshold.json`, when applicable
- `ensemble/ensemble-reconstruction-test.json`

The ensemble specification must store:

- ordered member IDs and artifact hashes;
- ordered prediction columns and class labels;
- preprocessing and pass-through transformer references;
- meta-model class, parameters, package versions, and random seed;
- meta-fold contract;
- selected decision threshold, if any;
- expected validation score and submission prediction hash; and
- numeric tolerances used by the verifier.

Main changes:

- Add a stacking-specific reconstruction path instead of falling through to weighted averaging.
- Rebuild the meta matrix from recorded base test predictions and the fixed member order.
- Load the serialized transformer and meta-model.
- Apply the recorded threshold only after probability reconstruction.
- Compare reconstructed probabilities, labels, submission schema, and hashes with stored evidence.
- Fail packaging or ensemble adoption when reconstruction parity does not pass.

Primary files:

- `src/swarmfold/artifacts.py`
- `src/swarmfold/package_verifier.py`
- `src/swarmfold/ensemble/engine.py`
- `src/swarmfold/ensemble/stacker.py`

Tests:

- fresh-process stack reconstruction parity;
- corrupted member order, class order, model artifact, or transformer hash fails verification;
- older non-stacking packages still verify under their schema version; and
- packaged final submission equals the verified reconstruction.

Acceptance criteria:

- `package_verifier` can reconstruct a stacking submission without live in-memory objects;
- parity evidence is included in the full package; and
- stacking becomes adoption-eligible only after this phase passes.

### Phase 7: Optimize thresholds for compatible label metrics

Goal: tune decision thresholds without contaminating probability metrics or validation evidence.

Main changes:

- Add `optimize_classification_threshold()` returning a typed result with threshold, metric,
  baseline score, optimized score, search method, candidate count, and provenance.
- Run threshold search only on cross-fitted OOF probabilities.
- Support binary label metrics such as F1, balanced accuracy, accuracy, precision, recall, and an
  explicitly configured cost function.
- Do not optimize a threshold for ROC-AUC, average precision, log-loss, Brier score, regression,
  multiclass scalar thresholds, or ranking metrics.
- Preserve both probabilities and thresholded labels in artifacts.
- Apply the chosen threshold consistently to test predictions and reconstruction.
- Treat 0.5 as the baseline threshold and reject an optimized threshold that does not improve the
  configured label metric under the gate policy.

Primary files:

- `src/swarmfold/ensemble/thresholds.py`
- `src/swarmfold/ensemble/stacker.py`
- `src/swarmfold/ensemble/engine.py`
- `src/swarmfold/schemas.py`

Tests:

- F1 threshold improves or equals the 0.5 baseline on controlled data;
- ROC-AUC and log-loss bypass threshold optimization;
- minimize and maximize semantics are correct;
- all-positive and all-negative edge cases remain valid;
- selected threshold is reproduced from packaged artifacts; and
- test submission uses the recorded threshold exactly once.

Acceptance criteria:

- threshold optimization is impossible for an incompatible metric;
- threshold evidence comes only from cross-fitted OOF probabilities; and
- label submission reconstruction matches the stored result.

### Phase 8: Activate all compatible methods in the Champion Gate

Goal: compare every safe ensemble method with the single-model champion only after Phases 4 through
7 provide trustworthy evidence.

Candidate methods:

- equal-weight averaging;
- pairwise blending;
- rank averaging when the metric supports ranks;
- constrained non-negative blending;
- Caruana greedy forward selection;
- cross-fitted stacking with optional pass-through;
- soft voting for compatible classification probabilities; and
- majority voting for compatible label outputs.

Compatibility matrix:

| Task or metric | Blend | Rank | Caruana | Stacking | Majority | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Binary ROC-AUC | Yes | Yes | Yes | Yes | No | No |
| Binary log-loss | Yes | No | Yes | Yes | No | No |
| Binary F1 or accuracy | Yes | No | Yes | Yes | Yes | Yes |
| Multiclass probability metric | Yes | No | Conditional | Yes | No | No |
| Regression | Yes | Metric-specific | Yes | Yes | No | No |
| Ranking | Metric-specific | Yes | Conditional | Conditional | No | No |

Main changes:

- Route methods through a single compatibility policy.
- Evaluate all methods on aligned OOF evidence using the central metric semantics.
- Compare against the best eligible single model, not merely the last lane result.
- Require minimum gain, stability, complete artifacts, and successful reconstruction.
- Record `ADOPTED`, `REJECTED`, `INELIGIBLE`, or `FAILED` for every attempted method.
- Preserve the original champion submission whenever no ensemble clears the gate.

Primary files:

- `src/swarmfold/ensemble/engine.py`
- `src/swarmfold/portfolio.py`
- `src/swarmfold/artifacts.py`
- `src/swarmfold/dashboard.py`

Tests:

- worse ensembles are rejected for both maximize and minimize metrics;
- compatible best ensemble is adopted with complete provenance;
- ineligible methods are not silently attempted;
- gate result is independent of candidate iteration order; and
- final package contains the actual adopted artifact or the retained single champion.

Acceptance criteria:

- all compatible methods are reachable through the default portfolio path;
- no method can bypass alignment, cross-fit, reconstruction, or stability checks; and
- UI and package explain why the final model was adopted.

### Phase 9: Schedule feature ablation and add fold-safe transform caching

Goal: close blockers 2 and 3 while reducing repeated work during HPO.

Main changes:

- Integrate feature ablation into runtime scheduling with explicit experiment lineage.
- Use ablation findings to inform later feature hypotheses without deleting the original evidence.
- Add a transform cache keyed by dataset fingerprint, fold ID, train-row hash, validation-row hash,
  transformer specification, fitted-state hash, feature schema, and software version.
- Never share target-dependent fitted state across folds.
- Expose cache hits, misses, and invalidation reason in experiment telemetry.

Tests and acceptance criteria:

- ablation experiments appear in the ledger and package;
- cached and uncached predictions are identical;
- changed folds, transformer parameters, row order, or package version invalidate the cache; and
- leakage sentinels prove fold boundaries are preserved.

### Phase 10: Add provider-operation retries

Goal: close blocker 5 without confusing infrastructure failures with modeling failures.

Main changes:

- Add a provider-operation wrapper for LLM and remote profiler calls.
- Retry only classified transient failures with bounded exponential backoff and jitter.
- Use idempotency keys where supported.
- Do not retry authentication, invalid request, deterministic schema, or model-training errors.
- Record attempts, latency, final classification, and fallback behavior.
- Keep this layer separate from the model debugger and training self-healing logic.

Tests and acceptance criteria:

- transient timeout and rate-limit simulations recover within policy;
- permanent failures fail quickly with actionable messages;
- retried operations do not duplicate run creation or artifacts; and
- local deterministic fallbacks remain available where defined.

### Phase 11: Complete multilabel and time-series end-to-end paths

Goal: close blocker 6 by turning declared adapters into truthful product capabilities.

Main changes:

- Add task-specific validation, metrics, baseline models, strong-model adapters, prediction shapes,
  ensemble compatibility, and submission reconstruction.
- Add multilabel class-column contracts and per-label or aggregate metric evidence.
- Add time-series cutoff, horizon, frequency, group, exogenous feature, and leakage contracts.
- Keep dashboard task activation disabled until the corresponding end-to-end suite passes.

Tests and acceptance criteria:

- one representative multilabel bundle and one forecasting bundle execute from upload to package;
- temporal tests prove future rows never enter training transforms;
- submissions match sample schema and preserve identifiers; and
- capability matrix marks a task supported only after its production route passes.

### Phase 12: Converge runtime, UI, documentation, and Graphify

Goal: close blockers 7 and 8 and publish one truthful system description.

Main changes:

- Audit `README.md`, `PROJECT_SPEC.md`, capability matrix, UI copy, API schemas, and actual runtime.
- Remove claims not demonstrated by production-path tests.
- Document Agent F, HPO budgets, ensemble rejection behavior, threshold compatibility, artifacts,
  and package reconstruction.
- Regenerate the Graphify index only after source and documentation converge.
- Add a release checklist that checks documentation links and capability claims.

Tests and acceptance criteria:

- Graphify reflects the five-lane plus Agent F topology;
- no documented feature is dashboard-blocked unless explicitly labeled planned;
- package contents match documentation; and
- full release gates pass.

## 5. Agent F input and output contract

### Inputs

- frozen validation contract;
- frozen metric and submission contracts;
- complete eligible prediction registry;
- base experiment metadata and fold scores;
- optional fold-safe pass-through train and test artifacts;
- ensemble stage configuration; and
- role-scoped ensemble knowledge retrieved for the current task.

### Outputs

- `ensemble/member-audit.json`
- `ensemble/diversity-report.json`
- `ensemble/ensemble-plan.json`
- `ensemble/ensemble-trials.jsonl`
- `ensemble/ensemble-decision.json`
- method-specific prediction and model artifacts;
- reconstruction verification evidence; and
- a concise explanation for the final package and UI.

### Suggested configuration

```yaml
ensemble_stage:
  enabled: true
  mode: production
  methods:
    - equal_weight
    - pairwise_blend
    - constrained_blend
    - rank_average
    - caruana
    - stacking
    - majority_vote
  max_member_pool: 40
  max_ensemble_trials: 100
  cross_fit_splits: 5
  feature_pass_through: auto
  optimize_label_threshold: true
  minimum_gain: 0.0
  require_reconstruction: true
```

`max_member_pool` and `max_ensemble_trials` are explicit compute controls, not hidden timeouts.
Setting either to `null` may allow an unbounded search when the user knowingly accepts the cost.

## 6. HPO controller contract

The controller owns study mechanics but does not own validation semantics.

Required interfaces:

```python
class HPOController:
    def suggest(self, trial_number: int) -> CandidateSpec: ...
    def report_fold(self, trial_id: str, fold_index: int, score: float) -> PruneDecision: ...
    def complete(self, trial_id: str, result: ExperimentResult) -> None: ...
    def fail(self, trial_id: str, reason: FailureRecord) -> None: ...
    def resume(self, study_id: str) -> HPOStudyState: ...
```

The first implementation should use seeded TPE with a persistent local study. Search-space
definitions must remain repository-owned and versioned so an old trial can be reconstructed even
if defaults later change.

## 7. Champion Gate policy

A candidate is eligible only when:

- its metric is computed from aligned OOF or cross-fitted predictions;
- its validation contract matches the current run;
- its metric direction and class order are explicit;
- required artifacts exist and pass reconstruction;
- no leakage or schema guard failed;
- its stability checks meet policy; and
- its score clears the best single-model champion by the configured minimum gain.

Tie behavior must be deterministic. By default, retain the simpler existing champion on an exact
tie unless the policy explicitly prefers an ensemble for stability.

## 8. Mapping of the eight existing blockers

| Blocker | Resolution phase | Proof required |
| --- | --- | --- |
| Local validation recommendation missing from confirmation UI | Phase 0 | UI/API integration test |
| Feature ablation not scheduled in runtime | Phase 9 | Ledger and package evidence |
| No fold-safe transform fingerprint cache | Phase 9 | Cache parity and invalidation tests |
| Portfolio uses only a subset of ensemble methods | Phases 4-8 | Production portfolio integration |
| Provider-operation retry layer absent | Phase 10 | Transient and permanent failure tests |
| Multilabel and time-series not end-to-end | Phase 11 | Upload-to-package tests |
| Graphify index is stale | Phase 12 | Regenerated topology query |
| README, spec, UI, matrix, and runtime have not converged | Phase 12 | Release truthfulness audit |

## 9. Pull request sequence

To keep reviews bounded and reversible, use this sequence:

1. Contracts and confirmation UI.
2. Full experiment prediction registry.
3. HPO controller and study persistence.
4. Agent F audit-only stage and UI.
5. Caruana activation and diagnostic-only stacking wiring.
6. Cross-fitted stacking and fold-safe pass-through.
7. Stacking artifact reconstruction.
8. Label-threshold optimizer.
9. Full Champion Gate activation.
10. Feature ablation runtime and transform cache.
11. Provider retry layer.
12. Multilabel and forecasting production paths.
13. Documentation, capability, and Graphify convergence.

Each pull request must include tests for its own safety boundary and must leave the previous stable
path usable behind a typed feature flag when rollout risk is material.

## 10. Release gates

Minimum automated gates:

- `uv run pytest -q`
- `uv run ruff check .`
- `git diff --check`
- unit tests for every search, ensemble, threshold, and reconstruction primitive;
- portfolio integration tests with multiple experiments per lane;
- leakage sentinels for base folds, meta folds, pass-through, group splits, and time splits;
- fresh-process package reconstruction tests;
- deterministic replay with fixed seeds;
- maximize and minimize metric tests; and
- backward verification of an older package schema.

A heavier marked test should exercise five lanes with multiple experiments, HPO pruning, Agent F,
and package reconstruction. Regular CI can use fewer experiments while preserving the same code
path.

## 11. Definition of done

This initiative is done only when all of the following are true:

- AI-A through AI-E execute independently and register every eligible experiment prediction.
- AI-B can use adaptive HPO with persistent, resumable, fold-aware pruning.
- Agent F runs after the shared registry barrier and has a distinct visible lifecycle.
- Caruana and cross-fitted stacking run through the production portfolio.
- feature pass-through is aligned, fold-safe, and reconstructable.
- threshold optimization runs only for compatible binary label metrics.
- every eligible ensemble method is evaluated by the same Champion Gate.
- no worse or unverifiable ensemble replaces the single-model champion.
- the final package can reconstruct the adopted submission in a fresh process.
- all eight previous blockers have their required runtime evidence.
- UI, README, project specification, capability matrix, package, and Graphify describe the same
  released behavior.

