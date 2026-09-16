---
title: SwarmFold Personal AutoML Roadmap
status: proposed
last_reviewed: 2026-09-16
---

# SwarmFold as a Personal AutoML System

## Current State

SwarmFold saat ini adalah agentic experiment orchestrator untuk kompetisi, belum adaptive
personal AutoML. Sistem sudah memiliki bundle profiling dengan human confirmation, frozen
contracts, lima lane behavior, LangGraph workflow, MLflow tracking, experiment artifacts,
preliminary fixed-member ensemble assessment pada single holdout, robustness report, notebook,
full package, dan memory package antar iterasi.

Execution core masih sempit:

- task executable hanya binary classification CSV;
- metric executable hanya accuracy dan ROC-AUC;
- validation masih satu stratified holdout, bukan true out-of-fold cross-validation;
- candidate queue dan parameter schedule masih hardcoded per lane;
- LLM planner hanya menulis hipotesis baseline, belum memilih pipeline dari knowledge corpus;
- static agent knowledge belum mempunyai retriever/runtime injection;
- capability registry menyebut CatBoost, LightGBM, dan XGBoost, tetapi execution adapter belum mengimplementasikan ketiganya;
- soft voting dan stacking adalah pipeline fixed-member, bukan ensemble yang belajar dari seluruh OOF experiment library;
- README dan graph lama masih memuat desain three-AI/three-human pada beberapa bagian sehingga source of truth belum tunggal.

## Target Architecture

```text
competition bundle
  -> problem signature and immutable contracts
  -> role-scoped hybrid knowledge retrieval
  -> typed experiment-plan compiler
  -> validation and leakage policy engine
  -> adaptive experiment DAG and resource scheduler
  -> feature/model/HPO adapters
  -> common OOF prediction store
  -> ensemble search and robustness gates
  -> champion registry and full package
  -> personal meta-memory for the next dataset and iteration
```

The LLM proposes and explains. Deterministic code validates, executes, scores, and publishes.

## Foundation Gate: Align System Truth

1. Make the five-lane dashboard, README, project spec, graph, schemas, and runtime use one topology.
2. Separate `declared`, `installed`, and `executable` model capability states.
3. Reject a proposal when no exact adapter exists even if its package is installed.
4. Version the task, metric, split, feature, model, and ensemble contracts independently.

Acceptance gate: every UI option resolves to a tested adapter and every runtime node is represented
once in the architecture documentation.

## Capability Track A: Knowledge Runtime

Compile every 20-field technique entry into a structured `TechniqueCard` with:

- stable technique ID and aliases;
- canonical owner role;
- supported task/data/metric tags;
- use/do-not-use conditions;
- preprocessing and leakage constraints;
- package, estimator, version, license, and executable status;
- parameter schema and initial search space;
- compute profile;
- hypothesis and experiment gates;
- source provenance and review date.

Use hybrid retrieval: deterministic metadata filters first, BM25 plus embeddings second, optional
reranking last. Deduplicate by technique ID/content hash and retrieve only a small evidence set per
role. Unverified entries may create research questions but may not enter the execution queue.

Acceptance gate: an agent can explain why each retrieved technique matched the current contract and
every proposed executable method maps to a verified adapter.

## Capability Track B: Generic Validation and Task Adapters

Add registries for:

- binary, multiclass, regression, count, ranking, time-series, survival, anomaly, and multilabel tasks;
- metric implementations including direction, output shape, grouping, clipping, and verification;
- holdout, repeated K-fold, group, stratified-group, temporal, purged, and nested validation;
- CSV, Parquet, JSON, multi-table, text-column, image-reference, and time-indexed bundles.

Build true OOF predictions with stable row IDs. Every fold-local operation, including target
encoding, selection, imputation, resampling, calibration, and early stopping, must be represented in
the pipeline DAG.

Acceptance gate: saved predictions reproduce the recorded metric and no preprocessing artifact is
fitted outside its permitted training fold.

## Capability Track C: Adaptive Search Controller

Replace fixed candidate loops with a staged search:

1. sanity baseline;
2. strong task-native baseline families;
3. feature-family ablations;
4. bounded HPO with pruning or multi-fidelity evaluation;
5. seed/fold confirmation for promising candidates;
6. ensemble search;
7. robustness and plateau innovation.

Use an Optuna-compatible persistent study or equivalent typed search service. Allocate trials from
evidence, not evenly by default. Track parent experiment, controlled change, predicted resource,
observed resource, failure reason, and keep/reject decision. A lane budget is an experiment target,
but duplicate or infeasible plans must be replaced rather than silently reducing the count.

Acceptance gate: the same state and seeds reproduce the queue, failed trials remain searchable, and
promotion requires a predeclared gate rather than the largest observed score alone.

## Capability Track D: OOF Ensemble Factory

Create a prediction registry shared by all lanes and test:

- equal and rank averaging;
- constrained weighted blending and NNLS;
- greedy ensemble selection;
- seed and fold averaging;
- calibration-aware probability blending;
- regularized stacking with OOF meta-features;
- task-specific forecast or survival aggregation.

Select members using score, fold stability, prediction correlation, error overlap, resource cost,
and marginal ensemble contribution. An ensemble is expected late in the workflow, but it is rejected
when it cannot beat the best member robustly.

Acceptance gate: weights/meta-models never see in-sample base predictions or final holdout labels,
and every ensemble can be reconstructed from member IDs and prediction hashes.

## Capability Track E: Personal Meta-Memory

The personal layer should learn priors, not copy winners blindly. Persist:

- dataset meta-features and problem signature;
- technique applicability and adapter failures;
- learning curves and resource curves;
- fold/seed score distributions;
- feature-family and model-family deltas;
- ensemble marginal contributions;
- rejected hypotheses and their failure mechanisms;
- package/environment compatibility;
- final local and leaderboard evidence as separate channels.

Use nearest-dataset retrieval plus a contextual bandit or simple Bayesian prior to reorder future
candidates. Current-dataset validation always overrides historical preference.

Acceptance gate: prior memory improves time-to-strong-baseline or final score on held-out projects
without changing the validation contract or repeating stale experiments.

## Capability Track F: Operations and Product Layer

- CPU/GPU-aware worker queues with bounded concurrency and no hidden lane timeout.
- Checkpoint/resume at experiment and fold level.
- Dataset/model/prediction caches keyed by immutable fingerprints.
- Secret scanning and artifact size policies before publication.
- UI views for plan DAG, retrieved evidence, fold metrics, prediction diversity, resource usage,
  experiment lineage, and explicit human approvals.
- Plugin interface for new tasks, metrics, models, feature generators, and ensemble strategies.

## Authoritative Implementation Sequence

The track letters group capabilities and do not define dependency order. Implement in this order:

1. Complete the Foundation Gate: classify every capability as declared, installed, or executable, and expose only already-tested adapters to planning.
2. Build true OOF storage and leakage-safe validation from Track B.
3. Add real CatBoost, LightGBM, and XGBoost adapters under the same capability contract.
4. Wire TechniqueCard compilation and role-scoped retrieval from Track A.
5. Build the OOF ensemble factory from Track D.
6. Add adaptive search and HPO from Track C.
7. Add personal meta-memory from Track E.
8. Harden scheduling, caching, secrets, UI, and plugin operations from Track F throughout the work.

Adaptive HPO and personal meta-learning are not trustworthy before the validation, adapter, and
OOF foundations pass their acceptance gates.
