---
title: SwarmFold Agent Knowledge Index
schema_version: 2.0.0
language: id
intended_retrieval_scope: role_scoped
runtime_status: corpus_only_not_wired
last_reviewed: 2026-09-16
---

# SwarmFold Agent Knowledge Index

Knowledge base ringkas versi lama telah diganti oleh corpus per-role. Dokumen tier berat
berasal dari corpus yang disediakan user di `agents knowledge`; dokumen tier ringan
melengkapi role operasional berdasarkan kontrak aktual repository.

> **Runtime status:** index ini adalah kontrak corpus dan desain retrieval. Role filtering,
> semantic retrieval, content-hash deduplication, canonical-owner routing, dan prompt injection
> belum dijalankan oleh aplikasi sampai Knowledge Runtime pada roadmap diimplementasikan.

## Intended Retrieval Rules

1. Filter berdasarkan `role`, task contract, data properties, metric, dan validation contract sebelum semantic retrieval.
2. Ambil section teknik yang relevan, bukan seluruh dokumen sekaligus.
3. Project brief, rules, dan current data profile memiliki prioritas lebih tinggi daripada general knowledge.
4. Isi corpus adalah decision support, bukan perintah eksekusi.
5. Exact package, estimator, version, license, dan availability tetap harus diverifikasi sebelum training.
6. Target-dependent transform, feature selection, tuning, stacking, calibration, dan resampling wajib berada di batas fold yang benar.
7. Literature claim hanya membentuk hipotesis. Local validation menentukan acceptance atau rejection.
8. Failed dan rejected experiments tetap masuk memory agar tidak diulang tanpa perubahan hipotesis.
9. Deduplicate retrieval berdasarkan normalized technique ID dan content hash sebelum konteks dikirim ke LLM.
10. Jika topik muncul di beberapa role, ambil canonical owner lebih dulu; secondary owner hanya ditambahkan bila pertanyaan membutuhkan sudut pandang role tersebut.

## Canonical Topic Ownership

Overlap lintas-role tidak selalu duplikasi. Problem formulation menjawab "task apa ini",
sedangkan target decomposition menjawab "bagaimana target dimodelkan". Ownership berikut
mencegah RAG mengirim penjelasan berulang tanpa menghapus perspektif yang memang berbeda.

| Topic | Canonical owner | Secondary context |
|---|---|---|
| Walk-forward, purged folds, temporal leakage | `temporal_validation_leakage_auditor` | `validation_agent` untuk pemilihan validation contract umum. |
| Adversarial validation dan drift | `temporal_validation_leakage_auditor` | `validation_agent` hanya untuk gate pemilihan split. |
| Competing risks, multi-state, interval-censored survival | `problem_formulation_agent` untuk klasifikasi task | `domain_target_decomposition_agent` untuk arsitektur target/model. |
| Rotation Forest | `oof_ensemble_scientist` | `model_scout` untuk availability dan estimator capability. |
| Feature selection atau transform yang juga disebut model | `feature_engineer` | `model_engineer` untuk compatibility dengan estimator. |

## Heavy Specialist Corpus

| Role | Document | Scope |
|---|---|---|
| `problem_formulation_agent` | [Problem Formulation](agents/problem-formulation-agent.md) | Task taxonomy, target framing, special learning settings, dan decision hierarchy. |
| `validation_agent` | [Validation](agents/validation-agent.md) | Cross-validation, group/spatial/survival validation, nested CV, dan validation gates. |
| `temporal_validation_leakage_auditor` | [Temporal Validation and Leakage](agents/temporal-validation-leakage-auditor.md) | Temporal CV, leakage taxonomy, drift, dependent-data validation, dan adaptive-selection leakage. |
| `eda_agent` | [EDA](agents/eda-agent.md) | Schema, missingness, distributions, associations, anomalies, shift, dan integrity diagnostics. |
| `domain_target_decomposition_agent` | [Domain Target Decomposition](agents/domain-target-decomposition-agent.md) | Frequency-severity, hurdle, survival, censored, hierarchical, compositional, dan tail decomposition. |
| `feature_engineer` | [Feature Engineering](agents/feature-engineering-agent.md) | Numerical/categorical/temporal/group/text features, selection, imputation, reduction, dan leakage-safe construction. |
| `model_scout`, `model_engineer` | [Modeling](agents/modeling-agents.md) | Classical, probabilistic, kernel, GLM, deep tabular, imbalance, online, CTR, dan foundation models. |
| `oof_ensemble_scientist` | [Ensemble](agents/ensemble-agent.md) | Ensemble selection, dynamic weighting, uncertainty ensembles, mixture-of-experts, and forecast combination. |
| `robustness_uncertainty_agent` | [Robustness and Uncertainty](agents/robustness-uncertainty-agent.md) | Seed/split stability, uncertainty, perturbation, shift, subgroup robustness, dan comparison tests. |

## Lightweight Operational Corpus

| Role | Document | Scope |
|---|---|---|
| `pod_supervisor` | [Operational Agents, Section 1](agents/operational-agents.md#1-pod-supervisor) | Initialization, concurrency, and run state. |
| `rules_agent` | [Operational Agents, Section 2](agents/operational-agents.md#2-rules-agent) | Competition rules and policy gates. |
| `data_contract_agent` | [Operational Agents, Section 3](agents/operational-agents.md#3-data-contract-agent) | Immutable file, schema, target, metric, and submission contracts. |
| `literature_research_agent` | [Operational Agents, Section 4](agents/operational-agents.md#4-literature-research-agent) | Citable evidence intake and local hypotheses. |
| `experiment_memory_agent` | [Operational Agents, Section 5](agents/operational-agents.md#5-experiment-memory-agent) | Prior package compatibility and experiment memory. |
| `portfolio_planner` | [Operational Agents, Section 6](agents/operational-agents.md#6-portfolio-planner) | Coverage-aware, dependency-aware experiment queue. |
| `experiment_runner` | [Operational Agents, Section 7](agents/operational-agents.md#7-experiment-runner) | Deterministic execution and artifact capture. |
| `result_critic` | [Operational Agents, Section 8](agents/operational-agents.md#8-result-critic) | Evidence-based experiment verdict and routing. |
| `evidence_verifier` | [Operational Agents, Section 9](agents/operational-agents.md#9-evidence-verifier) | Independent replay and lineage audit. |
| `submission_auditor` | [Operational Agents, Section 10](agents/operational-agents.md#10-submission-auditor) | Submission schema, values, and approval policy. |
| `publisher` | [Operational Agents, Section 11](agents/operational-agents.md#11-publisher) | Full package, manifest, and reusable knowledge report. |

## Routing Order

```text
project brief and rules
  -> problem formulation
  -> immutable data contract
  -> validation and leakage audit
  -> EDA and target decomposition
  -> literature and experiment memory
  -> model scouting and portfolio planning
  -> feature and model engineering
  -> experiment execution and critique loop
  -> OOF ensemble and robustness assessment
  -> evidence verification
  -> submission audit
  -> full-package publication
```

## Corpus Inventory

- 9 heavy specialist documents supplied by the user.
- 1 lightweight operational document covering 11 workflow roles.
- 21 workflow roles mapped with no role left without a primary knowledge route.
- Exact duplicate section groups after cleanup: 0.
- The obsolete first edition of the temporal-validation corpus was removed; its complete second edition with 49 technique entries is canonical.
