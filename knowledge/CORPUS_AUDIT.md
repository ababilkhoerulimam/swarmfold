---
title: SwarmFold Agent Knowledge Corpus Audit
audited: 2026-09-16
scope: knowledge/agents
---

# Agent Knowledge Corpus Audit

## Summary
- **Total Canonical Cards:** 468
- **Semantic Duplicates Remaining:** 0 (Zero title collision guarantee)
- **Merged Cross-Document / Overview Clusters:** 10
- **Directly Executable Runtime Cards:** 19
- **Declared Research / Advisory Cards:** 449

## Breakdown by Specialist Discipline
| Category | Canonical Cards Count | Primary Roles |
|---|---|---|
| `feature_engineering` | 120 | `feature engineering` |
| `model` | 89 | `model` |
| `validation` | 67 | `validation` |
| `robustness` | 57 | `robustness` |
| `problem_formulation` | 34 | `problem formulation` |
| `target_decomposition` | 34 | `target decomposition` |
| `ensemble` | 33 | `ensemble` |
| `eda` | 23 | `eda` |
| `operations` | 11 | `operations` |

## 20-Field Metadata Completeness
| Specification Field | Populated Cards | Completeness Rate |
|---|---|---|
| `title` | 468 / 468 | 100.0% |
| `rationale` | 468 / 468 | 100.0% |
| `fold_boundary_rule` | 468 / 468 | 100.0% |
| `leakage_risk` | 468 / 468 | 100.0% |
| `when_to_use` | 458 / 468 | 97.9% |
| `when_not_to_use` | 407 / 468 | 87.0% |
| `exact_package` | 468 / 468 | 100.0% |
| `example_hypothesis` | 419 / 468 | 89.5% |
| `acceptance_criteria` | 418 / 468 | 89.3% |
| `source` | 467 / 468 | 99.8% |

## Executable Model Families & Runtime State
- **Registered Executable Families:** `bagging, extra_trees, hist_gradient_boosting, lightgbm, logistic_regression, random_forest, sgd_log_loss, soft_voting, stacking`
- **Capability Separation Principle:** Every technique card without a tested adapter remains declared for advisory/research retrieval and is blocked from execution prior to trial allocation.
