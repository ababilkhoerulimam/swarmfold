---
title: SwarmFold Operational Agents Knowledge
schema_version: 1.0.0
language: id
retrieval_scope: role_scoped
last_reviewed: 2026-09-16
source_authority: repository_contract
---

# Operational Agents, Dokumentasi Teknis Tier Ringan

Dokumen ini melengkapi corpus tier berat dengan kontrak untuk agent yang bertugas mengatur,
memvalidasi, atau menerbitkan pekerjaan. Agent di sini tidak memilih metode statistik baru.
Mereka wajib menyerahkan keputusan metodologis kepada agent spesialis dan bekerja hanya dari
artefak yang dapat diaudit.

## 1. Pod Supervisor

**Role ID:** `pod_supervisor`

| Field | Detail |
|---|---|
| **Definisi** | Menginisialisasi satu pod eksperimen, status, budget policy, batas konkurensi, dan urutan kerja agent. |
| **Problem yang didukung** | Semua task yang sudah memiliki bundle dan project brief. |
| **Kapan digunakan** | Selalu pada awal run dan ketika run dilanjutkan dari checkpoint. |
| **Kapan tidak digunakan** | Tidak boleh merancang fitur, memilih model, atau menilai skor. |
| **Asumsi** | Run ID unik; konfigurasi dapat divalidasi; storage dan checkpoint tersedia. |
| **Input** | Run configuration, lane configuration, budget, prior snapshot ID, bundle fingerprint. |
| **Output** | State awal, role report `READY`, resource envelope, dan execution lineage. |
| **Dataset scale** | Agnostik terhadap ukuran data; hanya membaca metadata. |
| **Metric compatibility** | Agnostik; metric contract diteruskan tanpa diubah. |
| **Decision variables** | Lane aktif, konkurensi, resume/fresh run, status awal. |
| **Default awal** | Semua lane pilihan user aktif; status `RUNNING`; ensemble gate `PENDING`. |
| **CPU/GPU/RAM** | Sangat ringan; CPU dan RAM rendah. |
| **Risiko leakage** | Menggunakan state atau snapshot milik run lain. |
| **Failure modes** | Run ganda, checkpoint tidak cocok, budget invalid, state parsial dianggap lengkap. |
| **Teknik pembanding** | Validasi state deterministik dan dry-run graph. |
| **Agent compatibility** | Upstream seluruh agent; tidak menggantikan keputusan specialist. |
| **Exact implementation** | `swarmfold.roles.initialize_pod` dan `swarmfold.graph.build_experiment_graph`. |
| **Contoh hypothesis** | "State yang diinisialisasi ulang dari input identik menghasilkan kontrak dan fingerprint identik." |
| **Acceptance criteria** | Semua field wajib valid, role order lengkap, run ID dan fingerprint konsisten. |
| **Rejection criteria** | Konfigurasi ambigu, snapshot tidak dikenal, atau state lintas-run tercampur. |
| **Sumber** | Repository contracts dan graph, diperiksa 2026-09-16. |

## 2. Rules Agent

**Role ID:** `rules_agent`

| Field | Detail |
|---|---|
| **Definisi** | Mengekstrak aturan kompetisi dan membentuk policy gate untuk data eksternal, submission, lisensi, dan tindakan yang memerlukan approval manusia. |
| **Problem yang didukung** | Kompetisi, benchmark internal, atau project dengan rule document. |
| **Kapan digunakan** | Sebelum profiling, research, training, atau pembuatan submission. |
| **Kapan tidak digunakan** | Tidak boleh mengarang aturan yang tidak ada di brief. |
| **Asumsi** | Project Markdown atau dokumen aturan disediakan; ketidakjelasan dapat dicatat sebagai blocker. |
| **Input** | Project brief, filenames, metadata bundle, aturan user. |
| **Output** | Policy contract, allowed/forbidden actions, unresolved questions, dan citation locator. |
| **Dataset scale** | Agnostik; membaca dokumen dan metadata. |
| **Metric compatibility** | Memastikan metric resmi tercatat, tanpa mengubahnya. |
| **Decision variables** | External data, pretrained model, internet access, submission frequency, human approval. |
| **Default awal** | Hal yang tidak dinyatakan dianggap belum diizinkan, bukan otomatis dilarang atau dibolehkan. |
| **CPU/GPU/RAM** | Ringan. |
| **Risiko leakage** | Aturan dari kompetisi lain atau knowledge lama dianggap berlaku. |
| **Failure modes** | Citation hilang, rule conflict tidak diblokir, asumsi ditulis sebagai fakta. |
| **Teknik pembanding** | Manual rule checklist dan exact-text retrieval dari brief. |
| **Agent compatibility** | Memberi constraint kepada seluruh downstream agent. |
| **Exact implementation** | `swarmfold.roles.inspect_rules`; parser dokumen harus mempertahankan source locator. |
| **Contoh hypothesis** | "Semua tindakan terlarang dapat ditolak sebelum experiment queue dibentuk." |
| **Acceptance criteria** | Setiap policy memiliki status confirmed, unresolved, atau not-applicable beserta sumbernya. |
| **Rejection criteria** | Policy tanpa sumber, conflict yang disembunyikan, atau training dimulai saat blocker material masih terbuka. |
| **Sumber** | Repository role contract dan user-provided project brief, diperiksa 2026-09-16. |

## 3. Data Contract Agent

**Role ID:** `data_contract_agent`

| Field | Detail |
|---|---|
| **Definisi** | Membekukan peran file, target, ID, schema, task, metric, validation contract, dan submission contract. |
| **Problem yang didukung** | Semua bundle multi-file atau single-file yang sudah dikonfirmasi manusia. |
| **Kapan digunakan** | Setelah rules dan problem formulation, sebelum validasi dan EDA. |
| **Kapan tidak digunakan** | Tidak boleh menebak target atau file role tanpa evidence dan confirmation. |
| **Asumsi** | Semua file tersedia, dapat dibaca, dan memiliki fingerprint stabil. |
| **Input** | Structured profile hasil bundle analysis dan human corrections. |
| **Output** | Immutable `ContractBundle` serta fingerprint kontrak. |
| **Dataset scale** | Profiling berbasis metadata/sample untuk data besar; full scan hanya bila aman. |
| **Metric compatibility** | Metric harus cocok dengan task, output shape, dan submission contract. |
| **Decision variables** | File roles, target, IDs, dtypes, split, seed, metric direction. |
| **Default awal** | Tidak ada default target; unknown tetap blocker. |
| **CPU/GPU/RAM** | Ringan sampai sedang tergantung schema scan. |
| **Risiko leakage** | Test/holdout diberi role training, target terselip sebagai feature, sample submission dianggap label. |
| **Failure modes** | Schema drift, duplicate IDs, target dtype salah, train-test columns tidak selaras. |
| **Teknik pembanding** | Schema diff, fingerprint replay, dan manual confirmation summary. |
| **Agent compatibility** | Menjadi sumber tunggal kontrak untuk seluruh specialist dan executor. |
| **Exact implementation** | `swarmfold.contracts.freeze_contracts` dan model di `swarmfold.schemas`. |
| **Contoh hypothesis** | "Bundle yang sama menghasilkan contract fingerprint yang sama pada rerun." |
| **Acceptance criteria** | Semua role dan schema konsisten, metric valid, confirmation tercatat, fingerprint reproducible. |
| **Rejection criteria** | Target/ID/file role ambigu, train-test incompatibility, atau contract berubah diam-diam. |
| **Sumber** | Repository contract implementation, diperiksa 2026-09-16. |

## 4. Literature Research Agent

**Role ID:** `literature_research_agent`

| Field | Detail |
|---|---|
| **Definisi** | Mengambil bukti primer yang relevan dan menerjemahkannya menjadi hipotesis lokal, bukan menyalin klaim leaderboard. |
| **Problem yang didukung** | Semua kategori dengan kebutuhan metode, validasi, atau domain knowledge. |
| **Kapan digunakan** | Setelah task, schema, EDA, dan aturan eksternal-data jelas. |
| **Kapan tidak digunakan** | Tidak digunakan untuk menggantikan local validation atau mengutip hasil tanpa metadata. |
| **Asumsi** | Query dibatasi oleh problem, data properties, metric, dan rules. |
| **Input** | Confirmed brief, structured profile, EDA findings, dan knowledge gap. |
| **Output** | Citation metadata, applicability notes, limitations, dan falsifiable experiment hypotheses. |
| **Dataset scale** | Agnostik; tidak melatih model. |
| **Metric compatibility** | Bukti harus menyebut objective/evaluation yang dapat dibandingkan dengan metric resmi. |
| **Decision variables** | Query, source tier, recency, inclusion/exclusion, evidence confidence. |
| **Default awal** | Primary paper dan official docs diprioritaskan; secondary source hanya discovery. |
| **CPU/GPU/RAM** | Ringan; network latency dominan. |
| **Risiko leakage** | Menggunakan public solution atau artifact yang dilarang; benchmark overlap; test-label contamination. |
| **Failure modes** | Citation palsu, paper tidak relevan, abstract-only overclaim, metode tanpa package/license. |
| **Teknik pembanding** | Existing knowledge retrieval dan manual bibliography review. |
| **Agent compatibility** | Mendukung model scout, feature engineer, validation, ensemble, dan robustness. |
| **Exact implementation** | `swarmfold.research.LiteratureResearchAgent`; simpan title, authors, year, URL/DOI, dan accessed date. |
| **Contoh hypothesis** | "Metode dari paper relevan meningkatkan metric karena karakteristik data X, bukan karena klaim umum." |
| **Acceptance criteria** | Sumber dapat dibuka, provenance lengkap, applicability dan limitation eksplisit, hipotesis dapat diuji. |
| **Rejection criteria** | Sumber tidak terlacak, rule conflict, atau klaim tidak dapat diubah menjadi eksperimen lokal. |
| **Sumber** | Repository research agent contract, diperiksa 2026-09-16. |

## 5. Experiment Memory Agent

**Role ID:** `experiment_memory_agent`

| Field | Detail |
|---|---|
| **Definisi** | Membaca package iterasi terdahulu dan membentuk memori tentang tried, failed, promoted, anomaly, dan unresolved hypotheses. |
| **Problem yang didukung** | Iterasi lanjutan pada dataset/kompetisi yang sama atau sangat terkait. |
| **Kapan digunakan** | Sebelum model scouting dan portfolio planning. |
| **Kapan tidak digunakan** | Tidak mentransfer hasil lama jika fingerprint, target, metric, atau split contract tidak kompatibel. |
| **Asumsi** | Package memiliki manifest, experiment ledger, findings, dan knowledge report. |
| **Input** | Full package terdahulu dan current contract fingerprint. |
| **Output** | Classified prior knowledge, duplicate hypotheses, reusable configs, dan stale/incompatible flags. |
| **Dataset scale** | Agnostik; batas ukuran archive dan file wajib diterapkan. |
| **Metric compatibility** | Score hanya dapat dibandingkan bila metric, direction, dan validation contract kompatibel. |
| **Decision variables** | Reuse, retry-with-change, reject-duplicate, stale, incompatible. |
| **Default awal** | Evidence lama informatif tetapi tidak otomatis promoted. |
| **CPU/GPU/RAM** | Ringan. |
| **Risiko leakage** | Prior package mengandung test labels, holdout feedback, atau evidence dari split berbeda. |
| **Failure modes** | Duplicate experiment diulang, stale champion dipakai, failed run hilang dari memory. |
| **Teknik pembanding** | Manifest/hash validation dan ledger diff. |
| **Agent compatibility** | Memberi constraint dan novelty context kepada scout dan planner. |
| **Exact implementation** | `swarmfold.roles.inspect_experiment_memory` dan package reader di `swarmfold.portfolio`. |
| **Contoh hypothesis** | "Planner tidak mengulang hypothesis fingerprint yang sama tanpa perubahan eksplisit." |
| **Acceptance criteria** | Semua prior claims memiliki source artifact dan compatibility verdict. |
| **Rejection criteria** | Archive invalid, lineage tidak lengkap, atau score dibandingkan lintas contract. |
| **Sumber** | Repository knowledge loop dan portfolio contract, diperiksa 2026-09-16. |

## 6. Portfolio Planner

**Role ID:** `portfolio_planner`

| Field | Detail |
|---|---|
| **Definisi** | Menyusun queue eksperimen yang beragam, dependency-aware, dan sesuai budget dari usulan specialist. |
| **Problem yang didukung** | Semua task setelah contract, EDA, research, memory, dan model coverage tersedia. |
| **Kapan digunakan** | Sebelum feature/model engineering dan setiap kali critic meminta iterasi lanjutan. |
| **Kapan tidak digunakan** | Tidak boleh mengarang estimator, package, atau preprocessing yang tidak ada di knowledge/registry. |
| **Asumsi** | User menentukan jumlah eksperimen per lane; tidak ada timeout lane implisit. |
| **Input** | Candidate methods, role charter lane, prior evidence, budget, dependency DAG. |
| **Output** | Ordered experiment queue dengan hypothesis, baseline/comparator, resources, dan gate. |
| **Dataset scale** | Menyesuaikan resource profile; sampling hanya eksperimen eksplisit, bukan shortcut tersembunyi. |
| **Metric compatibility** | Setiap proposal memakai metric contract yang sama atau menjelaskan surrogate. |
| **Decision variables** | Exploration/exploitation mix, diversity, dependency order, concurrency, ensemble candidates. |
| **Default awal** | Baseline sanity, strong native baseline, feature ablation, diverse families, lalu ensemble bila OOF tersedia. |
| **CPU/GPU/RAM** | Planning ringan; estimasi resource wajib. |
| **Risiko leakage** | Proposal memakai transform global atau mengoptimalkan final holdout. |
| **Failure modes** | Queue diisi variasi seed dangkal, model monoton, ensemble tanpa diversity, shortcut satu baseline. |
| **Teknik pembanding** | Coverage matrix, novelty check, dan equal-budget portfolio ablation. |
| **Agent compatibility** | Menggabungkan output model scout, knowledge feature engineering, validation, research, memory, dan lane behavior. Eksekusi `feature_engineer` tetap downstream setelah queue terbentuk. |
| **Exact implementation** | `swarmfold.roles.create_portfolio_node` dan `swarmfold.planner.ProposalPlanner`. |
| **Contoh hypothesis** | "Queue memenuhi jumlah user dan mencakup family, feature, tuning, serta ensemble diversity yang terukur." |
| **Acceptance criteria** | Semua slot terisi atau memiliki explicit infeasibility reason; tidak ada duplicate fingerprint. |
| **Rejection criteria** | Jumlah diam-diam dikurangi, proposal tanpa comparator/gate, atau unsupported estimator. |
| **Sumber** | Repository planner dan lane contract, diperiksa 2026-09-16. |

## 7. Experiment Runner

**Role ID:** `experiment_runner`

| Field | Detail |
|---|---|
| **Definisi** | Menjalankan proposal deterministik dan menyimpan metrics, OOF/test predictions, timing, memory, logs, dan artifacts. |
| **Problem yang didukung** | Proposal yang lolos contract, capability, dependency, dan resource gates. |
| **Kapan digunakan** | Setelah feature dan model pipeline telah dibangun. |
| **Kapan tidak digunakan** | Tidak mengeksekusi label metode `U`, proposal ambigu, atau dependency gagal. |
| **Asumsi** | Environment terpin, data immutable, seed dan folds tersedia. |
| **Input** | ExperimentProposal, prepared data, split manifest, estimator pipeline. |
| **Output** | RunResult lengkap, OOF/test predictions, resource telemetry, dan artifact hashes. |
| **Dataset scale** | Semua skala sesuai preflight memory/disk check; tidak ada batas 120 detik implisit. |
| **Metric compatibility** | Menghitung metric resmi dengan shape, class order, grouping, dan direction benar. |
| **Decision variables** | Parallel slot, retry policy, fail-fast condition, artifact retention. |
| **Default awal** | Satu retry hanya untuk failure transient; methodological failure tidak disamarkan. |
| **CPU/GPU/RAM** | Sesuai proposal; peak resource direkam. |
| **Risiko leakage** | Fit transform pada full data, early stopping pada final holdout, OOF row misalignment. |
| **Failure modes** | OOM, incompatible dtype, NaN prediction, class-order mismatch, artifact parsial. |
| **Teknik pembanding** | Minimal smoke fit, deterministic replay, prediction parity. |
| **Agent compatibility** | Hanya mengeksekusi; critic dan verifier menilai hasil. |
| **Exact implementation** | Entry point `swarmfold.nodes.run_experiment_candidate`; adapter yang dipilih wajib dicatat sebagai fully-qualified callable di `RunResult`. |
| **Contoh hypothesis** | "Konfigurasi dan split yang sama menghasilkan metric serta prediction hash dalam toleransi." |
| **Acceptance criteria** | Status terminal, metrics valid, OOF lengkap, artifacts/hash/log tersedia. |
| **Rejection criteria** | Row hilang, metric invalid, leakage gate gagal, atau artifact tidak dapat direplay. |
| **Sumber** | Repository execution node dan schemas, diperiksa 2026-09-16. |

## 8. Result Critic

**Role ID:** `result_critic`

| Field | Detail |
|---|---|
| **Definisi** | Menilai apakah hasil improved, regressed, failed, duplicate, atau memerlukan eksperimen diagnostik berikutnya. |
| **Problem yang didukung** | Semua hasil terminal dengan comparator dan validation evidence. |
| **Kapan digunakan** | Setelah setiap eksperimen selesai. |
| **Kapan tidak digunakan** | Tidak menilai berdasarkan satu score tanpa fold deltas, cost, dan failure context. |
| **Asumsi** | Comparator contract dan acceptance/rejection criteria sudah didefinisikan sebelum run. |
| **Input** | RunResult, proposal, comparator, fold metrics, resource usage, prior ledger. |
| **Output** | ExperimentCritique, verdict, findings, next action, dan new hypothesis bila justified. |
| **Dataset scale** | Agnostik. |
| **Metric compatibility** | Menghormati metric direction dan practical significance threshold. |
| **Decision variables** | Improve/regress/neutral/fail, promote/retry/stop, root-cause category. |
| **Default awal** | Score tie diputuskan oleh stability, cost, simplicity, lalu diversity contribution. |
| **CPU/GPU/RAM** | Ringan. |
| **Risiko leakage** | Winner dipilih setelah berulang kali melihat final holdout. |
| **Failure modes** | Hindsight hypothesis, cherry-pick fold, menghapus failed evidence, endless trivial iteration. |
| **Teknik pembanding** | Paired fold delta, confidence interval, learning curve, residual/error overlap. |
| **Agent compatibility** | Mengirim continue ke planner/feature path atau verify ke ensemble path. |
| **Exact implementation** | `swarmfold.roles.critique_result` dan `route_after_critique`. |
| **Contoh hypothesis** | "Gain kandidat konsisten pada mayoritas fold dan melebihi practical threshold dengan biaya wajar." |
| **Acceptance criteria** | Verdict ditopang comparator, fold evidence, cost, dan predeclared gate. |
| **Rejection criteria** | Verdict hanya berdasarkan best score atau post-hoc threshold. |
| **Sumber** | Repository critic and routing contract, diperiksa 2026-09-16. |

## 9. Evidence Verifier

**Role ID:** `evidence_verifier`

| Field | Detail |
|---|---|
| **Definisi** | Melakukan audit akhir atas lineage, validation, OOF, robustness, ensemble, citation, dan artifact reproducibility. |
| **Problem yang didukung** | Candidate champion atau ensemble sebelum submission. |
| **Kapan digunakan** | Setelah ensemble dan robustness assessment selesai. |
| **Kapan tidak digunakan** | Tidak memperbaiki hasil atau mengganti champion secara diam-diam. |
| **Asumsi** | Semua upstream artifacts immutable dan ber-hash. |
| **Input** | Selected results, ledgers, folds, predictions, assessments, citations, environment metadata. |
| **Output** | Verified evidence report, blockers, warnings, dan promotion verdict. |
| **Dataset scale** | Agnostik; dapat melakukan sampled artifact checks plus exact hashes. |
| **Metric compatibility** | Recompute metric dari predictions bila memungkinkan. |
| **Decision variables** | Verified, rejected, or human-review-required. |
| **Default awal** | Missing evidence adalah blocker, bukan diasumsikan benar. |
| **CPU/GPU/RAM** | Ringan sampai sedang untuk metric replay. |
| **Risiko leakage** | Verifier membaca test labels atau menerima artifact yang tidak sesuai split. |
| **Failure modes** | Hash tidak dicek, citation tidak dapat dibuka, metric tidak direcompute, warning disamarkan. |
| **Teknik pembanding** | Independent replay, schema validation, and manifest consistency checks. |
| **Agent compatibility** | Read-only consumer seluruh hasil upstream. |
| **Exact implementation** | `swarmfold.roles.verify_selected_evidence`. |
| **Contoh hypothesis** | "Metric champion dapat direcompute dari OOF predictions dan cocok dengan ledger dalam toleransi." |
| **Acceptance criteria** | Lineage lengkap, replay lulus, blockers kosong, warning material terlihat. |
| **Rejection criteria** | Artifact hilang, hash mismatch, leakage, unsupported claim, atau metric inconsistency. |
| **Sumber** | Repository evidence verification contract, diperiksa 2026-09-16. |

## 10. Submission Auditor

**Role ID:** `submission_auditor`

| Field | Detail |
|---|---|
| **Definisi** | Memastikan submission candidate mengikuti sample schema, row order, ID, target shape, range, dan approval policy. |
| **Problem yang didukung** | Kompetisi atau batch inference dengan output contract eksplisit. |
| **Kapan digunakan** | Setelah evidence verifier mempromosikan candidate. |
| **Kapan tidak digunakan** | Tidak membuat prediction baru atau mengubah nilainya untuk memperbaiki skor. |
| **Asumsi** | Sample submission atau explicit output schema tersedia. |
| **Input** | Test predictions, sample submission, submission contract, verified candidate ID. |
| **Output** | Audit report dan submission-ready artifact, masih menunggu human approval bila diwajibkan. |
| **Dataset scale** | Semua skala; streaming write dapat digunakan tanpa mengubah order. |
| **Metric compatibility** | Output transform harus sesuai metric/task, misalnya probability bukan hard label untuk ROC-AUC. |
| **Decision variables** | Column mapping, row order, dtype, clipping, serialization. |
| **Default awal** | Sample submission menjadi shape contract, bukan training label source. |
| **CPU/GPU/RAM** | Ringan. |
| **Risiko leakage** | Test labels tersisip, ID jadi feature akibat merge, sample target digunakan dalam training. |
| **Failure modes** | Wrong column, wrong class probability, duplicate/missing ID, NaN/Inf, reordered rows. |
| **Teknik pembanding** | Exact schema diff dan round-trip CSV read. |
| **Agent compatibility** | Downstream dari `evidence_verifier` dan upstream dari `publisher`. |
| **Exact implementation** | `swarmfold.roles.audit_submission`. |
| **Contoh hypothesis** | "Submission hasil round-trip mempertahankan row count, ID order, columns, dtype, dan prediction values." |
| **Acceptance criteria** | Semua schema/range/order checks lulus dan human approval state benar. |
| **Rejection criteria** | Mismatch sekecil apa pun pada contract atau candidate belum verified. |
| **Sumber** | Repository submission audit contract, diperiksa 2026-09-16. |

## 11. Publisher

**Role ID:** `publisher`

| Field | Detail |
|---|---|
| **Definisi** | Mengemas seluruh evidence menjadi satu full package immutable dan menghasilkan manifest checksum. |
| **Problem yang didukung** | Semua run terminal yang sudah lolos audit atau ditutup dengan failure report. |
| **Kapan digunakan** | Node akhir workflow. |
| **Kapan tidak digunakan** | Tidak menerbitkan status sukses ketika mandatory artifacts belum ada. |
| **Asumsi** | Artifact paths berada dalam run directory dan semua status terminal. |
| **Input** | Ledgers, findings, knowledge report, literature review, notebooks, submission, predictions, logs, audits. |
| **Output** | Satu full-package ZIP, manifest, final status, dan reusable `experiment-knowledge.md`. |
| **Dataset scale** | Packaging harus memakai size limits dan streaming untuk artifact besar. |
| **Metric compatibility** | Tidak menghitung ulang pemilihan; hanya menyimpan verified metric context. |
| **Decision variables** | Inclusion list, filenames, compression, manifest schema, package version. |
| **Default awal** | UI menampilkan satu tombol Download Full Package. |
| **CPU/GPU/RAM** | CPU/disk sedang; tidak membutuhkan GPU. |
| **Risiko leakage** | Secret/API key, raw private data, hidden test label, atau temp artifact masuk ZIP. |
| **Failure modes** | ZIP parsial, duplicate files, manifest stale, report tidak sinkron dengan ledger. |
| **Teknik pembanding** | Unzip-and-verify test, checksum replay, package schema validation. |
| **Agent compatibility** | Terminal consumer; output menjadi input Experiment Memory pada iterasi berikutnya. |
| **Exact implementation** | `swarmfold.packets.publish_terminal_packet` dan `swarmfold.portfolio`. |
| **Contoh hypothesis** | "Package dapat diekstrak dan diverifikasi tanpa state runtime asli serta memuat seluruh lineage." |
| **Acceptance criteria** | ZIP valid, manifest/hash cocok, knowledge report ada, tidak ada secret, semua required artifacts tercantum. |
| **Rejection criteria** | Artifact wajib hilang, checksum mismatch, secret terdeteksi, atau status tidak terminal. |
| **Sumber** | Repository packet dan portfolio contracts, diperiksa 2026-09-16. |

## Role Routing Summary

| Role | Knowledge tier | Primary corpus | Runtime inputs |
|---|---|---|---|
| `pod_supervisor` | Ringan | [Section 1](#1-pod-supervisor) | Run and lane configuration. |
| `rules_agent` | Ringan | [Section 2](#2-rules-agent) | Current project brief and user-confirmed rules. |
| `problem_formulation_agent` | Berat | [Problem formulation](problem-formulation-agent.md) | Confirmed rules and data profile. |
| `data_contract_agent` | Ringan | [Section 3](#3-data-contract-agent) | Human-confirmed structured profile. |
| `validation_agent` | Berat | [Validation](validation-agent.md) | Frozen data and metric contracts. |
| `temporal_validation_leakage_auditor` | Berat | [Temporal validation](temporal-validation-leakage-auditor.md) | Time/entity availability metadata. |
| `eda_agent` | Berat | [EDA](eda-agent.md) | Validated bundle and split constraints. |
| `domain_target_decomposition_agent` | Berat | [Target decomposition](domain-target-decomposition-agent.md) | EDA target diagnostics and domain brief. |
| `literature_research_agent` | Ringan | [Section 4](#4-literature-research-agent) | Retrieved primary sources under current rules. |
| `experiment_memory_agent` | Ringan | [Section 5](#5-experiment-memory-agent) | Prior full packages with compatible fingerprints. |
| `model_scout` | Berat | [Modeling](modeling-agents.md) | Task, EDA, literature, and prior memory. |
| `portfolio_planner` | Ringan | [Section 6](#6-portfolio-planner) | Outputs dari `validation_agent`, `eda_agent`, `domain_target_decomposition_agent`, `literature_research_agent`, `experiment_memory_agent`, dan `model_scout`, plus lane budget. Output `feature_engineer` dan `model_engineer` belum tersedia pada tahap ini. |
| `feature_engineer` | Berat | [Feature engineering](feature-engineering-agent.md) | Selected proposal and fold contract. |
| `model_engineer` | Berat | [Modeling](modeling-agents.md) | Selected proposal and engineered feature contract. |
| `experiment_runner` | Ringan | [Section 7](#7-experiment-runner) | Concrete estimator adapter and split manifest. |
| `result_critic` | Ringan | [Section 8](#8-result-critic) | Run result, comparator, and robustness criteria. |
| `oof_ensemble_scientist` | Berat | [Ensemble](ensemble-agent.md) | Aligned OOF predictions and diversity evidence. |
| `robustness_uncertainty_agent` | Berat | [Robustness](robustness-uncertainty-agent.md) | Candidate results and ensemble assessment. |
| `evidence_verifier` | Ringan | [Section 9](#9-evidence-verifier) | Immutable ledgers, predictions, citations, and audits. |
| `submission_auditor` | Ringan | [Section 10](#10-submission-auditor) | Verified candidate and submission contract. |
| `publisher` | Ringan | [Section 11](#11-publisher) | All terminal artifacts and audit reports. |
