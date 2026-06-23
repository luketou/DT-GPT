# PICU DT-GPT Research Workflow Notes

Date: 2026-06-13

## Context

This note records the research discussion about using DT-GPT and LLMs for
trajectory forecasting in ICU and PICU data.

The current repository is the DT-GPT project for patient health trajectory
forecasting. Existing evidence in the repo shows:

- DT-GPT already targets patient trajectory forecasting with LLMs.
- The repo contains MIMIC-IV ICU experiments, critical care experiments, ADNI
  experiments, copy-forward baselines, Darts models, Time-LLM, LLMTime,
  BioMistral, and DT-GPT instruction-based experiments.
- Recent MIMIC work includes BioMistral-7B-DARE, DoRA/LoRA/QLoRA-style
  adaptation, Unsloth, vLLM evaluation, shard evaluation, and paper-style
  metric clarification.
- Metric interpretation is important: patient-level prediction R2, step-level
  scaled MAE, and paper-style correlation-preservation R2 are different and
  should not be mixed.

## User's Updated Research Assets

The user has access to a private hospital PICU dataset.

Important properties of this dataset:

- It is hospital-specific and not commonly available in the field.
- It contains rare PICU cases.
- It includes values that can look like statistical outliers but may be
  clinically plausible.
- These clinically plausible outliers may carry important trajectory signals
  rather than being simple data errors.

The user is considering two research directions:

- Use LLMs or DT-GPT to solve PICU trajectory forecasting.
- Optimize the current BioMistral-7B-DARE DT-GPT architecture because every
  new data batch requiring full fine-tuning is too slow and wasteful.

## Research Direction Evaluation

The strongest direction is not to treat these as two separate papers.

Recommended framing:

> Efficient Continual LLM Adaptation for Rare-but-Clinically-Valid PICU
> Trajectory Forecasting

Core claim:

> PICU trajectory forecasting contains clinically plausible extreme values that
> naive statistical preprocessing may remove as outliers. At the same time,
> hospital data arrives over time, and full fine-tuning BioMistral-7B-DARE or
> DT-GPT after every update is inefficient. The research goal is to build a
> clinically aware, parameter-efficient, continually adaptable LLM forecasting
> workflow that preserves rare-but-real trajectories while reducing update
> cost.

## Recommended Paper Positioning

Avoid this weak framing:

> We use BioMistral-7B-DARE to forecast PICU trajectories.

Prefer this stronger framing:

> We study PICU trajectory forecasting under clinically valid statistical
> outliers and propose a parameter-efficient continual adaptation framework
> that updates DT-GPT without full retraining while preserving rare but
> clinically meaningful trajectories.

Suggested title:

> Rare-but-Real: Efficient Continual LLM Adaptation for Clinically Plausible
> Outlier Trajectory Forecasting in Pediatric ICU

## Main Contributions

### 1. Problem Contribution

Define the PICU trajectory forecasting problem where some values are
statistical outliers but clinically plausible events.

This is important because:

- Standard preprocessing may delete clinically meaningful extreme cases.
- Average forecasting metrics may hide failure on rare high-risk trajectories.
- PICU differs from adult ICU and oncology trajectories because rare acute
  events may dominate clinical relevance.

### 2. Method Contribution

Use BioMistral-7B-DARE or DT-GPT with parameter-efficient adaptation instead of
repeated full fine-tuning.

Candidate methods:

- LoRA
- DoRA
- QLoRA
- Adapter replay
- Incremental adapter update
- Rare-case weighted fine-tuning
- Adapter merging or adapter routing by data batch, cohort, or clinical
  pattern

### 3. Evaluation Contribution

Evaluate not only average prediction quality but also rare-case robustness and
update efficiency.

Important metrics:

- Step-level scaled MAE
- Paper-style correlation-preservation R2
- Per-variable MAE and RMSE
- Normal-range performance
- Clinically plausible outlier performance
- Rare/extreme case error
- JSON validity and parse rate
- Update time
- GPU hours
- Forgetting on previous data after incremental updates

## Verdict

Strong version:

> Strong Accept if framed as PICU rare clinical trajectory forecasting plus
> efficient continual adaptation.

Medium version:

> Accept with revisions if framed only as BioMistral or DT-GPT fine-tuning on a
> private PICU dataset.

Weak version:

> Weak if framed only as optimizing fine-tuning speed, because LoRA, QLoRA, and
> continual PEFT already exist and the clinical contribution would be lost.

## Fatal Risks

### Risk 1: Novelty Risk

Simply saying "use LLMs for ICU or PICU trajectory forecasting" is too weak.
DT-GPT and related LLM patient trajectory work already exist.

Defense:

- Emphasize clinically plausible outliers.
- Emphasize PICU rarity and clinical validity.
- Emphasize continual adaptation and efficiency under incoming hospital data.

### Risk 2: Private Dataset Risk

Reviewers may say the dataset is private and cannot be reproduced.

Defense:

- Provide a MIMIC-IV sanity check or external validation where possible.
- Describe the PICU cohort and preprocessing transparently without exposing
  protected data.
- Release code, synthetic examples, and metric scripts.

### Risk 3: Metric Risk

Average metrics may hide failure on rare clinical extremes.

Defense:

- Use stratified evaluation.
- Separate normal-range cases, clinically plausible extreme cases, and invalid
  data errors.
- Do not compare incompatible R2 definitions.

### Risk 4: Clinical Interpretation Risk

AI cannot decide whether an outlier is clinically plausible.

Defense:

- Use clinician-defined plausibility rules.
- Record variable-specific acceptable ranges and exception conditions.
- If possible, include clinician annotation or review for a sample of extreme
  cases.

## Batch Goals

| Batch | Goal | Success standard | Stop condition |
|---|---|---|---|
| Batch 0 | Define research problem | One-sentence claim, target cohort, target variables, forecast horizon, and metrics are fixed | The contribution can be explained in one minute |
| Batch 1 | PICU dataset audit | Missingness, variable distribution, outlier profile, and clinically plausible extreme cases are summarized | Error outliers and clinically plausible outliers can be separated |
| Batch 2 | Baseline suite | Copy-forward, simple statistical baseline, Darts/Time-LLM/LLMTime, BioMistral/DT-GPT baseline are runnable | First result table exists |
| Batch 3 | DT-GPT PICU adaptation | BioMistral-7B-DARE plus LoRA/DoRA/QLoRA can train and evaluate on PICU data | Stable checkpoint and reproducible eval script exist |
| Batch 4 | Efficient update experiment | Full fine-tune, adapter update, and incremental adapter update are compared | Update time or GPU cost decreases without major performance collapse |
| Batch 5 | Rare-case robustness | Normal cases and clinically valid outlier cases are evaluated separately | The method is not only optimized for average MAE |
| Batch 6 | Paper figures | Figure 1, method overview, main result figure, and rare-case analysis figure are drafted | Each figure has one clear message |
| Batch 7 | Writing | Introduction, Method, Experiments, and Discussion have first drafts | Every claim maps to experiment output or confirmed literature |

## Immediate Two-Week Plan

| Day | Task |
|---|---|
| Day 1 | Write `research_claim.md`: one-sentence claim, three contributions, and three main experiments |
| Day 2-3 | Audit PICU data: missingness, variable distributions, extreme values, clinically plausible labels |
| Day 4-5 | Run minimal copy-forward and simple statistical baselines |
| Day 6-7 | Convert PICU samples into DT-GPT input/output text and manually inspect 20 examples |
| Day 8-10 | Run 100-500 patient BioMistral LoRA/DoRA smoke training |
| Day 11 | Run evaluation and inspect parse rate, MAE, and rare-case error |
| Day 12 | Build full fine-tune vs LoRA/DoRA estimated cost table |
| Day 13 | Draft Figure 1: how naive filtering removes clinically valid outliers |
| Day 14 | Write a one-page advisor memo |

## Weekly Research Workflow

| Time block | Phase | Activity | Tool | User check |
|---|---|---|---|---|
| Monday AM | Research planning | Update weekly hypothesis, experiment list, and expected outcome | Codex or ChatGPT | Confirm the week answers only one or two research questions |
| Monday PM-Tuesday | Coding | Implement preprocessing, adapter training, and evaluation scripts | Codex or Cursor | Run smoke tests for each script |
| Wednesday | Experiment | Run baseline, LoRA, DoRA, or incremental update jobs | Slurm and logs | Save config, seed, checkpoint, and metrics |
| Thursday AM | Analysis | Summarize metrics, failure cases, and rare outlier examples | Python, Pandas, Matplotlib | Trace every number to raw output |
| Thursday PM | Figure | Plot results and draft workflow diagrams | Matplotlib, Figma, PowerPoint | Use only real experiment data |
| Friday | Writing | Write weekly research memo | Markdown, Overleaf, AI polish | User writes skeleton first, AI only polishes language |
| Weekend | Advisor-ready review | Produce a one-page progress summary | Markdown or PDF | Next week's question is explicit |

## Figure Plan

### Figure 1: Motivating Example

Message:

> A statistically extreme PICU value can be clinically plausible and should not
> automatically be removed.

Panels:

- Patient trajectory with normal values and one rare extreme value.
- Naive statistical filter marks the value as an outlier.
- Clinical plausibility rule preserves it.
- Forecasting model with and without the value produces different predictions.

### Figure 2: Method Overview

Message:

> The system updates a frozen BioMistral/DT-GPT base model with efficient
> adapters rather than repeated full fine-tuning.

Panels:

- Incoming PICU data batch.
- Clinical plausibility-aware preprocessing.
- Text conversion.
- Adapter update.
- Evaluation on normal and rare-case subsets.

### Figure 3: Main Result

Message:

> Efficient adapter updates approach full fine-tuning performance at much lower
> update cost.

Suggested plot:

- x-axis: update method.
- y-axis left: forecasting error.
- y-axis right or separate panel: GPU hours or wall-clock time.

### Figure 4: Rare-Case Robustness

Message:

> Average metrics hide clinically important failures on rare trajectories.

Suggested plot:

- Normal cases vs clinically plausible outliers.
- Compare copy-forward, standard DT-GPT, and clinically aware efficient
  adaptation.

## Tool Recommendations

| Phase | Primary tool | Alternative | Reason |
|---|---|---|---|
| Coding | Codex | Cursor | Current repo is already in a Codex workspace and needs script/pipeline work |
| Experiment tracking | Markdown plus WandB/logs | MLflow | Keep tracking lightweight but reproducible |
| Literature | Google Scholar, arXiv, Zotero | Semantic Scholar | AI may organize, but user must verify citations |
| Figures | Matplotlib and Seaborn | Figma or PowerPoint | Result figures must be generated from real outputs |
| Writing | User skeleton plus AI polish | Grammarly | AI should polish wording, not own scientific claims |

## AI-Assisted Research Rules

- AI can help with literature organization, coding, debugging, and language
  polish.
- Research ideas, problem framing, method choice, experimental design, novelty
  claims, and conclusions must remain the user's own.
- Every AI-generated code block must be read and tested.
- Every AI-assisted paragraph must be verified sentence by sentence.
- AI must not fabricate citations.
- AI must not fabricate data, results, or procedures.
- Venue and school AI-disclosure rules must be checked before submission.

## Red-Line Reminders

- Do not let AI decide whether a PICU outlier is clinically plausible.
- Do not present private hospital data results as universal unless supported by
  external validation.
- Do not report only average MAE or R2.
- Do not paste AI-generated related work or citations directly into the paper.
- Do not claim full fine-tuning inefficiency as the sole novelty.
- The novelty is clinical rare-event retention under efficient continual
  adaptation.

## First Deliverable

The next concrete deliverable should be:

> `research_claim.md`

It should contain:

1. One-sentence paper claim.
2. Three contribution bullets.
3. Target cohort and target variables.
4. Forecast horizon.
5. Main metrics.
6. Baselines.
7. Three decisive experiments.

This should be completed before launching more large-scale training jobs.
