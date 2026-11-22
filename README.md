# Image-Based Dysgraphia Detection with Explainability Using CAMs and EDI

Reference implementation of the Explainable Dysgraphia Index (EDI), a screening
framework that pairs a convolutional classifier with an interpretable,
attention-weighted handwriting score.

## Publication

> V. Sreekumar, A. P. Anbumunee and S. Senthilkumar, "Image-Based Dysgraphia
> Detection with Explainability Using CAMs and EDI," *2026 International
> Conference on Computing, Communication, Control and Cyber-Physical Systems
> (I5CPS)*, NIT Puducherry, 26 February 2026, pp. 1-6.

- IEEE Xplore: https://ieeexplore.ieee.org/document/11452617/
- DOI: [10.1109/I5CPS67958.2026.11452617](https://doi.org/10.1109/I5CPS67958.2026.11452617)

The paper reports that the CNN and EDI fusion reached 92% accuracy and 0.98 AUC.
Those figures are the published result and are aggregated over multiple training
runs. This repository does not claim that any single committed run reproduces
them; see [Evaluation protocol](#evaluation-protocol) and
[Reproducing the results](#reproducing-the-results).

## Method

Dysgraphia is a learning disability that affects handwriting quality and is
normally identified by manual observation, which is subjective, or by
specialist tablet hardware, which does not scale. This framework works from
ordinary scanned notebook pages.

```
scanned page -> ResNet-18 -> Score-CAM -> attention-weighted metrics -> EDI -> fusion
```

Four metrics are taken from occupational-therapy handwriting assessment:

| Metric | What it measures |
| --- | --- |
| `spacing_irregularity` | dispersion of horizontal gaps between connected components |
| `baseline_wobble` | residual variance of text-line peak positions about a straight-line fit |
| `char_size_variance` | mean of bounding-box height and width variance |
| `stroke_width_std` | dispersion of the distance transform over ink pixels |

Each metric is computed twice: once over the whole page and once over the page
weighted by the Score-CAM activation map. The Explainable Dysgraphia Index
combines the attention-weighted values, in an equal-weight form that is directly
readable against clinical criteria and a CAM-weighted form used for fusion with
the classifier's own probability.

Score-CAM is used rather than Grad-CAM because it is gradient-free and produces
more consistent heatmaps across runs. Grad-CAM remains available via `--cam
gradcam` for fast iteration; it is roughly two orders of magnitude cheaper
because it needs no masked forward passes.

## Repository layout

```
edi/
  config.py       every tunable in one dataclass, so a config identifies a run
  data.py         dataset loading and subject-grouped splitting
  model.py        ResNet-18 / EfficientNet-B0 backbone with CAM hooks
  cam.py          Grad-CAM, Score-CAM, and measured heatmap stability
  features.py     handwriting metrics and EDI construction
  evaluate.py     out-of-fold evaluation of every head
  visualize.py    figures
  train.py        training loop with early stopping on validation AUC
  pipeline.py     orchestration
run.py            command-line entry point
notebooks/        walkthrough of the pipeline stage by stage
tests/            regression tests, PyTorch not required
data/             the dataset (see Dataset below)
results/          figures, tables and run manifests, regenerated on each run
```

## Dataset

The images under `data/` are the *Potential Dysgraphia Handwriting Dataset of
School-Age Children*:

> Ramlan, S. A. (2023). *Potential Dysgraphia Handwriting Dataset of School-Age
> Children* [Data set]. Mendeley Data. V1.
> https://doi.org/10.17632/39hr8dx76p.1

Redistributed under the dataset's **CC BY 4.0** licence, which requires
attribution; the citation above satisfies it. If you reuse the images, carry
that attribution forward.

249 scanned pages in total: 135 labelled low-potential dysgraphia (control) and
114 labelled potential dysgraphia. Labels were assigned by two assessors at the
Malaysian Dyslexia Association. See [DATASET.md](DATASET.md) for the full
provenance and ethics note.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9 or newer. A CUDA device is strongly recommended: Score-CAM performs
`scorecam_channels` masked forward passes per image.

## Usage

Command line:

```bash
python run.py                              # full study, Score-CAM, two seeds
python run.py --cam gradcam                # much faster, for iteration
python run.py --epochs 20 --seed 7
python run.py --stability-runs 1           # skip the CAM stability measurement
python run.py --data-dir /path/to/data --results-dir ./results
```

Python:

```python
from edi import PipelineConfig, run_pipeline

outcome = run_pipeline(PipelineConfig(data_dir="data"), cam_method="scorecam")
print(outcome["summary_table"])
print(outcome["stability"])
```

Google Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/Dysgraphia-Detection
!pip install -q -r requirements.txt
!python run.py --epochs 50 --stability-runs 2
```

Each run writes `results/edi_features.csv`, `results/results_table.csv`,
`results/results_summary.txt`, `results/run_manifest.json`, the trained weights,
and every figure. The manifest records the full config, the environment, the
partition sizes and all metrics, so a run is identifiable from the file alone.

## Evaluation protocol

Every number this repository reports is computed on the held-out test partition,
and no head is scored against a label it was fitted to.

- `cnn_only` uses the classifier's stored softmax probabilities. Nothing is
  fitted, so there is nothing to leak.
- `features_only`, `cam_features_only`, `edi_equal`, `edi_weighted` and
  `cnn_edi_fusion` are fitted with `StratifiedKFold` `cross_val_predict`. Every
  probability used for scoring comes from a fold in which that row was held out.
- Feature standardisation happens inside each fold's pipeline. EDI construction
  standardises the metric columns without reference to labels, so it is
  unsupervised and is verified as such by `test_construct_edi_ignores_labels`.
- AUC confidence intervals are 95% percentile bootstraps over 2000 resamples.
  With a test partition of about fifty images, a two-decimal AUC quoted without
  an interval overstates the precision available.

`tests/test_evaluate.py` exists specifically to hold this line. Fitting and
scoring on the same rows reaches a perfect 1.000 AUC on sixty columns of pure
noise, which is how an in-sample fit can be mistaken for a generalisation
result; `test_in_sample_would_have_leaked` demonstrates that failure and
`test_noise_features_do_not_score_high` asserts this implementation does not
reproduce it.

## Reproducing the results

Run `python run.py` and read `results/results_summary.txt`. The figures produced
are this repository's own out-of-fold measurements on one particular partition,
and they will differ from the aggregates reported in the paper: the paper's
numbers summarise multiple runs, while a single run on roughly fifty test images
carries a confidence interval wide enough to span several points of AUC.

Both are reported rather than reconciled, because they answer different
questions. The paper's aggregate is the published claim. The committed run
manifest is what this code produced, on this seed, in this environment.

The figures currently under `results/` predate the evaluation correction above
and are replaced on the next full run.

## Limitations

These are properties of the method and the data, recorded here so that nobody
has to rediscover them.

- **The split is not writer-independent in practice.** The dataset publishes no
  writer identifiers, so `load_dataset` derives a subject key from the filename
  and every page becomes its own subject. The splitting code groups by subject
  correctly and will honour real writer identities through
  `load_dataset(..., writer_map=...)`, but with this dataset the partition is
  equivalent to a random per-image split. `SplitReport` detects the condition
  and prints an explicit warning rather than describing the split as
  writer-independent.
- **The test partition is small.** Roughly fifty images. Confidence intervals
  are reported for this reason.
- **Binary screening only.** The output distinguishes potential from
  low-potential dysgraphia. It does not grade severity.
- **Single script and language.** The dataset is handwriting from
  Malay-speaking school children. Generalisation to other scripts is untested.
- **Screening, not diagnosis.** The index is designed to flag children for
  assessment by a qualified practitioner. It is not a diagnostic instrument and
  should not be used as one.

## Tests

```bash
python tests/test_evaluate.py
python tests/test_features.py
```

Both run without PyTorch installed. `tests/test_features.py` exercises the
metric extractors against the real dataset images when `data/` is present, and
falls back to synthetic pages otherwise.

## Licence

The code in this repository is released under the MIT Licence; see
[LICENSE](LICENSE). The dataset under `data/` is not covered by that licence and
remains CC BY 4.0, copyright its original authors.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{sreekumar2026image,
  title     = {Image-Based Dysgraphia Detection with Explainability Using {CAMs} and {EDI}},
  author    = {Sreekumar, Varsha and Anbumunee, Aadhavan P and Senthilkumar, Sudha},
  booktitle = {2026 International Conference on Computing, Communication, Control
               and Cyber-Physical Systems (I5CPS)},
  pages     = {1--6},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/I5CPS67958.2026.11452617}
}
```

And the dataset:

```bibtex
@dataset{ramlan2023potential,
  author    = {Ramlan, Siti Azura},
  title     = {Potential Dysgraphia Handwriting Dataset of School-Age Children},
  year      = {2023},
  publisher = {Mendeley Data},
  version   = {V1},
  doi       = {10.17632/39hr8dx76p.1},
  url       = {https://data.mendeley.com/datasets/39hr8dx76p/1}
}
```
