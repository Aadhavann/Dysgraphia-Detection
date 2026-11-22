# Dataset provenance and ethics

## Source

The images under `data/` are the *Potential Dysgraphia Handwriting Dataset of
School-Age Children*, redistributed here unmodified.

> Ramlan, S. A. (2023). *Potential Dysgraphia Handwriting Dataset of School-Age
> Children* [Data set]. Mendeley Data. V1.
> https://doi.org/10.17632/39hr8dx76p.1

| | |
| --- | --- |
| Publisher | Mendeley Data |
| Version | V1 |
| Year | 2023 |
| Licence | CC BY 4.0 |
| Licence URI | https://creativecommons.org/licenses/by/4.0/ |
| DOI | 10.17632/39hr8dx76p.1 |

## Licence obligations

CC BY 4.0 permits redistribution and derivative use provided attribution is
given, a link to the licence is included, and any changes are indicated. The
citation above is the required attribution. The images in this repository are
**unmodified**, so no change indication is needed.

The MIT licence covering the code in this repository does **not** extend to
`data/`. The dataset remains under CC BY 4.0 and copyright its original
authors.

If you fork this repository, keep this file and the attribution in the README.

## Contents

```
data/
  Low Potential Dysgraphia/    135 images, label 0, filenames LPD (n).jpg
  Potential Dysgraphia/        114 images, label 1, filenames PD (n).jpg
```

249 scanned notebook pages in total.

## How the labels were assigned

Per the dataset's own description, the pages were collected from primary school
children and from children undertaking interventions at the Malaysian Dyslexia
Association (PDM). Two PDM assessors then evaluated each sample and assigned it
to either *potential dysgraphia* or *low potential dysgraphia*.

The labels are therefore expert screening judgements, not clinical diagnoses.
That is consistent with how this repository frames its output: a screening
signal that flags a child for professional assessment, never a diagnosis.

## What the dataset does not provide

- **Writer identifiers.** Each file is one page and the filenames are a
  sequential index, not a writer code. There is consequently no way to group
  pages by the child who wrote them. See the first item under Limitations in the
  README: the split in this repository is subject-grouped in code but degenerates
  to a per-image split on this data, and says so at run time.
- **Age, sex, handedness or comorbidity.** No covariates accompany the images,
  so neither stratified analysis nor confounder adjustment is possible.
- **Severity grades.** Labelling is binary.
- **Collection protocol.** Pen, paper, scanner, resolution and lighting are not
  documented, so the sensitivity of the metrics to acquisition conditions cannot
  be quantified from this release.

## Ethics

The data consists of handwriting produced by children with and without a
learning disability. Three points follow for anyone reusing it.

The images are de-identified: no names or personal details appear in the files
or in this repository. Do not attempt to re-identify any contributor, and do not
add identifying metadata if you derive new artefacts from these images.

Figures generated from this dataset, including the case-study panels under
`results/`, show individual children's handwriting. They are appropriate in a
research context because the underlying release is open under CC BY 4.0, but
they should not be used in promotional or commercial material.

The labels reflect screening judgements rather than diagnoses, and the
classifier trained on them inherits that ceiling. A positive prediction means
the page resembles other pages that assessors flagged as potential dysgraphia.
It is not evidence that any particular child has dysgraphia, and it must not be
presented that way.
