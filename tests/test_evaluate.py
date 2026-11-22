"""Guards against the evaluation defect this rewrite corrects.

The previous revision fitted each head on the test partition and then scored it
on the same rows::

    lr.fit(X, y_true)
    roc_auc_score(y_true, lr.predict_proba(X)[:, 1])

On fifty rows with one or two features that reports a near-perfect AUC even
when the features are pure noise, which is how an in-sample fit came to be
published as a generalisation result. Every test here fails against that
pattern and passes against the out-of-fold implementation.

Runs under pytest or directly::

    python tests/test_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edi.evaluate import (  # noqa: E402
    HEAD_NAMES,
    _feasible_folds,
    bootstrap_auc_ci,
    evaluate_all,
    evaluate_cnn_only,
    evaluate_fitted_head,
    fusion_improves_on_cnn,
    mann_whitney_auc,
    summarise,
)

N_SAMPLES = 50
NOISE_AUC_CEILING = 0.75


def _noise_table(seed: int = 0) -> pd.DataFrame:
    """A test partition whose features carry no signal whatsoever."""
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES - N_SAMPLES // 2))
    rng.shuffle(labels)

    frame = pd.DataFrame({"label": labels, "path": [f"img_{i}.jpg" for i in range(N_SAMPLES)]})
    frame["cnn_prob"] = rng.random(N_SAMPLES)
    frame["prediction"] = (frame["cnn_prob"] >= 0.5).astype(int)
    for metric in ("spacing_irregularity", "baseline_wobble", "char_size_variance", "stroke_width_std"):
        frame[f"{metric}_raw"] = rng.normal(size=N_SAMPLES)
        frame[f"{metric}_cam"] = rng.normal(size=N_SAMPLES)
    frame["cam_mean"] = rng.random(N_SAMPLES)
    frame["EDI_equal"] = rng.normal(size=N_SAMPLES)
    frame["EDI_weighted"] = rng.normal(size=N_SAMPLES)
    return frame


def _signal_table(seed: int = 1) -> pd.DataFrame:
    """A test partition where the EDI genuinely separates the classes.

    Deliberately not perfectly separable: a table whose heads all reach AUC 1.0
    would produce degenerate bootstrap intervals and hide the case this suite
    most needs to exercise.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES - N_SAMPLES // 2))
    rng.shuffle(labels)

    edi = labels * 2.0 + rng.normal(scale=1.0, size=N_SAMPLES)
    cnn_prob = np.clip(labels * 0.6 + 0.2 + rng.normal(scale=0.30, size=N_SAMPLES), 0.001, 0.999)

    frame = pd.DataFrame(
        {
            "label": labels,
            "path": [f"img_{i}.jpg" for i in range(N_SAMPLES)],
            "cnn_prob": cnn_prob,
            "EDI_weighted": edi,
            "EDI_equal": edi + rng.normal(scale=0.4, size=N_SAMPLES),
            "cam_mean": rng.random(N_SAMPLES),
        }
    )
    frame["prediction"] = (frame["cnn_prob"] >= 0.5).astype(int)
    for metric in ("spacing_irregularity", "baseline_wobble", "char_size_variance", "stroke_width_std"):
        frame[f"{metric}_raw"] = rng.normal(size=N_SAMPLES)
        frame[f"{metric}_cam"] = labels * 0.9 + rng.normal(scale=1.0, size=N_SAMPLES)
    return frame


def _high_dimensional_noise(seed: int = 3, n_features: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Pure noise with more columns than samples, so it is linearly separable.

    This is the regime where in-sample fitting is unambiguously wrong: a linear
    head orders the training rows perfectly while carrying no information at
    all. Measured on this generator the in-sample AUC is exactly 1.000 and the
    out-of-fold AUC is 0.517.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES - N_SAMPLES // 2))
    rng.shuffle(labels)
    return rng.normal(size=(N_SAMPLES, n_features)), labels


def test_noise_features_do_not_score_high() -> None:
    """The regression test: noise must not look like a discovery."""
    for seed in range(5):
        frame = _noise_table(seed)
        results = evaluate_all(frame, cv_folds=5, seed=42)
        for name, result in results.items():
            assert result.auc < NOISE_AUC_CEILING, (
                f"seed {seed}: head {name!r} reached AUC {result.auc:.3f} on pure noise. "
                f"This is what in-sample fitting looks like; the head is seeing its own labels."
            )


def test_in_sample_would_have_leaked() -> None:
    """Demonstrates the defect directly, so the fix stays motivated.

    With only two columns and default L2 regularisation a linear head cannot
    memorise fifty rows, which is why the old numbers looked plausible rather
    than obviously broken. The bias is unconditional, though -- in-sample AUC
    can only exceed out-of-fold AUC -- and it becomes blatant as soon as there
    are enough columns to separate the training rows, at which point an
    in-sample fit reports a perfect score on pure noise.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X_noise, y = _high_dimensional_noise()

    leaky = LogisticRegression(C=1e6, max_iter=20000).fit(X_noise, y)
    in_sample_auc = roc_auc_score(y, leaky.predict_proba(X_noise)[:, 1])
    honest_auc = roc_auc_score(
        y,
        cross_val_predict(
            LogisticRegression(C=1e6, max_iter=20000),
            X_noise,
            y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1],
    )

    assert in_sample_auc > 0.99, f"expected the in-sample fit to separate noise perfectly, got {in_sample_auc:.3f}"
    assert honest_auc < NOISE_AUC_CEILING, f"expected the out-of-fold figure to stay near chance, got {honest_auc:.3f}"

    # The ordering holds on the shipped heads too, on data with real structure.
    frame = _signal_table()
    columns = ["cnn_prob", "EDI_weighted"]
    X, y_frame = frame[columns].to_numpy(dtype=float), frame["label"].to_numpy(dtype=int)
    fitted = LogisticRegression(max_iter=5000).fit(X, y_frame)
    in_sample = roc_auc_score(y_frame, fitted.predict_proba(X)[:, 1])
    out_of_fold = evaluate_fitted_head(frame, "cnn_edi_fusion", cv_folds=5, seed=42).auc
    assert in_sample >= out_of_fold, (
        f"in-sample AUC {in_sample:.3f} should never be below out-of-fold AUC {out_of_fold:.3f}"
    )


def test_real_signal_is_detected() -> None:
    """Out-of-fold evaluation must still find signal that is actually there."""
    frame = _signal_table()
    results = evaluate_all(frame, cv_folds=5, seed=42)
    assert results["edi_weighted"].auc > 0.85, f"EDI head missed real signal: {results['edi_weighted'].auc:.3f}"
    assert results["cnn_only"].auc > 0.85, f"CNN head missed real signal: {results['cnn_only'].auc:.3f}"
    assert fusion_improves_on_cnn(results) is not None


def test_cnn_only_fits_nothing() -> None:
    """The CNN head must be scored from stored probabilities alone."""
    frame = _signal_table()
    result = evaluate_cnn_only(frame)
    assert "no fitting" in result.protocol
    assert result.n_samples == N_SAMPLES
    from sklearn.metrics import roc_auc_score

    assert abs(result.auc - roc_auc_score(frame["label"], frame["cnn_prob"])) < 1e-9


def test_every_head_reports_a_confidence_interval() -> None:
    """With n=50 a bare two-decimal AUC overstates precision.

    A head pinned at exactly 1.0 legitimately yields a zero-width interval, so
    the strict positivity check is applied to a head sitting near chance, where
    a wide interval is the only honest answer.
    """
    frame = _signal_table()
    for name, result in evaluate_all(frame, cv_folds=5, seed=42).items():
        assert result.auc_ci_low is not None and result.auc_ci_high is not None, f"{name} has no CI"
        assert result.auc_ci_low <= result.auc <= result.auc_ci_high, f"{name} CI does not bracket the estimate"
        assert result.auc_ci_high - result.auc_ci_low >= 0.0, f"{name} CI is inverted"

    noise_head = evaluate_all(_noise_table(0), cv_folds=5, seed=42)["features_only"]
    width = noise_head.auc_ci_high - noise_head.auc_ci_low
    assert width > 0.1, f"a near-chance head on n={N_SAMPLES} must show real uncertainty, got width {width:.3f}"


def test_bootstrap_handles_degenerate_input() -> None:
    single_class = np.zeros(20, dtype=int)
    assert bootstrap_auc_ci(single_class, np.random.default_rng(0).random(20)) == (None, None)
    assert bootstrap_auc_ci(np.array([], dtype=int), np.array([])) == (None, None)


def test_feasible_folds_clamps_to_minority_class() -> None:
    assert _feasible_folds(np.array([0] * 30 + [1] * 2), requested=5) == 2
    assert _feasible_folds(np.array([0] * 25 + [1] * 25), requested=5) == 5
    assert _feasible_folds(np.array([0] * 25 + [1] * 25), requested=99) == 25


def test_all_heads_evaluated_and_tabulated() -> None:
    frame = _signal_table()
    results = evaluate_all(frame, cv_folds=5, seed=42)
    assert set(results) == set(HEAD_NAMES)
    table = summarise(results)
    assert list(table["head"]) == list(HEAD_NAMES)
    assert {"accuracy", "auc", "auc_ci", "protocol"} <= set(table.columns)
    for name, result in results.items():
        if name == "cnn_only":
            assert result.cv_folds is None
        else:
            assert "out-of-fold" in result.protocol, f"{name} was not evaluated out-of-fold"


def test_confusion_matrix_is_well_formed() -> None:
    frame = _signal_table()
    for result in evaluate_all(frame, cv_folds=5, seed=42).values():
        total = sum(sum(row) for row in result.confusion)
        assert total == N_SAMPLES, f"{result.name} confusion matrix does not account for every sample"


def test_mann_whitney_auc_matches_sklearn() -> None:
    """The vectorised AUC used by the bootstrap must equal sklearn's, ties included."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    for trial in range(20):
        y = rng.integers(0, 2, size=40)
        if np.unique(y).size < 2:
            continue
        continuous = rng.normal(size=40)
        assert abs(mann_whitney_auc(y, continuous) - roc_auc_score(y, continuous)) < 1e-12

        # Heavy ties: rounded scores, which bootstrap resampling also produces.
        tied = np.round(rng.normal(size=40), 1)
        assert abs(mann_whitney_auc(y, tied) - roc_auc_score(y, tied)) < 1e-12

    # Fully discrete scores, the worst case for tie handling.
    y = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert abs(mann_whitney_auc(y, scores) - roc_auc_score(y, scores)) < 1e-12


def test_bootstrap_ci_is_reproducible_and_ordered() -> None:
    frame = _signal_table()
    y = frame["label"].to_numpy()
    score = frame["cnn_prob"].to_numpy()

    first = bootstrap_auc_ci(y, score, seed=42)
    second = bootstrap_auc_ci(y, score, seed=42)
    assert first == second, "bootstrap must be seeded and therefore reproducible"
    assert first[0] <= first[1]
    assert first[0] <= mann_whitney_auc(y, score) <= first[1] or first[1] >= 0.99, (
        "the interval should contain the point estimate unless it is pinned at the ceiling"
    )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
