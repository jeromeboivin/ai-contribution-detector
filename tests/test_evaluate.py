import numpy as np

from aicontrib.model.evaluate import per_language_report


def test_per_language_report_splits_rows_by_language():
    names = ["human", "co_authored", "ai"]
    y_true = np.array([0, 1, 2, 0, 2, 2])
    y_pred = np.array([0, 1, 2, 2, 0, 2])
    languages = ["JavaScript", "JavaScript", "JavaScript", "C#", "C#", "C#"]

    lines = per_language_report(y_true, y_pred, languages, names).splitlines()

    assert lines[0].split()[:4] == ["language", "rows", "accuracy", "macro-F1"]
    csharp, javascript = (line.split() for line in lines[1:])
    assert javascript[:3] == ["JavaScript", "3", "1.000"]
    # C#: human 0/1 right, no co_authored rows, ai 1/2 right.
    assert csharp[:3] == ["C#", "3", "0.333"]
    assert csharp[-3:] == ["0.000", "0.000", "0.500"]
