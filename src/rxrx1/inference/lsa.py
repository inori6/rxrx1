import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def build_plate_groups(train_manifest, label_to_index, num_plates=4):
    df = train_manifest[["experiment", "plate", "well", "sirna"]].drop_duplicates()
    num_classes = len(label_to_index)

    if num_classes % num_plates:
        raise ValueError(f"{num_classes} classes cannot be divided into {num_plates} plates.")

    group_size = num_classes // num_plates
    groups = set()

    for _, plate_df in df.groupby(["experiment", "plate"]):
        labels = plate_df["sirna"].unique()
        if len(labels) != group_size:
            continue
        groups.add(tuple(sorted(label_to_index[label] for label in labels)))

    groups = sorted(groups)

    if len(groups) != num_plates:
        raise ValueError(
            f"Expected {num_plates} unique plate groups, found {len(groups)}. "
            "Use a complete full-class treatment manifest."
        )

    group_sets = [set(group) for group in groups]
    for i in range(num_plates):
        for j in range(i + 1, num_plates):
            if group_sets[i] & group_sets[j]:
                raise ValueError("Plate groups overlap.")

    expected = set(range(num_classes))
    if set().union(*group_sets) != expected:
        raise ValueError("Plate groups do not cover all model classes.")

    return groups


def infer_plate_groups(predictions, groups):
    assignments = {}
    num_groups = len(groups)
    group_sets = [np.asarray(group) for group in groups]

    for experiment, exp_df in predictions.groupby("experiment"):
        plates = sorted(exp_df["plate"].unique())

        if len(plates) != num_groups:
            raise ValueError(
                f"{experiment}: expected {num_groups} plates, found {len(plates)}."
            )

        scores = np.zeros((num_groups, num_groups), dtype=np.float64)

        for i, plate in enumerate(plates):
            plate_df = exp_df[exp_df["plate"] == plate]
            logits = _stack_logits(plate_df)
            pred = logits.argmax(axis=1)

            for j, group in enumerate(group_sets):
                scores[i, j] = np.isin(pred, group).sum()

        denom = scores.sum(axis=0, keepdims=True)
        scores = scores / np.maximum(denom, 1.0)

        rows, cols = linear_sum_assignment(-scores)

        for row, col in zip(rows, cols):
            assignments[(experiment, plates[row])] = int(col)

    return assignments


def apply_lsa(predictions, groups, assignments, label_to_index):
    num_classes = len(label_to_index)
    group_size = num_classes // len(groups)
    index_to_label = {index: label for label, index in label_to_index.items()}
    results = []

    if predictions["id_code"].duplicated().any():
        raise ValueError(
            "LSA expects one row per well. Aggregate site/TTA logits before calling apply_lsa()."
        )

    for (experiment, plate), plate_df in predictions.groupby(
        ["experiment", "plate"], sort=False
    ):
        key = (experiment, plate)

        if key not in assignments:
            raise ValueError(f"No plate-group assignment for {key}.")

        if len(plate_df) != group_size:
            raise ValueError(
                f"{key}: expected {group_size} wells, found {len(plate_df)}."
            )

        candidates = np.asarray(groups[assignments[key]])
        logits = _stack_logits(plate_df)

        if logits.shape[1] != num_classes:
            raise ValueError(
                f"Expected {num_classes} logits, got {logits.shape[1]}."
            )

        scores = logits[:, candidates]
        rows, cols = linear_sum_assignment(-scores)

        assigned = np.full(len(plate_df), -1, dtype=np.int64)
        assigned[rows] = candidates[cols]

        for id_code, class_index in zip(plate_df["id_code"], assigned):
            results.append({
                "id_code": id_code,
                "sirna": index_to_label[int(class_index)],
            })

    return pd.DataFrame(results)


def lsa_predict(predictions, train_manifest, label_to_index):
    groups = build_plate_groups(train_manifest, label_to_index)
    assignments = infer_plate_groups(predictions, groups)
    submission = apply_lsa(predictions, groups, assignments, label_to_index)
    return submission, assignments


def _stack_logits(df):
    logits = np.stack(df["logits"].to_numpy())
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}.")
    return logits