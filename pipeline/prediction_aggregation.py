import numpy as np
import pandas as pd


def build_prediction_cube(prediction_frames, target_columns):
    numeric_arrays = []

    for prediction_df in prediction_frames:
        numeric_df = prediction_df.loc[:, target_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric_arrays.append(
            np.expand_dims(
                numeric_df.to_numpy(dtype=np.float32),
                axis=2,
            )
        )

    return np.concatenate(numeric_arrays, axis=2)


def aggregate_prediction_cube(prediction_cube, sample_merging_strategy):
    if sample_merging_strategy == "mean":
        valid_counts = np.sum(~np.isnan(prediction_cube), axis=2)
        value_sums = np.nansum(prediction_cube, axis=2)
        aggregated = np.divide(
            value_sums,
            valid_counts,
            out=np.full(value_sums.shape, np.nan, dtype=np.float32),
            where=valid_counts > 0,
        )
        return aggregated.astype(np.float32)

    if sample_merging_strategy == "50th percentile":
        return np.percentile(prediction_cube, 50, axis=2).astype(np.float32)

    raise Exception("Experiment: unknown sample_merging_strategy provided!")
