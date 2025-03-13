from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import processing
import ruptures as rpt
from qgis.core import QgsFeature, QgsVectorLayer, edit

if t.TYPE_CHECKING:
    from qgis.core import (
        QgsGeometry,
        QgsProcessingContext,
        QgsProcessingFeedback,
    )

    from .raster import DEM


def estimate_gully_simpler(
    prev: np.ndarray,
    after: np.ndarray,
    changepoints: c.Sequence[int],
    debug_out_file: Path | None = None,
    true_values: np.ndarray | None = None,
):
    def debug_estimation(before: np.ndarray, estimation: np.ndarray):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(15, 5))

        x = list(range(before.shape[0] - estimation.shape[0], before.shape[0]))[
            :100
        ]
        ax.plot(
            x,
            estimation[:100],
            c='#00ff00',
            label='2019 estimated channel',
            linewidth=2,
        )
        ax.plot(
            before[:100],
            c='#5795dc',
            label='2012 flow path profile',
            linewidth=2,
        )
        if true_values is not None:
            x = list(range(x[0], true_values.shape[0] + x[0]))[:100]
            ax.plot(
                x,
                true_values[:100],
                c='yellow',
                label='2019 flow path profile (truth)',
                linewidth=2,
            )
        for i, changepoint in enumerate(
            [point for point in changepoints if point <= 100]
        ):
            ax.axvline(
                changepoint,
                color='r',
                ls='--',
                linewidth=2,
                label='changepoint' if i == 0 else None,
            )
        ax.legend(prop={'size': 15})
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Elevation (m)', size=15)
        plt.tight_layout()
        plt.savefig(debug_out_file, dpi=300)
        plt.close()

    head = prev[: changepoints[0]]
    diff = after.shape[0] - prev.shape[0]
    after[: diff + head.shape[0]] = None
    after[: head.shape[0]] = head

    invalid = np.isnan(after)
    x = np.arange(after.shape[0])
    after[invalid] = np.interp(x[invalid], x[~invalid], after[~invalid])
    debug_estimation(prev, after)
    return after


def estimate_gully(
    values_before: np.ndarray,
    values_after: np.ndarray,
    changepoints: c.Sequence[int],
    debug_out_file: Path | None = None,
    true_values: np.ndarray | None = None,
):
    # Only the first changepoint is considered.
    # The others are used, for now, for plotting purposes (debug)

    def poly_fit(x, y, d=5):
        if x.shape[0] <= 5:
            return y
        return np.polynomial.polynomial.Polynomial.fit(x, y, d)(x)

    def fill_head_with_nan(y: np.ndarray, changepoint: int):
        y = y.copy()
        y[:changepoint] = np.nan
        return y

    def pad(y1: np.ndarray, y2: np.ndarray):
        y1 = y1.copy()
        nans = np.empty(y2.shape[0])
        nans[:] = np.nan
        return np.concatenate([nans, y1])

    def normalize(array, min_, max_):
        y_min = min(array)
        y_max = max(array)
        return min_ + (array - y_min) * (max_ - min_) / (y_max - y_min)

    def exponential(y):
        return 5**y

    def estimate_nan(y: np.ndarray):
        y = y.copy()
        nan_indices = np.argwhere(np.isnan(y))
        nan_len = nan_indices.shape[0]
        max_, min_ = y[nan_indices[0] - 1], y[nan_indices[-1] + 1]
        # estimated = normalize(exponential(np.linspace(max_, min_, nan_len)), min_, max_)
        # dont include max min
        linear_sequence = np.linspace(max_, min_, nan_len + 2)[1:-1]
        min_, max_ = linear_sequence[0], linear_sequence[-1]
        estimated = normalize(exponential(linear_sequence), max_, min_)
        y[nan_indices] = estimated
        return y

    def fill_polyfit(padded: np.ndarray, y_poly: np.ndarray, y2: np.ndarray):
        padded = padded.copy()
        y_poly += np.abs(y2[0] - y_poly[0])
        padded[: y_poly.shape[0]] = y_poly
        return padded

    def debug_estimation(before: np.ndarray, estimation: np.ndarray):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(15, 5))

        x = range(before.shape[0] - estimation.shape[0], before.shape[0])
        ax.plot(
            x,
            estimation,
            c='#00ff00',
            label='2019 estimated channel',
            linewidth=2,
        )
        ax.plot(
            before, c='#5795dc', label='2012 flow path profile', linewidth=2
        )
        if true_values is not None:
            x = list(range(x.start, true_values.shape[0] + x.start))
            print(len(x), len(true_values))
            ax.plot(
                x,
                true_values,
                c='yellow',
                label='2019 flow path profile (truth)',
                linewidth=2,
            )
        for i, changepoint in enumerate(changepoints):
            ax.axvline(
                changepoint,
                color='r',
                ls='--',
                linewidth=2,
                label='changepoint' if i == 0 else None,
            )
        ax.legend(prop={'size': 15})
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Elevation (m)', size=15)
        plt.tight_layout()
        plt.savefig(debug_out_file, dpi=300)
        plt.close()

    def debug_poly_fit(head: np.ndarray, head_splline_fitted: np.ndarray):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5, 5))

        x = range(head.shape[0])
        ax.plot(x, head, c='#5795dc')
        ax.plot(head_splline_fitted, c='#00ff00')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Elevation (m)', size=15)
        plt.tight_layout()
        plt.savefig(
            debug_out_file.with_name(f'{debug_out_file.stem}_poly_fit.png'),
            dpi=300,
        )
        plt.close()
        # plt.show()

    y1 = values_before.copy()
    y1_head = y1[: changepoints[0]]
    y1_poly = poly_fit(np.arange(y1_head.shape[0]), y1_head)
    if debug_out_file is not None:
        debug_poly_fit(y1_head, y1_poly)
    y2 = values_after.copy()
    no_head = fill_head_with_nan(y1, changepoints[0])
    padded = pad(no_head, y2)
    with_head = fill_polyfit(padded, y1_poly, y2)
    estimation = estimate_nan(with_head)
    # estimation[: changepoints[0]] = poly_fit(
    #     np.arange(estimation[: changepoints[0]].shape[0]),
    #     estimation[: changepoints[0]],
    # )
    # estimation = poly_fit(np.arange(estimation.shape[0]), estimation)
    if debug_out_file is not None:
        debug_estimation(y1, estimation)
    return estimation


def filter_on_id_line(
    line_id: int, samples: QgsVectorLayer
) -> list[QgsFeature]:
    iterator = samples.getFeatures(expression=f'"ID_LINE" = {line_id}')
    if iterator.compileFailed():
        raise Exception('Failed to filter the samples on ID_LINE.')
    return list(iterator)  # type: ignore


def samples_to_ndarray(samples: c.Iterable[QgsFeature]) -> np.ndarray:
    return np.array([sample.attribute('Z') for sample in samples])


def get_changepoints(samples: np.ndarray, penalty: int = 10):
    algorithm = rpt.Pelt(model='rbf').fit(samples)
    # the last value should not be returned.
    return algorithm.predict(pen=penalty)[:-1]


def estimate_gully_head(
    profile_samples: c.Sequence[QgsFeature],
    profile_to_estimate_samples: c.Sequence[QgsFeature],
    profile_truth_samples: c.Sequence[QgsFeature],
    line_id: int,
) -> list[QgsFeature]:
    profile_samples_ndarray = samples_to_ndarray(profile_samples)
    profile_to_estimate_samples_ndarray = samples_to_ndarray(
        profile_to_estimate_samples
    )
    profile_truth_ndarray = samples_to_ndarray(profile_truth_samples)
    estimated = estimate_gully_simpler(
        profile_samples_ndarray,
        profile_to_estimate_samples_ndarray,
        get_changepoints(profile_to_estimate_samples_ndarray),
        Path(__file__).parent.parent
        / 'data'
        / 'test_data'
        / 'plots'
        / f'{line_id}.png',
        profile_truth_ndarray,
    )

    estimated_features: list[QgsFeature] = []
    for feature, z in zip(profile_to_estimate_samples, estimated):
        estimated_feature = QgsFeature(feature)
        estimated_feature.setAttribute('Z', z)
        estimated_features.append(estimated_feature)
    return estimated_features


def estimate_gully_heads(
    sampled_profiles: QgsVectorLayer,
    sampled_profiles_to_estimate: QgsVectorLayer,
    sampled_profiles_truth: QgsVectorLayer,
    profile_count: int,
) -> list[QgsFeature]:
    estimated: list[QgsFeature] = []
    for line_id in range(profile_count):
        profiles_subset = filter_on_id_line(line_id, sampled_profiles)
        profiles_to_estimate_subset = filter_on_id_line(
            line_id, sampled_profiles_to_estimate
        )
        profiles_truth_subset = filter_on_id_line(
            line_id, sampled_profiles_truth
        )
        estimated.extend(
            estimate_gully_head(
                profiles_subset,
                profiles_to_estimate_subset,
                profiles_truth_subset,
                line_id,
            )
        )
    return estimated


@dataclass
class Samples:
    profiles: QgsVectorLayer
    profiles_to_estimate: QgsVectorLayer
    boundary: QgsVectorLayer
    estimated: QgsVectorLayer  # estimated profiles + boundary


def get_estimated_samples(
    dem: DEM,
    dem_truth: DEM,
    profiles: c.Sequence[QgsGeometry],
    profiles_to_estimate: c.Sequence[QgsGeometry],
    boundary: QgsGeometry,
    context: QgsProcessingContext,
    feedback: QgsProcessingFeedback,
) -> Samples:
    sampled_profiles = dem.sample(profiles, feedback=feedback, context=context)
    sampled_profiles_to_estimate = dem.sample(
        profiles_to_estimate, feedback=feedback, context=context
    )
    sampled_profiles_truth = dem_truth.sample(
        profiles_to_estimate, feedback=feedback, context=context
    )
    sampled_boundary = dem.sample(
        [boundary], feedback=feedback, context=context
    )
    estimated = estimate_gully_heads(
        sampled_profiles,
        sampled_profiles_to_estimate,
        sampled_profiles_truth,
        len(profiles),
    )
    layer = QgsVectorLayer('Point', 'estimated', 'memory')
    provider = layer.dataProvider()
    assert provider is not None
    provider.addAttributes(estimated[0].fields())
    layer.updateFields()
    with edit(layer):
        layer.addFeatures(estimated)
        layer.addFeatures(list(sampled_boundary.getFeatures()))  # type: ignore

    return Samples(
        sampled_profiles, sampled_profiles_to_estimate, sampled_boundary, layer
    )


def aggregate_samples(samples: QgsVectorLayer) -> QgsVectorLayer:
    return processing.run(
        'native:aggregate',
        {
            'INPUT': samples,
            'GROUP_BY': 'to_string("X") || \',\' || to_string("Y")',
            'AGGREGATES': [
                {
                    'aggregate': 'first_value',
                    'delimiter': ',',
                    'input': '"ID_LINE"',
                    'length': 16,
                    'name': 'ID_LINE',
                    'precision': 0,
                    'sub_type': 0,
                    'type': 4,
                    'type_name': 'int8',
                },
                {
                    'aggregate': 'first_value',
                    'delimiter': ',',
                    'input': '"ID_POINT"',
                    'length': 16,
                    'name': 'ID_POINT',
                    'precision': 0,
                    'sub_type': 0,
                    'type': 4,
                    'type_name': 'int8',
                },
                {
                    'aggregate': 'first_value',
                    'delimiter': ',',
                    'input': '"X"',
                    'length': 18,
                    'name': 'X',
                    'precision': 10,
                    'sub_type': 0,
                    'type': 6,
                    'type_name': 'double precision',
                },
                {
                    'aggregate': 'first_value',
                    'delimiter': ',',
                    'input': '"Y"',
                    'length': 18,
                    'name': 'Y',
                    'precision': 10,
                    'sub_type': 0,
                    'type': 6,
                    'type_name': 'double precision',
                },
                {
                    'aggregate': 'minimum',
                    'delimiter': ',',
                    'input': '"Z"',
                    'length': 18,
                    'name': 'Z',
                    'precision': 10,
                    'sub_type': 0,
                    'type': 6,
                    'type_name': 'double precision',
                },
            ],
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )['OUTPUT']
