from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
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
        plt.savefig(debug_out_file)
        plt.close()

    head = prev[: changepoints[0]]
    z_diff = after[0] - prev[0]
    diff = after.shape[0] - prev.shape[0]
    after[: diff + head.shape[0]] = None
    after[: head.shape[0]] = head + z_diff
    invalid = np.isnan(after)
    x = np.arange(after.shape[0])
    after[invalid] = np.interp(x[invalid], x[~invalid], after[~invalid])
    # debug_estimation(prev, after)
    return after


def filter_on_id_line(
    line_id: int, samples: QgsVectorLayer
) -> list[QgsFeature]:
    iterator = samples.getFeatures(expression=f'"ID_LINE" = {line_id}')
    if iterator.compileFailed():
        raise Exception('Failed to filter the samples on ID_LINE.')
    return list(iterator)  # type: ignore


def samples_to_ndarray(samples: c.Iterable[QgsFeature]) -> np.ndarray:
    return np.array([sample.attribute('Z') for sample in samples])


def get_changepoints(samples: np.ndarray, penalty: int):
    algorithm = rpt.Pelt(model='rbf').fit(samples)
    # the last value should not be returned.
    return algorithm.predict(pen=penalty)[:-1]


def estimate_gully_head(
    profile_samples: c.Sequence[QgsFeature],
    profile_to_estimate_samples: c.Sequence[QgsFeature],
    line_id: int,
    profile_truth_samples: c.Sequence[QgsFeature] | None = None,
    changepoint_penalty: int = 10,
) -> list[QgsFeature] | None:
    profile_samples_ndarray = samples_to_ndarray(profile_samples)
    profile_to_estimate_samples_ndarray = samples_to_ndarray(
        profile_to_estimate_samples
    )
    changepoints = get_changepoints(
        profile_to_estimate_samples_ndarray, penalty=changepoint_penalty
    )
    if not changepoints:
        return None
    path_to_plots = (
        Path(__file__).parent.parent
        / 'data'
        / 'test_data'
        / 'plots'
        / f'{line_id}.png'
    )
    estimated = estimate_gully_simpler(
        profile_samples_ndarray,
        profile_to_estimate_samples_ndarray,
        changepoints,
        path_to_plots,
        samples_to_ndarray(profile_truth_samples)
        if profile_truth_samples is not None
        else None,
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
    profile_count: int,
    sampled_profiles_truth: QgsVectorLayer | None = None,
    changepoint_penalty: int = 10,
) -> list[QgsFeature]:
    def handle_line_id(line_id: int):
        profiles_subset = filter_on_id_line(line_id, sampled_profiles)
        profiles_to_estimate_subset = filter_on_id_line(
            line_id, sampled_profiles_to_estimate
        )
        profiles_truth_subset = None
        if sampled_profiles_truth is not None:
            profiles_truth_subset = filter_on_id_line(
                line_id, sampled_profiles_truth
            )
        estimated_gully_head = estimate_gully_head(
            profiles_subset,
            profiles_to_estimate_subset,
            line_id,
            profile_truth_samples=profiles_truth_subset,
            changepoint_penalty=changepoint_penalty,
        )
        if estimated_gully_head is not None:
            return estimated_gully_head

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        results = executor.map(handle_line_id, range(profile_count))
    return list(
        itertools.chain.from_iterable(
            result for result in results if result is not None
        )
    )


@dataclass
class Samples:
    profiles: QgsVectorLayer
    profiles_to_estimate: QgsVectorLayer
    boundary: QgsVectorLayer
    estimated: QgsVectorLayer  # estimated profiles + boundary


def get_estimated_samples(
    dem: DEM,
    profiles: c.Sequence[QgsGeometry],
    profiles_to_estimate: c.Sequence[QgsGeometry],
    boundary: QgsGeometry,
    context: QgsProcessingContext,
    feedback: QgsProcessingFeedback,
    dem_truth: DEM | None = None,
    changepoint_penalty: int = 10,
) -> Samples:
    sampled_profiles = dem.sample(profiles, feedback=feedback, context=context)
    sampled_profiles_to_estimate = dem.sample(
        profiles_to_estimate, feedback=feedback, context=context
    )

    sampled_profiles_truth = None
    if dem_truth is not None:
        sampled_profiles_truth = dem_truth.sample(
            profiles_to_estimate, feedback=feedback, context=context
        )
    sampled_boundary = dem.sample(
        [boundary], feedback=feedback, context=context
    )
    estimated = estimate_gully_heads(
        sampled_profiles,
        sampled_profiles_to_estimate,
        len(profiles),
        sampled_profiles_truth=sampled_profiles_truth,
        changepoint_penalty=changepoint_penalty,
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


def aggregate_samples(
    samples: QgsVectorLayer, aggregate_method: str = 'minimum'
) -> QgsVectorLayer:
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
                    'aggregate': aggregate_method,
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
