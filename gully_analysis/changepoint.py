from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
import typing as t
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import processing
import ruptures as rpt
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsVectorLayer,
    edit,
)
from scipy.interpolate import PchipInterpolator

if t.TYPE_CHECKING:
    from qgis.core import (
        QgsGeometry,
        QgsProcessingContext,
        QgsProcessingFeedback,
    )

    from .raster import DEM


def debug_estimation(
    profile: np.ndarray,
    estimation: np.ndarray,
    changepoints: c.Sequence[int],
    out_file: Path,
    values_slice: None | slice = None,
    true_values: np.ndarray | None = None,
) -> None:
    if values_slice is None:
        values_slice = slice(0, profile.shape[0], 1)
    _, ax = plt.subplots(figsize=(8, 5))
    diff = profile.shape[0] - estimation.shape[0]
    x = list(range(diff, profile.shape[0]))[values_slice]
    ax.plot(
        x,
        estimation[values_slice],
        c='#00ff00',
        label='estimated flow path',
        linewidth=2,
    )
    ax.plot(
        profile[values_slice],
        c='#5795dc',
        label='flow path',
        linewidth=2,
    )
    if true_values is not None:
        x = list(range(x[0], true_values.shape[0] + x[0]))[values_slice]
        ax.plot(
            x,
            true_values[values_slice],
            c='yellow',
            label='flow path profile (truth)',
            linewidth=2,
        )
    for i, changepoint in enumerate(
        [point for point in changepoints if point <= values_slice.stop]
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
    plt.savefig(out_file)
    plt.close()


def estimate_gully_simpler(
    profile: np.ndarray,
    profile_to_estimate: np.ndarray,
    changepoints: c.Sequence[int],
    debug_out_file: Path | None = None,
    true_values: np.ndarray | None = None,
    gully_cover: np.ndarray | None = None,
) -> np.ndarray:
    head = profile[: changepoints[0]]
    if gully_cover is None:
        z_diff = profile_to_estimate[0] - profile[0]
    else:
        z_diff = gully_cover[0] - profile[0]
    diff = profile_to_estimate.shape[0] - profile.shape[0]
    abs_diff = np.abs(diff)
    buffer = np.max([0, diff])
    profile_to_estimate[:] = None
    if profile_to_estimate.shape[0] >= head.shape[0]:
        profile_to_estimate[: buffer + head.shape[0]] = None
        profile_to_estimate[: head.shape[0]] = head + z_diff
    else:
        profile_to_estimate[0] = profile[0] + z_diff

    assert profile.shape[0] == profile_to_estimate.shape[0] - diff, (
        f'{profile.shape}, {profile_to_estimate.shape}'
    )

    for changepoint in changepoints[1:-1]:
        if diff < 0:
            with_buffer = changepoint - abs_diff
        else:
            with_buffer = changepoint + abs_diff
        if np.abs(with_buffer) < head.shape[0]:
            continue
        profile_to_estimate[with_buffer] = profile[changepoint]
    profile_to_estimate[-1] = profile[-1]

    invalid = np.isnan(profile_to_estimate)
    x = np.arange(profile_to_estimate.shape[0])
    x_valid = x[~invalid]
    y_valid = profile_to_estimate[~invalid]
    spline = PchipInterpolator(x_valid, y_valid, extrapolate=False)
    profile_to_estimate[invalid] = spline(x[invalid])

    if debug_out_file is not None:
        debug_estimation(
            profile,
            profile_to_estimate,
            true_values=true_values,
            changepoints=changepoints,
            out_file=debug_out_file,
        )
    return profile_to_estimate


def filter_on_id_line(
    line_id: int, samples: QgsVectorLayer
) -> list[QgsFeature]:
    col = 'ID_LINE'
    iterator = samples.getFeatures(expression=f'"{col}" = {line_id}')
    if iterator.compileFailed():
        raise Exception(f'Failed to filter the samples on {col}.')
    return list(iterator)  # type: ignore


def samples_to_ndarray(samples: c.Iterable[QgsFeature]) -> np.ndarray:
    return np.array([sample.attribute('Z') for sample in samples])


def get_changepoints(samples: np.ndarray, penalty: int) -> list[int]:
    algorithm = rpt.Pelt(model='l2').fit(samples)
    return algorithm.predict(pen=penalty)


def estimate_gully_head(
    profile_samples: c.Sequence[QgsFeature],
    profile_to_estimate_samples: c.Sequence[QgsFeature],
    line_id: int,
    profile_gully_cover: c.Sequence[QgsFeature] | None = None,
    profile_truth_samples: c.Sequence[QgsFeature] | None = None,
    changepoint_penalty: int = 10,
) -> list[QgsFeature] | None:
    profile_samples_ndarray = samples_to_ndarray(profile_samples)
    profile_to_estimate_samples_ndarray = samples_to_ndarray(
        profile_to_estimate_samples
    )
    profile_gully_cover_ndarray = (
        samples_to_ndarray(profile_gully_cover)
        if profile_gully_cover is not None
        else None
    )
    profile_truth_samples_ndarray = (
        samples_to_ndarray(profile_truth_samples)
        if profile_truth_samples is not None
        else None
    )
    changepoints = get_changepoints(
        profile_samples_ndarray, penalty=changepoint_penalty
    )
    if not changepoints:
        return None
    (
        Path(__file__).parent.parent
        / 'data'
        / 'test_data'
        / 'plots'
        / f'{line_id}.png'
    )
    estimated = estimate_gully_simpler(
        profile=profile_samples_ndarray,
        profile_to_estimate=profile_to_estimate_samples_ndarray,
        changepoints=changepoints,
        # debug_out_file=path_to_plots,
        true_values=profile_truth_samples_ndarray,
        gully_cover=profile_gully_cover_ndarray,
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
    sampled_profiles_gully_cover: QgsVectorLayer | None = None,
    changepoint_penalty: int = 10,
) -> list[QgsFeature]:
    def handle_line_id(line_id: int) -> None | list[QgsFeature]:
        profiles_subset = filter_on_id_line(line_id, sampled_profiles)
        profiles_to_estimate_subset = filter_on_id_line(
            line_id, sampled_profiles_to_estimate
        )
        profiles_gully_cover_subset = None
        if sampled_profiles_gully_cover is not None:
            profiles_gully_cover_subset = filter_on_id_line(
                line_id, sampled_profiles_gully_cover
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
            profile_gully_cover=profiles_gully_cover_subset,
            profile_truth_samples=profiles_truth_subset,
            changepoint_penalty=changepoint_penalty,
        )
        if estimated_gully_head is not None:
            return estimated_gully_head
        return None

    # results = []
    # for i in range(profile_count):
    #    results.append(handle_line_id(i))
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
    crs: QgsCoordinateReferenceSystem,
    boundary: QgsGeometry | None = None,
    sampled_boundary: QgsVectorLayer | None = None,
    context: QgsProcessingContext | None = None,
    feedback: QgsProcessingFeedback | None = None,
    gully_cover: DEM | None = None,  # for predicting past
    dem_truth: DEM | None = None,  # for debug
    changepoint_penalty: int = 10,
) -> Samples:
    sampled_profiles = dem.sample(profiles, feedback=feedback, context=context)
    sampled_profiles_to_estimate = dem.sample(
        profiles_to_estimate, feedback=feedback, context=context
    )
    context.project().addMapLayer(sampled_profiles_to_estimate)

    sampled_profiles_truth = None
    if dem_truth is not None:
        sampled_profiles_truth = dem_truth.sample(
            profiles_to_estimate, feedback=feedback, context=context
        )
    sampled_profiles_gully_cover = None
    if gully_cover is not None:
        sampled_profiles_gully_cover = gully_cover.sample(
            profiles_to_estimate, feedback=feedback, context=context
        )
    if sampled_boundary is None and boundary is not None:
        sampled_boundary = dem.sample(
            [boundary], feedback=feedback, context=context
        )
    estimated = estimate_gully_heads(
        sampled_profiles,
        sampled_profiles_to_estimate,
        len(profiles),
        sampled_profiles_truth=sampled_profiles_truth,
        changepoint_penalty=changepoint_penalty,
        sampled_profiles_gully_cover=sampled_profiles_gully_cover,
    )
    layer = QgsVectorLayer('Point', 'estimated', 'memory')
    provider = layer.dataProvider()
    assert provider is not None
    provider.addAttributes(estimated[0].fields())
    layer.updateFields()
    with edit(layer):
        layer.addFeatures(estimated)
        layer.addFeatures(list(sampled_boundary.getFeatures()))  # type: ignore
        layer.setCrs(crs)

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
