from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import typing as t
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cache
from pathlib import Path

import processing
from qgis.core import QgsVectorLayer

sys.path.append(str(Path(__file__).parent.parent / 'gully_analysis'))
import gully_analysis  # noqa


DATA_DIR = Path(__file__).parent / 'data'
TEMP_DIR = Path(tempfile.gettempdir())


class Site(Enum):
    SOLDANESTI_AMONTE = auto()
    SAVENI_AMONTE = auto()
    SOLDANESTI_AVAL = auto()
    SAVENI_AVAL = auto()


class Model(Enum):
    FUTURE = auto()
    PAST = auto()


@cache
def get_centerline(
    site: Site,
    polygon: Path,
    centerline_smoothness: int = 20,
    centerline_thin: int = 0,
):
    out_file = TEMP_DIR / f'{site.name.lower()}_centerline.shp'
    with gully_analysis.utils.timeit(
        f'Computing centerlines for {site.name} with smoothness {centerline_smoothness} and thin {centerline_thin}'
    ):
        layer = QgsVectorLayer(polygon.as_posix(), 'polygon', 'ogr')
        return gully_analysis.geometry.Centerlines.compute(
            layer,
            smoothness=centerline_smoothness,
            thin=centerline_thin,
            output=out_file.as_posix(),
        )._layer


@cache
def get_sink_removed(site: Site, model: Model, dem: Path):
    out_file = (
        TEMP_DIR
        / f'{site.name.lower()}_{model.name.lower()}_dem_sink_removed.tif'
    )
    with gully_analysis.utils.timeit(
        f'Removing sinks from DEM for {site.name}'
    ):
        return (
            gully_analysis.raster.DEM.from_file(dem)
            .remove_sinks(output=out_file.as_posix())
            .layer
        )


def get_directory_files(site: Site) -> dict[str, Path]:
    directory = DATA_DIR / site.name.lower()
    return {
        'future': directory / 'future.fgb',
        'past': directory / 'past.fgb',
        'past_google_earth': directory / 'past_google_earth.fgb',
        'past_dem': directory / 'past.tif',
        'future_dem': directory / 'future.tif',
        'estimation_surface': directory / 'estimation_surface.fgb',
    }


def get_advanced_parameters():
    return {
        'CHANGEPOINT_PENALTY': 10,
        'CENTERLINE_SMOOTHNESS': 20,
        'CENTERLINE_THIN': 0,
        'SAMPLE_AGGREGATE': 'maximum',
        'MULTILEVEL_B_SPLINE_LEVEL': 14,
    }


def advanced_parameter_combinations():
    # changepoint_penalty = (5, 10, 15, 20)
    # changepoint_penalty = (30,)
    changepoint_penalty = range(5, 51, 5)
    # sample_aggregate = ('maximum', 'minimum', 'mean')
    sample_aggregate = ('maximum',)
    return itertools.product(changepoint_penalty, sample_aggregate)


def get_past_model_parameters(
    files: dict[str, str], output_dir: Path, site: Site, model: Model
) -> dict[str, t.Any]:
    return {
        'GULLY_BOUNDARY': files['future'],
        'GULLY_ELEVATION': get_sink_removed(
            site=site, dem=Path(files['future_dem']), model=model
        ),
        'GULLY_ELEVATION_SINK_REMOVED': True,
        'GULLY_PAST_ELEVATION': files['past_dem'],
        'GULLY_PAST_BOUNDARY': files['past'],
        'ESTIMATION_SURFACE': files['estimation_surface'],
        'CENTERLINES': get_centerline(site=site, polygon=Path(files['future'])),
        'DEBUG_MODE': False,
        'ESTIMATED_DEM': (output_dir / 'estimated_dem_past.tif').as_posix(),
        'ESTIMATION_SURFACE_OUTPUT': (
            output_dir / 'estimation_surface_output_past.fgb'
        ).as_posix(),
        **get_advanced_parameters(),
    }


def get_future_model_parameters(
    files: dict[str, str], output_dir: Path, site: Site, model: Model
) -> dict[str, t.Any]:
    return {
        'GULLY_BOUNDARY': files['past'],
        'GULLY_ELEVATION': get_sink_removed(
            site=site, dem=Path(files['past_dem']), model=model
        ),
        'GULLY_ELEVATION_SINK_REMOVED': True,
        'GULLY_FUTURE_ELEVATION': files['future_dem'],
        'GULLY_FUTURE_BOUNDARY': files['future'],
        'ESTIMATION_SURFACE': files['estimation_surface'],
        'CENTERLINES': get_centerline(site=site, polygon=Path(files['future'])),
        'DEBUG_MODE': False,
        'ESTIMATED_DEM': (output_dir / 'estimated_dem_future.tif').as_posix(),
        'ESTIMATION_SURFACE_OUTPUT': (
            output_dir / 'estimation_surface_output_future.fgb'
        ).as_posix(),
        **get_advanced_parameters(),
    }


def get_model_parameters(
    model: Model, site: Site, trial_dir: Path
) -> dict[str, t.Any]:
    files = get_directory_files(site)
    files_as_str = {k: v.as_posix() for k, v in files.items()}
    out_dir = trial_dir / site.name / model.name
    os.makedirs(out_dir, exist_ok=True)
    if model is Model.PAST:
        return get_past_model_parameters(
            files_as_str, out_dir, site, model=model
        )
    elif model is Model.FUTURE:
        return get_future_model_parameters(
            files_as_str, out_dir, site, model=model
        )


@dataclass
class Trial:
    TRIAL_DIR_NAME = 'trial_{id}'
    output_dir: Path
    id: int = field(init=False)
    trial_dir: Path = field(init=False)

    def __post_init__(self):
        self.id = self.get_trial_id(self.output_dir)
        self.trial_dir = self.output_dir / self.TRIAL_DIR_NAME.format(
            id=self.id
        )
        self.trial_dir.mkdir()

    @staticmethod
    def get_trial_id(path: Path) -> int:
        trial_dirs = [
            file
            for file in path.iterdir()
            if file.is_dir() and file.name.startswith('trial')
        ]
        if trial_dirs:
            return (
                max(map(lambda path: int(path.name.split('_')[1]), trial_dirs))
                + 1
            )
        return 0

    def save_configs(self, out_file: Path | None = None, **kwargs):
        if out_file is None:
            out_file = self.trial_dir / 'configs.json'
        with out_file.open(mode='w') as config_file:
            configs = self.__dict__.copy()
            configs.update(kwargs)

            json.dump(make_serializable(configs), config_file, indent=2)

    def run(self, model: Model, site: Site, parameters: dict[str, t.Any]):
        try:
            print(f'Running {self!r} {model!r} {site!r}')
            processing.run(
                f'gullyanalysis:estimate{model.name.lower()}',
                parameters,
            )
        except Exception as e:
            print(f'Failed {self!r} {site!r} {model!r}: {e}')


def make_serializable(obj: t.Any):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if not is_serializable(obj):
        return str(obj)
    return obj


def is_serializable(obj: t.Any):
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


def main():
    trials_dir = Path(__file__).parent.parent / 'trials'
    trials_dir.mkdir(exist_ok=True)
    for (
        changepoint_penalty,
        aggregate_method,
    ) in advanced_parameter_combinations():
        trial = Trial(trials_dir)

        trial.save_configs(
            changepoint_penalty=changepoint_penalty,
            aggregate_method=aggregate_method,
        )

        for site in Site:
            for model in Model:
                parameters = get_model_parameters(model, site, trial.trial_dir)
                parameters['CHANGEPOINT_PENALTY'] = changepoint_penalty
                parameters['SAMPLE_AGGREGATE'] = aggregate_method
                trial.save_configs(
                    trial.trial_dir / site.name / model.name / 'configs.json',
                    parameters=parameters,
                )
                trial.run(model, site, parameters)


if __name__ == '__main__':
    main()
