from __future__ import annotations

import itertools
import json
import os
import typing as t
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import processing

DATA_DIR = Path(__file__).parent / 'data'


class Site(Enum):
    SAVENI_AMONTE = auto()
    SOLDANESTI_AMONTE = auto()
    SOLDANESTI_AVAL = auto()
    SAVENI_AVAL = auto()


class Model(Enum):
    PAST = auto()
    FUTURE = auto()


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
    changepoint_penalty = (5, 10, 15, 20)
    sample_aggregate = ('maximum', 'minimum', 'mean')
    return itertools.product(changepoint_penalty, sample_aggregate)


def get_past_model_parameters(
    files: dict[str, str], output_dir: Path
) -> dict[str, t.Any]:
    return {
        'GULLY_BOUNDARY': files['future'],
        'GULLY_ELEVATION': files['future_dem'],
        'GULLY_ELEVATION_SINK_REMOVED': False,
        'GULLY_PAST_ELEVATION': files['past_dem'],
        'GULLY_PAST_BOUNDARY': files['past'],
        'ESTIMATION_SURFACE': files['estimation_surface'],
        'CENTERLINES': None,
        'DEBUG_MODE': True,
        'ESTIMATED_DEM': (output_dir / 'estimated_dem_past.tif').as_posix(),
        'ESTIMATION_SURFACE_OUTPUT': (
            output_dir / 'estimation_surface_output_past.fgb'
        ).as_posix(),
        **get_advanced_parameters(),
    }


def get_future_model_parameters(
    files: dict[str, str], output_dir: Path
) -> dict[str, t.Any]:
    return {
        'GULLY_BOUNDARY': files['past'],
        'GULLY_ELEVATION': files['past_dem'],
        'GULLY_ELEVATION_SINK_REMOVED': False,
        'GULLY_FUTURE_ELEVATION': files['future_dem'],
        'GULLY_FUTURE_BOUNDARY': files['future'],
        'ESTIMATION_SURFACE': files['estimation_surface'],
        'CENTERLINES': None,
        'DEBUG_MODE': True,
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
        return get_past_model_parameters(files_as_str, out_dir)
    elif model is Model.FUTURE:
        return get_future_model_parameters(files_as_str, out_dir)


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
        except Exception:
            print(f'Failed {self!r} {site!r} {model!r}')


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
