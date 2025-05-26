from __future__ import annotations

from enum import Enum
from pathlib import Path


class Displayable(Enum):
    def display_name(self) -> str:
        return ' '.join([part.capitalize() for part in self.name.split('_')])


class PluginName(Displayable):
    GULLY_ANALYSIS = 'gullyanalysis'


class Algorithm(Displayable):
    ESTIMATE_EROSION_FUTURE = 'estimatefuture'
    ESTIMATE_EROSION_PAST = 'estimatepast'


class AlgorithmGroup(Displayable):
    ESTIMATORS = 'estimators'


class Assets(Enum):
    FOLDER = Path(__file__).parent.parent / 'assets'
    ICON = FOLDER / 'icon.png'
