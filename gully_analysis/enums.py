from enum import Enum


class PluginName(Enum):
    DISPLAY_NAME = 'Gully Analysis'
    SHORT_NAME = 'gullyanalysis'


class Algorithm(Enum):
    ESTIMATE_EROSION_FUTURE = 'estimatefuture'

    @classmethod
    def display_name(cls) -> str:
        return ' '.join(
            [part.capitalize() for part in cls._name_.split('_')]
        )
