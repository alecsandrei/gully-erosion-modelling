from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon  # type: ignore

from .algorithms import EstimateErosionFuture
from ..enums import (
    PluginName,
    Assets
)


class GullyAnalysisProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(EstimateErosionFuture())

    def id(self, *args, **kwargs):
        return PluginName.GULLY_ANALYSIS.value

    def name(self, *args, **kwargs):
        return self.tr(PluginName.GULLY_ANALYSIS.display_name())

    def icon(self):
        return QIcon(
            Assets.ICON.value.as_posix()
        )
