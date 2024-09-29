from qgis.core import QgsProcessingProvider

from ..enums import PluginName
from .algorithms import EstimateErosionFuture


class GullyAnalysisProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        self.addAlgorithm(EstimateErosionFuture())

    def id(self, *args, **kwargs):
        return PluginName.GULLY_ANALYSIS.value

    def name(self, *args, **kwargs):
        return self.tr(PluginName.GULLY_ANALYSIS.display_name())

    def icon(self):
        return QgsProcessingProvider.icon(self)
