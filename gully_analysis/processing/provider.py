from ..enums import PluginName

from qgis.core import QgsProcessingProvider


class GullyAnalysisProvider(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        ...

    def id(self, *args, **kwargs):
        return PluginName.SHORT.value

    def name(self, *args, **kwargs):
        return self.tr(PluginName.LONG.value)

    def icon(self):
        return QgsProcessingProvider.icon(self)
