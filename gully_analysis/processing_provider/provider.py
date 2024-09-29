from gully_analysis.enums import PluginName

from qgis.core import QgsProcessingProvider


class GullyErosionEstimator(QgsProcessingProvider):

    def loadAlgorithms(self, *args, **kwargs):
        ...

    def id(self, *args, **kwargs):
        return PluginName.SHORT.value

    def name(self, *args, **kwargs):
        return self.tr(PluginName.LONG.value)

    def icon(self):
        """Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        return QgsProcessingProvider.icon(self)
