from qgis.core import QgsApplication

from .processing_provider.provider import GullyAnalysisProvider


class GullyAnalysis():

    def __init__(self):
        self.provider = None

    def initProcessing(self):
        self.provider = GullyAnalysisProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)
