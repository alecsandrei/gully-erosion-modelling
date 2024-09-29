import os

from qgis.core import QgsApplication
import osgeo.ogr


if QgsApplication.instance() is None:
    osgeo.ogr.UseExceptions()
    QgsApplication.setPrefixPath("/home/alex/miniforge3/envs/qgis", True)
    qgs = QgsApplication([], True)
    qgs.initQgis()

    def init_native_alg():
        from qgis.analysis import QgsNativeAlgorithms
        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    def init_processing():
        from processing.core.Processing import Processing
        Processing.initialize()

    init_native_alg()
    init_processing()


DEBUG = int(os.getenv('DEBUG', 0))
CACHE = int(os.getenv('CACHE', 0))
MODEL = os.getenv('MODEL', 'all')
