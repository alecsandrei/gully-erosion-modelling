import os

from qgis.core import *
import osgeo.ogr


osgeo.ogr.UseExceptions()

# NOTE: some paths are for now


QgsApplication.setPrefixPath("/home/alex/miniforge3/envs/qgis", True)
qgs = QgsApplication([], False)
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
