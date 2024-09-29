from __future__ import annotations

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsFeatureSink,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFeatureSink
)

from ...enums import (
    Algorithm,
    AlgorithmGroup
)


class EstimateErosionFuture(QgsProcessingAlgorithm):

    GULLY_BOUNDARY = 'GULLY_BOUNDARY'
    GULLY_ELEVATION = 'GULLY_ELEVATION'
    GULLY_FUTURE_BOUNDARY = 'GULLY_FUTURE_BOUNDARY'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return EstimateErosionFuture()

    def name(self):
        return Algorithm.ESTIMATE_EROSION_FUTURE.value

    def displayName(self):
        return self.tr(Algorithm.ESTIMATE_EROSION_FUTURE.display_name())

    def group(self):
        return self.tr(AlgorithmGroup.ESTIMATORS.display_name())

    def groupId(self):
        return AlgorithmGroup.ESTIMATORS.value

    def shortHelpString(self):
        return self.tr(
            'Used to estimate the eroded volume near the gully head '
            'for a future date.'
        )

    def initAlgorithm(self, config=None):  # type: ignore

        # Gully boundary
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GULLY_BOUNDARY,
                self.tr('The gully boundary'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.GULLY_ELEVATION,
                self.tr('The gully elevation raster'),
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GULLY_FUTURE_BOUNDARY,
                self.tr('The gully future boundary'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )

        # TODO: add output

    def processAlgorithm(self, parameters, context, feedback):

        gully_boundary = parameters[self.GULLY_BOUNDARY]
        gully_elevation = parameters[self.GULLY_ELEVATION]
        gully_future_boundary = parameters[self.GULLY_FUTURE_BOUNDARY]

        print(gully_boundary, gully_elevation, gully_future_boundary)
