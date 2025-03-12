from __future__ import annotations

import typing as t
from enum import Enum, auto
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,  # type: ignore
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication  # type: ignore

from ...enums import Algorithm, AlgorithmGroup
from ...geometry import (
    Centerlines,
    Endpoints,
    intersection_points,
    polygon_to_line,
    remove_duplicated,
)
from ...graph import ProfilePathMapper, get_shortest_paths
from ...raster import DEM
from ...utils import (
    geometries_to_layer,
    get_first_geometry,
    remove_layers_from_project,
)


class Layers(Enum):
    CENTERLINES = auto()
    DIFFERENCE = auto()
    SHORTEST_PATHS_START_POINTS = auto()
    FLOW_PATH_PROFILES = auto()
    DEM_NO_SINKS = auto()
    SHORTEST_PATHS = auto()
    POINTS_INTERSECTING_GULLY = auto()
    FLOW_PATH_PROFILE_POUR_POINTS = auto()
    MAPPED_PROFILES = auto()


class EstimateErosionFuture(QgsProcessingAlgorithm):
    GULLY_BOUNDARY = 'GULLY_BOUNDARY'
    GULLY_ELEVATION = 'GULLY_ELEVATION'
    GULLY_ELEVATION_SINK_REMOVED = 'GULLY_ELEVATION_SINK_REMOVED'
    GULLY_FUTURE_BOUNDARY = 'GULLY_FUTURE_BOUNDARY'
    CENTERLINES = 'CENTERLINES'
    DEBUG_MODE = 'DEBUG_MODE'
    OUTPUT = 'OUTPUT'

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
                [QgsProcessing.TypeVectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.GULLY_ELEVATION,
                self.tr('The gully elevation raster'),
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.GULLY_ELEVATION_SINK_REMOVED,
                self.tr('Sink removed?'),
                False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GULLY_FUTURE_BOUNDARY,
                self.tr('The gully future boundary'),
                [QgsProcessing.TypeVectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CENTERLINES,
                self.tr('The gully future boundary centerlines'),
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DEBUG_MODE,
                self.tr('Debug mode (more logs and intermediary layers)'),
            )
        )

        # TODO: add output

    def processAlgorithm(
        self,
        parameters: dict[str, t.Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback | None = None,
    ):
        # TODO: prevent the code from breaking if feedback is None
        assert feedback is not None
        project = context.project()
        if project is None:
            QgsProcessingException('Failed to retrieve the project.')
        assert project is not None
        crs = project.crs()
        if crs is None:
            QgsProcessingException('Failed to fetch the CRS from the project.')
        gully_boundary = self.parameterAsVectorLayer(
            parameters, self.GULLY_BOUNDARY, context
        )
        gully_elevation = self.parameterAsRasterLayer(
            parameters, self.GULLY_ELEVATION, context
        )
        gully_elevation_is_sink_removed = self.parameterAsBool(
            parameters, self.GULLY_ELEVATION_SINK_REMOVED, context
        )
        eps = gully_elevation.rasterUnitsPerPixelX()
        gully_future_boundary = self.parameterAsVectorLayer(
            parameters, self.GULLY_FUTURE_BOUNDARY, context
        )
        centerlines = self.parameterAsVectorLayer(
            parameters, self.CENTERLINES, context
        )
        debug_mode = self.parameterAsBool(parameters, self.DEBUG_MODE, context)
        if debug_mode:
            remove_layers_from_project(project, Layers._member_names_)
        gully_polygon = get_first_geometry(gully_boundary).coerceToType(
            Qgis.WkbType.Polygon
        )[0]
        gully_limit = polygon_to_line(gully_polygon)
        gully_future_polygon = get_first_geometry(
            gully_future_boundary
        ).coerceToType(Qgis.WkbType.Polygon)[0]

        difference = gully_future_polygon.difference(gully_polygon)
        if debug_mode:
            difference_layer = geometries_to_layer(
                [difference], Layers.DIFFERENCE.name
            )
            difference_layer.setCrs(crs)
            project.addMapLayer(difference_layer)
        limit_difference = polygon_to_line(difference)

        if centerlines is None:
            # Leaving a 'TEMPORARY_OUTPUT' here will create a geopackage
            # instead, which breaks the output file for me (QGIS 3.38.3)
            # By creating a temporay shapefile path via
            # QgsProcessingParameterFileDestination, we make sure that the
            # output will be valid
            temp_path = Path(
                QgsProcessingParameterFileDestination(
                    name='centerlines'
                ).generateTemporaryDestination(context)
            ).with_suffix('.shp')
            centerlines = Centerlines.compute(
                gully_future_boundary, context, feedback, temp_path
            )
            if debug_mode:
                feedback.pushDebugInfo(f'Saved centerlines at {temp_path}')
                centerlines._layer.setName(Layers.CENTERLINES.name)
                project.addMapLayer(centerlines._layer)  # type: ignore
                feedback.pushDebugInfo('Added centerlines in the project')

        else:
            centerlines = Centerlines.from_layer(centerlines)

        pour_points = []
        for centerline in centerlines.intersects(difference):
            first, _ = Endpoints.from_linestring(centerline).as_qgis_geometry()
            if (
                first.intersects(limit_difference)
                and first.distance(gully_limit) > eps
            ):
                pour_points.append(first)
        if debug_mode:
            pour_points_layer = geometries_to_layer(
                pour_points, Layers.SHORTEST_PATHS_START_POINTS.name
            )
            pour_points_layer.setCrs(crs)
            project.addMapLayer(pour_points_layer)

        centerlines_valid = [
            centerline
            for centerline in centerlines
            if not centerline.isEmpty() and centerline.isGeosValid()
        ]
        centerlines_valid_layer = geometries_to_layer(
            centerlines_valid, 'CENTERLINES'
        )
        centerlines = Centerlines(centerlines_valid, centerlines_valid_layer)
        points_intersecting_gully_boundary = intersection_points(
            centerlines, gully_polygon
        )
        if debug_mode:
            points_intersecting_gully_layer = geometries_to_layer(
                points_intersecting_gully_boundary,
                Layers.POINTS_INTERSECTING_GULLY.name,
            )
            points_intersecting_gully_layer.setCrs(crs)
            project.addMapLayer(points_intersecting_gully_layer)
            feedback.pushDebugInfo('Building graph to merge the centerlines.')
        shortest_paths = get_shortest_paths(
            pour_points,
            centerlines._layer,
            points_intersecting_gully_boundary,
            feedback,
        )
        if debug_mode:
            feedback.pushDebugInfo('Graph built.')
        if debug_mode:
            shortest_paths_as_layer = geometries_to_layer(
                [line.path for line in shortest_paths],
                Layers.SHORTEST_PATHS.name,
            )
            shortest_paths_as_layer.setCrs(crs)
            project.addMapLayer(shortest_paths_as_layer)
        if not gully_elevation_is_sink_removed:
            sink_removed = DEM(gully_elevation).remove_sinks(
                context, feedback if debug_mode else None
            )
            if debug_mode:
                sink_removed.layer.setName(Layers.DEM_NO_SINKS.name)
                project.addMapLayer(sink_removed.layer)
        else:
            sink_removed = DEM(gully_elevation)
        profile_pour_points = [
            QgsGeometry.fromPointXY(path.end) for path in shortest_paths
        ]
        profile_pour_points_dedup = remove_duplicated(profile_pour_points)
        if debug_mode:
            profile_pour_points_layer = geometries_to_layer(
                profile_pour_points_dedup,
                Layers.FLOW_PATH_PROFILE_POUR_POINTS.name,
            )
            profile_pour_points_layer.setCrs(crs)
            project.addMapLayer(profile_pour_points_layer)

        profiles = sink_removed.flow_path_profiles_from_points(
            profile_pour_points_dedup,
            eps=eps,
            context=context,
            feedback=feedback if debug_mode else None,
        )
        assert len(profile_pour_points_dedup) == len(profiles), (
            'Pour point count does not match flow path profile count.'
        )
        for profile in profiles:
            profile.convertToMultiType()
        if debug_mode:
            profiles_layer = geometries_to_layer(
                profiles, Layers.FLOW_PATH_PROFILES.name
            )
            profiles_layer.setCrs(crs)
            project.addMapLayer(profiles_layer)
        mapper = ProfilePathMapper(
            profile_pour_points_dedup, profiles, shortest_paths
        )
        mapped_profiles = mapper.get_mapped_profiles()

        if debug_mode:
            mapped_profiles_layer = geometries_to_layer(
                mapped_profiles, name=Layers.MAPPED_PROFILES.name
            )
            mapped_profiles_layer.setCrs(crs)
            project.addMapLayer(mapped_profiles_layer)
        return {self.OUTPUT: None}
