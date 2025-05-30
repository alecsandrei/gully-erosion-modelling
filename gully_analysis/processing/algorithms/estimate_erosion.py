from __future__ import annotations

import typing as t
from dataclasses import dataclass
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
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
)
from qgis.PyQt.QtCore import QCoreApplication  # type: ignore

from ...changepoint import aggregate_samples, get_estimated_samples
from ...enums import Algorithm, AlgorithmGroup
from ...geometry import (
    Centerlines,
    Endpoints,
    intersection_points,
    remove_duplicated,
    to_linestring,
)
from ...graph import ProfileCenterlineMapper, ProfileMapper, get_shortest_paths
from ...raster import (
    DEM,
    BackcastVolumeEvaluator,
    ForecastVolumeEvaluator,
    multilevel_b_spline,
)
from ...utils import (
    delete_holes,
    dissolve_layer,
    export,
    geometries_to_layer,
    get_first_geometry,
    timeit,
)


class Layers(Enum):
    CENTERLINES = auto()
    DIFFERENCE = auto()
    SHORTEST_PATHS_START_POINTS = auto()
    FLOW_PATH_PROFILES = auto()
    RIDGE_LINES = auto()
    DEM_NO_SINKS = auto()
    SHORTEST_PATHS = auto()
    POINTS_INTERSECTING_GULLY = auto()
    FLOW_PATH_PROFILE_POUR_POINTS = auto()
    MAPPED_PROFILES = auto()
    SAMPLES = auto()
    AGGREGATED_SAMPLES = auto()
    ESTIMATED_DEM = auto()
    GULLY_COVER = auto()
    ESTIMATED_SURFACES = auto()


@dataclass
class AdvancedParameters:
    changepoint_penalty: int
    centerline_smoothness: float
    centerline_thin: float
    sample_aggregation: str
    multilevel_b_spline_level: int

    def __str__(self):
        return '\n'.join([f'{k}: {v}' for k, v in self.__dict__.items()])

    def to_file(self, out_file: Path):
        with out_file.open(mode='a') as file:
            file.writelines(['\n' + str(self)])


class EstimateErosionFuture(QgsProcessingAlgorithm):
    GULLY_BOUNDARY = 'GULLY_BOUNDARY'
    GULLY_ELEVATION = 'GULLY_ELEVATION'
    GULLY_ELEVATION_SINK_REMOVED = 'GULLY_ELEVATION_SINK_REMOVED'
    GULLY_FUTURE_ELEVATION = 'GULLY_FUTURE_ELEVATION'
    GULLY_FUTURE_BOUNDARY = 'GULLY_FUTURE_BOUNDARY'
    ESTIMATION_SURFACE = 'ESTIMATION_SURFACE'
    CENTERLINES = 'CENTERLINES'
    DEBUG_MODE = 'DEBUG_MODE'
    CHANGEPOINT_PENALTY = 'CHANGEPOINT_PENALTY'
    CENTERLINE_THIN = 'CENTERLINE_THIN'
    CENTERLINE_SMOOTHNESS = 'CENTERLINE_SMOOTHNESS'
    SAMPLE_AGGREGATE = 'SAMPLE_AGGREGATE'
    MULTILEVEL_B_SPLINE_LEVEL = 'MULTILEVEL_B_SPLINE_LEVEL'
    ESTIMATED_DEM = 'ESTIMATED_DEM'
    ESTIMATION_SURFACE_OUTPUT = 'ESTIMATION_SURFACE_OUTPUT'

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
            'for a future moment.'
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
            QgsProcessingParameterRasterLayer(
                self.GULLY_FUTURE_ELEVATION,
                self.tr('The gully elevation future raster'),
                optional=True,
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
                self.ESTIMATION_SURFACE,
                self.tr(
                    'The area where the estimation is going to be calculated'
                ),
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

        changepoint_penalty_parameter = QgsProcessingParameterNumber(
            self.CHANGEPOINT_PENALTY,
            self.tr(
                'The penalty for the PELT algorithm used for detecting the changepoint for the gully head'
            ),
            defaultValue=10,
        )
        changepoint_penalty_parameter.setFlags(
            changepoint_penalty_parameter.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(changepoint_penalty_parameter)

        centerline_smoothness_param = QgsProcessingParameterNumber(
            self.CENTERLINE_SMOOTHNESS,
            self.tr('The r.voronoi.skeleton smoothness parameter'),
            defaultValue=20,
        )
        centerline_smoothness_param.setFlags(
            centerline_smoothness_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(centerline_smoothness_param)

        centerline_thin_param = QgsProcessingParameterNumber(
            self.CENTERLINE_THIN,
            self.tr('The r.voronoi.skeleton thin parameter'),
            defaultValue=0,
        )
        centerline_thin_param.setFlags(
            centerline_thin_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(centerline_thin_param)

        aggregate_samples_parameter = QgsProcessingParameterString(
            self.SAMPLE_AGGREGATE,
            self.tr(
                'How to aggregate overlapping samples: minimum, maximum, mean etc. (used for interpolating the DEM)'
            ),
            defaultValue='minimum',
        )
        aggregate_samples_parameter.setFlags(
            aggregate_samples_parameter.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(aggregate_samples_parameter)

        multilevel_b_spline_level_param = QgsProcessingParameterNumber(
            self.MULTILEVEL_B_SPLINE_LEVEL,
            self.tr('Multilevel B-Spline interpolation level parameter'),
            defaultValue=14,
        )
        multilevel_b_spline_level_param.setFlags(
            multilevel_b_spline_level_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(multilevel_b_spline_level_param)
        estimated_dem = QgsProcessingParameterRasterDestination(
            self.ESTIMATED_DEM, self.tr('The resulting estimated elevation')
        )
        self.addParameter(estimated_dem)
        estimation_surface_output = QgsProcessingParameterVectorDestination(
            self.ESTIMATION_SURFACE_OUTPUT,
            self.tr('The computed estimated volume in the estimation surfaces'),
        )
        self.addParameter(estimation_surface_output)

    def getAdvancedParameters(
        self, parameters: dict[str, t.Any], context: QgsProcessingContext
    ) -> AdvancedParameters:
        changepoint_penalty = self.parameterAsInt(
            parameters, self.CHANGEPOINT_PENALTY, context
        )
        centerline_smoothness = self.parameterAsDouble(
            parameters, self.CENTERLINE_SMOOTHNESS, context
        )
        centerline_thin = self.parameterAsDouble(
            parameters, self.CENTERLINE_THIN, context
        )
        aggregate_samples = self.parameterAsString(
            parameters, self.SAMPLE_AGGREGATE, context
        )
        multilevel_b_spline_level = self.parameterAsInt(
            parameters, self.MULTILEVEL_B_SPLINE_LEVEL, context
        )
        return AdvancedParameters(
            changepoint_penalty,
            centerline_smoothness,
            centerline_thin,
            aggregate_samples,
            multilevel_b_spline_level,
        )

    def processAlgorithm(
        self,
        parameters: dict[str, t.Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback | None = None,
    ) -> dict[str, t.Any]:
        # TODO: prevent the code from breaking if feedback is None
        assert feedback is not None
        estimated_dem_output = self.parameterAsOutputLayer(
            parameters, self.ESTIMATED_DEM, context
        )
        estimation_surface_output = self.parameterAsOutputLayer(
            parameters, self.ESTIMATION_SURFACE_OUTPUT, context
        )
        out_dir = Path(estimated_dem_output).parent / 'interim_files_future'
        out_dir.mkdir(exist_ok=True)
        project = context.project()
        if project is None:
            QgsProcessingException('Failed to retrieve the project.')
        assert project is not None
        crs = project.crs()
        if crs is None:
            QgsProcessingException('Failed to fetch the CRS from the project.')
        advanced_params = self.getAdvancedParameters(parameters, context)
        gully_boundary = dissolve_layer(
            self.parameterAsVectorLayer(
                parameters, self.GULLY_BOUNDARY, context
            )
        )  # type: ignore
        estimation_surface = self.parameterAsVectorLayer(
            parameters, self.ESTIMATION_SURFACE, context
        )  # type: ignore
        gully_elevation = DEM(
            self.parameterAsRasterLayer(
                parameters, self.GULLY_ELEVATION, context
            )
        )  # type: ignore

        gully_future_elevation = DEM(
            self.parameterAsRasterLayer(
                parameters, self.GULLY_FUTURE_ELEVATION, context
            )
        ).align_to(gully_elevation)
        gully_elevation_is_sink_removed = self.parameterAsBool(
            parameters, self.GULLY_ELEVATION_SINK_REMOVED, context
        )
        cell_size = gully_elevation.layer.rasterUnitsPerPixelX()
        gully_future_boundary = delete_holes(
            dissolve_layer(
                self.parameterAsVectorLayer(
                    parameters, self.GULLY_FUTURE_BOUNDARY, context
                ),
            ),
            0.001,
        )
        centerlines = self.parameterAsVectorLayer(
            parameters, self.CENTERLINES, context
        )
        debug_mode = self.parameterAsBool(parameters, self.DEBUG_MODE, context)
        gully_polygon = get_first_geometry(gully_boundary).coerceToType(
            Qgis.WkbType.Polygon
        )[0]
        gully_limit = to_linestring(gully_polygon)
        gully_future_polygon = get_first_geometry(
            gully_future_boundary
        ).coerceToType(Qgis.WkbType.Polygon)[0]
        gully_future_limit = to_linestring(gully_future_polygon)

        difference = gully_future_polygon.difference(gully_polygon)
        if debug_mode:
            difference_layer = geometries_to_layer(
                [difference], Layers.DIFFERENCE.name
            )
            difference_layer.setCrs(crs)
            _, difference_layer = export(
                difference_layer,
                (out_dir / Layers.DIFFERENCE.name).with_suffix('.fgb'),
            )
            project.addMapLayer(difference_layer)
        limit_difference = to_linestring(difference)

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
            ).with_suffix('.fgb')
            centerlines = Centerlines.compute(
                gully_future_boundary,
                context,
                feedback,
                temp_path.as_posix(),
                smoothness=advanced_params.centerline_smoothness,
                thin=advanced_params.centerline_thin,
                # thin='adaptive',
            )
            centerlines = Centerlines.from_layer(
                export(
                    centerlines._layer, out_file=out_dir / 'centerlines.fgb'
                )[1]
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
                and first.distance(gully_limit) > cell_size
            ):
                pour_points.append(first)
        if debug_mode:
            pour_points_layer = geometries_to_layer(
                pour_points, Layers.SHORTEST_PATHS_START_POINTS.name
            )
            pour_points_layer.setCrs(crs)
            _, pour_points_layer = export(
                pour_points_layer,
                (out_dir / Layers.SHORTEST_PATHS_START_POINTS.name).with_suffix(
                    '.fgb'
                ),
            )
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
            centerlines, gully_polygon, 0
        )
        if debug_mode:
            points_intersecting_gully_layer = geometries_to_layer(
                points_intersecting_gully_boundary,
                Layers.POINTS_INTERSECTING_GULLY.name,
            )
            points_intersecting_gully_layer.setCrs(crs)
            _, points_intersecting_gully_layer = export(
                points_intersecting_gully_layer,
                (out_dir / Layers.POINTS_INTERSECTING_GULLY.name).with_suffix(
                    '.fgb'
                ),
            )
            project.addMapLayer(points_intersecting_gully_layer)
            feedback.pushDebugInfo('Building graph to merge the centerlines.')
        shortest_paths = get_shortest_paths(
            pour_points,
            centerlines._layer,
            points_intersecting_gully_boundary,
            feedback=feedback if debug_mode else None,
        )
        if debug_mode:
            shortest_paths_as_layer = geometries_to_layer(
                [line.path for line in shortest_paths],
                Layers.SHORTEST_PATHS.name,
            )
            shortest_paths_as_layer.setCrs(crs)
            _, shortest_paths_as_layer = export(
                shortest_paths_as_layer,
                (out_dir / Layers.SHORTEST_PATHS.name).with_suffix('.fgb'),
            )
            project.addMapLayer(shortest_paths_as_layer)
        if not gully_elevation_is_sink_removed:
            sink_removed = gully_elevation.remove_sinks(
                context,
                feedback if debug_mode else None,
                (out_dir / 'sink_removed.tif').as_posix(),
            )
            if debug_mode:
                sink_removed.layer.setName(Layers.DEM_NO_SINKS.name)
                project.addMapLayer(sink_removed.layer)
        else:
            sink_removed = gully_elevation
        sink_removed.layer.setCrs(crs)
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
            _, profile_pour_points_layer = export(
                profile_pour_points_layer,
                (
                    out_dir / Layers.FLOW_PATH_PROFILE_POUR_POINTS.name
                ).with_suffix('.fgb'),
            )
            project.addMapLayer(profile_pour_points_layer)
        profiles = sink_removed.flow_path_profiles_from_points(
            profile_pour_points_dedup,
            tolerance=cell_size,
            context=context,
            feedback=feedback if debug_mode else None,
        )
        for profile in profiles:
            profile.convertToMultiType()
        if debug_mode:
            profiles_layer = geometries_to_layer(
                profiles, Layers.FLOW_PATH_PROFILES.name
            )
            profiles_layer.setCrs(crs)
            _, profiles_layer = export(
                profiles_layer,
                (out_dir / Layers.FLOW_PATH_PROFILES.name).with_suffix('.fgb'),
            )
            project.addMapLayer(profiles_layer)
        mapper = ProfileCenterlineMapper(
            profile_pour_points_dedup, profiles, shortest_paths
        )
        mapped_profiles = mapper.get_mapped_profiles()
        if debug_mode:
            mapped_profiles_layer = geometries_to_layer(
                [profile['mapped'] for profile in mapped_profiles],
                name=Layers.MAPPED_PROFILES.name,
            )
            mapped_profiles_layer.setCrs(crs)
            _, mapped_profiles_layer = export(
                mapped_profiles_layer,
                (out_dir / Layers.MAPPED_PROFILES.name).with_suffix('.fgb'),
            )
            project.addMapLayer(mapped_profiles_layer)

        samples = get_estimated_samples(
            dem=sink_removed,
            # dem_truth=gully_future_elevation.remove_sinks(),
            profiles=[
                profiles[profile['profile_index']]
                for profile in mapped_profiles
            ],
            profiles_to_estimate=[
                profile['mapped'] for profile in mapped_profiles
            ],
            crs=crs,
            boundary=gully_future_limit,
            context=context,
            feedback=feedback if debug_mode else None,
            changepoint_penalty=advanced_params.changepoint_penalty,
        )
        aggregated = aggregate_samples(
            samples.estimated, advanced_params.sample_aggregation
        )
        aggregated.setCrs(crs)
        aggregated.setName(Layers.SAMPLES.name)
        if debug_mode:
            _, aggregated = export(
                aggregated,
                (out_dir / Layers.SAMPLES.name).with_suffix('.shp'),
                driver_name='ESRI Shapefile',
            )
            project.addMapLayer(aggregated)
        gully_elevation.layer.setCrs(crs)
        gully_future_elevation.layer.setCrs(crs)
        estimated_dem = (
            multilevel_b_spline(
                aggregated,
                cell_size,
                level=advanced_params.multilevel_b_spline_level,
                context=context,
                feedback=feedback if debug_mode else None,
            )
            .align_to(gully_elevation)
            .gaussian_filter(output=estimated_dem_output)
        )
        gully_cover = (
            multilevel_b_spline(
                samples.boundary,
                cell_size,
                level=advanced_params.multilevel_b_spline_level,
                context=context,
                feedback=feedback if debug_mode else None,
            )
            .align_to(
                gully_elevation, output=(out_dir / 'gully_cover.tif').as_posix()
            )
            .with_name(Layers.GULLY_COVER.name)
        )
        gully_cover.layer.setCrs(crs)
        if debug_mode:
            project.addMapLayer(gully_cover.layer)
        validation_gully_cover = None
        if gully_future_elevation is not None:
            gully_cover_samples = gully_future_elevation.sample(
                [gully_future_limit], feedback=feedback, context=context
            )
            validation_gully_cover = multilevel_b_spline(
                gully_cover_samples,
                cell_size,
                level=14,
                context=context,
                feedback=feedback if debug_mode else None,
            ).align_to(gully_elevation)

        results = {self.ESTIMATED_DEM: estimated_dem_output}
        if estimation_surface is not None:
            evaluator = ForecastVolumeEvaluator(
                computation_surface=estimation_surface,
                gully_cover=gully_cover,
                validation_gully_cover=validation_gully_cover,
                out_file=Path(estimation_surface_output),
                estimated_dem=estimated_dem,
                boundary=gully_boundary,
                dem=gully_elevation,
                validation_dem=gully_future_elevation,
                project=project if debug_mode else None,
            )
            evaluator.evaluate()
            results[self.ESTIMATION_SURFACE_OUTPUT] = estimation_surface_output

        return results


class EstimateErosionPast(QgsProcessingAlgorithm):
    GULLY_BOUNDARY = 'GULLY_BOUNDARY'
    GULLY_ELEVATION = 'GULLY_ELEVATION'
    GULLY_ELEVATION_SINK_REMOVED = 'GULLY_ELEVATION_SINK_REMOVED'
    GULLY_PAST_ELEVATION = 'GULLY_PAST_ELEVATION'
    GULLY_PAST_BOUNDARY = 'GULLY_PAST_BOUNDARY'
    ESTIMATION_SURFACE = 'ESTIMATION_SURFACE'
    CENTERLINES = 'CENTERLINES'
    DEBUG_MODE = 'DEBUG_MODE'
    CHANGEPOINT_PENALTY = 'CHANGEPOINT_PENALTY'
    CENTERLINE_THIN = 'CENTERLINE_THIN'
    CENTERLINE_SMOOTHNESS = 'CENTERLINE_SMOOTHNESS'
    SAMPLE_AGGREGATE = 'SAMPLE_AGGREGATE'
    MULTILEVEL_B_SPLINE_LEVEL = 'MULTILEVEL_B_SPLINE_LEVEL'
    ESTIMATED_DEM = 'ESTIMATED_DEM'
    ESTIMATION_SURFACE_OUTPUT = 'ESTIMATION_SURFACE_OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return EstimateErosionPast()

    def name(self):
        return Algorithm.ESTIMATE_EROSION_PAST.value

    def displayName(self):
        return self.tr(Algorithm.ESTIMATE_EROSION_PAST.display_name())

    def group(self):
        return self.tr(AlgorithmGroup.ESTIMATORS.display_name())

    def groupId(self):
        return AlgorithmGroup.ESTIMATORS.value

    def shortHelpString(self):
        return self.tr(
            'Used to estimate the eroded volume near the gully head '
            'for a past moment.'
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
            QgsProcessingParameterRasterLayer(
                self.GULLY_PAST_ELEVATION,
                self.tr('The gully elevation past raster'),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GULLY_PAST_BOUNDARY,
                self.tr('The gully past boundary'),
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.ESTIMATION_SURFACE,
                self.tr(
                    'The area where the estimation is going to be calculated'
                ),
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CENTERLINES,
                self.tr('The gully past boundary centerlines'),
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

        changepoint_penalty_parameter = QgsProcessingParameterNumber(
            self.CHANGEPOINT_PENALTY,
            self.tr(
                'The penalty for the PELT algorithm used for detecting the changepoint for the gully head'
            ),
            defaultValue=10,
        )
        changepoint_penalty_parameter.setFlags(
            changepoint_penalty_parameter.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(changepoint_penalty_parameter)

        centerline_smoothness_param = QgsProcessingParameterNumber(
            self.CENTERLINE_SMOOTHNESS,
            self.tr('The r.voronoi.skeleton smoothness parameter'),
            defaultValue=20,
        )
        centerline_smoothness_param.setFlags(
            centerline_smoothness_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(centerline_smoothness_param)

        centerline_thin_param = QgsProcessingParameterNumber(
            self.CENTERLINE_THIN,
            self.tr('The r.voronoi.skeleton thin parameter'),
            defaultValue=0,
        )
        centerline_thin_param.setFlags(
            centerline_thin_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(centerline_thin_param)

        aggregate_samples_parameter = QgsProcessingParameterString(
            self.SAMPLE_AGGREGATE,
            self.tr(
                'How to aggregate overlapping samples: minimum, maximum, mean etc. (used for interpolating the DEM)'
            ),
            defaultValue='maximum',
        )
        aggregate_samples_parameter.setFlags(
            aggregate_samples_parameter.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(aggregate_samples_parameter)

        multilevel_b_spline_level_param = QgsProcessingParameterNumber(
            self.MULTILEVEL_B_SPLINE_LEVEL,
            self.tr('Multilevel B-Spline interpolation level parameter'),
            defaultValue=14,
        )
        multilevel_b_spline_level_param.setFlags(
            multilevel_b_spline_level_param.flags()
            | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(multilevel_b_spline_level_param)

        estimated_dem = QgsProcessingParameterRasterDestination(
            self.ESTIMATED_DEM, self.tr('Estimated elevation')
        )
        self.addParameter(estimated_dem)

        estimation_surface_output = QgsProcessingParameterVectorDestination(
            self.ESTIMATION_SURFACE_OUTPUT,
            self.tr('Estimated volume'),
        )
        self.addParameter(estimation_surface_output)

    def getAdvancedParameters(
        self, parameters: dict[str, t.Any], context: QgsProcessingContext
    ) -> AdvancedParameters:
        changepoint_penalty = self.parameterAsInt(
            parameters, self.CHANGEPOINT_PENALTY, context
        )
        centerline_smoothness = self.parameterAsDouble(
            parameters, self.CENTERLINE_SMOOTHNESS, context
        )
        centerline_thin = self.parameterAsDouble(
            parameters, self.CENTERLINE_THIN, context
        )
        aggregate_samples = self.parameterAsString(
            parameters, self.SAMPLE_AGGREGATE, context
        )
        multilevel_b_spline_level = self.parameterAsInt(
            parameters, self.MULTILEVEL_B_SPLINE_LEVEL, context
        )
        return AdvancedParameters(
            changepoint_penalty,
            centerline_smoothness,
            centerline_thin,
            aggregate_samples,
            multilevel_b_spline_level,
        )

    def processAlgorithm(
        self,
        parameters: dict[str, t.Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback | None = None,
    ) -> dict[str, t.Any]:
        # TODO: prevent the code from breaking if feedback is None
        assert feedback is not None
        eps = 0.001
        estimated_dem_output = self.parameterAsOutputLayer(
            parameters, self.ESTIMATED_DEM, context
        )
        estimation_surface_output = self.parameterAsOutputLayer(
            parameters, self.ESTIMATION_SURFACE_OUTPUT, context
        )
        out_dir = Path(estimated_dem_output).parent / 'interim_files_past'
        out_dir.mkdir(exist_ok=True)
        project = context.project()
        if project is None:
            QgsProcessingException('Failed to retrieve the project.')
        assert project is not None
        crs = project.crs()
        if crs is None:
            QgsProcessingException('Failed to fetch the CRS from the project.')
        advanced_params = self.getAdvancedParameters(parameters, context)
        gully_boundary = dissolve_layer(
            self.parameterAsVectorLayer(
                parameters, self.GULLY_BOUNDARY, context
            )
        )
        estimation_surface = self.parameterAsVectorLayer(
            parameters, self.ESTIMATION_SURFACE, context
        )
        gully_elevation = DEM(
            self.parameterAsRasterLayer(
                parameters, self.GULLY_ELEVATION, context
            )
        )
        gully_past_elevation = DEM(
            self.parameterAsRasterLayer(
                parameters, self.GULLY_PAST_ELEVATION, context
            )
        ).align_to(gully_elevation)
        gully_elevation_is_sink_removed = self.parameterAsBool(
            parameters, self.GULLY_ELEVATION_SINK_REMOVED, context
        )
        cell_size = gully_elevation.layer.rasterUnitsPerPixelX()
        gully_past_boundary = delete_holes(
            dissolve_layer(
                self.parameterAsVectorLayer(
                    parameters, self.GULLY_PAST_BOUNDARY, context
                ),
            ),
            eps,
        )
        centerlines = self.parameterAsVectorLayer(
            parameters, self.CENTERLINES, context
        )
        debug_mode = self.parameterAsBool(parameters, self.DEBUG_MODE, context)
        gully_polygon = get_first_geometry(gully_boundary).coerceToType(
            Qgis.WkbType.Polygon
        )[0]
        gully_limit = to_linestring(gully_polygon)
        gully_past_polygon = get_first_geometry(
            gully_past_boundary
        ).coerceToType(Qgis.WkbType.Polygon)[0]
        gully_past_limit = to_linestring(gully_past_polygon)

        difference = gully_polygon.difference(gully_past_polygon)
        difference_layer = geometries_to_layer(
            [difference], Layers.DIFFERENCE.name
        )
        if debug_mode:
            difference_layer.setCrs(crs)
            _, difference_layer = export(
                difference_layer,
                (out_dir / Layers.DIFFERENCE.name).with_suffix('.fgb'),
            )
            project.addMapLayer(difference_layer)
        limit_difference = to_linestring(difference)

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
            ).with_suffix('.fgb')
            centerlines = Centerlines.compute(
                gully_boundary,
                context,
                feedback,
                temp_path.as_posix(),
                smoothness=advanced_params.centerline_smoothness,
                thin=advanced_params.centerline_thin,
            )
            centerlines = Centerlines.from_layer(
                export(
                    centerlines._layer, out_file=out_dir / 'centerlines.fgb'
                )[1]
            )
            if debug_mode:
                feedback.pushDebugInfo(f'Saved centerlines at {temp_path}')
                centerlines._layer.setName(Layers.CENTERLINES.name)
                project.addMapLayer(centerlines._layer)  # type: ignore
                feedback.pushDebugInfo('Added centerlines in the project')

        else:
            centerlines = Centerlines.from_layer(centerlines)

        pour_points = []
        for centerline in centerlines.intersects(limit_difference):
            first, _ = Endpoints.from_linestring(centerline).as_qgis_geometry()
            if first.intersects(limit_difference):
                pour_points.append(first)
        if debug_mode:
            pour_points_layer = geometries_to_layer(
                pour_points, Layers.SHORTEST_PATHS_START_POINTS.name
            )
            pour_points_layer.setCrs(crs)
            _, pour_points_layer = export(
                pour_points_layer,
                (out_dir / Layers.SHORTEST_PATHS_START_POINTS.name).with_suffix(
                    '.fgb'
                ),
            )
            project.addMapLayer(pour_points_layer)

        with timeit('Sink removal'):
            if not gully_elevation_is_sink_removed:
                sink_removed = gully_elevation.remove_sinks(
                    context,
                    feedback if debug_mode else None,
                    (out_dir / 'sink_removed.tif').as_posix(),
                )
                sink_removed.layer.setCrs(crs)
                if debug_mode:
                    sink_removed.layer.setName(Layers.DEM_NO_SINKS.name)
                    project.addMapLayer(sink_removed.layer)
            else:
                sink_removed = gully_elevation
        with timeit('Flow path profiles'):
            profiles = sink_removed.flow_path_profiles_from_points(
                pour_points,
                tolerance=cell_size,
                context=context,
                feedback=feedback if debug_mode else None,
            )
        profiles = [
            profile.intersection(gully_polygon).coerceToType(
                Qgis.WkbType.MultiLineString
            )[0]
            for profile in profiles
        ]
        for profile in profiles:
            if profile.wkbType() == Qgis.WkbType.GeometryCollection:
                profile.convertGeometryCollectionToSubclass(
                    Qgis.GeometryType.Line
                )

        profiles_layer = geometries_to_layer(
            profiles, Layers.FLOW_PATH_PROFILES.name
        )
        profiles_layer.setCrs(crs)
        if debug_mode:
            _, profiles_layer = export(
                profiles_layer,
                (out_dir / Layers.FLOW_PATH_PROFILES.name).with_suffix('.fgb'),
            )
            project.addMapLayer(profiles_layer)
        with timeit('Profile mapping'):
            mapped_profiles = ProfileMapper(
                profiles, difference
            ).get_mapped_profiles()
            mapped_profiles_list = [
                profile_map['mapped'] for profile_map in mapped_profiles
            ]
            for mapped_profile in mapped_profiles_list:
                mapped_profile.convertToMultiType()
        if debug_mode:
            mapped_profiles_layer = geometries_to_layer(
                mapped_profiles_list,
                Layers.MAPPED_PROFILES.name,
            )
            mapped_profiles_layer.setCrs(crs)
            _, mapped_profiles_layer = export(
                mapped_profiles_layer,
                (out_dir / Layers.MAPPED_PROFILES.name).with_suffix('.fgb'),
            )
            project.addMapLayer(mapped_profiles_layer)

        with timeit('Sample limit elevation'):
            gully_limit_sampled = gully_elevation.sample(
                [gully_limit], feedback=feedback, context=context
            )
        with timeit('Get gully cover'):
            gully_cover = multilevel_b_spline(
                gully_limit_sampled,
                cell_size,
                level=14,
                context=context,
                feedback=feedback if debug_mode else None,
            ).align_to(
                gully_elevation, output=(out_dir / 'gully_cover.tif').as_posix()
            )
        if debug_mode:
            gully_cover.layer.setCrs(crs)
            gully_cover.layer.setName(Layers.GULLY_COVER.name)
            project.addMapLayer(gully_cover.layer)
        with timeit('Sample gully cover'):
            gully_cover_sampled = gully_cover.sample(
                [gully_past_limit],
                feedback=feedback,
                context=context,
            )

        with timeit('Get flow path samples'):
            flow_path_samples = get_estimated_samples(
                dem=sink_removed,
                gully_cover=gully_cover,
                dem_truth=gully_past_elevation.remove_sinks(),
                profiles=[
                    profiles[profile['profile_index']]
                    for profile in mapped_profiles
                ],
                crs=crs,
                profiles_to_estimate=[
                    profile['mapped'] for profile in mapped_profiles
                ],
                sampled_boundary=gully_cover_sampled,
                context=context,
                feedback=feedback if debug_mode else None,
                changepoint_penalty=advanced_params.changepoint_penalty,
            )
        with timeit('Aggregate flow path samples'):
            aggregated = aggregate_samples(
                flow_path_samples.estimated,
                advanced_params.sample_aggregation,
            )
        if debug_mode:
            aggregated.setCrs(crs)
            aggregated.setName(Layers.AGGREGATED_SAMPLES.name)
            _, aggregated = export(
                aggregated,
                (out_dir / Layers.AGGREGATED_SAMPLES.name).with_suffix('.shp'),
                driver_name='ESRI Shapefile',
            )
            project.addMapLayer(aggregated)
        gully_elevation.layer.setCrs(crs)
        gully_past_elevation.layer.setCrs(crs)
        with timeit('Estimate DEM'):
            estimated_dem = (
                multilevel_b_spline(
                    aggregated,
                    cell_size,
                    level=advanced_params.multilevel_b_spline_level,
                    context=context,
                    feedback=feedback if debug_mode else None,
                )
                .align_to(gully_elevation)
                .gaussian_filter(output=estimated_dem_output)
            )
        estimated_dem.layer.setCrs(crs)
        estimated_dem.layer.setName(Layers.ESTIMATED_DEM.name)

        validation_gully_cover = None
        if gully_past_elevation is not None:
            gully_cover_samples = gully_past_elevation.sample(
                [gully_past_limit], feedback=feedback, context=context
            )
            validation_gully_cover = multilevel_b_spline(
                gully_cover_samples,
                cell_size,
                level=14,
                context=context,
                feedback=feedback if debug_mode else None,
            ).align_to(gully_elevation)

        results = {self.ESTIMATED_DEM: estimated_dem_output}
        if estimation_surface is not None:
            with timeit('Evaluate volume'):
                evaluator = BackcastVolumeEvaluator(
                    computation_surface=estimation_surface,
                    gully_cover=gully_cover,
                    out_file=Path(estimation_surface_output),
                    project=project if debug_mode else None,
                    dem=gully_elevation,
                    estimated_dem=estimated_dem,
                    estimated_boundary=gully_past_boundary,
                    validation_dem=gully_past_elevation,
                    validation_gully_cover=validation_gully_cover,
                )

                evaluator.evaluate()
            results[self.ESTIMATION_SURFACE_OUTPUT] = estimation_surface_output

        return results
