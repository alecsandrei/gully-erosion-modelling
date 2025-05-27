from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import processing
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)

from .exceptions import InvalidCoordinateSystem
from .utils import (
    difference,
    geometries_to_layer,
    get_geometries_from_layer,
    get_geometries_from_path,
    intersection,
)

if t.TYPE_CHECKING:
    from qgis.core import QgsGeometry


class Extent(t.NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    epsg: str

    @staticmethod
    def from_raster(raster: Raster, epsg: str | None = None):
        extent = raster.layer.extent()
        if epsg is None:
            epsg = raster.crs().geographicCrsAuthId()
        if not epsg:
            raise InvalidCoordinateSystem('EPSG code not provided.')
        return Extent(
            extent.xMinimum(),
            extent.xMaximum(),
            extent.yMinimum(),
            extent.yMaximum(),
            epsg,
        )

    def __str__(self):
        """Makes the object usable as an input to QGIS processing."""
        return ' '.join(
            [','.join(str(coord) for coord in self[:4]), f'[EPSG:{self.epsg}]']
        )


T = t.TypeVar('T', bound='Raster')


@dataclass
class Raster:
    layer: QgsRasterLayer

    def __sub__(self: T, other: T) -> T:
        return self.difference(other)

    def crs(self) -> QgsCoordinateReferenceSystem:
        crs = self.layer.metadata().crs()
        if not crs.isValid():
            crs = self.layer.crs()
        return crs

    def difference(self: T, other: Raster) -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'native:rastercalc',
                    {
                        'LAYERS': [
                            self.layer,
                            other.layer,
                        ],
                        'EXPRESSION': f'"{self.layer.name()}@1"-"{other.layer.name()}@1"',
                        'EXTENT': None,
                        'CELL_SIZE': None,
                        'CRS': None,
                        'OUTPUT': 'TEMPORARY_OUTPUT',
                    },
                )['OUTPUT'],
                self.layer.name(),
                'gdal',
            )
        )

    # def statistics(self) -> dict[str, t.Any]:
    #     return processing.run(
    #         'native:rasterlayerstatistics',
    #         {
    #             'INPUT': self.layer,
    #             'BAND': 1,
    #             'OUTPUT_HTML_FILE': 'TEMPORARY_OUTPUT',
    #         },
    #     )

    def invert(self: T) -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'sagang:invertgrid',
                    {
                        'GRID': self.layer,
                        'INVERSE': 'TEMPORARY_OUTPUT',
                    },
                )['INVERSE']
            )
        )

    def mean(self: T, other: Raster) -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'native:rastercalc',
                    {
                        'LAYERS': [
                            self.layer,
                            other.layer,
                        ],
                        'EXPRESSION': f'("{self.layer.name()}@1"+"{other.layer.name()}@1") / 2',
                        'EXTENT': None,
                        'CELL_SIZE': None,
                        'CRS': None,
                        'OUTPUT': 'TEMPORARY_OUTPUT',
                    },
                )['OUTPUT'],
                self.layer.name(),
                'gdal',
            )
        )

    @property
    def extent(self) -> Extent:
        return Extent.from_raster(self)

    def fillna(
        self: T, value: float = 0, output: str = 'TEMPORARY_OUTPUT'
    ) -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'native:fillnodata',
                    {
                        'INPUT': self.layer,
                        'BAND': 1,
                        'FILL_VALUE': value,
                        'OUTPUT': output,
                    },
                )['OUTPUT']
            )
        )

    @classmethod
    def from_rasters(
        cls, rasters: c.Sequence[Raster], output: str = 'TEMPORARY_OUTPUT'
    ) -> Raster:
        return cls(
            QgsRasterLayer(
                processing.run(
                    'gdal:merge',
                    {
                        'INPUT': [raster.layer for raster in rasters],
                        'PCT': False,
                        'SEPARATE': False,
                        'NODATA_INPUT': None,
                        'NODATA_OUTPUT': None,
                        'OPTIONS': '',
                        'EXTRA': '',
                        'DATA_TYPE': 5,
                        'OUTPUT': output,
                    },
                )['OUTPUT'],
                'merged',
                'gdal',
            )
        )

    def apply_mask(
        self: T, mask: QgsVectorLayer, output: str = 'TEMPORARY_OUTPUT'
    ) -> T:
        # masked = processing.run(
        #     'gdal:cliprasterbymasklayer',
        #     {
        #         'INPUT': self.layer,
        #         'MASK': mask,
        #         'SOURCE_CRS': self.layer.crs(),
        #         'TARGET_CRS': self.layer.crs(),
        #         'TARGET_EXTENT': None,
        #         'NODATA': None,
        #         'ALPHA_BAND': False,
        #         'CROP_TO_CUTLINE': True,
        #         'KEEP_RESOLUTION': False,
        #         'SET_RESOLUTION': False,
        #         'X_RESOLUTION': None,
        #         'Y_RESOLUTION': None,
        #         'MULTITHREADING': False,
        #         'OPTIONS': None,
        #         'DATA_TYPE': 0,
        #         'EXTRA': '',
        #         'OUTPUT': output,
        #     },
        # )
        # return type(self)(
        #     QgsRasterLayer(masked['OUTPUT'], self.layer.name(), 'gdal')
        # )
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'sagang:cliprasterwithpolygon',
                    {
                        'INPUT': [
                            self.layer,
                        ],
                        'OUTPUT': output,
                        'POLYGONS': mask,
                        'EXTENT': 2,
                    },
                )['OUTPUT'],
                self.layer.name(),
                'gdal',
            )
        )

    def rank_filter(
        self: T,
        rank: int,
        kernel_radius: int = 2,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
        output: str = 'TEMPORARY_OUTPUT',
    ) -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'sagang:rankfilter',
                    {
                        'INPUT': self.layer,
                        'RESULT': output,
                        'RANK': rank,
                        'KERNEL_TYPE': 1,
                        'KERNEL_RADIUS': kernel_radius,
                    },
                )['RESULT'],
                self.layer.name(),
                'gdal',
            ),
        )

    def align_to(
        self: T,
        raster: Raster,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
        output: str = 'TEMPORARY_OUTPUT',
    ) -> T:
        aligned = processing.run(
            'gdal:warpreproject',
            {
                'INPUT': self.layer,
                'SOURCE_CRS': None,
                'TARGET_CRS': None,
                'RESAMPLING': 1,
                'NODATA': None,
                'TARGET_RESOLUTION': None,
                'OPTIONS': None,
                'DATA_TYPE': 0,
                'TARGET_EXTENT': str(raster.extent),
                'TARGET_EXTENT_CRS': None,
                'MULTITHREADING': False,
                'EXTRA': '',
                'OUTPUT': output,
            },
            context=context,
            feedback=feedback,
        )
        return type(self)(
            QgsRasterLayer(aligned['OUTPUT'], self.layer.name(), 'gdal')
        )

    def with_name(self: T, name: str) -> T:
        self.layer.setName(name)
        return self

    def raster_volume(self: T) -> T:
        cell_area = (
            self.layer.rasterUnitsPerPixelX()
            * self.layer.rasterUnitsPerPixelY()
        )
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'gdal:rastercalculator',
                    {
                        'INPUT_A': self.layer,
                        'BAND_A': 1,
                        'INPUT_B': None,
                        'BAND_B': None,
                        'INPUT_C': None,
                        'BAND_C': None,
                        'INPUT_D': None,
                        'BAND_D': None,
                        'INPUT_E': None,
                        'BAND_E': None,
                        'INPUT_F': None,
                        'BAND_F': None,
                        'FORMULA': f'A*{cell_area}',
                        'NO_DATA': None,
                        'EXTENT_OPT': 0,
                        'PROJWIN': None,
                        'RTYPE': 5,
                        'OPTIONS': '',
                        'EXTRA': '',
                        'OUTPUT': 'TEMPORARY_OUTPUT',
                    },
                )['OUTPUT'],
                self.layer.name(),
                'gdal',
            )
        )

    def get_raster_surface_volume(self) -> float:
        feedback = QgsProcessingFeedback()
        results = processing.run(
            'native:rastersurfacevolume',
            {
                'INPUT': self.layer,
                'BAND': 1,
                'LEVEL': 0,
                'METHOD': 0,
                'OUTPUT_HTML_FILE': 'TEMPORARY_OUTPUT',
            },
            feedback=feedback,
        )
        return results['VOLUME']

    def gaussian_filter(self: T, output: str = 'TEMPORARY_OUTPUT') -> T:
        return type(self)(
            QgsRasterLayer(
                processing.run(
                    'sagang:gaussianfilter',
                    {
                        'INPUT': self.layer,
                        'RESULT': output,
                        'KERNEL_RADIUS': 2,
                        'SIGMA': 50,
                    },
                )['RESULT'],
                self.layer.name(),
                'gdal',
            )
        )

    def statistics(self) -> dict[str, t.Any]:
        return processing.run(
            'native:rasterlayerstatistics',
            {
                'INPUT': self.layer,
                'BAND': 1,
                'OUTPUT_HTML_FILE': 'TEMPORARY_OUTPUT',
            },
        )

    def zonal_statistics(
        self, vector: QgsVectorLayer, column_prefix: str = 'stat_'
    ) -> QgsVectorLayer:
        return processing.run(
            'native:zonalstatisticsfb',
            {
                'INPUT': vector,
                'INPUT_RASTER': '/media/alex/alex/python-modules-packages-utils/gully-analysis/data/date/soldanesti_amonte/soldanesti_amonte_2012.tif',
                'RASTER_BAND': 1,
                'COLUMN_PREFIX': column_prefix,
                'STATISTICS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']


@dataclass
class DEM(Raster):
    def remove_sinks(
        self,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
        output: str = 'TEMPORARY_OUTPUT',
    ) -> DEM:
        sink_route = processing.run(
            'sagang:sinkdrainageroutedetection',
            {
                'ELEVATION': self.layer,
                'SINKROUTE': 'TEMPORARY_OUTPUT',
                'THRESHOLD': False,
                'THRSHEIGHT': 100,
            },
            context=context,
            feedback=feedback,
        )
        dem_preproc = processing.run(
            'sagang:sinkremoval',
            {
                'DEM': self.layer,
                'SINKROUTE': sink_route['SINKROUTE'],
                'DEM_PREPROC': output,
                'METHOD': 1,
                'THRESHOLD': False,
                'THRSHEIGHT': 100,
            },
            context=context,
            feedback=feedback,
        )
        return DEM(
            QgsRasterLayer(
                dem_preproc['DEM_PREPROC'],
                f'{self.layer.name()}_no_sinks',
                'gdal',
            )
        )

    def flow_accumulation(self) -> Raster:
        return Raster(
            QgsRasterLayer(
                processing.run(
                    'sagang:flowaccumulationparallelizable',
                    {
                        'DEM': self.layer,
                        'FLOW': 'TEMPORARY_OUTPUT',
                        'UPDATE': 100,
                        'METHOD': 0,
                        'CONVERGENCE': 1.1,
                    },
                )['FLOW'],
                f'{self.layer.name()}_flow_accumulation',
                'gdal',
            )
        )

    def channel_network(
        self,
        flow_accumulation: Raster | None = None,
        output: str = 'TEMPORARY_OUTPUT',
    ) -> QgsVectorLayer:
        if flow_accumulation is None:
            flow_accumulation = self.flow_accumulation()
        return QgsVectorLayer(
            processing.run(
                'sagang:channelnetwork',
                {
                    'ELEVATION': self.layer,
                    'SINKROUTE': None,
                    'CHNLNTWRK': 'TEMPORARY_OUTPUT',
                    'CHNLROUTE': 'TEMPORARY_OUTPUT',
                    'SHAPES': output,
                    'INIT_GRID': flow_accumulation.layer,
                    'INIT_METHOD': 2,
                    'INIT_VALUE': flow_accumulation.statistics()['MEAN'],
                    # 'INIT_VALUE': 0,
                    'DIV_GRID': None,
                    'DIV_CELLS': 5,
                    'TRACE_WEIGHT': None,
                    'MINLEN': 10,
                },
            )['SHAPES'],
            f'{self.layer.name()}_channel_network',
            'ogr',
        )

    def flow_path_profiles_from_points(
        self,
        points: c.Sequence[QgsGeometry],
        tolerance: float = 0.5,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
    ) -> list[QgsGeometry]:
        """Get flow path profiles from pour points."""
        points_as_layer = geometries_to_layer(points, 'pour_points')
        points_as_layer.setCrs(self.layer.crs())
        profiles = processing.run(
            'sagang:leastcostpaths',
            {
                'SOURCE': points_as_layer,
                'DEM': self.layer,
                'VALUES': None,
                'POINTS': 'TEMPORARY_OUTPUT',
                'LINE': 'TEMPORARY_OUTPUT',
            },
            context=context,
            feedback=feedback,
        )
        points_layer = geometries_to_layer(points)

        profile_paths = list(Path(profiles['LINE']).parent.glob('*.shp'))
        if len(profile_paths) > 1:
            profile_paths = sorted(
                profile_paths,
                key=lambda path: int(path.stem.replace('LINE', '')),
            )
        profiles = list(get_geometries_from_path(*profile_paths))
        valid = [profile for profile in profiles if profile.isGeosValid()]
        profiles_layer = geometries_to_layer(valid)
        snapped_profiles = processing.run(
            'native:snapgeometries',
            {
                'INPUT': profiles_layer,
                'REFERENCE_LAYER': points_layer,
                'TOLERANCE': tolerance,
                'BEHAVIOR': 6,
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']
        return list(get_geometries_from_layer(snapped_profiles))

    def sample(
        self,
        lines: c.Sequence[QgsGeometry] | QgsVectorLayer,
        feedback: QgsProcessingFeedback | None = None,
        context: QgsProcessingContext | None = None,
        output: str = 'TEMPORARY_OUTPUT',
    ) -> QgsVectorLayer:
        if isinstance(lines, c.Sequence):
            lines = geometries_to_layer(lines)
        lines.setCrs(self.layer.crs())

        profiles = processing.run(
            'sagang:profilesfromlines',
            {
                'DEM': self.layer,
                'VALUES': None,
                'LINES': lines,
                'NAME': 'FID',
                'PROFILE': output,
                'PROFILES': 'TEMPORARY_OUTPUT',
                'SPLIT': False,
            },
            feedback=feedback,
            context=context,
        )
        return QgsVectorLayer(profiles['PROFILE'], '', 'ogr')


def multilevel_b_spline(
    points: QgsVectorLayer,
    cell_size: float,
    level: int = 10,
    context: QgsProcessingContext | None = None,
    feedback: QgsProcessingFeedback | None = None,
    output: str = 'TEMPORARY_OUTPUT',
) -> DEM:
    return DEM(
        QgsRasterLayer(
            processing.run(
                'sagang:multilevelbspline',
                {
                    'SHAPES': points,
                    'FIELD': 'Z',
                    'TARGET_USER_SIZE': cell_size,
                    'TARGET_USER_FITS': 0,
                    'TARGET_OUT_GRID': output,
                    'METHOD': 0,
                    'LEVEL_MAX': level,
                },
                context=context,
                feedback=feedback,
            )['TARGET_OUT_GRID'],
            '',
            'gdal',
        )
    )


def inverse_distance_weighted(
    points: QgsVectorLayer,
    cell_size: float,
    power: int = 2,
    weighting: int = 1,
    output: str = 'TEMPORARY_OUTPUT',
    context: QgsProcessingContext | None = None,
    feedback: QgsProcessingFeedback | None = None,
) -> DEM:
    return DEM(
        QgsRasterLayer(
            processing.run(
                'sagang:inversedistanceweightedinterpolation',
                {
                    'POINTS': points,
                    'FIELD': 'Z',
                    'CV_METHOD': 0,
                    'CV_SUMMARY': 'TEMPORARY_OUTPUT',
                    'CV_RESIDUALS': 'TEMPORARY_OUTPUT',
                    'CV_SAMPLES': 10,
                    'TARGET_USER_SIZE': cell_size,
                    'TARGET_USER_FITS': 0,
                    'TARGET_OUT_GRID': output,
                    'SEARCH_RANGE': 1,
                    'SEARCH_RADIUS': 1000,
                    'SEARCH_POINTS_ALL': 1,
                    'SEARCH_POINTS_MIN': 1,
                    'SEARCH_POINTS_MAX': 20,
                    'DW_WEIGHTING': weighting,
                    'DW_IDW_POWER': power,
                    'DW_BANDWIDTH': 1,
                },
                context=context,
                feedback=feedback,
            )['TARGET_OUT_GRID'],
            '',
            'gdal',
        )
    )


# @dataclass
# class VolumeEvaluatorFuture:
#     estimation_dem: DEM
#     truth_dem: DEM
#     gully_cover: DEM
#     estimation_surface: QgsVectorLayer
#     out_dir: Path
#     crs: QgsCoordinateReferenceSystem

#     def __post_init__(self):
#         self.estimation_dem = self.estimation_dem.apply_mask(
#             self.estimation_surface,
#             output=(
#                 self.out_dir / f'masked_{self.estimation_dem.layer.name()}.tif'
#             ).as_posix(),
#         )
#         self.truth_dem = self.truth_dem.apply_mask(
#             self.estimation_surface,
#             output=(
#                 self.out_dir / f'masked_{self.truth_dem.layer.name()}.tif'
#             ).as_posix(),
#         )
#         self.gully_cover = self.gully_cover.apply_mask(
#             self.estimation_surface,
#             output=(
#                 self.out_dir / f'masked_{self.gully_cover.layer.name()}.tif'
#             ).as_posix(),
#         )
#         for layer in (
#             self.estimation_dem,
#             self.truth_dem,
#             self.gully_cover,
#         ):
#             layer.layer.setCrs(self.crs)

#     def get_masked(self) -> list[DEM]:
#         return [
#             dem.apply_mask(self.estimation_surface)
#             for dem in (
#                 self.estimation_dem,
#                 self.truth_dem,
#                 self.gully_cover,
#             )
#         ]

#     def evaluate(self) -> QgsVectorLayer:
#         # NOTE: This does not account for the values which are negative
#         truth_diff = self.gully_cover - self.truth_dem
#         truth_volume = truth_diff.raster_volume()
#         estimated_diff = self.gully_cover - self.estimation_dem
#         estimated_volume = estimated_diff.raster_volume()
#         truth_zonal = processing.run(
#             'native:zonalstatisticsfb',
#             {
#                 'INPUT': self.estimation_surface,
#                 'INPUT_RASTER': truth_volume.layer,
#                 'RASTER_BAND': 1,
#                 'COLUMN_PREFIX': 'truth_',
#                 'STATISTICS': [1],  # this is sum
#                 'OUTPUT': 'TEMPORARY_OUTPUT',
#             },
#         )['OUTPUT']
#         estimated_zonal = processing.run(
#             'native:zonalstatisticsfb',
#             {
#                 'INPUT': truth_zonal,
#                 'INPUT_RASTER': estimated_volume.layer,
#                 'RASTER_BAND': 1,
#                 'COLUMN_PREFIX': 'estimated_',
#                 'STATISTICS': [1],  # this is sum
#                 'OUTPUT': 'TEMPORARY_OUTPUT',
#             },
#         )['OUTPUT']

#         with_error = processing.run(
#             'native:fieldcalculator',
#             {
#                 'INPUT': estimated_zonal,
#                 'FIELD_NAME': 'error',
#                 'FIELD_TYPE': 0,
#                 'FIELD_LENGTH': 0,
#                 'FIELD_PRECISION': 0,
#                 'FORMULA': 'abs("truth_sum" -  "estimated_sum") / "truth_sum"',
#                 'OUTPUT': 'TEMPORARY_OUTPUT',
#             },
#         )['OUTPUT']

#         return with_error


@dataclass
class VolumeEvaluator:
    past_dem: DEM
    future_dem: DEM
    future_limit: QgsGeometry
    past_boundary: QgsVectorLayer
    estimation_surface: QgsVectorLayer
    gully_cover: DEM
    out_file: Path
    validation_future_dem: DEM | None = None
    validation_past_dem: DEM | None = None
    validation_gully_cover: DEM | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.validation_past_dem is not None:
            self.validation_past_dem.layer.rasterUnitsPerPixelX()

            self.validation_gully_cover = (
                self.validation_past_dem.rank_filter(rank=50)
                .align_to(self.validation_past_dem)
                .with_name('validation_gully_cover')
            )
            # self.validation_gully_cover = (
            #    multilevel_b_spline(
            #        self.validation_past_dem.sample(
            #            [self.future_limit],
            #            output=(
            #                self.out_dir / 'validation_past_dem_samples.shp'
            #            ).as_posix(),
            #        ),
            #        cell_size,
            #        level=14,
            #        output=(
            #            self.out_dir / 'validation_gully_cover.sdat'
            #        ).as_posix(),
            #    )
            #    .align_to(self.validation_past_dem)
            #    .with_name('validation_gully_cover')
            # )

    def evaluate(self, project: QgsProject) -> QgsVectorLayer:
        assert self.gully_cover is not None
        if self.validation_gully_cover:
            project.addMapLayer(self.validation_gully_cover.layer)
        past = intersection(self.estimation_surface, self.past_boundary)
        future = difference(self.estimation_surface, self.past_boundary)
        gully_future_volume = (
            (
                self.gully_cover.apply_mask(future)
                - self.future_dem.apply_mask(future)
            )
            .raster_volume()
            .with_name('gully_future_volume')
        )
        gully_past_volume = (
            (self.past_dem.apply_mask(past) - self.future_dem.apply_mask(past))
            .raster_volume()
            .with_name('gully_past_volume')
        )
        truth = None
        if self.validation_past_dem is not None:
            # assert self.validation_gully_cover is not None
            # gully_past_volume_truth = (
            #    (
            #        self.validation_past_dem.apply_mask(past)
            #        - self.future_dem.apply_mask(past)
            #    )
            #    .raster_volume()
            #    .with_name('gully_past_volume_truth')
            # )
            # gully_future_volume_truth = (
            #    (
            #        self.validation_gully_cover.apply_mask(future)
            #        - self.future_dem.apply_mask(future)
            #    )
            #    .raster_volume()
            #    .with_name('gully_future_volume_truth')
            # )
            # truth = Raster.from_rasters(
            #    [gully_future_volume_truth, gully_past_volume_truth]
            # ).with_name('truth')
            truth = (
                (
                    self.validation_past_dem.apply_mask(self.estimation_surface)
                    - self.future_dem.apply_mask(self.estimation_surface)
                )
                .raster_volume()
                .with_name('truth')
            )
        elif self.validation_future_dem is not None:
            assert self.gully_cover is not None
            gully_future_volume_truth = (
                (
                    self.gully_cover.apply_mask(future)
                    - self.validation_future_dem.apply_mask(future)
                )
                .raster_volume()
                .with_name('gully_future_volume_truth')
            )
            gully_past_volume_truth = (
                (
                    self.past_dem.apply_mask(past)
                    - self.validation_future_dem.apply_mask(past)
                )
                .raster_volume()
                .with_name('gully_past_volume')
            )
            truth = Raster.from_rasters(
                [gully_future_volume_truth, gully_past_volume_truth]
            ).with_name('truth')
        estimation = Raster.from_rasters(
            [gully_future_volume, gully_past_volume]
        ).with_name('estimation')
        project.addMapLayer(estimation.layer)
        zonal_statistics = processing.run(
            'native:zonalstatisticsfb',
            {
                'INPUT': self.estimation_surface,
                'INPUT_RASTER': estimation.layer,
                'RASTER_BAND': 1,
                'COLUMN_PREFIX': 'estimated_',
                'STATISTICS': [1],  # this is sum
                'OUTPUT': 'TEMPORARY_OUTPUT'
                if truth is None
                else self.out_file.as_posix(),
            },
        )['OUTPUT']
        if truth is not None:
            project.addMapLayer(truth.layer)
            zonal_statistics = processing.run(
                'native:zonalstatisticsfb',
                {
                    'INPUT': zonal_statistics,
                    'INPUT_RASTER': truth.layer,
                    'RASTER_BAND': 1,
                    'COLUMN_PREFIX': 'truth_',
                    'STATISTICS': [1],  # this is sum
                    'OUTPUT': 'TEMPORARY_OUTPUT',
                },
            )['OUTPUT']

            with_error = processing.run(
                'native:fieldcalculator',
                {
                    'INPUT': zonal_statistics,
                    'FIELD_NAME': 'error',
                    'FIELD_TYPE': 0,
                    'FIELD_LENGTH': 0,
                    'FIELD_PRECISION': 0,
                    'FORMULA': 'abs("truth_sum" -  "estimated_sum") / "truth_sum"',
                    'OUTPUT': self.out_file.as_posix(),
                },
            )['OUTPUT']
            return with_error
        return zonal_statistics
