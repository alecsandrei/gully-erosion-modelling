from __future__ import annotations

import abc
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
    merge_vector_layers,
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

    def least_cost_paths(
        self,
        source_points: QgsVectorLayer,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
    ) -> dict[str, t.Any]:
        return processing.run(
            'sagang:leastcostpaths',
            {
                'SOURCE': source_points,
                'DEM': self.layer,
                'VALUES': None,
                'POINTS': 'TEMPORARY_OUTPUT',
                'LINE': 'TEMPORARY_OUTPUT',
            },
            context=context,
            feedback=feedback,
        )

    def flow_path_profiles_from_points(
        self,
        source_points: c.Sequence[QgsGeometry],
        tolerance: float = 0.5,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
    ) -> list[QgsGeometry]:
        """Get flow path profiles from pour points."""
        source_points_as_layer = geometries_to_layer(
            source_points, 'pour_points'
        )
        source_points_as_layer.setCrs(self.layer.crs())
        source_points_layer = geometries_to_layer(source_points)
        profiles_least_cost = self.least_cost_paths(
            source_points_layer,
            context=context,
            feedback=feedback,
        )
        profile_paths = list(
            Path(profiles_least_cost['LINE']).parent.glob('*.shp')
        )
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
                'REFERENCE_LAYER': source_points_as_layer,
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


@dataclass(kw_only=True)
class VolumeEvaluator(abc.ABC):
    computation_surface: QgsVectorLayer
    gully_cover: DEM
    out_file: Path
    project: QgsProject | None = None

    @abc.abstractmethod
    def compute_delta(self) -> tuple[Raster, Raster | None]:
        """Computes the volume difference between the estimated and the truth DEM."""
        ...

    @abc.abstractmethod
    def compute_volume(self) -> tuple[Raster, Raster | None]:
        """Computes the volume of the estimated DEM."""
        ...

    @abc.abstractmethod
    def evaluate(self) -> QgsVectorLayer: ...

    def zonal_statistics(
        self,
        input: QgsVectorLayer,
        estimation: Raster,
        truth: Raster | None = None,
        prefix: str = '',
    ) -> QgsVectorLayer:
        zonal_statistics = processing.run(
            'native:zonalstatisticsfb',
            {
                'INPUT': input,
                'INPUT_RASTER': estimation.layer,
                'RASTER_BAND': 1,
                'COLUMN_PREFIX': f'{prefix}estimated_',
                'STATISTICS': [1],  # this is sum
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']
        if truth is not None:
            if self.project is not None:
                self.project.addMapLayer(truth.layer)
            zonal_statistics_truth = processing.run(
                'native:zonalstatisticsfb',
                {
                    'INPUT': zonal_statistics,
                    'INPUT_RASTER': truth.layer,
                    'RASTER_BAND': 1,
                    'COLUMN_PREFIX': f'{prefix}truth_',
                    'STATISTICS': [1],  # this is sum
                    'OUTPUT': 'TEMPORARY_OUTPUT',
                },
            )['OUTPUT']

            zonal_statistics = processing.run(
                'native:fieldcalculator',
                {
                    'INPUT': zonal_statistics_truth,
                    'FIELD_NAME': f'{prefix}error',
                    'FIELD_TYPE': 0,
                    'FIELD_LENGTH': 0,
                    'FIELD_PRECISION': 0,
                    'FORMULA': f'abs("{prefix}truth_sum" -  "{prefix}estimated_sum") / "{prefix}truth_sum"',
                    'OUTPUT': 'TEMPORARY_OUTPUT',
                },
            )['OUTPUT']
        return zonal_statistics


@dataclass(kw_only=True)
class ForecastVolumeEvaluator(VolumeEvaluator):
    dem: DEM
    estimated_dem: DEM
    boundary: QgsVectorLayer
    validation_dem: DEM | None = None
    validation_gully_cover: DEM | None = None
    boundary_intersect: QgsVectorLayer = field(init=False)
    boundary_difference: QgsVectorLayer = field(init=False)

    def __post_init__(self) -> None:
        self.boundary_intersect = intersection(
            self.computation_surface, self.boundary
        )
        self.boundary_difference = difference(
            self.computation_surface, self.boundary
        )

    def evaluate(self) -> QgsVectorLayer:
        estimation_delta, truth_delta = self.compute_delta()
        delta_statistics = self.zonal_statistics(
            self.computation_surface,
            estimation_delta,
            truth_delta,
            prefix='delta_',
        )
        estimation_volume, truth_volume = self.compute_volume()
        volume_statistics = self.zonal_statistics(
            self.computation_surface,
            estimation_volume,
            truth_volume,
            prefix='volume_',
        )
        return processing.run(
            'native:joinattributesbylocation',
            {
                'INPUT': delta_statistics,
                'PREDICATE': [2],  # equals
                'JOIN': volume_statistics,
                'JOIN_FIELDS': [],
                'METHOD': 0,
                'DISCARD_NONMATCHING': False,
                'PREFIX': '',
                'OUTPUT': self.out_file.as_posix(),
            },
        )['OUTPUT']

    def compute_delta(self) -> tuple[Raster, Raster | None]:
        estimation = (
            (
                self.dem.apply_mask(self.computation_surface)
                - self.estimated_dem.apply_mask(self.computation_surface)
            )
            .raster_volume()
            .with_name('estimation_delta')
        )
        truth = None
        if self.validation_dem is not None:
            truth = (
                (
                    self.dem.apply_mask(self.computation_surface)
                    - self.validation_dem.apply_mask(self.computation_surface)
                )
                .raster_volume()
                .with_name('truth_delta')
            )
        return (estimation, truth)

    def compute_volume(self) -> tuple[Raster, Raster | None]:
        estimation = (
            (
                self.gully_cover.apply_mask(self.computation_surface)
                - self.estimated_dem.apply_mask(self.computation_surface)
            )
            .raster_volume()
            .with_name('estimation_volume')
        )
        truth = None
        if (
            self.validation_dem is not None
            and self.validation_gully_cover is not None
        ):
            truth = (
                (
                    self.validation_gully_cover.apply_mask(
                        self.computation_surface
                    )
                    - self.validation_dem.apply_mask(self.computation_surface)
                )
                .raster_volume()
                .with_name('truth_volume')
            )
        return (estimation, truth)


@dataclass(kw_only=True)
class BackcastVolumeEvaluator(VolumeEvaluator):
    dem: DEM
    estimated_dem: DEM
    estimated_boundary: QgsVectorLayer
    validation_dem: DEM | None = None
    validation_gully_cover: DEM | None = None
    boundary_intersect: QgsVectorLayer = field(init=False)
    boundary_difference: QgsVectorLayer = field(init=False)

    def __post_init__(self) -> None:
        self.boundary_intersect = intersection(
            self.computation_surface, self.estimated_boundary
        )  # past
        self.boundary_difference = difference(
            self.computation_surface, self.estimated_boundary
        )  # future

    def evaluate(self) -> QgsVectorLayer:
        estimation_delta, truth_delta = self.compute_delta()
        delta_statistics = self.zonal_statistics(
            self.computation_surface,
            estimation_delta,
            truth_delta,
            prefix='delta_',
        )
        estimation_volume, truth_volume = self.compute_volume()
        volume_statistics = self.zonal_statistics(
            self.boundary_intersect,
            estimation_volume,
            truth_volume,
            prefix='volume_',
        )
        return merge_vector_layers(
            [delta_statistics, volume_statistics],
            output=self.out_file.as_posix(),
        )

    def compute_delta(self) -> tuple[Raster, Raster | None]:
        """Computes the eroded volume difference between the known DEM and the estimated DEM."""
        boundary_difference_volume = (
            self.gully_cover.apply_mask(self.boundary_difference)
            - self.dem.apply_mask(self.boundary_difference)
        ).raster_volume()
        boundary_intersect_volume = (
            self.estimated_dem.apply_mask(self.boundary_intersect)
            - self.dem.apply_mask(self.boundary_intersect)
        ).raster_volume()
        estimation = Raster.from_rasters(
            [boundary_intersect_volume, boundary_difference_volume]
        ).with_name('estimation_delta')
        truth = None
        if self.validation_dem is not None:
            truth = (
                (
                    self.validation_dem.apply_mask(self.computation_surface)
                    - self.dem.apply_mask(self.computation_surface)
                )
                .raster_volume()
                .with_name('truth_delta')
            )
        if self.project is not None:
            self.project.addMapLayer(estimation.layer)
            if truth is not None:
                self.project.addMapLayer(truth.layer)
        return (estimation, truth)

    def compute_volume(self) -> tuple[Raster, Raster | None]:
        estimation = (
            (
                self.gully_cover.apply_mask(self.boundary_intersect)
                - self.estimated_dem.apply_mask(self.boundary_intersect)
            )
            .raster_volume()
            .with_name('estimation_volume')
        )
        truth = None
        if (
            self.validation_dem is not None
            and self.validation_gully_cover is not None
        ):
            truth = (
                (
                    self.validation_gully_cover.apply_mask(
                        self.computation_surface
                    )
                    - self.validation_dem.apply_mask(self.computation_surface)
                )
                .raster_volume()
                .with_name('truth_volume')
            )
        return (estimation, truth)
