from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass
from pathlib import Path

import processing
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRasterLayer,
    QgsVectorLayer,
)

from .exceptions import InvalidCoordinateSystem
from .geometry import snap_to_geometry
from .utils import geometries_to_layer, get_geometries_from_path

if t.TYPE_CHECKING:
    from qgis.core import QgsCoordinateReferenceSystem, QgsGeometry


class Extent(t.NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    epsg: str

    @staticmethod
    def from_raster(raster: QgsRasterLayer, epsg: str | None = None):
        extent = raster.extent()
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


    @property
    def extent(self) -> Extent:
        return Extent.from_raster(self.layer)

    def apply_mask(
        self: T, mask: QgsVectorLayer, output: str = 'TEMPORARY_OUTPUT'
    ) -> T:
        masked = processing.run(
            'gdal:cliprasterbymasklayer',
            {
                'INPUT': self.layer,
                'MASK': mask,
                'SOURCE_CRS': self.layer.crs(),
                'TARGET_CRS': self.layer.crs(),
                'TARGET_EXTENT': None,
                'NODATA': None,
                'ALPHA_BAND': False,
                'CROP_TO_CUTLINE': True,
                'KEEP_RESOLUTION': False,
                'SET_RESOLUTION': False,
                'X_RESOLUTION': None,
                'Y_RESOLUTION': None,
                'MULTITHREADING': False,
                'OPTIONS': None,
                'DATA_TYPE': 0,
                'EXTRA': '',
                'OUTPUT': output,
            },
        )
        return type(self)(
            QgsRasterLayer(masked['OUTPUT'], self.layer.name(), 'gdal')
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
                dem_preproc['DEM_PREPROC'], 'DEM_sink_removed', 'gdal'
            )
        )

    def flow_path_profiles_from_points(
        self,
        points: t.Sequence[QgsGeometry],
        eps: float,
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
        profile_paths = list(Path(profiles['LINE']).parent.glob('*.shp'))
        if len(profile_paths) > 1:
            profile_paths = sorted(
                profile_paths,
                key=lambda path: int(path.stem.replace('LINE', '')),
            )
        profiles = list(get_geometries_from_path(*profile_paths))
        return [
            next(snap_to_geometry([profile], [pour_point], tolerance=eps))
            for profile, pour_point in zip(profiles, points)
        ]

    def sample(
        self,
        lines: c.Sequence[QgsGeometry],
        feedback: QgsProcessingFeedback | None = None,
        context: QgsProcessingContext | None = None,
    ) -> QgsVectorLayer:
        layer = geometries_to_layer(lines)

        profiles = processing.run(
            'sagang:profilesfromlines',
            {
                'DEM': self.layer,
                'VALUES': None,
                'LINES': layer,
                'NAME': 'FID',
                'PROFILE': 'TEMPORARY_OUTPUT',
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
                    'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
                    'SEARCH_RANGE': 1,
                    'SEARCH_RADIUS': 1000,
                    'SEARCH_POINTS_ALL': 1,
                    'SEARCH_POINTS_MIN': 1,
                    'SEARCH_POINTS_MAX': 20,
                    'DW_WEIGHTING': 1,
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


@dataclass
class VolumeEvaluator:
    dem: DEM
    estimation_dem: DEM
    truth_dem: DEM
    gully_cover: DEM
    estimation_surface: QgsVectorLayer
    out_dir: Path
    crs: QgsCoordinateReferenceSystem

    def __post_init__(self):
        self.dem = self.dem.apply_mask(
            self.estimation_surface,
            output=(
                self.out_dir / f'masked_{self.dem.layer.name()}.tif'
            ).as_posix(),
        )
        self.estimation_dem = self.estimation_dem.apply_mask(
            self.estimation_surface,
            output=(
                self.out_dir / f'masked_{self.estimation_dem.layer.name()}.tif'
            ).as_posix(),
        )
        self.truth_dem = self.truth_dem.apply_mask(
            self.estimation_surface,
            output=(
                self.out_dir / f'masked_{self.truth_dem.layer.name()}.tif'
            ).as_posix(),
        )
        self.gully_cover = self.gully_cover.apply_mask(
            self.estimation_surface,
            output=(
                self.out_dir / f'masked_{self.gully_cover.layer.name()}.tif'
            ).as_posix(),
        )
        for layer in (
            self.dem,
            self.estimation_dem,
            self.truth_dem,
            self.gully_cover,
        ):
            layer.layer.setCrs(self.crs)

    def get_masked(self) -> list[DEM]:
        return [
            dem.apply_mask(self.estimation_surface)
            for dem in (
                self.dem,
                self.estimation_dem,
                self.truth_dem,
                self.gully_cover,
            )
        ]

    def evaluate(self) -> QgsVectorLayer:
        # NOTE: This does not account for the values which are negative
        truth_diff = self.gully_cover - self.truth_dem
        truth_volume = truth_diff.raster_volume()
        estimated_diff = self.gully_cover - self.estimation_dem
        estimated_volume = estimated_diff.raster_volume()
        truth_zonal = processing.run(
            'native:zonalstatisticsfb',
            {
                'INPUT': self.estimation_surface,
                'INPUT_RASTER': truth_volume.layer,
                'RASTER_BAND': 1,
                'COLUMN_PREFIX': 'truth_',
                'STATISTICS': [1],  # this is sum
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']
        estimated_zonal = processing.run(
            'native:zonalstatisticsfb',
            {
                'INPUT': truth_zonal,
                'INPUT_RASTER': estimated_volume.layer,
                'RASTER_BAND': 1,
                'COLUMN_PREFIX': 'estimated_',
                'STATISTICS': [1],  # this is sum
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']

        with_error = processing.run(
            'native:fieldcalculator',
            {
                'INPUT': estimated_zonal,
                'FIELD_NAME': 'error',
                'FIELD_TYPE': 0,
                'FIELD_LENGTH': 0,
                'FIELD_PRECISION': 0,
                'FORMULA': 'abs("truth_sum" -  "estimated_sum") / "truth_sum"',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
        )['OUTPUT']

        return with_error
