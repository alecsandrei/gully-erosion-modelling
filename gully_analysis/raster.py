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
    from qgis.core import QgsGeometry


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

    def __sub__(self, other: Raster) -> Raster:
        return self.difference(other)

    def difference(self: T, other: Raster) -> T:
        return type(self)(
            processing.run(
                'sagang:rasterdifference',
                {'A': self.layer, 'B': other.layer, 'C': 'TEMPORARY_OUTPUT'},
            )['C']
        )

    @property
    def extent(self) -> Extent:
        return Extent.from_raster(self.layer)

    def apply_mask(self: T, mask: QgsVectorLayer) -> T:
        masked = processing.run(
            'sagang:cliprasterwithpolygon',
            {
                'INPUT': self.layer,
                'OUTPUT': 'TEMPORARY_OUTPUT',
                'POLYGONS': mask,
                'EXTENT': 1,
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
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            context=context,
            feedback=feedback,
        )
        return type(self)(
            QgsRasterLayer(aligned['OUTPUT'], self.layer.name(), 'gdal')
        )

    def volume(self) -> float:
        feedback = QgsProcessingFeedback()
        processing.run(
            'sagang:rastervolume',
            {'GRID': self.layer, 'METHOD': 0, 'LEVEL': 3},
            feedback=feedback,
        )

        log = feedback.textLog().splitlines()

        def is_result_line(string):
            return 'Volume:' in string

        try:
            result_line = next(filter(is_result_line, log))
        except StopIteration:
            raise ValueError(f'Failed to compute the raster volume for {self}')
        else:
            return float(result_line.split(': ')[-1])


@dataclass
class DEM(Raster):
    def remove_sinks(
        self,
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
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
                'DEM_PREPROC': 'TEMPORARY_OUTPUT',
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
        profile_paths = sorted(
            Path(profiles['LINE']).parent.glob('*.shp'),
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
    context: QgsProcessingContext | None = None,
    feedback: QgsProcessingFeedback | None = None,
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
                    'TARGET_OUT_GRID': 'TEMPORARY_OUTPUT',
                    'METHOD': 0,
                    'LEVEL_MAX': 14,
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
class Evaluator:
    dem: DEM
    estimation_dem: DEM
    truth_dem: DEM
    gully_cover: DEM
    estimation_surface: QgsVectorLayer

    def __post_init__(self):
        self.dem = self.dem.apply_mask(self.estimation_surface)
        self.estimation_dem = self.estimation_dem.apply_mask(
            self.estimation_surface
        )
        self.truth_dem = self.truth_dem.apply_mask(self.estimation_surface)
        self.gully_cover = self.gully_cover.apply_mask(self.estimation_surface)

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

    def evaluate(self, feedback: QgsProcessingFeedback):
        gully_cover_volume = self.gully_cover.volume()
        truth_volume = gully_cover_volume - self.truth_dem.volume()
        estimation_volume = gully_cover_volume - self.estimation_dem.volume()
        error = abs(truth_volume - estimation_volume) / truth_volume
        feedback.pushInfo(
            f'Estimation: {estimation_volume:.3f}\n'
            f'Truth: {truth_volume:.3f}\n'
            f'Error: {error:.3%}'
        )
