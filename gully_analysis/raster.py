from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path

import processing
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRasterLayer,
)

from .utils import geometries_to_layer

if t.TYPE_CHECKING:
    from qgis.core import QgsGeometry


@dataclass
class Raster:
    layer: QgsRasterLayer


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
        context: QgsProcessingContext | None = None,
        feedback: QgsProcessingFeedback | None = None,
    ):
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
        profiles = list(Path(profiles['LINE']).parent.glob('*.shp'))
        merged_profiles = processing.run(
            'native:mergevectorlayers',
            {
                'LAYERS': [profile.as_posix() for profile in profiles],
                'CRS': self.layer.crs(),
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            context=context,
            feedback=feedback,
        )['OUTPUT']
        return merged_profiles
