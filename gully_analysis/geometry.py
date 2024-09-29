from __future__ import annotations

import typing as t
import collections.abc as c
from collections import UserList
from pathlib import Path
import itertools
from functools import partial
from collections import deque

import processing
from qgis.core import (
    QgsVectorLayer,
    Qgis,
    QgsGeometry,
    QgsVertexIterator,
    QgsPoint
)
import numpy as np

from . import DEBUG
from .utils import geometries_to_layer

if t.TYPE_CHECKING:
    from qgis.core import (
        QgsGeometry,
    )


class Endpoints(t.NamedTuple):
    first: QgsPoint
    last: QgsPoint

    @staticmethod
    def from_linestring(linestring: QgsGeometry) -> Endpoints:
        vertices = linestring.vertices()
        first = next(vertices)
        # Fastest way to drain an iterator
        last = deque(vertices, 1).pop()
        return Endpoints(first, last)

    def as_qgis_geometry(self):
        return (
            QgsGeometry.fromPoint(self.first),
            QgsGeometry.fromPoint(self.last)
        )


def endpoints_gen(
    linestrings: c.Iterable[QgsGeometry]
) -> c.Generator[Endpoints]:
    """Generator which expects linestrings."""
    for linestring in linestrings:
        yield Endpoints.from_linestring(linestring)


def intersection_points(
    lines: c.Iterable[QgsGeometry],
    polygon: QgsGeometry
) -> list[QgsGeometry]:
    """
    Finds the intersection points between the lines and the polygon exterior.

    The endpoints of the lines which intersect the polygon exterior
        rings are not returned.
    """
    as_line = polygon_to_line(polygon)
    intersections: list[QgsGeometry] = []
    for line in lines:
        endpoints = Endpoints.from_linestring(line)
        intersection = as_line.intersection(line)
        if intersection.isEmpty():
            continue
        # It could be a MultiPoint.
        point_list = intersection.coerceToType(Qgis.WkbType.Point)
        expected = (Qgis.WkbType.Point, Qgis.WkbType.MultiPoint)
        error = check_incorrect_geometry(expected, intersections)
        if error:
            raise error
        for intersection in point_list:
            if intersection.constGet() not in endpoints:
                intersections.append(intersection)
    return intersections


def convert_to_single_part(
    geometries: c.Sequence[QgsGeometry],
    errors: t.Literal['raise', 'ignore'] = 'raise'
) -> None:
    """Converts geometries to single part, in place."""
    for geometry in geometries:
        if not geometry.isMultipart():
            continue
        parts = list(geometry.parts())
        if len(parts) > 1 and errors == 'raise':
            raise MultipartGeometryFound(
                f'{MultipartGeometryFound.default_message} '
                'To suppress this error, pass errors="ignore".'
            )
        geometry.convertToSingleType()


def single_part_gen(
    geometries: c.Sequence[QgsGeometry]
) -> c.Generator[QgsGeometry]:
    for geometry in geometries:
        if geometry.isMultipart():
            coerced = geometry.coerceToType(Qgis.WkbType.LineString)
            for part in coerced:
                yield part
        else:
            yield geometry


class Centerlines(UserList[QgsGeometry]):

    def __init__(self, initlist=None):
        super().__init__(initlist)

    @staticmethod
    def from_polygon(
        polygon: QgsGeometry,
        epsg: str,
        output: str | Path = 'TEMPORARY_OUTPUT'
    ):
        if isinstance(output, Path):
            output = output.as_posix()
        as_layer = geometries_to_layer(
            [polygon],
            epsg=epsg
        )
        geometry_fixed = fix_geometry(as_layer)
        if DEBUG:
            print('Creating centerline from polygon')
        print(polygon)
        centerline = processing.run('grass:v.voronoi.skeleton', {
            'input': geometry_fixed,
            'smoothness': 0.1,
            'thin': 1,
            '-a': False,
            '-s': True,
            '-l': False,
            '-t': False,
            'output': output,
            'GRASS_REGION_PARAMETER': None,
            'GRASS_SNAP_TOLERANCE_PARAMETER': -1,
            'GRASS_MIN_AREA_PARAMETER': 0.0001,
            'GRASS_OUTPUT_TYPE_PARAMETER': 0,
            'GRASS_VECTOR_DSCO': '',
            'GRASS_VECTOR_LCO': '',
            'GRASS_VECTOR_EXPORT_NOCAT': False
        })['output']
        return Centerlines(
            list(
                get_geometries(QgsVectorLayer(centerline, 'centerline', 'ogr'))
            )
        )

    @staticmethod
    def post_processing(geometries: t.Sequence[QgsGeometry]):
        not_empty = [
            centerline for centerline in geometries if not centerline.isEmpty()
        ]
        return list(single_part_gen(not_empty))

    def difference(self, geometry: QgsGeometry) -> Centerlines:
        difference = [
            centerline.difference(geometry) for centerline in self.data
        ]
        return Centerlines(self.post_processing(difference))

    def intersects(self, geometry: QgsGeometry) -> Centerlines:
        return Centerlines(
            [centerline for centerline in self.data
             if centerline.intersects(geometry)]
        )


def fix_geometry(layer: QgsVectorLayer) -> QgsVectorLayer:
    """Always converts to multi-geometry layer."""
    if DEBUG:
        print('Fixing geometry of layer', layer.name())
    return processing.run('native:fixgeometries', {
        'INPUT': layer,
        'METHOD': 1,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']


def has_invalid_geometry(geometries: c.Sequence[QgsGeometry]):
    return all(geometry.isGeosValid() for geometry in geometries)


def get_geometries(layer: QgsVectorLayer) -> c.Generator[QgsGeometry]:
    ids = layer.allFeatureIds()
    for id in ids:
        yield layer.getFeature(id).geometry()


def check_incorrect_geometry(
    expected_types: c.Sequence[Qgis.WkbType],
    geometries: c.Iterable[QgsGeometry],
) -> IncorrectGeometry | None:
    for geometry in geometries:
        if not (type_ := geometry.wkbType()) in expected_types:
            return IncorrectGeometry(type_, expected_types)
    return None


def polygon_to_line(polygon: QgsGeometry) -> QgsGeometry:
    """Coerces a polygon to a MultiLineString."""
    return polygon.coerceToType(Qgis.WkbType.MultiLineString)[0]


class IncorrectGeometry(Exception):

    def __init__(
        self,
        found: Qgis.WkbType,
        expected: c.Sequence[Qgis.WkbType] | Qgis.WkbType
    ):
        self.message = f'Expected geometry {expected}, found {found}.'
        super().__init__(self.message)


class MultipartGeometryFound(Exception):
    default_message = 'Expected single part geometry, found multi part.'

    def __init__(self, message: str | None = None):
        self.message = self.default_message if message is None else message
        super().__init__(self.message)
