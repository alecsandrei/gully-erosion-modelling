from __future__ import annotations

import collections.abc as c
import typing as t
from collections import UserList, deque
from pathlib import Path

import processing
from qgis.core import Qgis, QgsGeometry, QgsPoint, QgsVectorLayer

from gully_analysis.utils import get_geometries_from_layer

if t.TYPE_CHECKING:
    from qgis.core import QgsProcessingContext, QgsProcessingFeedback


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
            QgsGeometry.fromPoint(self.last),
        )


def intersection_points(
    lines: c.Iterable[QgsGeometry], polygon: QgsGeometry
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


class Centerlines(UserList[QgsGeometry]):
    def __init__(self, initlist=None, layer: QgsVectorLayer | None = None):
        self._layer = layer
        super().__init__(initlist)

    def as_multilinestring(self) -> QgsGeometry:
        geom = QgsGeometry()
        for part in self.data:
            add_part_result = geom.addPartGeometry(part)
            if not add_part_result == Qgis.GeometryOperationResult.Success:
                raise GeometryError(
                    f'Failed to convert {self!r} to a MultiLineString QgsGeometry.'
                )
        return geom.coerceToType(Qgis.WkbType.MultiLineString)[0]

    @staticmethod
    def from_layer(centerlines: QgsVectorLayer):
        return Centerlines(
            list(get_geometries_from_layer(centerlines)), centerlines
        )

    @staticmethod
    def compute(
        polygon_layer: QgsVectorLayer,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        output: str | Path = 'TEMPORARY_OUTPUT',
    ):
        if isinstance(output, Path):
            output = output.as_posix()
        geometry_fixed = fix_geometry(polygon_layer, context, feedback)
        centerline = processing.run(
            'grass:v.voronoi.skeleton',
            {
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
                'GRASS_VECTOR_EXPORT_NOCAT': False,
            },
            context=context,
            feedback=feedback,
        )['output']
        # Removing duplicate vertices is required to make a valid network
        without_duplicate_vertices = processing.run(
            'native:removeduplicatevertices',
            {
                'INPUT': centerline,
                'TOLERANCE': 1e-06,
                'USE_Z_VALUE': False,
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            context=context,
            feedback=feedback,
        )['OUTPUT']
        if not isinstance(without_duplicate_vertices, QgsVectorLayer):
            without_duplicate_vertices = QgsVectorLayer(
                without_duplicate_vertices, 'centerline', 'ogr'
            )
        geoms = list(get_geometries_from_layer(without_duplicate_vertices))
        return Centerlines(geoms, without_duplicate_vertices)

    def intersects(self, geometry: QgsGeometry) -> Centerlines:
        return Centerlines(
            [
                centerline
                for centerline in self.data
                if centerline.intersects(geometry)
            ]
        )


def fix_geometry(
    layer: QgsVectorLayer,
    context: QgsProcessingContext,
    feedback: QgsProcessingFeedback,
) -> QgsVectorLayer:
    """Always converts to multi-geometry layer."""
    return processing.run(
        'native:fixgeometries',
        {'INPUT': layer, 'METHOD': 1, 'OUTPUT': 'TEMPORARY_OUTPUT'},
        context=context,
        feedback=feedback,
    )['OUTPUT']


def check_incorrect_geometry(
    expected_types: c.Sequence[Qgis.WkbType],
    geometries: c.Iterable[QgsGeometry],
) -> IncorrectGeometryType | None:
    for geometry in geometries:
        if (type_ := geometry.wkbType()) not in expected_types:
            return IncorrectGeometryType(type_, expected_types)
    return None


def polygon_to_line(polygon: QgsGeometry) -> QgsGeometry:
    """Coerces a polygon to a MultiLineString."""
    return polygon.coerceToType(Qgis.WkbType.MultiLineString)[0]


class GeometryError(Exception): ...


class IncorrectGeometryType(GeometryError):
    def __init__(
        self,
        found: Qgis.WkbType,
        expected: c.Sequence[Qgis.WkbType] | Qgis.WkbType,
    ):
        self.message = f'Expected geometry {expected}, found {found}.'
        super().__init__(self.message)


class MultipartGeometryFound(GeometryError):
    default_message = 'Expected single part geometry, found multi part.'

    def __init__(self, message: str | None = None):
        self.message = self.default_message if message is None else message
        super().__init__(self.message)
