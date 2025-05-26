from __future__ import annotations

import collections.abc as c
import typing as t
from collections import UserList, deque
from math import pi

import numpy as np
import processing
from qgis.analysis import QgsGeometrySnapper
from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsPointXY,
    QgsVectorLayer,
)

from .utils import dissolve_layer, get_first_geometry, get_geometries_from_layer

if t.TYPE_CHECKING:
    from qgis.core import QgsProcessingContext, QgsProcessingFeedback


class Endpoints(t.NamedTuple):
    first: QgsPointXY
    last: QgsPointXY

    @staticmethod
    def from_linestring(linestring: QgsGeometry) -> Endpoints:
        vertices = linestring.vertices()
        first = next(vertices)
        if not vertices.hasNext():
            raise ValueError('Linestring only has one vertex.')
        # Fastest way to drain an iterator
        last = deque(vertices, 1).pop()
        return Endpoints(QgsPointXY(first), QgsPointXY(last))

    def as_qgis_geometry(self):
        return (
            QgsGeometry.fromPointXY(self.first),
            QgsGeometry.fromPointXY(self.last),
        )


def intersection_points(
    lines: c.Iterable[QgsGeometry], geometry: QgsGeometry, tolerance: float
) -> list[QgsGeometry]:
    """
    Finds the intersection points between the lines and the polygon exterior.

    The endpoints of the lines which intersect the polygon exterior
    rings are not returned.
    """
    as_line = to_linestring(geometry)
    intersections: list[QgsGeometry] = []
    for line in lines:
        endpoints = Endpoints.from_linestring(line)
        intersection = as_line.intersection(line)
        if intersection.isEmpty():
            continue
        # It could be a MultiPoint.
        expected = (Qgis.WkbType.Point, Qgis.WkbType.MultiPoint)
        error = check_incorrect_geometry(expected, intersections)
        if error:
            raise error
        point_list = intersection.coerceToType(Qgis.WkbType.Point)
        for intersection in point_list:
            if intersection.constGet() not in endpoints:
                intersections.append(intersection)
    return intersections


def get_longest(geometries: c.Sequence[QgsGeometry]) -> QgsGeometry:
    return sorted(geometries, key=QgsGeometry.length)[-1]


def remove_duplicated(
    geometries: c.Sequence[QgsGeometry],
) -> list[QgsGeometry]:
    """Returns non-duplicated geometries while keeping the first occurance."""
    non_duplicated = []
    for i, geom in enumerate(geometries):
        if not any(
            geom.equals(other)
            for j, other in enumerate(geometries[:i])
            if j != i
        ):
            non_duplicated.append(geom)
    return non_duplicated


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
    def get_thin_from_roundness(roundness: float):
        # 0.05 <= uses thin 5, 0.3 >= thin 0
        # inbetween geomspaced
        space = np.geomspace(0.2, 0.05, 5)
        return int(np.digitize([roundness], space)[0])

    @staticmethod
    def from_layer(centerlines: QgsVectorLayer):
        return Centerlines(
            list(get_geometries_from_layer(centerlines)), centerlines
        )

    @classmethod
    def compute(
        cls,
        polygon_layer: QgsVectorLayer,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
        output: str = 'TEMPORARY_OUTPUT',
        smoothness: float = 5,
        thin: float | t.Literal['adaptive'] = 1,
    ):
        geometry_fixed = fix_geometry(polygon_layer, context, feedback)
        if thin == 'adaptive':
            dissolved = dissolve_layer(geometry_fixed)
            polygon = get_first_geometry(dissolved)
            thin = cls.get_thin_from_roundness(roundness(polygon))

        centerline = processing.run(
            'grass:v.voronoi.skeleton',
            {
                'input': geometry_fixed,
                'smoothness': smoothness,
                'thin': thin,
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


def intersects(geometry: QgsGeometry, others: c.Sequence[QgsGeometry]) -> bool:
    return any(geometry.intersects(other) for other in others)


def roundness(geometry: QgsGeometry) -> float:
    area = geometry.area()
    perimeter = geometry.length()
    return (4 * pi * area) / perimeter**2


def fix_geometry(
    layer: QgsVectorLayer,
    context: QgsProcessingContext | None = None,
    feedback: QgsProcessingFeedback | None = None,
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


def to_linestring(polygon: QgsGeometry) -> QgsGeometry:
    """Coerces a polygon to a MultiLineString."""
    return polygon.coerceToType(Qgis.WkbType.MultiLineString)[0]


def snap_to_geometry(
    geometries: c.Sequence[QgsGeometry],
    snap_to: c.Sequence[QgsGeometry],
    tolerance: float,
    mode: QgsGeometrySnapper.SnapMode = QgsGeometrySnapper.SnapMode.PreferClosest,
) -> c.Generator[QgsGeometry, None, None]:
    for geometry in geometries:
        yield QgsGeometrySnapper.snapGeometry(
            geometry, tolerance, snap_to, mode=mode
        )


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


def get_line_intersections(lines: QgsVectorLayer) -> QgsVectorLayer:
    union = processing.run(
        'native:union',
        {
            'INPUT': lines,
            'OVERLAY': None,
            'OVERLAY_FIELDS_PREFIX': '',
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'GRID_SIZE': None,
        },
    )['OUTPUT']
    cleaned = processing.run(
        'native:deleteduplicategeometries',
        {
            'INPUT': union,
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )['OUTPUT']

    line_intersections = processing.run(
        'native:lineintersections',
        {
            'INPUT': cleaned,
            'INTERSECT': cleaned,
            'INPUT_FIELDS': [],
            'INTERSECT_FIELDS': [],
            'INTERSECT_FIELDS_PREFIX': '',
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )
    return line_intersections['OUTPUT']


def multipart_to_singlepart(geometries: QgsVectorLayer) -> QgsVectorLayer:
    return processing.run(
        'native:multiparttosingleparts',
        {
            'INPUT': geometries,
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )['OUTPUT']
