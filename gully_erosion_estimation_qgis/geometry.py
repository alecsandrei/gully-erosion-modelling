from __future__ import annotations

import typing as t
import collections.abc as c
from pathlib import Path
import itertools
from collections import deque

import processing
from qgis.core import (
    QgsVectorLayer,
    Qgis,
    QgsGeometry,
    QgsVertexIterator,
    QgsPoint
)

from . import DEBUG

if t.TYPE_CHECKING:
    from qgis.core import (
        QgsGeometry,
    )


class Endpoints(t.NamedTuple):
    first: QgsPoint
    last: QgsPoint


def endpoints_gen(lines: c.Iterable[QgsGeometry]) -> c.Generator[Endpoints]:
    """Generator which expects linestrings."""
    for line in lines:
        vertices = line.vertices()
        first = next(vertices)
        # Fastest way to drain an iterator
        last = deque(vertices, 1).pop()
        yield Endpoints(first, last)


def intersection_points(polygon: QgsGeometry, lines: c.Iterable[QgsGeometry]):
    as_line = polygon.coerceToType(Qgis.WkbType.MultiLineString)
    assert len(as_line) == 1
    intersections = [
        intersection for line in lines
        if not (intersection := as_line[0].intersection(line)).isEmpty()
    ]
    if (
        error := check_incorrect_geometry([Qgis.WkbType.Point], intersections)
    ):
        raise error
    return intersections


def convert_to_single_part(
    geometries: c.Iterable[QgsGeometry],
    errors: t.Literal['raise', 'ignore'] = 'raise'
):
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
            for part in geometry.parts():
                yield part
        else:
            yield geometry


def create_centerline(
    layer: QgsVectorLayer,
    output: str | Path = 'TEMPORARY_OUTPUT'
):
    if isinstance(output, Path):
        output = output.as_posix()
    fixed = fix_geometry(layer)
    if DEBUG:
        print('Creating centerline for layer', layer.name())
    centerline = processing.run('grass:v.voronoi.skeleton', {
        'input': fixed,
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
    return QgsVectorLayer(centerline, 'centerline', 'ogr')


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
