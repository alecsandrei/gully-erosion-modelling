from __future__ import annotations

import collections.abc as c
import itertools
import typing as t

import processing
from qgis.core import (
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    edit,
)

if t.TYPE_CHECKING:
    from pathlib import Path

    from qgis.core import QgsGeometry

ExportResult = tuple[
    QgsVectorFileWriter.WriterError, str | None, str | None, str | None
]


def delete_holes(layer: QgsVectorLayer, min_area: float = 0) -> QgsVectorLayer:
    return processing.run(
        'native:deleteholes',
        {
            'INPUT': layer,
            'MIN_AREA': min_area,
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )['OUTPUT']


def dissolve_layer(layer: QgsVectorLayer) -> QgsVectorLayer:
    return processing.run(
        'native:dissolve',
        {
            'INPUT': layer,
            'FIELD': [],
            'SEPARATE_DISJOINT': False,
            'OUTPUT': 'TEMPORARY_OUTPUT',
        },
    )['OUTPUT']


def get_first_geometry(layer: QgsVectorLayer) -> QgsGeometry:
    return layer.getFeature(layer.allFeatureIds()[0]).geometry()


def get_geometries_from_path(*paths: Path):
    return itertools.chain.from_iterable(
        get_geometries_from_layer(
            QgsVectorLayer(path.as_posix(), 'temp', 'ogr')
        )
        for path in paths
    )


def get_geometries_from_layer(
    layer: QgsVectorLayer,
) -> c.Generator[QgsGeometry]:
    ids = layer.allFeatureIds()
    for id in ids:
        yield layer.getFeature(id).geometry()


def export(
    layer: QgsVectorLayer, out_file: Path, driver_name: str = 'FlatGeobuf'
) -> tuple[ExportResult, QgsVectorLayer]:
    # NOTE: this function is unused. Should be dropped
    # later on if no use is found
    options = QgsVectorFileWriter.SaveVectorOptions()
    ctx = QgsCoordinateTransformContext()
    options.driverName = driver_name
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, out_file.as_posix(), ctx, options
    )
    layer = QgsVectorLayer(out_file.as_posix(), layer.name(), 'ogr')
    return (result, layer)


def remove_layers_from_project(
    project: QgsProject, layer_names: list[str]
) -> None:
    for map_layer in itertools.chain.from_iterable(
        project.mapLayersByName(layer_name) for layer_name in layer_names
    ):
        # NOTE: I also tried removeMapLayers to avoid
        # a for loop but I was getting
        # TypeError: index 0 has type 'QgsVectorLayer' but 'str' is expected
        if not map_layer.isValid():
            continue
        project.removeMapLayers([map_layer.id()])


def geometries_to_layer(
    geoms: c.Sequence[QgsGeometry], name: str | None = None
) -> QgsVectorLayer:
    """Converts input geometries to a layer.

    The resulting layer will not have a CRS.
    """
    primitives = [geom.constGet() for geom in geoms]
    if any(primitive is None for primitive in primitives):
        raise Exception(
            'Failed to retrieve the underyling primitive from a geometry.'
        )
    geom_types = set(primitive.geometryType() for primitive in primitives)  # type: ignore
    if len(geom_types) > 1:
        raise Exception(
            'All of the geometries should have the same type. Found', geom_types
        )
    geom_type = list(geom_types)[0]
    layer = QgsVectorLayer(geom_type, '' if name is None else name, 'memory')

    def get_feature(idx: int, geom: QgsGeometry):
        feature = QgsFeature(idx)
        feature.setGeometry(geom)
        return feature

    with edit(layer):
        features = [
            get_feature(i, geom) for i, geom in enumerate(geoms, start=1)
        ]
        for feature in features:
            layer.addFeature(feature)

    assert layer.featureCount() == len(geoms)
    return layer
