from __future__ import annotations

import collections.abc as c
import typing as t

from qgis.core import (
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsVectorFileWriter,
    QgsVectorLayer,
    edit,
)

if t.TYPE_CHECKING:
    from pathlib import Path

    from qgis.core import QgsGeometry


def get_first_geometry(layer: QgsVectorLayer) -> QgsGeometry:
    return layer.getFeature(layer.allFeatureIds()[0]).geometry()


ExportResult = tuple[
    QgsVectorFileWriter.WriterError, str | None, str | None, str | None
]


def export(
    layer: QgsVectorLayer, out_file: Path, driver_name: str = 'ESRI Shapefile'
) -> ExportResult:
    # NOTE: this function is unused. Should be dropped
    # later on if no use is found
    options = QgsVectorFileWriter.SaveVectorOptions()
    ctx = QgsCoordinateTransformContext()
    options.driverName = driver_name
    return QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, out_file.as_posix(), ctx, options
    )


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
