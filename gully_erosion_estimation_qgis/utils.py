from __future__ import annotations

import typing as t
import collections.abc as c

from qgis.core import edit
from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext
)

if t.TYPE_CHECKING:
    from qgis.core import (
        QgsGeometry
    )
    from pathlib import Path


def get_first_geometry(layer: QgsVectorLayer) -> QgsGeometry:
    return layer.getFeature(layer.allFeatureIds()[0]).geometry()


def export(
    layer: QgsVectorLayer,
    out_file: Path,
    driver_name: str = 'ESRI Shapefile'
) -> tuple[QgsVectorFileWriter.WriterError, str | None]:
    options = QgsVectorFileWriter.SaveVectorOptions()
    ctx = QgsCoordinateTransformContext()
    options.driverName = driver_name
    return QgsVectorFileWriter.writeAsVectorFormatV3(
        layer,
        out_file.as_posix(),
        ctx,
        options
    )


def layer_path_from_geometry_type(wkt_type_str: str, epsg: str):
    return f'{wkt_type_str}?crs=EPSG:{epsg}'


def geometries_to_layer(
    geoms: c.Sequence[QgsGeometry],
    epsg: str,
    name: str | None = None
):
    geom_types = set(geom.constGet().geometryType() for geom in geoms)
    if len(geom_types) > 1:
        raise Exception(
            'All the geometries should have the same type. '
            'Found', geom_types
        )
    geom_type = list(geom_types)[0]
    layer = QgsVectorLayer(
        layer_path_from_geometry_type(geom_type, epsg),
        '' if name is None else name,
        'memory'
    )

    def get_feature(idx: int, geom: QgsGeometry):
        feature = QgsFeature(idx)
        feature.setGeometry(geom)
        return feature

    with edit(layer):
        features = [get_feature(i, geom) for i, geom in enumerate(geoms, start=1)]
        for feature in features:
            layer.addFeature(feature)

    return layer
