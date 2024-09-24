import os
from pathlib import Path

from qgis.core import (
    QgsVectorLayer,
    QgsPolygon,
    QgsMultiPolygon,
    QgsProject,
    Qgis
)


from gully_erosion_estimation_qgis.utils import (
    get_first_geometry,
    export,
    geometries_to_layer
)
from gully_erosion_estimation_qgis.geometry import (
    create_centerline,
    intersection_points,
    get_geometries,
    convert_to_single_part
)
from gully_erosion_estimation_qgis import (
    DEBUG,
    CACHE
)


DATA_DIR = Path('/media/alex/alex/python-modules-packages-utils/gully-erosion-estimation/data/derived')


def construct_gpkg_path(gpkg: Path, layer_name: str):
    return f'{gpkg}|layername={layer_name}'


def main():
    model = os.getenv('MODEL')
    model_dir = DATA_DIR / model
    cache_dir = model_dir
    gpkg = model_dir / f'{model}.gpkg'
    layer_2012 = QgsVectorLayer(
        construct_gpkg_path(gpkg, '2012'), '2012', providerLib='ogr'
    )
    layer_2019 = QgsVectorLayer(
        construct_gpkg_path(gpkg, '2019'), '2019', providerLib='ogr'
    )
    assert layer_2019.featureCount() == 1
    assert layer_2012.featureCount() == 1

    polygon_2012 = get_first_geometry(layer_2012)
    polygon_2019 = get_first_geometry(layer_2019)
    assert (
        isinstance(polygon_2012.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2012)}'
    assert (
        isinstance(polygon_2019.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2019)}'

    polygon_difference = polygon_2019.difference(polygon_2012)


    if CACHE:
        centerline = QgsVectorLayer(
            (cache_dir / 'centerline.shp').as_posix(), 'centerline', 'ogr'
        )
    else:
        centerline = create_centerline(
            layer_2019, output=cache_dir / 'centerline.shp'
        )
    geoms = list(get_geometries(centerline))
    convert_to_single_part(geoms)
    points = intersection_points(polygon_2012, geoms)
    boundary_2019 = polygon_2019.coerceToType(Qgis.WkbType.MultiLineString)[0]
    points_filtered = [
        geometry for geometry in points
        if not geometry.intersects(boundary_2019)
    ]
    points_as_layer = geometries_to_layer(
        points_filtered,
        epsg=layer_2012.crs().geographicCrsAuthId()
    )
    export(points_as_layer, cache_dir / 'intersection_points.shp')


if __name__ == '__main__':
    main()
