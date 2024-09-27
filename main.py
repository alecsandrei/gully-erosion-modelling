from pathlib import Path

from qgis.core import (
    QgsVectorLayer,
    QgsPolygon,
    QgsMultiPolygon,
    QgsProject,
    QgsGeometry,
    Qgis
)


from gully_erosion_estimation_qgis.utils import (
    get_first_geometry,
    export,
    geometries_to_layer
)
from gully_erosion_estimation_qgis.geometry import (
    Centerlines,
    Endpoints,
    intersection_points,
    get_geometries,
)
from gully_erosion_estimation_qgis import (
    DEBUG,
    CACHE,
    MODEL
)


DATA_DIR = Path(__file__).parent / 'data'


def construct_gpkg_path(gpkg: Path, layer_name: str):
    return f'{gpkg}|layername={layer_name}'


def main(model):
    model_dir = DATA_DIR / model
    cache_dir = model_dir
    gpkg = model_dir / f'{model}.gpkg'
    layer_2012 = QgsVectorLayer(
        construct_gpkg_path(gpkg, '2012'), '2012', providerLib='ogr'
    )
    layer_2019 = QgsVectorLayer(
        construct_gpkg_path(gpkg, '2019'), '2019', providerLib='ogr'
    )
    epsg=layer_2012.crs().geographicCrsAuthId()
    assert layer_2019.featureCount() == 1
    assert layer_2012.featureCount() == 1

    polygon_2012 = get_first_geometry(layer_2012)
    polygon_2019 = get_first_geometry(layer_2019)
    difference = polygon_2019.difference(polygon_2012)
    assert (
        isinstance(polygon_2012.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2012)}'
    assert (
        isinstance(polygon_2019.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2019)}'

    if CACHE:
        geoms = list(
            get_geometries(
                QgsVectorLayer(
                    (cache_dir / 'centerline.shp').as_posix(), 'centerline', 'ogr'
                )
            )
        )
        centerlines = Centerlines(geoms)
    else:
        centerlines = Centerlines.from_layer(
            layer_2019, output=cache_dir / 'centerline.shp'
        )

    centerlines_subset = [
        centerline for centerline in centerlines.intersects(difference)
        if QgsGeometry.fromPoint(Endpoints.from_linestring(centerline).first).intersects(difference.coerceToType(Qgis.WkbType.MultiLineString)[0])
    ]
    export(
        geometries_to_layer(centerlines_subset, epsg=epsg),
        cache_dir / 'centerline_erosion.shp'
    )

    # pour_points = intersection_points(centerlines, polygon_2012)
    # export(
    #     geometries_to_layer(pour_points, epsg),
    #     cache_dir / 'centerline_difference.shp'
    # )
    # export(
    #     geometries_to_layer(pour_points, epsg),
    #     cache_dir / 'pour_points.shp'
    # )
    # print(geoms_difference[0])
    # points = intersection_points(polygon_2012, geoms_difference)
    # boundary_2019 = polygon_2019.coerceToType(Qgis.WkbType.MultiLineString)[0]
    # points_filtered = [
    #     geometry for geometry in points
    #     if not geometry.intersects(boundary_2019)
    # ]

    # points_as_layer = geometries_to_layer(
    #     points,
    #     epsg=layer_2012.crs().geographicCrsAuthId()
    # )
    # export(points_as_layer, cache_dir / 'intersection_points.shp')

if __name__ == '__main__':
    if MODEL == 'all':
        for model in [
            'soldanesti_aval',
            'saveni_aval',
            'saveni_amonte',
            'soldanesti_amonte'
        ]:
            if DEBUG:
                print('Running model', model)
            main(model)
    else:
        main(MODEL)
