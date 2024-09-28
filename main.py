from pathlib import Path
import importlib

from qgis.utils import iface
from qgis.core import (
    QgsVectorLayer,
    QgsPolygon,
    QgsProject,
    QgsMultiPolygon,
    QgsProject,
    QgsGeometry,
    Qgis,
)

import gully_erosion_estimation_qgis.graph
from gully_erosion_estimation_qgis.utils import (
    get_first_geometry,
    export,
    geometries_to_layer
)
from gully_erosion_estimation_qgis.geometry import (
    Centerlines,
    Endpoints,
    intersection_points,
    convert_to_single_part,
    get_geometries,
)
from gully_erosion_estimation_qgis import (
    DEBUG,
    CACHE,
    MODEL
)
from gully_erosion_estimation_qgis.graph import build_graph

importlib.reload(gully_erosion_estimation_qgis.graph)

# Whether or not the script is running from the QGIS Python console
RUNNING_FROM_CONSOLE = __name__ == '__console__'
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
    epsg = layer_2012.crs().geographicCrsAuthId()
    assert layer_2019.featureCount() == 1
    assert layer_2012.featureCount() == 1

    polygon_2012 = (
        get_first_geometry(layer_2012)
        .coerceToType(Qgis.WkbType.Polygon)[0]
    )
    polygon_2019 = (
        get_first_geometry(layer_2019)
        .coerceToType(Qgis.WkbType.Polygon)[0]
    )
    difference = polygon_2019.difference(polygon_2012)
    limit_2012 = polygon_2012.coerceToType(Qgis.WkbType.MultiLineString)[0]
    limit_difference = difference.coerceToType(Qgis.WkbType.MultiLineString)[0]
    assert (
        isinstance(polygon_2012.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2012)}'
    assert (
        isinstance(polygon_2019.constGet(), (QgsPolygon, QgsMultiPolygon))
    ), f'Expected type {QgsPolygon}, found {type(polygon_2019)}'

    if CACHE:
        print('Reading cached features.')
        geoms = list(
            get_geometries(
                QgsVectorLayer(
                    (cache_dir / 'centerline.shp').as_posix(),
                    'centerline',
                    'ogr'
                )
            )
        )
        centerlines = Centerlines(geoms)
    else:
        centerlines = Centerlines.from_polygon(
            polygon_2019, epsg=epsg, output=cache_dir / 'centerline.shp'
        )

    start_points = []
    for centerline in centerlines.intersects(difference):
        first, last = Endpoints.from_linestring(centerline).as_qgis_geometry()
        if (
            first.intersects(limit_difference)
            and not last.intersects(limit_2012)
            and not first.intersects(limit_2012)
        ):
            start_points.append(first)

    points = intersection_points(centerlines, polygon_2012)
    centerlines_as_layer = geometries_to_layer(
        centerlines, epsg=epsg, name='centerlines')
    points_as_layer = geometries_to_layer(
        points, epsg=epsg, name='intersection_points')
    start_points_as_layer = geometries_to_layer(
        start_points, epsg=epsg, name='graph_start_points')
    shortest_paths = list(build_graph(
        start_points,
        centerlines_as_layer,
        points
    ))
    shortest_paths_as_layer = geometries_to_layer(
        shortest_paths, epsg, 'shortest_paths')
    if RUNNING_FROM_CONSOLE:
        instance = QgsProject.instance()
        instance.removeAllMapLayers()
        instance.addMapLayer(centerlines_as_layer)
        instance.addMapLayer(points_as_layer)
        instance.addMapLayer(start_points_as_layer)
        instance.addMapLayer(shortest_paths_as_layer)
    # export(
    #     geometries_to_layer(centerlines_subset, epsg=epsg),
    #     cache_dir / 'centerline_erosion.shp'
    # )



if __name__ in ['__main__', '__console__']:
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
