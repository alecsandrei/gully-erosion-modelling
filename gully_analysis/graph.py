from __future__ import annotations

import itertools
import typing as t

from qgis.analysis import (
    QgsGraphAnalyzer,
    QgsGraphBuilder,
    QgsVectorLayerDirector,
)
from qgis.core import QgsGeometry, QgsPointXY

if t.TYPE_CHECKING:
    from qgis.core import QgsProcessingFeedback, QgsVectorLayer

Path = QgsGeometry
Start = QgsPointXY
End = QgsPointXY
ShortestPath = tuple[Start, End, Path]
ShortestPaths = list[ShortestPath]


def get_shortest_paths(
    start_points: list[QgsGeometry],
    lines: QgsVectorLayer,
    destination_points: list[QgsGeometry],
    feedback: QgsProcessingFeedback | None = None,
) -> ShortestPaths:
    """Computes shortest paths from the start points to the destination points.

    The shortest path is the path from the start point to the closest
    destination point in the network.
    """
    # TODO: find a faster way to compute this
    director = QgsVectorLayerDirector(
        lines, -1, '', '', '', QgsVectorLayerDirector.Direction.DirectionBoth
    )
    builder = QgsGraphBuilder(lines.crs())

    start_points_as_xy = map(QgsGeometry.asPoint, start_points)
    destination_points_as_xy = map(QgsGeometry.asPoint, destination_points)
    tied_points = director.makeGraph(
        builder, itertools.chain(start_points_as_xy, destination_points_as_xy)
    )
    tied_start_points = tied_points[: len(start_points)]
    tied_end_points = tied_points[len(start_points) :]
    shortest_paths: ShortestPaths = []
    for i, tied_start_point in enumerate(tied_start_points, start=1):
        graph = builder.graph()
        start_idx = graph.findVertex(tied_start_point)
        shortest_route = None
        route_end_point = None
        for j, tied_end_point in enumerate(tied_end_points, start=1):
            end_idx = graph.findVertex(tied_end_point)
            tree, _ = QgsGraphAnalyzer.dijkstra(graph, start_idx, 0)
            route = [graph.vertex(end_idx).point()]

            replace = True
            while end_idx != start_idx:
                if tree[end_idx] == -1:
                    if feedback is not None:
                        feedback.pushWarning(
                            f'No route for start id {i}, end id {j}?, {route, end_idx, start_idx}'
                        )
                    replace = False
                    break
                end_idx = graph.edge(tree[end_idx]).fromVertex()
                route.insert(0, graph.vertex(end_idx).point())
                if shortest_route is not None and len(route) >= len(
                    shortest_route
                ):
                    replace = False
                    break

            if replace:
                route_end_point = tied_end_point
                shortest_route = route

        if shortest_route:
            assert route_end_point is not None
            shortest_paths.append(
                (
                    tied_start_point,
                    route_end_point,
                    QgsGeometry.fromPolylineXY(shortest_route),
                )
            )

    return shortest_paths
