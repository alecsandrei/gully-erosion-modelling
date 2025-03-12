from __future__ import annotations

import collections.abc as c
import itertools
import typing as t
from dataclasses import dataclass

from qgis.analysis import (
    QgsGraphAnalyzer,
    QgsGraphBuilder,
    QgsVectorLayerDirector,
)
from qgis.core import Qgis, QgsGeometry, QgsPoint, QgsPointXY

from .geometry import Endpoints
from .utils import get_geometries_from_layer

if t.TYPE_CHECKING:
    from qgis.core import QgsProcessingFeedback, QgsVectorLayer


@dataclass
class ProfilePathMapper:
    profile_pour_points: c.Sequence[QgsGeometry]
    profiles: c.Sequence[QgsGeometry]
    shortest_paths: c.Sequence[ShortestPath]

    def get_mapped_profiles(self) -> c.Generator[QgsGeometry]:
        """Maps the flow path profiles with the shortest paths (centerlines).

        This is done using the flow path profile pour points, which coincide
        with the destination point (the second value in the tuple) of the
        ShortestPath. Yields N lines merged from the shortest path and the flow
        path profile which continues "downstream", where N is the length of
        shortest_paths.
        """

        def get_matched_flow_path_profile(point: QgsPointXY) -> QgsGeometry:
            for i, pour_point in enumerate(self.profile_pour_points):
                if pour_point.asPoint() == point:
                    return self.profiles[i]
            assert False, 'Should not have reached here.'

        for shortest_path in self.shortest_paths:
            start, end, path = shortest_path
            flow_path_profile = get_matched_flow_path_profile(end)
            flow_path_profile_copy = QgsGeometry(flow_path_profile)
            assert flow_path_profile_copy.intersects(path)
            flow_path_profile_copy.addPartGeometry(path)
            merged = flow_path_profile_copy.mergeLines()
            merged_endpoints = Endpoints.from_linestring(merged)
            _flow_path_profile_endpoints = Endpoints.from_linestring(
                flow_path_profile
            )
            assert merged_endpoints[0] == start
            assert merged_endpoints[1] == _flow_path_profile_endpoints[1]
            yield merged


class ShortestPath(t.NamedTuple):
    start: QgsPointXY
    end: QgsPointXY
    path: QgsGeometry


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
    qgs_lines = list(get_geometries_from_layer(lines))
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
                ShortestPath(
                    tied_start_point,
                    route_end_point,
                    QgsGeometry.fromPolylineXY(shortest_route),
                ),
            )

    return shortest_paths
