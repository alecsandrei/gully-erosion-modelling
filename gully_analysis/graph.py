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
from qgis.core import Qgis, QgsGeometry, QgsPointXY

from .geometry import Endpoints

if t.TYPE_CHECKING:
    from qgis.core import QgsProcessingFeedback, QgsVectorLayer


@dataclass
class ProfilePathMapper:
    profile_pour_points: c.MutableSequence[QgsGeometry]
    profiles: list[QgsGeometry]
    shortest_paths: list[ShortestPath]

    def fix_shortest_path_if_intersects(
        self, shortest_path: ShortestPath
    ) -> bool:
        """The shortest path can intersect the profile multiple times.

        This function stops the shortest path at the first intersection with the
        longitudinal profile.
        """
        intersecting_lines = [
            line
            for line in self.profiles
            if Endpoints.from_linestring(line).first == shortest_path.end
        ]
        if len(intersecting_lines) != 1:
            if len(intersecting_lines) > 1:
                print('Error: expected at most one intersecting line.')
            return False
        intersecting_line = intersecting_lines[0]
        intersection = intersecting_line.intersection(shortest_path.path)
        if intersection.wkbType() != Qgis.WkbType.MultiPoint:
            return False
        point_intersections = intersection.coerceToType(Qgis.WkbType.Point)
        intersection_point: QgsPointXY = [
            point.asPoint()
            for point in point_intersections
            if point.asPoint() != shortest_path.end
        ][0]
        shortest_path.path.splitGeometry([intersection_point], False)
        shortest_path.path = shortest_path.path.coerceToType(
            Qgis.WkbType.LineString
        )[0]
        shortest_path.end = intersection_point
        return True

    def get_mapped_profiles(self) -> list[QgsGeometry]:
        """Maps the flow path profiles with the shortest paths (centerlines).

        This is done using the flow path profile pour points, which coincide
        with the destination point (the second value in the tuple) of the
        ShortestPath. Returns N lines merged from the shortest path and the flow
        path profile which continues "downstream", where N is the length of
        shortest_paths.
        """

        def get_matched_flow_path_profile(point: QgsPointXY) -> QgsGeometry:
            pour_point = None
            for i, pour_point in enumerate(self.profile_pour_points):
                if pour_point.asPoint() == point:
                    return self.profiles[i]
            raise Exception(
                f'Should not have reached here: {pour_point, point}'
            )

        merged_lines = []
        for i, shortest_path in enumerate(self.shortest_paths):
            flow_path_profile = get_matched_flow_path_profile(shortest_path.end)
            updated = self.fix_shortest_path_if_intersects(shortest_path)
            flow_path_profile_copy = QgsGeometry(flow_path_profile)
            assert flow_path_profile_copy.intersects(shortest_path.path)
            if updated:
                # stubs say it is a List[QgsPoint] | List[QgsPointXY],
                # but a QgsGeometry is actually returned
                splitted_line = t.cast(
                    QgsGeometry,
                    flow_path_profile_copy.splitGeometry(
                        [shortest_path.end], False
                    )[1][0],
                )
                flow_path_profile_copy = splitted_line.coerceToType(
                    Qgis.WkbType.LineString
                )[0]
                assert not flow_path_profile_copy.isMultipart()
            flow_path_profile_copy.addPartGeometry(shortest_path.path)
            merged = flow_path_profile_copy.mergeLines()
            merged_endpoints = Endpoints.from_linestring(merged)
            _flow_path_profile_endpoints = Endpoints.from_linestring(
                flow_path_profile
            )
            assert merged_endpoints[0] == shortest_path.start, (
                merged_endpoints[0],
                shortest_path.start,
            )
            assert merged_endpoints[1] == _flow_path_profile_endpoints[1], (
                merged_endpoints[1],
                _flow_path_profile_endpoints[1],
            )
            merged_lines.append(merged)
        return merged_lines


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
