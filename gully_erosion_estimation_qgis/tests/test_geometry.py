import unittest

from qgis.core import (
    QgsGeometry,
    QgsPoint,
)

from gully_erosion_estimation_qgis.geometry import (
    endpoints_gen,
    Endpoints
)


class TestGeometry(unittest.TestCase):

    def test_endpoints_gen(self):

        points = [
            QgsPoint(x=47.175243, y=27.5742408),
            QgsPoint(x=47.185789, y=27.5569165),
            QgsPoint(x=47.1839887, y=27.5587854)
        ]
        geometry = QgsGeometry.fromPolyline(points)
        first, last = points[0], points[-1]
        self.assertEqual(
            next(endpoints_gen([geometry])),
            Endpoints(first, last)
        )
