# ruff: noqa
# type: ignore
from __future__ import annotations

import sys
from pathlib import Path

from qgis.core import QgsApplication

qgs = QgsApplication([], False)
qgs.initQgis()
# TODO: Find a way to get this programaticaly
sys.path.append(
    '/home/alex/.local/share/QGIS/QGIS3/profiles/default/python/plugins'
)
from processing.core.Processing import Processing
from processing_saga_nextgen.saga_nextgen_plugin import (
    SagaNextGenAlgorithmProvider,
)

from gully_analysis.processing.provider import GullyAnalysisProvider

TEST_DATA_DIR = Path(__file__).parent.parent / 'test_data'

Processing.initialize()
gully_analysis_provider = GullyAnalysisProvider()
gully_analysis_provider.loadAlgorithms()
saga_provider = SagaNextGenAlgorithmProvider()
saga_provider.loadAlgorithms()
qgs.processingRegistry().addProvider(provider=saga_provider)
qgs.processingRegistry().addProvider(provider=gully_analysis_provider)
