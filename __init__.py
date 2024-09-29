def classFactory(iface):
    """Load GullyAnalysis class from file GullyAnalysis"""
    from .gully_analysis.gully_analysis import GullyAnalysis
    return GullyAnalysis(iface)
