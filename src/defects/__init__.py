"""
Defects module for failure analysis, clustering, and reproducer synthesis.
"""

from .defect_clustering import (
    DefectDetectionSystem,
    DefectFingerprint,
    DefectCluster,
    MinimalReproducer,
    DefectSeverity,
    DefectCategory
)

__all__ = [
    "DefectDetectionSystem",
    "DefectFingerprint", 
    "DefectCluster",
    "MinimalReproducer",
    "DefectSeverity",
    "DefectCategory"
]
