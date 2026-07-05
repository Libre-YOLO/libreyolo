"""Local bounding-box annotation tool for LibreYOLO (``libreyolo label``).

A zero-third-party-dependency web annotator served from Python's standard-library
HTTP server -- the same pattern as :mod:`libreyolo.ui`. The page is an embedded
HTML/JS app served from :mod:`libreyolo.label.page`; labels are read from and
written to LibreYOLO's *native* YOLO ``labels/*.txt`` files, so a labelled
dataset trains with zero conversion.

v1 scope is deliberately small (the 20/80 cut): axis-aligned **bounding boxes**
(the ``detect`` task) only. Polygons, OBB and classification are future work and
are explicitly *not* edited here -- a label file containing non-box rows is
treated as read-only so it can never be clobbered.

See :mod:`libreyolo.label.server`.
"""

from .server import serve

__all__ = ["serve"]
