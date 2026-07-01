"""The embedded LibreLabel single-page app (HTML + CSS + vanilla JS).

Served verbatim at ``/`` by :mod:`libreyolo.label.server`. The page is
assembled -- without any build step -- from the parts in
:mod:`libreyolo.label.web`; this module just re-exports the final string.
"""

from .web import INDEX_HTML  # noqa: F401
