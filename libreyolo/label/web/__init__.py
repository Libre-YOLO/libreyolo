"""The embedded LibreLabel single-page app, assembled from parts.

No build step, no framework, no third-party JS -- the page is still ONE
self-contained HTML string served verbatim at ``/``. It is merely AUTHORED
in reviewable pieces (CSS, markup, JS split by concern) instead of a single
3,000-line literal; every part is concatenated verbatim, in order, at import
time. Keep each part cut at top-level statement boundaries so any part still
parses on its own.
"""

from .html_head import PART as _html_head
from .css import PART as _css
from .html_body import PART as _html_body
from .js_state import PART as _js_state
from .js_app import PART as _js_app
from .js_editor import PART as _js_editor
from .js_canvas import PART as _js_canvas
from .js_power import PART as _js_power
from .js_home import PART as _js_home
from .html_tail import PART as _html_tail

INDEX_HTML = "".join((
    _html_head,
    _css,
    _html_body,
    _js_state,
    _js_app,
    _js_editor,
    _js_canvas,
    _js_power,
    _js_home,
    _html_tail,
))
