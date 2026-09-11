"""Inert salvage from the ``z-anubis`` line: storage allotments and adapter routing.

Nothing in this package is imported or registered by the running application.
Each module defines a FastAPI ``APIRouter`` that ``src/api/webapp.py`` never
calls ``include_router`` on, so the routes do not exist at runtime and the
package changes no behaviour. See ``README.md`` beside this file for what was
salvaged, why, and the checklist for activating it.
"""
