#!/usr/bin/env python
"""Write docs/openapi.json from the FastAPI app (no models or data needed)."""

import json

from visualsearch.api.app import create_app
from visualsearch.config import PROJECT_ROOT, Settings

app = create_app(settings=Settings())
out = PROJECT_ROOT / "docs" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2))
print(f"wrote {out} ({len(app.openapi()['paths'])} paths)")
