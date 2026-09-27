"""Source-compatible BRCA DriverMONI models, graph inputs, and three-fold CV."""

import os

# Set this before any DGL import, including library use outside the CLI. Otherwise
# a fresh DGL install writes ~/.dgl/config.json or uses an unrelated saved backend.
if os.environ.get("DGLBACKEND", "pytorch") != "pytorch":
    raise RuntimeError("DriverMONI requires DGLBACKEND=pytorch; unset or correct DGLBACKEND before running")
os.environ["DGLBACKEND"] = "pytorch"
