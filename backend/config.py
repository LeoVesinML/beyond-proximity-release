from pathlib import Path

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "scenes"

# Spatial graph
SPATIAL_SCALE_FACTOR = 3.5   # × median nearest-neighbour distance

# MCP
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001

# API
API_HOST = "127.0.0.1"
API_PORT = 8000

# Depth capture
DEPTH_NEAR = 0.1    # metres, must match THREE.js camera near
DEPTH_FAR  = 100.0  # metres, must match THREE.js camera far
