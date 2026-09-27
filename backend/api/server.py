"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.api.routes import scenes, captures, tree, query, query_log

app = FastAPI(title="SemanticSplat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenes.router,   prefix="/api/scenes",   tags=["scenes"])
app.include_router(captures.router, prefix="/api/captures", tags=["captures"])
app.include_router(tree.router,     prefix="/api/tree",     tags=["tree"])
app.include_router(query.router,     prefix="/api/query",      tags=["query"])
app.include_router(query_log.router, prefix="/api/query-log",  tags=["query-log"])

# Serve PLY / splat files from the project-root `scenes/` directory.
# Path: backend/api/server.py -> backend/api -> backend -> project_root -> scenes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCENES_DIR   = _PROJECT_ROOT / "scenes"
_SCENES_DIR.mkdir(exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(_SCENES_DIR)), name="scenes")


@app.get("/api/health")
def health():
    scenes_dir = str(_SCENES_DIR)
    ply_files = list(_SCENES_DIR.glob("*.ply"))
    return {
        "status": "ok",
        "scenes_dir": scenes_dir,
        "ply_files": [f.name for f in ply_files],
    }


if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT
    # reload=False to avoid subprocess path issues with StaticFiles
    uvicorn.run("backend.api.server:app", host=API_HOST, port=API_PORT, reload=False)
