"""
FastAPI application bootstrap with: \n
- Lifespan-managed initialization of the LLM pipeline (indexes + reranker client) \n
- CORS configured for the frontend \n
- Authenticated WebSocket endpoint (cookie-based token) \n
- Static file serving for built frontend \n
- Catch-all route to support React Router \n

Environment contract (from `settings`): \n
- INIT_MODE: if 'runtime', preload indexes & reranker during app startup. \n
- FRONTEND_URL: allowed CORS origin. \n
"""

from fastapi import FastAPI, WebSocket, Cookie, WebSocketDisconnect
from backend.api.fast_api import router
from backend.api.utils import verify_token
from fastapi.middleware.cors import CORSMiddleware
from backend.database.config.config import settings
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging
from contextlib import asynccontextmanager
from backend.api.llm_pipeline import initialize_indexes, load_reranker_model, LLM_Pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan manager.

    Notes
    ------------
    - On startup (before yielding):
        * If INIT_MODE == 'runtime':
            - Build vector indexes (top_k=10).
            - Load Cohere client + reranker (retry loop until available).
            - Construct and initialize `LLM_Pipeline`, attach to `app.state`.
    - On shutdown (after yielding):
        * If a pipeline was created, call `shutdown()` to release resources.
    """
    print("⚙️  Loading vector index...")
    pipeline = None

    if settings.INIT_MODE == "runtime":
        indexes = None
        cohere_client = None
        reranker = None

        # Retry until reranker is ready (e.g., cold start, network hiccup)
        while reranker is None:
            indexes = initialize_indexes(top_k=10)
            # cohere_client, reranker = load_reranker_model(type = 'cohere')
            reranker = load_reranker_model(type = 'cross-encoder')['reranker_model']
            
        pipeline = LLM_Pipeline(indexes, reranker, cohere_client)
        app.state.pipeline = pipeline
        app.state.app = pipeline.initialize_workflow()
        app.state.user_data_dict = {}
        print("✅ Vector index and pipeline loaded.")
    else:
        print(f"⏭️  Skipping runtime init (INIT_MODE={settings.INIT_MODE}).")

    try:
        yield
    finally:
        # Guard against shutdown when pipeline wasn't created
        if hasattr(app.state, "pipeline") and app.state.pipeline is not None:
            app.state.pipeline.shutdown()
            print("🛑 Pipeline shutdown complete.")
        else:
            print("🛑 App shutting down (no pipeline to release).")


# Instantiate the FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)
"""Instatiates a FastAPI application object
    The lifespan=lifespan argument registers a custom startup/shutdown lifecycle manager that:\n
        - On startup: initializes the LLM pipeline (indexes + reranker) if INIT_MODE == 'runtime'.\n
        - On shutdown: gracefully releases pipeline resources. \n
"""

logger = logging.getLogger("uvicorn")
"""Logger instance for capturing and emitting Uvicorn server logs."""

# -----------------------
# CORS configuration
# -----------------------
url = settings.FRONTEND_URL
"""The allowed frontend origin (URL) used for CORS configuration.
This value comes from application settings and represents the domain that is permitted to interact with the backend via cross-origin requests.
"""


app.add_middleware(
    CORSMiddleware,
    allow_origins=[url],      # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# API routes
# -----------------------
app.include_router(router)

# -----------------------
# Static assets (built frontend)
# -----------------------
app.mount("/assets", StaticFiles(directory="frontend/dist/assets", html=True), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Cookie(None)):
    """
    Authenticated WebSocket endpoint.

    Auth
    ----
    - Expects a cookie named `token`.
    - `verify_token(token)` returns a username on success; otherwise connection is closed.

    Protocol
    --------
    - Upon connect: accepts and greets the authenticated user.
    - Echo loop: echoes back text messages.
    - On disconnect: logs a simple message (replace with real cleanup if needed).

    Close Codes
    -----------
    - 1008: Policy Violation (used when auth fails).
    """
    await websocket.accept()
    username = verify_token(token)
    if not username:
        await websocket.close(code=1008)
        return

    await websocket.send_text(f"Hello {username}! You are authenticated.")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"You said: {data}")
    except WebSocketDisconnect:
        print(f"{username} disconnected")


# ✅ Catch-all route for React Router (must come after mounting static)
@app.get("/")
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str = ""):
    """
    Serve the frontend's index.html for all non-API routes to support client-side routing.
    """
    return FileResponse(os.path.join("frontend", "dist", "index.html"))
