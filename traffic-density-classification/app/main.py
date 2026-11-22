"""
TrafficVision - FastAPI Main Application Entrypoint
Wires middleware, routers, static frontend serving, and lifecycle events.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.dependencies import get_model_service
from app.routes import health, model_info, prediction


app = FastAPI(
    title="TrafficVision API",
    description="Real-Time Traffic Density & Congestion Classification Using Deep Learning (YOLO11)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(prediction.router)

# Mount frontend static directory if exists
frontend_path = Path(settings.FRONTEND_DIRECTORY)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="frontend_static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend_root():
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "TrafficVision API is operational. Visit /docs for Swagger UI."}


@app.on_event("startup")
async def startup_event():
    print("[TrafficVision] Starting application services...")
    # Pre-initialize model on startup
    model_svc = get_model_service()
    if model_svc.is_loaded():
        print(f"[TrafficVision] YOLO11 backbone ready on device: {model_svc.device}")
    else:
        print("[TrafficVision] [WARNING] YOLO11 backbone failed to initialize on startup.")


@app.on_event("shutdown")
async def shutdown_event():
    print("[TrafficVision] Shutting down application services.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
