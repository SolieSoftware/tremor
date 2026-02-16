import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tremor.api import events, monitor, signals
from tremor.causal.network import load_network
from tremor.config import settings
from tremor.models.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        load_network(settings.CAUSAL_NETWORK_PATH)
        logger.info("Causal network loaded from %s", settings.CAUSAL_NETWORK_PATH)
    except FileNotFoundError:
        logger.warning(
            "Causal network file not found at %s — starting without network",
            settings.CAUSAL_NETWORK_PATH,
        )
    yield


app = FastAPI(
    title="Tremor",
    description="Event-driven causal shock monitor for financial markets",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(signals.router)
app.include_router(monitor.router)
