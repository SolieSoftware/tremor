import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tremor.api import causal_tests, events, monitor, signals
from tremor.causal.baselines import load_baselines
from tremor.causal.network import load_network
from tremor.config import settings
from tremor.models.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    for path in (settings.CAUSAL_NETWORK_PATH, settings.GRANGER_RESULTS_PATH):
        try:
            load_network(path)
            logger.info("Causal network loaded from %s", path)
            break
        except FileNotFoundError:
            continue
    else:
        logger.warning(
            "No causal network found at %s or %s — starting without network",
            settings.CAUSAL_NETWORK_PATH,
            settings.GRANGER_RESULTS_PATH,
        )
    try:
        load_baselines(settings.IRF_BASELINES_PATH)
        logger.info("IRF baselines loaded from %s", settings.IRF_BASELINES_PATH)
    except FileNotFoundError:
        logger.warning(
            "IRF baselines not found at %s — propagation directions default to positive",
            settings.IRF_BASELINES_PATH,
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
app.include_router(causal_tests.router)
