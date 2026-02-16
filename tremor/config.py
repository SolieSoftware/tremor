from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./tremor.db"
    CAUSAL_NETWORK_PATH: str = "data/causal_network.graphml"
    GRANGER_RESULTS_PATH: str = "data/granger_results.csv"
    IRF_BASELINES_PATH: str = "data/irf_baselines.json"
    DEFAULT_SHOCK_THRESHOLD_SD: float = 2.0
    PROPAGATION_BUFFER_WEEKS: int = 2
    MIN_EVENTS_FOR_CAUSAL_TEST: int = 5
    DEFAULT_PRE_WINDOW_DAYS: int = 5
    DEFAULT_POST_WINDOW_DAYS: int = 5
    DEFAULT_OVERLAP_BUFFER_DAYS: int = 10
    CAUSAL_SIGNIFICANCE_LEVEL: float = 0.05

    model_config = {"env_prefix": "TREMOR_"}


settings = Settings()
