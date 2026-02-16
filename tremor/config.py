from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./tremor.db"
    CAUSAL_NETWORK_PATH: str = "data/causal_network.graphml"
    GRANGER_RESULTS_PATH: str = "data/granger_results.csv"
    IRF_BASELINES_PATH: str = "data/irf_baselines.json"
    DEFAULT_SHOCK_THRESHOLD_SD: float = 2.0
    PROPAGATION_BUFFER_WEEKS: int = 2

    model_config = {"env_prefix": "TREMOR_"}


settings = Settings()
