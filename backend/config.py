from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CAT Operator Guardian API"
    app_version: str = "1.0.0"
    database_url: str = "sqlite:///./cat_guardian.db"
    models_dir: str = "../ml/saved_models"
    ml_module_dir: str = "../ml"

    class Config:
        env_file = ".env"


settings = Settings()
