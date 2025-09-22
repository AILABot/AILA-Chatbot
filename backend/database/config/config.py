"""
Configuration — Pydantic v2 Settings (env / .env)
=================================================

Purpose
-------
Centralized, strongly-typed application configuration using:
- Pydantic v2 `BaseSettings` for environment-driven values
- `pydantic-settings` v2 for `.env` loading and model config

Load Order & Behavior
---------------------
- Values are read from the environment; if not present, `.env` is used.
- Missing required fields raise a validation error at import time.
- `extra="ignore"`: unknown env vars are ignored (not an error).

Usage
-----
from backend.database.config.config import settings

# Example
db_host = settings.DB_HOST
openai_model = settings.OPEN_AI_MODEL

Security
--------
- Never commit secrets or the `.env` file to source control.
- Prefer runtime environment variables in production (K8s/Secrets Manager/etc.).
"""


from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables
    or a `.env` file. Provides strongly typed access to environment values.
    """

    # Pydantic v2 config (replaces inner class Config)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # or "forbid"/"allow"
    )

    OLLAMA_SERVER_URL: str = Field(..., description="URL of the Ollama server for model inference.")
    FRONTEND_URL: str = Field(..., description="Base URL of the frontend client application.")
    DB_USERNAME: str = Field(..., description="Database username credential.")
    DB_PASSWORD: str = Field(..., description="Database password credential.")
    DB_HOST: str = Field(..., description="Hostname or IP address of the database server.")
    DB_DATABASE_NAME: str = Field(..., description="Name of the application’s database.")
    DB_DRIVER_NAME: str = Field(..., description="Database driver (e.g., `postgresql`, `mysql`, `sqlite`).")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(..., description="Duration (in minutes) before access tokens expire.")
    API_KEY: str = Field(..., description="OPEN API key for application-level integrations.")
    SECRET_KEY: str = Field(..., description="Secret key for signing tokens and securing sensitive operations.")
    ALGORITHM: str = Field(..., description="Cryptographic algorithm used for JWT or token signing (e.g., `HS256`).")
    VITE_API_URL: str = Field(..., description="API base URL injected into the frontend (e.g., Vite builds).")
    APP_PASSWORD: str = Field(..., description="Application-specific password (e.g., for email sending).")
    SENDER_EMAIL: str = Field(..., description="Default email address used for sending application emails.")
    COHERE_API_KEY: str = Field(..., description="API key for accessing Cohere’s services.")
    COHERE_MODEL_ID: str = Field(..., description="Identifier of the Cohere model to use.")
    INIT_MODE: str = Field(..., description="Initialization mode (e.g., `dev`, `prod`, `test`).")
    OPEN_AI_MODEL: str = Field(..., description="OpenAI model name (e.g., `gpt-4o-mini`).")
    TAVILY_API_KEY: str = Field(..., description="API key for Tavily API integration.")
    AWS_PROFILE: str = Field(..., description="AWS named profile.")
    AWS_ACCESS_KEY: str = Field(..., description="AWS access key ID.")
    AWS_SECRET_KEY: str = Field(..., description="AWS secret access key.")
    BUCKET_NAME: str = Field(..., description="Default S3 bucket name.")
    REGION: str = Field(..., description="AWS region name (e.g., `eu-central-1`).")
    HF_TOKEN: str = Field(..., description="Hugging Face Token for accessing the embedding and reranking models")

# Singleton instance of Settings, ready to be imported across the app
settings = Settings()
"""Defines a Settings object that contains the contents of the .env file"""