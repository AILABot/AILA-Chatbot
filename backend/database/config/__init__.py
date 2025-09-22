"""
The `config` package provides two core building blocks for establishing and managing database connections.

Contents:
    - config: Configuration layer - strongly typed app settings loaded from environment variables (with .env support), exposed through a singleton Settings object
    - connection_engine: Database layer - SQLAlchemy bootstrap that constructs a connection URL from those settings, creates the Engine, shared MetaData, and the declarative base for ORM models

Together they provide secure, environment-driven configuration and a clean ORM foundation.
"""