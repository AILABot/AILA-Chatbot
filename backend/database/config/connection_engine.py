"""
Connection Engine (SQLAlchemy)

Purpose
-------
Centralizes database initialization for the application:
- Builds a secure SQLAlchemy connection URL from environment-backed settings.
- Creates the Engine (connection pool + SQL execution entry point).
- Defines shared MetaData for table and schema objects.
- Exposes a Declarative Base class for ORM models.

Notes
-----
- Uses `URL.create(...)` to avoid hardcoding credentials and to keep configuration
  environment-driven (e.g., via `.env`, container secrets, or deployment vars).
- Engine settings can be tuned per deployment (pool size, health checks, SSL).
- All ORM models must inherit from `declarativeBase` to participate in schema reflection
  and enable ORM features.
"""


from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base
from sqlalchemy.schema import MetaData
from backend.database.config.config import settings

# --------------------------------------------------------------------
# Construct the SQLAlchemy connection URL using values from Settings.
# This ensures credentials and connection details are loaded securely
# from environment variables or a `.env` file.
# --------------------------------------------------------------------
connection_url = URL.create(
    drivername=settings.DB_DRIVER_NAME,   # e.g., "postgresql", "mysql", "sqlite"
    username=settings.DB_USERNAME,        # Database username
    password=settings.DB_PASSWORD,        # Database password
    host=settings.DB_HOST,                # Hostname or IP of the DB server
    database=settings.DB_DATABASE_NAME    # Name of the database
)
"""Constructs the SQLAlchemy connection URL using values from Settings. 
This ensures credentials and connection details are loaded securely from environment variables or a `.env` file.
"""

# --------------------------------------------------------------------
# Engine object: core interface to the database.
# Responsible for managing connections, executing SQL, and pooling.
# --------------------------------------------------------------------
connection_engine = create_engine(connection_url)
"""Engine object: Core interface to the database.
Responsible for managing connections, executing SQL, and pooling.
"""

# --------------------------------------------------------------------
# Metadata object: stores schema-level information about tables,
# constraints, indexes, etc. Shared across all models.
# --------------------------------------------------------------------
metadata = MetaData()
"""
Metadata object: Stores schema-level information about tables, constraints, indexes, etc. Shared across all models.
"""
# --------------------------------------------------------------------
# Declarative Base: root class for ORM models.
# All model classes should inherit from this to gain ORM features
# and automatic schema generation.
# --------------------------------------------------------------------
declarativeBase = declarative_base(metadata=metadata)
"""Declarative Base: Root class for ORM models.
All model classes should inherit from this to gain ORM features and automatic schema generation.
 """
