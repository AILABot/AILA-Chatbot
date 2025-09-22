"""
The `database` package is responsible for all interactions with the application's database.
It provides configuration, entity definitions, CRUD operations, and utility functions 
that ensure smooth integration between the application and its data layer.

Contents:
    - config:
        Configuration files and settings required to establish and manage 
        database connections.
        
    - entities:
        SQLAlchemy entity models representing the database tables and schemas.
        
    - daos:
        Data Access Objects (DAOs) providing CRUD operations for the entities.
        
    - core:
        Core functions that connect application routers with the database 
        and orchestrate higher-level operations.
        
    - helpers:
        Utility helpers to manage database transactions, sessions, and 
        other low-level operations.
"""
