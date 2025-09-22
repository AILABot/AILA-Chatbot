"""
The `helpers` package provides utility functions and decorators
that support database operations and cross-cutting concerns.

These helpers simplify transaction handling, ensure consistent
session management, and reduce boilerplate across DAOs and services.

Contents
--------
- transactionManagement
    Provides tools for database transaction management:
        - Context variable (`db_session_context`) for propagating the active session across function calls without explicit passing
        - `@transactional` decorator for wrapping functions in a managed transaction:
            - Reuses an existing session if one is active in context
            - Creates, commits, and closes a new session otherwise
            - Rolls back the session on errors
            - Simplifies DAO/service logic by handling lifecycle automatically
"""