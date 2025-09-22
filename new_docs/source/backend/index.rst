Backend (FastAPI) Documentation
===============================

Welcome to the documentation for the **Backend** of the *AILA Chatbot*!  

This section provides a complete reference to the backend system, which is built with **FastAPI**.  
It explains the architectural design, core functionality, and the public API surface generated from inline docstrings across the backend modules and packages.  

The backend is responsible for:  

- Managing authentication and user sessions  
- Handling real-time WebSocket communication for chat  
- Integrating the Retrieval-Augmented Generation (RAG) pipeline  
- Serving API endpoints to the frontend and external clients  

Overview
--------
High-level notes about the backend go here.  
(You can expand this section with architecture diagrams, flowcharts, or explanations of key modules like `auth`, `chat`, and `retrieval`.)  

API Reference
-------------
The following tree contains the full API reference for the backend package.  
It is auto-generated from the FastAPI codebase using `sphinx-autoapi`.  

- :doc:`backend/api </backend_api/backend/api/index>`
- :doc:`backend/reinforcement_learning </backend_api/backend/reinforcement_learning/index>`
- :doc:`backend/cache_models </backend_api/backend/cache_models/index>`
- :doc:`backend/crypt </backend_api/backend/crypt/index>`
- :doc:`backend/database </backend_api/backend/database/index>`
- :doc:`backend/main </backend_api/backend/main/index>`


.. [#f1] API documentation automatically generated with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_.

