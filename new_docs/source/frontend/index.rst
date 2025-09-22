Frontend Typescript Documentation
=================================

Welcome to the documentation of the **AILA-Chatbot** frontend!

This documentation provides a complete overview of the **AILA-Chatbot** application.  
It covers the core functionality used to build the frontend of the application,  
and includes a reference to the public API generated from docstrings across the  
project’s modules and packages.

Overview
--------

The AILA-Chatbot frontend is a **TypeScript + React** application that serves as the user-facing interface for interacting with the chatbot.  
It is designed with modular components, a context-based authentication system, and a clean architecture for scalability.

**Key Features**:

- **Authentication** (Login, Register, Persistent Session)  
- **Chat Interface** with real-time WebSocket support  
- **Reusable Components** with a consistent design system  
- **TypeScript Models** for strongly-typed development  
- **Service Layer** for API communication (Axios-based)  
- **Context API** for managing authentication state  


Project Structure
------------------

The project is organized into modular packages for maintainability and clarity:

.. toctree::
   :maxdepth: 1
   :glob:

   api/api/axios/README
   api/App/README
   api/components/Template/README
   api/context/AuthContext/README
   api/main/README
   api/models/Types/README
   api/pages/Chat/README
   api/pages/Login/README
   api/pages/Register/README
   api/services/AuthService/README
