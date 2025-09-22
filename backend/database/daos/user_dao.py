
"""
User DAO

Purpose
-------
Thin data-access layer for the `User` ORM entity. Provides:
- Creation with password hashing
- Lookup by username or email
- Token retrieval and updates
- Verification status and verification-code updates

Design
------
- The DAO expects an active SQLAlchemy `Session` supplied by the caller.
- Business logic (validation, authorization, transactions) should live in a
  higher-level service; the DAO focuses on persistence operations.
- Passwords are hashed using `EncryptionDec.hash_password(...)` before insert.

Entity (expected columns)
-------------------------
- id: uuid / primary key
- user_name: str (unique)
- email: str (unique)
- password: str (hashed)
- role : str
- verified: bool
- verification_code: str | None
- code_created_on: datetime | None
- session_id: str | None  (JWT or session token)

Usage
-----
.. code-block:: python

    from sqlalchemy.orm import Session
    from backend.database.connection_engine import connection_engine
    from backend.database.entities.user import User
    from backend.database.daos.user_dao import UserDao  # adjust path as needed

    dao = UserDao()
    with Session(connection_engine) as session:
        # Create user (password will be hashed)
        u = User(user_name="roman", email="roman@tribalchief.com", password="Spear#123")
        dao.createUser(session, u)
        session.commit()  # caller controls commit here

        # Fetch by username
        users = dao.fetchUser(session, "roman")     # returns list[User], at most 1 due to limit
        # Fetch by email
        users2 = dao.fetchUserByEmail(session, "roman@tribalchief.com")

        # Update verification status
        dao.updateVerified(session, "roman")

        # Update verification code and timestamp
        from datetime import datetime, timezone
        dao.updateVerCode(session, "roman", code="123456", code_created_on=datetime.now(timezone.utc))

        # Update token (e.g., after login)
        dao.updateToken(session, users[0].id, token="jwt-or-session-token")

Error Handling
--------------
- Each method catches generic `Exception`, prints a message, and re-raises.
  Consider replacing `print(...)` with structured logging (e.g., `logger.exception(...)`).
- Methods using `.one()` can raise `NoResultFound` or `MultipleResultsFound`.
  Upstream code should be ready to handle these.

Return Values
-------------
- createUser(...) -> bool
- fetchUser(...) -> list[User] (at most one row due to limit(1))
- fetchUserByEmail(...) -> list[User] (at most one row due to limit(1))
- fetchUserToken(...) -> str (intended), but see "Caveats" below
- updateVerified(...), updateVerCode(...), updateToken(...) -> None (commit inside)
"""

from sqlalchemy.orm import Session
from backend.database.entities.user import User
import uuid
from backend.crypt.encrypt_decrypt import EncryptionDec

class UserDao:
    """
    Data Access Object (DAO) for managing User entities.
    Provides methods for creating users, retrieving user data,
    and updating authentication/verification fields.
    """

    def createUser(self, session: Session, user_data: User) -> bool:
        """
        Create a new user in the database with a hashed password.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        user_data : User
            User entity object containing user details.

        Returns
        -------
        bool
            True if user creation is successful.

        Raises
        ------
        Exception
            If hashing or insertion fails.
        """
        try:
            enc = EncryptionDec()
            user_data.password = enc.hash_password(text=user_data.password)
            session.add(user_data)
            return True
        except Exception as e:
            print(f"Error in UserDao.createUser. Error Message: {e}")
            raise e

    def fetchUser(self, session: Session, username: str):
        """
        Fetch a user by username.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        username : str
            The username of the user.

        Returns
        -------
        list[User]
            A list containing the matching user (at most one due to limit(1)).

        Raises
        ------
        Exception
            If query fails.
        """
        try:
            users = session.query(User).filter(User.user_name == username).limit(1).all()
            return users
        except Exception as e:
            print(f"Error in UserDao.fetchUser. Error Message: {e}")
            raise e

    def fetchUserByEmail(self, session: Session, email: str):
        """
        Fetch a user by email.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        email : str
            Email address of the user.

        Returns
        -------
        list[User]
            A list containing the matching user (at most one due to limit(1)).

        Raises
        ------
        Exception
            If query fails.
        """
        try:
            users = session.query(User).filter(User.email == email).limit(1).all()
            return users
        except Exception as e:
            print(f"Error in UserDao.fetchUserByEmail. Error Message: {e}")
            raise e

    def fetchUserToken(self, session: Session, username: str):
        """
        Fetch the session token of a user by username.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        username : str
            The username of the user.

        Returns
        -------
        str
            The session token of the user.

        Raises
        ------
        Exception
            If user does not exist or query fails.
        """
        try:
            users = session.query(User).filter(User.user_name == username)
            return users.session_id
        except Exception as e:
            print(f"Error in UserDao.fetchUserToken. Error Message: {e}")
            raise e

    def updateVerified(self, session: Session, username: str):
        """
        Mark a user as verified.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        username : str
            The username of the user.

        Raises
        ------
        Exception
            If user cannot be found or update fails.
        """
        try:
            user = session.query(User).filter(User.user_name == username).one()
            user.verified = True
            session.commit()
        except Exception as e:
            print(f"Error in UserDao.updateVerified. Error Message: {e}")
            raise e

    def updateVerCode(self, session: Session, username: str, code: str, code_created_on):
        """
        Update a user's verification code and creation timestamp.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        username : str
            The username of the user.
        code : str
            The new verification code.
        code_created_on : datetime
            Timestamp when the code was generated.

        Raises
        ------
        Exception
            If update fails.
        """
        try:
            user = session.query(User).filter(User.user_name == username.strip()).one()
            user.verification_code = code
            user.code_created_on = code_created_on
            session.commit()
        except Exception as e:
            print(f"Error in UserDao.updateVerCode. Error Message: {e}")
            raise e

    def updateToken(self, session: Session, user_id: uuid.UUID, token: str):
        """
        Update a user's session token.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        user_id : uuid.UUID
            Unique identifier of the user.
        token : str
            The new session token.

        Raises
        ------
        Exception
            If update fails.
        """
        try:
            user = session.query(User).filter(User.id == user_id).one()
            user.session_id = token
            session.commit()
        except Exception as e:
            print(f"Error in UserDao.updateToken. Error Message: {e}")
            raise e
