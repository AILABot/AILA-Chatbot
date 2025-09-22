"""
User ORM Model
==============

The ``User`` ORM model represents a registered user in the system. It maps to the
``app_user`` table and contains authentication, verification, and role information.

Key features
~~~~~~~~~~~~
- PostgreSQL-native UUID primary key (``id``)
- Username, password, and session token storage
- User role management (e.g., ``admin``, ``member``)
- Email verification with codes and timestamps
- Verification state tracking

"""

from backend.database.config.connection_engine import declarativeBase
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy import VARCHAR, Boolean, TEXT, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID
import uuid
from datetime import datetime, timezone

class User(declarativeBase):
    """
    ORM model for the `app_user` table.
    Represents a registered user in the system.

    Attributes
    ----------
    id : UUID
        Primary key. Unique identifier for the user.
    user_name : str
        Username chosen by the user (max 255 chars).
    password : str
        Hashed password of the user.
    session_id : str
        Current session token for the user.
    role : str
        Role of the user (e.g., "admin", "member").
    email : str
        Email address of the user.
    verified : bool
        Whether the user's email has been verified.
    verification_code : str
        Code sent to the user for verification.
    code_created_on : datetime
        Timestamp when the verification code was generated.
    """

    __tablename__ = "app_user"

    id: Mapped[UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True
    )
    """Primary key. UUID of the user."""

    user_name: Mapped[str] = mapped_column(
        VARCHAR(255), nullable=False
    )
    """Username of the user (max length 255)."""

    password: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Hashed password of the user."""

    session_id: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Session token string associated with the user."""

    role: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Role assigned to the user (e.g., admin, member)."""

    email: Mapped[str] = mapped_column(
        VARCHAR(255), nullable=False
    )
    """Email address of the user (max length 255)."""

    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False
    )
    """Boolean flag indicating if the user has been verified."""

    verification_code: Mapped[str] = mapped_column(
        TEXT, nullable=False
    )
    """Verification code used for confirming user identity."""

    code_created_on: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now(timezone.utc)
    )
    """Datetime when the verification code was created. Defaults to current UTC time."""

    def __init__(
        self,
        user_name: str,
        password: str,
        role: str,
        email: str,
        verification_code: str,
        date_created_on,
        session_id: str,
    ):
        """
        Initialize a new User object.

        Parameters
        ----------
        user_name : str
            Username of the user.
        password : str
            Hashed password of the user.
        role : str
            Role of the user (e.g., "admin", "member").
        email : str
            Email address of the user.
        verification_code : str
            Verification code assigned to the user.
        date_created_on : datetime | str
            Creation timestamp for the verification code.
            Can be a datetime object or ISO8601 string.
        session_id : str
            Session token string for the user.
        """
        self.id = uuid.uuid4()
        self.session_id = session_id
        self.user_name = user_name
        self.password = password
        self.role = role
        self.email = email
        self.verified = False
        self.verification_code = verification_code
        if isinstance(date_created_on, str):
            self.code_created_on = datetime.fromisoformat(date_created_on)
        else:
            self.code_created_on = date_created_on

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the user.

        Returns
        -------
        str
            A formatted string containing the user ID, username, and (hashed) password.
        """
        return (
            f"Agent: id:{self.id}, username: {self.user_name}, password:{self.password}"
        )
