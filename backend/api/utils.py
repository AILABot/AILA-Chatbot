"""
JWT utilities for issuing and verifying access tokens.

Functions
---------
create_access_token(data: dict) -> str
    Creates a signed JWT access token with an expiration (`exp`) claim.
verify_token(token: str) -> str | None
    Verify a JWT's signature & expiration and return the subject (`sub`) if valid.

Environment contract (from `settings`)
--------------------------------------
SECRET_KEY : str
    HMAC signing key for JWTs.
ALGORITHM : str
    JWT signing algorithm (e.g., "HS256").
ACCESS_TOKEN_EXPIRE_MINUTES : int
    Token lifetime window in minutes.
"""

from datetime import datetime
from typing import Optional

from jose import jwt, JWTError
from backend.database.config.config import settings


def create_access_token(data: dict) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ----------
    data : dict
        Claims to embed in the token
        Retrieves it in `verify_token` function.

    Returns
    -------
    str
        Encoded JWT string.

    Notes
    ----------
    - Adds an `exp` (expiration) claim calculated from ACCESS_TOKEN_EXPIRE_MINUTES.
    - Uses `settings.SECRET_KEY` and `settings.ALGORITHM` for signing.   


    """
    expiration_time = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    encoding = data.copy()
    # exp is a NumericDate (seconds since epoch)
    expires = int(datetime.now().timestamp()) + (int(expiration_time) * 60)
    encoding.update({"exp": expires})
    return jwt.encode(encoding, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT and return its subject.


    Parameters
    ----------
    token : str
        Encoded JWT string from the client (e.g., cookie or Authorization header).

    Returns
    ----------
    str | None
        The `sub` claim (subject) if the token is valid, otherwise None.
    
    Notes
    ----------
    - Decodes and validates the signature and expiration using SECRET_KEY/ALGORITHM.
    - Returns `payload.get('sub')` for downstream authentication logic.
    - On any JWTError (invalid signature, expired, malformed), returns None.    

    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except JWTError as e:
        # Consider replacing print with structured logging in production
        print(e)
        return None
