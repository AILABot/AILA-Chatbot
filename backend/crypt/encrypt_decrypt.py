import bcrypt
import re
import random
import string

class EncryptionDec:
    """
    Utility class for password hashing, validation, and verification code generation.
    
    Methods
    -------
    hash_password(text: str) -> str
        Hashes a plaintext password using bcrypt with a generated salt.
    check_passwords(plain_text: str, passwd: str) -> bool
        Verifies a plaintext password against a hashed password.
    is_valid_password(password: str) -> bool
        Validates that a password meets security requirements:
        - At least 8 characters
        - At least one lowercase letter
        - At least one uppercase letter
        - At least one digit
        - At least one special character
    generate_verification_code(length: int = 6) -> str
        Generates a numeric verification code of given length (default: 6 digits).
    """

    def __init__(self):
        """Initialize the EncryptionDec utility."""
        pass

    def hash_password(self, text: str) -> str:
        """
        Hash a plaintext password using bcrypt.

        Parameters
        ----------
        text : str
            The plaintext password.

        Returns
        -------
        str
            The bcrypt-hashed password (UTF-8 decoded).
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(text.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def check_passwords(self, plain_text: str, passwd: str) -> bool:
        """
        Verify if a plaintext password matches a hashed password.

        Parameters
        ----------
        plain_text : str
            The plaintext password to check.
        passwd : str
            The previously hashed password to verify against.

        Returns
        -------
        bool
            True if the password matches, False otherwise.
        """
        return bcrypt.checkpw(plain_text.encode("utf-8"), passwd.encode("utf-8"))

    def is_valid_password(self, password: str) -> bool:
        """
        Validate that a password meets security complexity rules.

        Parameters
        ----------
        password : str
            The plaintext password to validate.

        Returns
        -------
        bool
            True if password is valid, False otherwise.

        Notes
        -----
        - Minimum length: 8 characters
        - Must contain at least:
          - one lowercase letter
          - one uppercase letter
          - one digit
          - one special character (!@#$%^&*(),.?":{}|<>)
        """
        if len(password) < 8:
            return False

        has_lower = re.search(r"[a-z]", password)
        has_upper = re.search(r"[A-Z]", password)
        has_digit = re.search(r"\d", password)
        has_special = re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)

        return all([has_lower, has_upper, has_digit, has_special])

    def generate_verification_code(self, length: int = 6) -> str:
        """
        Generate a numeric verification code of specified length.

        Parameters
        ----------
        length : int, optional
            Number of digits in the code. Default is 6.

        Returns
        -------
        str
            A randomly generated numeric code of given length.

        Example
        -------
        >>> enc = EncryptionDec()
        >>> enc.generate_verification_code()
        '493027'
        """
        return "".join(random.choices(string.digits, k=length))
