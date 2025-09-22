"""
The `crypt` package provides cryptographic utilities that secure
authentication workflows and user data.

It centralizes password hashing, password validation, and verification
code generation in order to enforce consistent security practices
across the application.

Contents
--------
- encrypt_decrypt
    Utility module exposing the `EncryptionDec` class:
        * `hash_password` — securely hashes plaintext passwords using bcrypt
        * `check_passwords` — verifies a plaintext password against a hashed one
        * `is_valid_password` — validates password complexity rules:
            - minimum length (8+)
            - must include lowercase, uppercase, digit, and special character
        * `generate_verification_code` — produces numeric verification codes (default length: 6 digits) for email verification or 2FA workflows
"""
