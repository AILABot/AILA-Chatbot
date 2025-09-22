"""
Service-layer operations for authentication, conversations, and messages.

All functions are wrapped with the `@transactional` decorator, which manages
SQLAlchemy sessions and transactions automatically. Each function accepts (and
uses) an injected `session: Session` provided by the decorator.

This module provides high-level operations that orchestrate DAO calls and
auxiliary services (encryption, email).
"""

from backend.database.helpers.transactionManagement import transactional
from sqlalchemy.orm import Session
import uuid
from backend.database.daos.user_dao import UserDao
from backend.database.daos.conversation_dao import ConversationDao
from backend.database.daos.user_message_dao import UserMessagesDao
from backend.database.daos.document_feedback_dao import DocumentFeedbackDao
from backend.crypt.encrypt_decrypt import EncryptionDec
from backend.database.entities.messages import UserMessage
from backend.database.entities.conversations import Conversation
from backend.database.entities.user import User
from backend.database.entities.document_feedback import DocumentFeedback
from backend.api.models import UserAuthentication
from datetime import datetime, timezone, timedelta
from backend.database.config.config import settings
import smtplib
from backend.api.models import DocumentFeedbackDetails
from email.mime.text import MIMEText


@transactional
def login_user(session: Session, username: str, password: str) -> UserAuthentication:
    """
    Authenticate a user by username and password.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Username to authenticate.
    password : str
        Plaintext password to verify.

    Returns
    -------
    UserAuthentication
        A dict-like structure with:
          - authenticated (bool): True if credentials are valid.
          - detail (str): Error or info message.
          - user_details (dict | None): On success: {username, email, verified}; on failure may contain partial info.

    Notes
    -----
    - Uses EncryptionDec.check_passwords for constant-time verification.
    - If the username does not exist, returns authenticated=False with a message.
    """
    user_dao = UserDao()
    enc = EncryptionDec()
    users_fetched = user_dao.fetchUser(session, username)
    if len(users_fetched) == 0:
        return {
            "authenticated": False,
            "detail": "No user was found with that username",
            "user_details": None,
        }
    for user in users_fetched:
        if enc.check_passwords(password, user.password):
            return {
                "authenticated": True,
                "detail": "",
                "user_details": {
                    "username": user.user_name,
                    "email": user.email,
                    "verified": user.verified,
                    "role":user.role
                },
            }
        else:
            return {
                "authenticated": False,
                "detail": "Password is wrong",
                "user_details": {"username": user.user_name, "email": user.email},
            }


@transactional
def check_create_user_instance(session: Session, username: str, password: str, email: str, role:str):
    """
    Validate uniqueness, validate password policy, create a new user, and send a verification code.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Desired username (must be unique).
    password : str
        Plaintext password to be validated and hashed at DAO level.
    email : str
        Email address (must be unique).

    Returns
    -------
    dict
        - On success: {'res': True, 'detail': <verification_code>} \n
        - On failure: {'res': False, 'detail': <reason>} \n

    Notes
    --------
    - Checks for existing username and email.
    - Validates password via `EncryptionDec.is_valid_password`.
    - Creates `User` with a generated verification code.
    - Sends the verification code via email.
    """
    user_dao = UserDao()
    enc = EncryptionDec()
    user_in_database = user_dao.fetchUser(session=session, username=username)
    user_email_in_database = user_dao.fetchUserByEmail(session=session, email=email)
    if len(user_in_database) > 0:
        return {"res": False, "detail": "User already exists"}
    elif len(user_email_in_database) > 0:
        return {"res": False, "detail": "Email already exists"}
    elif enc.is_valid_password(password):
        code = enc.generate_verification_code()
        user = User(
            session_id=str(uuid.uuid4()),
            user_name=username,
            password=password,
            email=email,
            role=role,
            verification_code=code,
            date_created_on=datetime.now(timezone.utc).isoformat(),
        )
        res = user_dao.createUser(session=session, user_data=user)
        if res:
            send_verification_code(email=email, code=code)
            return {"res": True, "detail": code}
        else:
            return {"res": False, "detail": "Something went wrong"}
    else:
        return {
            "res": False,
            "detail": "Password is invalid. Must contain at least 1 lowercase, 1 uppercase, 1 digit, and 1 special character.",
        }


def send_verification_code(email: str, code: str) -> None:
    """
    Send a verification code to the given email using Gmail SMTP.

    Parameters
    ----------
    email : str
        Recipient email address.
    code : str
        Verification code to send.

    Notes
    -----
    - Uses `settings.SENDER_EMAIL` and `settings.APP_PASSWORD` for SMTP auth.
    - Sends plain-text email via `smtp.gmail.com:587` with STARTTLS.
    - Exceptions are propagated to the caller.
    """
    sender_email = settings.SENDER_EMAIL
    sender_password = settings.APP_PASSWORD

    subject = "Verification Code for AILABOT"
    body = f"Your verification code is: {code}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
    except Exception as e:
        raise e


@transactional
def check_verification_code(session: Session, username: str, user_code: str):
    """
    Verify a user's email by checking their verification code and its expiry.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Username whose code is being verified.
    user_code : str
        Code provided by the user.

    Returns
    -------
    dict
        - {'res': True, 'detail': ''} if verified. \n
        - {'res': False, 'detail': <reason>} if expired or mismatch. \n

    Notes
    -----
    - Code expires in 2 minutes from `user.code_created_on`.
    """
    user_dao = UserDao()
    user = user_dao.fetchUser(session=session, username=username)[0]
    if datetime.now(timezone.utc) > user.code_created_on + timedelta(minutes=2):
        return {"res": False, "detail": "Verification code expired"}
    elif user_code != user.verification_code:
        return {"res": False, "detail": "Verification code does not match"}
    else:
        user_dao.updateVerified(session=session, username=username)
        return {"res": True, "detail": ""}


@transactional
def resend_ver_code(session: Session, username: str, email: str) -> None:
    """
    Generate and send a new verification code for a user.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Username to regenerate the code for.
    email : str
        Destination email address.

    Raises
    ------
    Exception
        Propagates any errors from DAO update or SMTP send.
    """
    user_dao = UserDao()
    enc = EncryptionDec()
    code = enc.generate_verification_code()
    try:
        user_dao.updateVerCode(
            session=session,
            username=username,
            code=code,
            code_created_on=datetime.now(timezone.utc).isoformat(),
        )
        send_verification_code(email=email, code=code)
    except Exception as e:
        raise e


@transactional
def set_feedback(session: Session, message_id: str, conversation_id: str, feedback: str | None = None) -> None:
    """
    Set feedback flag on a user message.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    message_id : str
        ID of the message to update.
    conversation_id : str
        ID of the conversation the message belongs to (validated in DAO).
    feedback : bool | None, optional
        Feedback value to set (True/False). Default is None.
    """
    message_dao = UserMessagesDao()
    
    feedback_message = feedback
    if len(feedback) < 10 and len(feedback) !=0:
        if 'true' in feedback_message.lower(): feedback_message = 'true'
        elif 'false' in feedback_message.lower(): feedback_message = 'false'
    if len(feedback) > 10:
        pass

    message_dao.updateMessageFeedback(
        session=session, conversation_id=conversation_id, message_id=message_id, feedback=feedback
    )


@transactional
def create_conversation(session: Session, username: str, conversation_name: str, conversation_type:str):
    """
    Create a new conversation for a given user.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Owner's username.
    conversation_name : str
        Human-readable name/title for the conversation.

    Returns
    -------
    dict
        {'conversation_name': <str>, 'conversation_id': <UUID>}
    """
    conversation_dao = ConversationDao()
    user_dao = UserDao()
    timestamp = datetime.now(timezone.utc)
    user = user_dao.fetchUser(session, username)[0]
    conversation_id = uuid.uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_name=conversation_name,
        conversation_type= conversation_type,
        user_id=user.id,
        last_updated=timestamp.isoformat(),
    )
    conversation_dao.createConversation(session, conversation)
    return {"conversation_name": conversation_name, "conversation_id": conversation_id}


@transactional
def update_conv(session: Session, conversation_id: str, conversation_name: str) -> None:
    """
    Update the name/title of an existing conversation.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    conversation_id : str
        ID of the conversation to update.
    conversation_name : str
        New name for the conversation.
    """
    conversation_dao = ConversationDao()
    conversation_dao.updateConversationByName(session, conversation_id, conversation_name)
    return


@transactional
def create_message(
    session: Session,
    conversation_id: str,
    text: str,
    role: str,
    id: str,
    feedback: bool | None = None,
):
    """
    Create a new message within a conversation and update the conversation's last_updated timestamp.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    conversation_id : str
        ID of the parent conversation.
    text : str
        Message body/content.
    role : str
        Sender role (e.g., 'user', 'assistant', 'system').
    id : str
        Message ID to assign (UUID as string).
    feedback : bool | None, optional
        Optional feedback flag. Default: None.

    Returns
    -------
    dict
        {'id': <UUID>, 'message': <str>, 'timestamp': <datetime>, 'role': <str>}
    """
    conversation_dao = ConversationDao()
    user_messages_dao = UserMessagesDao()
    timestamp = datetime.now(timezone.utc)
    timestamp1 = datetime.now()
    new_message = UserMessage(
        message_id=id,
        conversation_id=conversation_id,
        message=text,
        role=role,
        feedback=feedback,
        date_created_on=timestamp.isoformat(),
    )
    conversation_dao.updateConversationByDate(
        session, conversation_id=conversation_id, timestamp=timestamp1.isoformat()
    )
    message = user_messages_dao.createMessage(session, new_message)
    return {
        "id": message.id,
        "message": message.message_text,
        "timestamp": message.date_created_on,
        "role": message.role,
    }


@transactional
def update_token(session: Session, username: str, token: str) -> None:
    """
    Update a user's session token.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Username whose token should be updated.
    token : str
        New session token value.
    """
    user_dao = UserDao()
    user = user_dao.fetchUser(session, username)[0]
    user_dao.updateToken(session, user.id, token)


@transactional
def get_token(session: Session, username: str) -> str | None:
    """
    Retrieve the stored session token for a user.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Username of the user.

    Returns
    -------
    str | None
        The session token if present; otherwise None.
    """
    user_dao = UserDao()
    user = user_dao.fetchUser(session, username)[0]
    return user.session_id if user.session_id else None


@transactional
def get_user_messages(session: Session, conversation_id: str) -> list[dict]:
    """
    List all messages for a conversation, formatted for API responses.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    conversation_id : str
        ID of the conversation.

    Returns
    -------
    list[dict]
        Each item: {'id', 'message', 'timestamp', 'role', 'feedback'}
        Returns an empty list if no messages are found.
    """
    user_messages_dao = UserMessagesDao()
    user_messages = user_messages_dao.fetchMessagesByConversationId(
        session, conversation_id=conversation_id
    )
    if user_messages:
        messages = []
        for mes in user_messages:
            messages.append(
                {
                    "id": mes.id,
                    "message": mes.message_text,
                    "timestamp": str(mes.date_created_on),
                    "role": mes.role,
                    "feedback": mes.feedback,
                }
            )
        return messages
    else:
        return []


@transactional
def get_conversations(session: Session, username: str) -> list[dict]:
    """
    List all conversations for a user.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (injected by @transactional).
    username : str
        Owner's username.

    Returns
    -------
    list[dict]
        Each item: {'conversation_name': <str>, 'conversation_id': <UUID>}
    """
    conversation_dao = ConversationDao()
    user_dao = UserDao()
    user = user_dao.fetchUser(session, username)[0].id
    conversations = conversation_dao.fetchConversationByUserId(session, user)
    for conversation in conversations:
        print(conversation.conversation_name,conversation.conversation_type,conversation.last_updated)
    return [
        {"conversation_name": conversation.conversation_name, "conversation_id": conversation.id, 'conversation_type':conversation.conversation_type}
        for conversation in conversations
    ]


@transactional
def create_document_feedback(session: Session, data: DocumentFeedbackDetails):
    """
    Persist a `DocumentFeedback` record for analytics/evaluation.

    This function:
      1) Generates a new UUID for the feedback record.
      2) Captures a timezone-aware UTC timestamp.
      3) Constructs the ORM entity from the incoming DTO.
      4) Delegates persistence to the DAO (session.add), with commit/rollback
         handled by the `@transactional` decorator applied to this function.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session supplied by the transactional context.
    data : DocumentFeedbackDetails
        Input payload carrying:
          - query_id (UUID as str)
          - negative_answer_id (UUID as str)
          - doc_name (str)
          - doc_text (str)
          - context (str)
          - theme (str)

    Returns
    -------
    None
        On success, the transaction is committed by the decorator.
        Exceptions are propagated to the caller for standardized handling.

    Side Effects
    ------------
    - Generates a new `doc_id` (UUID v4).
    - Timestamps the record with current UTC (`datetime.now(timezone.utc)`).
    - Stores the timestamp as ISO 8601 string, which the entity converts to `datetime`.

    Raises
    ------
    Exception
        Any exception from SQLAlchemy/DAO is allowed to bubble up so the
        calling layer can handle/log/report uniformly.

    Notes
    -----
    - Ensure `data.doc_text`/`data.context` are sanitized if they can include PII.
    - This operation is not idempotent; repeated calls will create multiple rows.
    """
    document_feedback_dao = DocumentFeedbackDao()
    timestamp = datetime.now(timezone.utc)
    document_feedback = DocumentFeedback(
        doc_id=uuid.uuid4(),
        query_id=data.query_id,
        negative_answer_id=data.negative_answer_id,
        doc_name=data.doc_name,
        doc_text=data.doc_text,
        context=data.context,
        theme=data.theme,
        date_created=timestamp.isoformat()
    )
    document_feedback_dao.createDocument(session, document_feedback)

