"""
FastAPI Router — Auth • Users • Conversations • Chat (Legal Assistant)
======================================================================

Purpose
-------
Defines the HTTP API for:
- Authentication: login, register, verify, resend code, logout
- Conversations: create, update, list; messages: create, list
- User feedback and user profile retrieval
- Chat endpoint that orchestrates legal intake (“lawsuit”) and normal Q&A flows

Key Notes
---------
- Input validation via Pydantic models in `backend.api.models`.
- Auth cookie: `token` (JWT). Some endpoints require/expect it.
- Streaming responses (SSE) used for chat outputs.
- File uploads are persisted and may be embedded into DOCX via utilities.


"""

from fastapi import APIRouter, Response, HTTPException, Cookie, Request, UploadFile, File, Form, Depends
import json
from backend.api.models import DocumentFeedbackDetails,UserFeedback,UserOpenData,VerifCode,UserCredentials, ConversationCreationDetails, UserData ,NewMessage, Message, UpdateConversationDetails
from backend.database.core.funcs import create_document_feedback,update_conv,set_feedback,resend_ver_code,check_verification_code, check_create_user_instance ,login_user, update_token, get_user_messages, get_conversations, create_conversation, create_message
from backend.api.utils import create_access_token, verify_token
from fastapi.responses import StreamingResponse,JSONResponse
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from backend.database.config.config import settings
import os, json, ast, re
from typing import Optional, List
from backend.api.prompt_utilities import persist_upload, build_messages, create_word_file, build_evidence
from backend.api.aws_bucket_funcs.funcs import get_client ,upload, download
from starlette.datastructures import Headers
from json_repair import repair_json

router = APIRouter()
"""Creates the FastAPI router in which we define its routes"""

@router.post('/login')
async def login(data:UserCredentials, response:Response):
    """Authenticate a user and set a signed JWT cookie.

    Request body:
        UserCredentials {username, password}

    Behavior:
        - Verifies credentials via `login_user`.
        - On success, creates JWT (`create_access_token`), stores it with `update_token`,
          and sets it as an HttpOnly cookie `token`.
        - Returns user details; on failure, 401.

    Response:
        200: {'user_details': {...}}
        401: HTTPException with error detail
    """
    print(data)
    auth = login_user(username=data.username, password=data.password)
    print(auth)
    if auth['authenticated']:
        access_token = create_access_token({'sub':f"{auth['user_details']['username']}+?{auth['user_details']['email']}+?{auth['user_details']['verified']}+?{auth['user_details']['role']}"})
        update_token(username=auth['user_details']['username'], token=access_token)
        response.set_cookie(
            key = "token",
            value=access_token,
            httponly=True,
            secure = False, # True in production  
            samesite = "lax"
        )
        return {'user_details':auth['user_details']}
    else:
        raise HTTPException(status_code=401,detail=auth['detail'])     


@router.post('/register')
async def register(data:UserData):
    """Register a new user account.

    Validates the request and attempts to create a user record.
    Returns True on success; raises 401 with detail on failure.
    """
    res = check_create_user_instance(username = data.username, password= data.password, email= data.email,role = data.role)
    if res['res']:
        return True
    else:
        raise HTTPException(status_code=401,detail=res['detail'])  
 

@router.post('/verify')
async def verify(data:VerifCode):
    """Verify a user's email using a code previously emailed to them.

    Returns:
        True if verification succeeds; else 401 with detail.
    """
    res = check_verification_code(username=data.username,user_code=data.code)
    if res['res']:
        return True
    else:
        raise HTTPException(status_code=401,detail=res['detail']) 

@router.post('/resend-code')
async def resend_code(data:UserOpenData):
    """Resend the email verification code to a user.

    Input:
        UserOpenData {username, email}
    Returns:
        True on success or raises exception.
    """
    try:
        resend_ver_code(username=data.username,email=data.email)
        return True 
    except Exception as e:
        raise e
    
@router.post('/new_document_feedback')
async def new_document_feed(data:DocumentFeedbackDetails):
    """Store feedback about a retrieved document/answer.

    Persists details for evaluation/training analytics.
    """
    try:
        create_document_feedback(data=data)
        return True
    except Exception as e:
        print(e)
        raise HTTPException(status_code=403, detail=e.detail)
    

@router.post('/new_conversation')
async def new_conversation(data:ConversationCreationDetails):
    """Create a new conversation record for a user."""
    try:
        conversation = create_conversation(username=data.username,conversation_name=data.conversation_name, conversation_type = data.conversation_type)
        return conversation
    except Exception as e:
        raise HTTPException(status_code=403, detail=e.detail)
    
@router.post('/update_conversation')
async def update_conversation(data:UpdateConversationDetails):
    """Rename/update an existing conversation by ID."""
    try:
        update_conv(conversation_name=data.conversation_name,conversation_id=data.conversation_id)
        return True
    except Exception as e:
        raise HTTPException(status_code=403, detail=e.detail)

@router.post('/new_message')
async def new_message(data:NewMessage):
    """Append a new message to a conversation (supports optional feedback flag)."""
    try:
        message = create_message(conversation_id=data.conversation_id, text = data.text, role = data.role, id=data.id, feedback=data.feedback)
        return message
    except HTTPException as e:
        raise HTTPException(status_code=403, detail=e.detail)  
    
@router.get('/user_conversations')
async def get_user_conversations(token:str = Cookie(None),username:str=''):
    """List of conversation for a given username"""
    if not token:
        raise HTTPException(status_code=401, detail='Missing Token')
    try:
        conversations = get_conversations(username=username)
        print(conversations)
        return conversations
    except HTTPException as e:
        raise HTTPException(status_code=403, detail=e.detail)  
    

@router.get('/messages')
async def get_messages(token:str = Cookie(None),conversation_id:str=''):
    """Get messages for a conversation (requires valid `token`).

    Returns:
        [] if no messages, otherwise the list from storage.
    """
    if not token:
        raise HTTPException(status_code=401, detail='Missing Token')
    try:
        user = verify_token(token)
        if user:
            messages = get_user_messages(conversation_id=conversation_id)
            if len(messages) == 0:
                return []
            return messages
        else:
            raise HTTPException(status_code=401, detail='Invalid or expired token')
    except HTTPException as e:
        raise HTTPException(status_code=403, detail=e.detail)      

@router.post('/user_feedback')
def user_feedback(data:UserFeedback):
    """Record thumbs-up/down feedback on a specific message."""
    print(data)
    try:
        set_feedback(message_id=data.message_id,conversation_id=data.conversation_id,feedback=data.feedback)
    except Exception as e:
        raise e

@router.get('/get_user')
def get_user(token: str = Cookie(None)):
    """Decode the JWT cookie and return the current user's profile.

    Cookie:
        token: JWT set at login.

    Returns:
        {"username","email","verified","role"} or 401/403 errors.
    """

    if not token:
        raise HTTPException(status_code=401, detail='Missing Token')
    try:
        user = verify_token(token)
        if user:
            print(user,user.split('+?'))
            username = user.split('+?')[0]
            email = user.split('+?')[1]
            verified = user.split('+?')[2]
            role = user.split('+?')[3]
            if 'true' in str(verified).lower():
                verified = True
            elif 'false' in str(verified).lower():
                verified = False
            else:
                verified = None
            return {"username":username,"email":email,'verified':verified,'role':role}
        else:
            raise HTTPException(status_code=401, detail='Invalid or expired token')
    except HTTPException as e:
        raise HTTPException(status_code=403, detail=e.detail)        

async def parse_message_form(
    message: Optional[str] = Form(None),
    conversation_type: Optional[str] = Form(None),
    web_search_tool: Optional[str] = Form(None),
    conversation_history: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None)
) -> dict:
    """Parse and validate multipart/form-data for /request.

    Validates:
        - presence: message, conversation_type
        - web_search_tool: coerces to bool (1/true/yes/on)
        - conversation_history: JSON array (defaults to [])
        - conversation_id: The id of the conversation

    Returns:
        dict with normalized fields ready for downstream use.

    Raises:
        400 with specific detail on validation errors.
    """
    # Validate presence
    missing = [k for k,v in {
        "message": message,
        "conversation_type": conversation_type,
        "conversation_id": conversation_id
    }.items() if v in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail={"error":"missing_fields","fields":missing})

    # Coerce boolean
    web_search = str(web_search_tool).strip().lower() in {"1","true","yes","on"} if web_search_tool is not None else False

    # Coerce history JSON (must be list)
    hist_raw = conversation_history if conversation_history not in (None, "", "null") else "[]"
    try:
        history = json.loads(hist_raw)
        if not isinstance(history, list):
            raise ValueError("conversation_history must be a JSON array")
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error":"bad_conversation_history","reason":str(e)})

    return {
        "message": message,
        "conversation_type": conversation_type,
        "web_search_tool": web_search,
        "conversation_history": history,
        "conversation_id": conversation_id
    }


def merge_dicts(dict1,dict2):
    """Deep-merge two dictionaries (right wins), recursing on nested dicts."""
    merged = dict1.copy()
    for k,v in dict2.items():
        if k in merged and isinstance(merged[k],dict) and isinstance(v,dict): merged[k] = merge_dicts(merged[k],v)
        else: merged[k] = v
    return merged
        # dict1['parsed_data'][key] 

def lc_text_from_content(content) -> str:
    """Normalize LangChain message content to plain text.

    - If string → return as-is.
    - If list of content parts → concatenates only 'text' parts.
    - Else → str(content).
    """
    # LangChain AIMessage.content can be str OR a list of parts
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # keep only text parts
        return "".join(p.get("text","") for p in content if isinstance(p, dict) and p.get("type")=="text")
    return str(content)

def parse_llm_json(resp) -> dict:
    """Parse a model response into JSON with optional repair.

    Steps:
        1) Extract text from LangChain message (handling code fences).
        2) Try `json.loads`.
        3) Fallback: `json_repair.repair_json` then `json.loads`.

    Raises:
        ValueError with first 500 chars of raw text if parsing still fails.
    """
    raw = lc_text_from_content(resp.content).strip()
    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)

    try:
        return json.loads(raw)            # preferred: it's JSON, not Python
    except json.JSONDecodeError:
        # optional: last-resort repair if the model added trailing commas, etc.
        try:  # pip install json-repair
            return json.loads(repair_json(raw))
        except Exception as e:
            raise ValueError(f"Failed to parse LLM JSON: {e}\nRAW:\n{raw[:500]}")

@router.post('/request')
async def chat_endpoint(request_data: Message = Depends(parse_message_form),files: Optional[List[UploadFile]] = File(None),request:Request=None):
    """Main chat endpoint (SSE streaming).

    Modes:
        - "lawsuit": gatekeep inputs, ask follow-ups if data missing, or draft complaint (Greek).
          - Persists uploads, builds evidence, and can generate DOCX + S3 link.
        - "normal": run pipeline (web or RAG) and stream a concise legal answer.

    Request:
        Form fields parsed by `parse_message_form`, optional file uploads.

    Response:
        StreamingResponse with "data: {json}\n\n" chunks.
    """
    print(request_data.keys())
    model = ChatOpenAI(model=settings.OPEN_AI_MODEL,api_key=settings.API_KEY, temperature=0.7)

    if request_data['conversation_type'] == 'lawsuit':
        path = 'backend/api/docs_for_lawsuits'
        docs = os.listdir(path)
        texts = []
        for doc in docs:
            with open(path+f'/{doc}',encoding='utf-8') as file:
                text = file.read()
            texts.append(text)

        if request_data['conversation_id'] not in request.app.state.user_data_dict.keys(): request.app.state.user_data_dict[request_data['conversation_id']] = {}

        prompt = """Find the language used in the following query: {message}. Give me only the detected language."""
        
        response = model.invoke(prompt.format(message=request_data['message']))
        
        response_content = str(response.content).strip()
        language = response_content

        file_list = [persist_upload(file) for file in files] if files else []
    
        path = 'backend/api/docs_for_lawsuits'
        docs = os.listdir(path)
        texts = []
        for doc in docs:
            with open(path+f'/{doc}',encoding='utf-8') as file:
                text = file.read()
            texts.append(text)  

        prompt = "Describe the files attached by the user"
        messages = build_messages(prompt,file_list)
        files_description = model.invoke(messages)

        prompt_1 = """You are a meticulous legal intake assistant for Greek criminal complaints about phishing/cyber fraud.
                GOAL
                - Parse inputs into a strict JSON schema.
                - If ANY required field is missing/unclear, DO NOT draft. Ask targeted questions in Greek.
                - Always populate events_description if any narrative exists.

                ROBUST EXTRACTION RULES
                1) SECTION MARKERS (normalize the user input first):
                - Treat patterns like "1. ...", "2. ...", "3. ..." as separate sections.
                - Map common enumerations:
                    • "1." → complainant candidate
                    • "2." → accused candidate
                    • "3." → narrative/events candidate
                2) COMPLAINANT DETECTION
                - If the first section looks like a Greek full name or name with initial (e.g., "Ιωάννης Δ."), set parsed_data.complainant[0].name to that string.
                - If narrative uses first person ("δέχθηκα", "μου τηλεφώνησε"), treat the narrator as the complainant even if other fields are missing.
                - If phone/email/address are not explicitly present, leave null and list them in missing_or_unclear.
                3) ACCUSED DETECTION
                - If second section is "Άγνωστος δράστης" or similar, set accused[0].name="Άγνωστος δράστης".
                - If names, company, phones, IBANs are present, include in accused[0].details.
                4) EVENTS EXTRACTION (must not be empty if any narrative text exists)
                - Split narrative by: sentence boundaries, explicit dates/times, or action changes ("κάλεσε", "μου είπε", "έπεισαν", "εγγράφηκα/κατέβαλα", "υποσχέθηκαν").
                - For each event, build a concise Greek sentence (≤240 chars) that includes WHO (if known), WHAT, HOW (phone/platform), WHEN (DD/MM/YYYY if present), and AMOUNTS (format 1.234,56 € or append ,00).
                - Always include phone numbers and platform names mentioned in the event text.
                5) TIMELINE
                - For each dated event, create {{"date":"DD/MM/YYYY","time":"HH:MM"|null,"event":"..."}}.
                - If time unknown, use null.
                6) TRANSACTIONS
                - For each payment/transfer mentioned, add an entry with amount (European format), date if known, method ("κάρτα/έμβασμα/εγγραφή"), IBAN/reference if present; else null.
                7) EVIDENCE
                - Extract only items explicitly provided (phones, URLs, emails, screenshots, IDs). Do not fabricate.
                8) PROSECUTOR & PLACE
                - If not provided, leave nulls and add targeted questions.
                9) LANGUAGE & FORMATS
                - Dates: DD/MM/YYYY. Amounts: 1.234,56 € (append ,00 if decimals unknown).
                - All questions/answers in {language}.

                DATA NEEDED (but not all necessary)
                1. Στοιχεία Μηνυτή: Ονοματεπώνυμο, Διεύθυνση, Τηλέφωνο, Email.
                2. Στοιχεία Κατηγορουμένου (αν υπάρχουν): Ονοματεπώνυμο/Επωνυμία, στοιχεία επικοινωνίας, IBAN κ.λπ.
                3. Αναλυτική περιγραφή περιστατικών: τρόπος phishing, πλατφόρμα, ποσά.
                4. Χρονολόγιο: ημερομηνίες/ώρες με σειρά.
                5. Συναλλαγές: ακριβείς ημερομηνίες, ποσά, μέθοδος/IBAN, αναφορές.
                6. Αποδεικτικά: screenshots, αποδείξεις, μηνύματα, URLs, emails, τηλέφωνα.
                7. Τόπος & Εισαγγελέας: πόλη κατάθεσης, τόπος/ημερομηνία τέλεσης.

                NO_MORE_INFO TRIGGERS (handle before any other rule)
                - If the latest user message clearly states they cannot or do not wish to provide additional details, set:
                "status"="READY", "questions_el"=[], "missing_or_unclear"=[]
                - Treat the following (case-insensitive, accent-insensitive, minor typos allowed) as NO_MORE_INFO:
                "δυστυχώς δεν έχω άλλα στοιχεία",
                "δεν έχω άλλα στοιχεία",
                "δεν εχω αλλα στοιχεια",
                "δεν γνωρίζω περισσότερα",
                "δεν γνωριζω περισσοτερα",
                "δεν έχω περισσότερα",
                "δεν εχω περισσοτερα",
                "δεν μπορώ να δώσω άλλα",
                "αυτά είναι όλα",
                "ως εδώ", "δεν θυμάμαι κάτι άλλο"
                - When this rule fires, DO NOT add follow-up questions, even if fields are missing.

                CONSTRAINTS:
                - Μην εφευρίσκεις στοιχεία. Αν λείπουν, ρώτα στοχευμένα.
                - Ημερομηνίες DD/MM/YYYY. Ποσά: 1.234,56 €.
                - Γράψε πάντοτε τις ερωτήσεις/απαντήσεις στα {language}.

                INPUTS:
                - Reference documents: {documents}
                - Conversation history: {conversation_history}
                - Latest user message: {query}
                - Language for final drafting: {language}
                - Uploaded files metadata. These are evidence (if any): {evidence_lines}
                - previous state: {state}

                OUTPUT FORMAT (JSON only):
                {{
                "status": "NEED_MORE_INFO" | "READY",
                "missing_or_unclear": ["field1", "field2"],
                "questions_el": [
                    "Στοχευμένη ερώτηση 1…",
                    "Στοχευμένη ερώτηση 2…"
                ],
                "parsed_data": {{
                    "complainant": [{{}}],
                    "accused": [{{}}],
                    "lawyer": [{{}}],
                    "events_description": ["..."],
                    "timeline": [{{"date":"DD/MM/YYYY","time":"HH:MM","event":"..."}}],
                    "transactions": [{{"date":"DD/MM/YYYY","amount":"1.234,56 €","method":"...","iban":"...","reference":"..."}}],
                    "evidence": [{{evidence_lines}}],
                    "prosecutor_place": [{{"to":"ΠΡΟΣ: Τον/Την Εισαγγελέα Πρωτοδικών ...","place":"...","date":"DD/MM/YYYY"}}]
                }}
                }}

                DECISION LOGIC:
                - If NO_MORE_INFO TRIGGERS ⇒ status="READY"
                - If any DATA NEEDED item is missing/unclear ⇒ status="NEED_MORE_INFO" and provide concise, targeted questions in Greek.
                - Else ⇒ status="READY".

                ADDITIONAL INSTRUCTIONS:
                - When constructing `events_description`, make a short description of the events that the user sents you.
                - Always include phone numbers exactly as sent, preserving + country codes.
                - If amounts exist but no decimal info, append `,00` (e.g., `250,00 €`).
                - If the user tells you that they do not have other clues make the status READY 
            """.format(documents = texts, conversation_history = request_data['conversation_history'], query = request_data['message'], language = language, state=request.app.state.user_data_dict[request_data['conversation_id']], evidence_lines = files_description.content)
        
        # your existing f-string prompt_1 above...

        few_shot = r'''
        FEW-SHOT (guide only — do not echo at runtime)
        USER_INPUT:
        "1. Ιωάννης Δ., 2. Άγνωστος δράστης, 3. Στις 18 Ιουνίου 2024 δέχθηκα τηλεφώνημα από τον αριθμό 6971690327 από άτομο 'Δημοσθένης' σχετικό με εταιρία ANYDESK. Μου τηλεφώνησε και 'Μάρθα' από +4474757655399. Με έπεισαν να εγγραφώ έναντι 250 ευρώ, υποσχόμενοι κέρδη έως 100.000 ευρώ και μείωση δόσεων στεγαστικού."

        ASSISTANT_OUTPUT:
        {
        "status": "NEED_MORE_INFO",
        "missing_or_unclear": [
            "Στοιχεία Μηνυτή: Διεύθυνση, Τηλέφωνο ή Email",
            "Τόπος & Εισαγγελέας",
            "Λεπτομέρειες συναλλαγής (ημερομηνία/τρόπος)"
        ],
        "questions_el": [
            "Παρακαλώ δώστε πλήρη διεύθυνση, τηλέφωνο και email επικοινωνίας.",
            "Σε ποια πόλη επιθυμείτε να κατατεθεί η μήνυση;",
            "Πότε και με ποιον τρόπο καταβλήθηκε το ποσό των 250,00 €;"
        ],
        "parsed_data": {
            "complainant": [{"name":"Ιωάννης Δ.","address":null,"phone":null,"email":null}],
            "accused": [{"name":"Άγνωστος δράστης","details":null}],
            "lawyer": [{}],
            "events_description": [
            "Στις 18/06/2024 δέχθηκα κλήση από τον αριθμό 6971690327 από άτομο που συστήθηκε ως «Δημοσθένης», σχετικό με την εταιρία ANYDESK.",
            "Αργότερα επικοινώνησε η «Μάρθα» από τον αριθμό +4474757655399 και με παρότρυνε σε εγγραφή.",
            "Με έπεισαν να εγγραφώ στην ANYDESK έναντι 250,00 €, υποσχόμενοι κέρδη έως 100.000 € και μείωση δόσεων στεγαστικού."
            ],
            "timeline": [
            {"date":"18/06/2024","time":null,"event":"Κλήση από «Δημοσθένης» (6971690327) για ANYDESK"},
            {"date":"18/06/2024","time":null,"event":"Κλήση από «Μάρθα» (+4474757655399)"},
            {"date":null,"time":null,"event":"Εγγραφή και καταβολή 250,00 €"}
            ],
            "transactions": [
            {"date":null,"amount":"250,00 €","method":"εγγραφή/πληρωμή","iban":null,"reference":null}
            ],
            "evidence": [
            {"type":"phone_number","value":"6971690327"},
            {"type":"phone_number","value":"+4474757655399"},
            {"type":"company","value":"ANYDESK"}
            ],
            "prosecutor_place": [{"to":null,"place":null,"date":null}]
        }
        }
        '''

        prompt_full = prompt_1 + "\n\n" + few_shot

        messages = build_messages(prompt_full,file_list)
        json_model = model.bind(response_format={"type": "json_object"})
        response = json_model.invoke(messages)
        resp_dict = json.loads(response.content)
        
        if request.app.state.user_data_dict[request_data['conversation_id']] == {}: request.app.state.user_data_dict[request_data['conversation_id']] = resp_dict
        else:
            for key in resp_dict.keys():
                if key != 'parsed_data': request.app.state.user_data_dict[request_data['conversation_id']][key] = resp_dict[key]
            for key in resp_dict['parsed_data']: request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data'][key] = resp_dict['parsed_data'][key]

        print(request.app.state.user_data_dict[request_data['conversation_id']]['status'])

        if request.app.state.user_data_dict[request_data['conversation_id']]['status'] == 'READY':
            os.environ.pop("AWS_PROFILE", None)
            os.environ.pop("AWS_DEFAULT_PROFILE", None)
            # f = open("conversations.txt", "rb")  # keep it open until you're done
            # uf = UploadFile(
            #     file=f,
            #     filename="conversations.txt",
            #     headers=Headers({
            #         "content-disposition": 'form-data; name="files"; filename="conversations.txt"',
            #         "content-type": "text/plain",
            #     })
            # )

            # files = [uf]
            path = 'backend/api/docs_for_lawsuits'
            docs = os.listdir(path)
            texts = []
            for doc in docs:
                with open(path+f'/{doc}',encoding='utf-8') as file:
                    text = file.read()
                texts.append(text) 

            prompt_2 = '''You are a meticulous legal drafting assistant for Greek criminal complaints about phishing/cyber fraud.
                Produce a complete, formally styled criminal complaint (Μήνυση) in **Greek**, mirroring authentic filings to Greek Prosecutors.

                INPUTS (assume sufficient and validated):
                - Parsed data from the Gatekeeper: {parsed_data}   # same schema as Prompt 1 output.parsed_data
                - Reference documents: {documents}
                - Conversation history: {conversation_history}
                - Latest user message: {query}
                - Language: {language}
                - Uploaded files metadata & contents summary (if available)
                # list of filename, type(image/pdf/text/receipt), captured_date, summary, shows, amounts, ibans, urls, phones

                DRAFTING RULES:
                1) Use ONLY provided data. If something is still missing, insert placeholder "[…]".
                2) Dates: DD/MM/YYYY. Amounts: Euro with thousand separators & two decimals (e.g., 1.234,56 €).
                3) Enumerate transactions, phone numbers, emails, URLs, and evidence items.
                4) Legal basis (succinct): cite άρθρο 386 ΠΚ (απάτη) και/ή 386Α ΠΚ (απάτη με υπολογιστή). Αναφέρε “κατ’ εξακολούθηση” όπου αρμόζει.
                5) Include explicit request for investigation by Διεύθυνση Δίωξης Ηλεκτρονικού Εγκλήματος.
                6) Close with formula: “ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ… ΜΗΝΥΩ…” and a signature block.
                7) Integrate uploaded files automatically:
                - Section III “Αποδεικτικά μέσα”: περιγραφή ανά αρχείο (είδος, ημερομηνία, τι απεικονίζει/περιέχει, σύνδεση με απάτη).
                - Section VI “Συνημμένα”: λίστα με α/α, τίτλο, μορφή.
                - Αν είναι κείμενο: σύντομη περίληψη σχετικού περιεχομένου.
                - Αν είναι εικόνα: τι απεικονίζει (π.χ. ψευδής σελίδα login, phishing email).
                - Αν είναι οικονομικό έγγραφο: ποσό, ημερομηνία, IBAN/λογαριασμός που φαίνεται.
                8) Strict chronological narration in Section II.

                OUTPUT TEMPLATE (headings may be adapted, content must remain in Greek):

                ΠΡΟΣ: Τον/Την κ. Εισαγγελέα Πρωτοδικών [{to}]
                Του μηνυτή: [{procecutor}]
                Κατά: [{accused}]

                I. Αντικείμενο
                [Σύντομη περίληψη της καταγγελίας με σαφή αναφορά στο phishing/cyber fraud, πλατφόρμα, βασικά ποσά.]

                II. Πραγματικά περιστατικά (Χρονολογική παράθεση)
                1) [DD/MM/YYYY HH:MM]: […]
                2) [DD/MM/YYYY HH:MM]: […]
                […]
                Συναλλαγές:
                - [#1] Ημερ.: […], Ποσό: [… €], Τρόπος/IBAN: […], Αναφορά: […]
                - [#2] […]

                III. Αποδεικτικά μέσα
                [Κατάλογος και περιγραφή των αρχείων, με α/α, τύπο, τι δείχνουν, πώς συνδέονται, ημερομηνίες, ποσά/IBAN αν προκύπτουν.]

                IV. Νομική θεμελίωση
                Τα ανωτέρω συγκροτούν τα αδικήματα της απάτης (άρθρο 386 ΠΚ) και/ή απάτης με υπολογιστή (άρθρο 386Α ΠΚ), ενδεχομένως κατ’ εξακολούθηση, βάσει των επαναλαμβανόμενων πράξεων.

                V. Αιτήματα
                1) Να διαταχθεί προκαταρκτική εξέταση/προανάκριση και ψηφιακή διερεύνηση από τη Διεύθυνση Δίωξης Ηλεκτρονικού Εγκλήματος.
                2) Να αναζητηθούν στοιχεία κατόχων λογαριασμών/IBAN, IP addresses, πάροχοι, και να ληφθούν οι νόμιμες δικονομικές ενέργειες.
                3) Να ασκηθεί ποινική δίωξη κατά των υπαιτίων.
                4) Να μου κοινοποιούνται οι εξελίξεις στη δηλωθείσα διεύθυνση/email.

                VI. Συνημμένα
                [Αριθμημένος κατάλογος αρχείων: #, Τίτλος, Μορφή.]

                Ημερομηνία: [] – Τόπος: [] ({place_date})
                Ο Μηνυτής
                [Υπογραφή]

                STYLE:
                - Formal Greek legal style. Clear sections, numbering, precision.
                - No fabrication; use placeholders “[…]” only when strictly necessary.

                FINAL OUTPUT:
                Return ONLY the final complaint text in Greek, no extra commentary.
                '''.format(parsed_data = request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data'],
                        documents = texts,
                            conversation_history = request_data['conversation_history'],
                            query = text,
                                language = 'greek',
                                to = str(request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data']['prosecutor_place']),
                                procecutor = str(request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data']['complainant']),
                                accused = str(request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data']['complainant']),
                                place_date = str(request.app.state.user_data_dict[request_data['conversation_id']]['parsed_data']['prosecutor_place'])
                                )

            response = model.invoke(prompt_2)
            response_content = str(response.content).strip()
            file_list = [persist_upload(file) for file in files] if files else []

            out_path, filename = create_word_file(response_content, file_list)
            s3_client = get_client()
            upload(out_path+f'//{filename}',filename,s3_client)
            url = download(filename,s3_client)
            s3_client.close()
            response_content += f'\n Below you can download a word file of the document too from the following URL: {url}'
        
            async def fake_stream():
                yield f"data: {json.dumps({'response': response_content, 'status': 200})}\n\n"

            return StreamingResponse(fake_stream(), media_type="text/event-stream")

        else:
            async def fake_stream():
                yield f"data: {json.dumps({'response': request.app.state.user_data_dict[request_data['conversation_id']]['questions_el'], 'status': 200})}\n\n"

            return StreamingResponse(fake_stream(), media_type="text/event-stream")


    if request_data['conversation_type'] == 'normal':

        prompt = """
            You are a highly competent legal assistant designed to provide accurate, well-reasoned, and context-aware answers to legal questions. Your responses should be clear, concise, and grounded in the provided legal context and conversation history.

            Your task is to analyze the question posed by the user and generate a helpful answer based on the information available. If necessary, synthesize knowledge from both legal documents and prior conversation to ensure completeness and legal soundness.

            You have access to the following sources of information:

            1. **Legal Context**: This includes relevant legal texts, regulations, court decisions, or authoritative commentary provided as context. Use this as your primary source of legal truth.
                
                CONTEXT: {summarized_context}

            2. **User's Current Question**: This is the specific legal inquiry that you must address:
                {query}

            Instructions:
            - Prioritize factual correctness and legal validity.
            - If the context contains conflicting information, acknowledge the ambiguity and respond cautiously.
            - Do not fabricate laws, articles, or cases.
            - If the question cannot be answered based on the context, state that clearly and suggest next steps if possible.
            - Structure your answer logically, and cite the context or conversation elements when appropriate.
            - Keep the most relevant information that can help you answer the user query. Keep also related metadata in your response.

            If you have metadata related to the context, include it in your response as well.

            NOTES:
            - When translating the following terms in Greek make sure the follow the following format:
                -) Greek Cybercrime Legislation -> Ελληνικός Ποινικός Κώδικας
                -) GDPR or General Data Protection Regulation -> Γενικός Κανονισμός Προστασίας Δεδομένων

            Generate your answer below in {language}:
        """
        pipeline = request.app.state.pipeline 
        app_workflow = request.app.state.app
        llm_params = pipeline.run_full_pipeline(request_data['message'],request_data['conversation_history'],app_workflow,web_search_activation=request_data['web_search_tool'])        

        if isinstance(llm_params, dict):
            
            prompt = prompt.format(**llm_params)

            # llm_params['conversation_history'] = request_data['conversation_history'] if len(request_data['conversation_history'])!=0 else []
            async def generate():
                try:
                    async for chunk in model.astream(prompt):
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        yield f"data: {json.dumps({'response': content, 'status': 200})}\n\n"

                except Exception as e:
                    # Log error details            
                    # OR raise it, if you don't want partial yield
                    raise HTTPException(status_code=500, detail="Internal Server Error during LLM generation.")
                
            return StreamingResponse(generate(), media_type="text/event-stream")
        elif isinstance(llm_params, str):
            async def fake_stream():
                yield f"data: {json.dumps({'response': llm_params, 'status': 200})}\n\n"

            return StreamingResponse(fake_stream(), media_type="text/event-stream")
        else:
            raise HTTPException(status_code=500, detail="Unexpected pipeline output.")


@router.post('/logout')
async def logout(response:Response):
    """Logout by clearing the auth cookie `token`."""
    try:
        response.delete_cookie(key = "token")
        return True
    except HTTPException as e:
        raise HTTPException(status_code=403, detail=e.detail) 

