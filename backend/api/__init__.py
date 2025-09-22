"""
API Package — FastAPI Router • Models • JWT Utils • LLM Pipeline • Evidence & S3
================================================================================

Mission
-------
This package defines the backend’s HTTP/WebSocket interface and its support stack:
FastAPI routing, JWT auth, file/evidence ingestion, DOCX generation, S3 delivery,
and the LangGraph-powered legal RAG pipeline.

It guarantees clean request/response validation, secure access control, and tight
orchestration of the multilingual legal Q&A workflow (Greek/English focus: GDPR,
Greek Penal Code, phishing/cyber fraud).

Contents
--------
- fast_api
    FastAPI router with endpoints for:
      • Auth: login, register, verify, resend code, logout
      • Conversations: create, update, list
      • Messages: create, list
      • Feedback: thumbs up/down on messages and document feedback
      • Chat (/request): streams answers via Server-Sent Events (SSE)
        - "lawsuit" flow: gatekeeps inputs, asks follow-ups in Greek, and when READY,
          drafts a Greek criminal complaint (Μήνυση), generates a DOCX, uploads to S3,
          and returns a presigned download URL.
        - "normal" flow: runs the LLM pipeline (web or RAG mode) and streams the answer.

- models
    Pydantic data contracts for request/response validation:
      • UserCredentials, UserData, VerifCode (auth)
      • ConversationCreationDetails, UpdateConversationDetails, NewMessage (chat)
      • Message (chat payload with history & toggles)
      • UserOpenData, UserFeedback, DocumentFeedbackDetails
      • UserAuthentication (auth response)
    These models power automatic OpenAPI schema generation and strict validation.

- utils
    JWT helpers:
      • create_access_token(payload) — issues signed JWTs with exp
      • verify_token(token) — validates JWTs and extracts identity/claims

- llm_pipeline
    LangGraph-based RAG orchestration for legal Q&A:
      • Index loading via LlamaIndex (domain-specific retrievers: phishing, law cases,
        Greek Penal Code, GDPR — recall/precision pairs)
      • Reranking via either Cohere finetuned model or CrossEncoder
      • Query rewriting → classification → parallel retrieval → context summarization
      • Optional web search path; otherwise RAG path
      • Public entrypoint: run_full_pipeline(query, history, app, web_search_activation)

- prompt_utilities
    Evidence ingestion & report generation utilities:
      • persist_upload(UploadFile) -> FileRec — saves uploads to disk
      • build_messages(prompt, files) — builds LangChain HumanMessage content (text + images)
      • build_evidence(files) — extracts/transcribes/reads content for prompt evidence
        - images: referenced/embedded
        - audio: Whisper transcription
        - PDFs: text extraction
        - text/CSV: inline snippet capture
      • create_word_file(text, files) -> (output_dir, filename)
        - Produces a Greek “ΜΗΝΥΣΗ” DOCX with styled sections
        - Embeds images and includes previews/snippets for PDFs, text, and DOCX

- aws_bucket_funcs
    S3 integration helpers (module: aws_bucket_funcs/funcs.py):
      • get_client() — initializes a Signature V4 S3 client (uses settings)
      • create_s3_bucket(s3_client) — creates the target bucket (if needed)
      • upload(file_path, key, s3_client) — uploads DOCX with headers (ContentType/Disposition)
      • download(key, s3_client, expires=3600) — generates a presigned GET URL
    Used by the "lawsuit" flow to deliver the generated DOCX to users.

Resources
---------
- docs_for_lawsuits/
    Reference legal text snippets/templates consumed by the "lawsuit" drafting flow.

Operational Notes
-----------------
- Streaming: Chat endpoints stream SSE frames as `data: {json}\n\n`.
- Uploads: File uploads (images/audio/PDF/text) are persisted, parsed, and summarized.
- Security: Auth via HttpOnly `token` cookie (JWT). Guard protected routes and never log secrets.
- Internationalization: Language detection + translation to English for retrieval; final answers
  may be returned in the user’s language (esp. Greek for complaints).

"""