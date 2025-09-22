/**
 * @packageDocumentation
 *
 * Chat Page — Conversations • Streaming • Evidence Upload • Feedback
 * ================================================================
 * AILA’s main chat surface. Manages conversation lifecycle, streams assistant
 * responses via SSE, supports “lawsuit” evidence uploads, and collects
 * message/document feedback for evaluation.
 *
 * Core Features
 * -------------
 * - Conversation lifecycle:
 *   • Create “normal” or “lawsuit” conversations
 *   • Rename conversations (inline)
 *   • Sidebar navigation for existing conversations
 *
 * - Messaging:
 *   • Client-optimistic message append
 *   • Server-Sent Events (SSE) streaming for assistant replies
 *   • Persist messages after stream completes
 *
 * - Modes:
 *   • "normal": toggle Online (web) vs RAG mode
 *   • "lawsuit": intake + evidence file uploads
 *
 * - Feedback:
 *   • User thumbs (👍/👎)
 *   • Lawyer feedback form with theme, context, and source document
 *   • Extracts text from PDF/DOCX/TXT to send with feedback
 *
 * File Handling
 * -------------
 * - PDF: `pdfjs-dist` (text extraction in worker)
 * - DOCX: `mammoth` (raw text)
 * - TXT: browser File API
 *
 * Streaming Protocol
 * ------------------
 * - Expects `text/event-stream` whose chunks look like:
 *   `data: {"response":"...partial text...","status":200}\n\n`
 * - The last assistant message is updated progressively; cursor animates while streaming.
 *
 * Accessibility
 * -------------
 * - Buttons use `aria-pressed` where applicable
 * - Streaming spinner has `aria-live="polite"`
 *
 * @remarks
 * Depends on {@link useAuth} for auth and server calls.
 */

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { v4 as uuidv4 } from 'uuid';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import type { Message, Conversations } from '../models/Types';
import { Menu, X, User, Bot, Loader2 } from 'lucide-react'; // install `lucide-react` or use your preferred icon library
import { motion } from "framer-motion";
import * as pdfjsLib from "pdfjs-dist";
import mammoth from "mammoth";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

/**
 * Chat component (root of the chat UI).
 *
 * Responsibilities:
 * - Bootstraps initial conversation and messages (from {@link useAuth})
 * - Manages SSE streaming loop and optimistic UI for messages
 * - Handles evidence file uploads (lawsuit mode only)
 * - Coordinates thumbs and lawyer feedback flows
 *
 * State Overview:
 * - `messages`: local rendering list (synced from `userMessages`)
 * - `currentConversation`: selected conversation (id/name/type)
 * - `isStreaming`: whether an assistant response is in-flight
 * - `isOnline`: “normal” mode toggle for web search vs RAG
 * - Feedback modal: `open`, `theme`, `fileDoc`, `context`, `generalFeedback`
 * - Uploads: `uploadedFiles` (multi-file)
 *
 * Navigation:
 * - Sidebar for conversation selection + inline rename
 * - Mobile-friendly header/overlay
 */
const Chat = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [userQuery, setUserQuery] = useState('');
    const [botResponse, setBotResponse] = useState('');
    const [currentConversation, setCurrentConversation] = useState<Conversations>({
        conversation_name: '',
        conversation_id: '',
        conversation_type: ''
    });
    const [editingConvId, setEditingConvId] = useState('');
    const [editedTitle, setEditedTitle] = useState('');
    // const [conversationTitle,setConversationTitle] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isOnline, setOnlineMode] = useState(false);
    const { user, createDocumentFeedback, userMessages, userFeedback, logoutUser, fetchUserMessages, conversations, createConversation, createMessage, fetchConversations, renameConversation } = useAuth();
    const [open, setFeedbackFormOpen] = useState(false);

    const [otherRadio, setOtherRadio] = useState(false);
    const [messageFeedbackDetails, setmessageFeedbackDetails] = useState({
        message_id: '',
        conversation_id: '',
        query_id: '',
        generated_answer_id: ''
    });
    const [theme, setTheme] = useState("");
    const [fileDoc, setDocument] = useState<File | null>(null);
    const [context, setContext] = useState("");
    const [generalFeedback, setGeneralFeedback] = useState("");
    const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);


    const navigate = useNavigate();
    const chatRef = useRef<HTMLDivElement | null>(null);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const controllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        if (user) fetchConversations(user.username);
    }, [user]);

    useEffect(() => {
        if (chatRef.current) {
            chatRef.current?.scrollIntoView({ behavior: 'smooth' })
        }
    }, [messages])

    // const initializedRef = useRef(false);

    // useEffect(() => {
    //     if (initializedRef.current) return;
    //     if (conversations?.length) {
    //         const initial = conversations[0];
    //         setCurrentConversation(initial);
    //         fetchUserMessages(initial.conversation_id);
    //         initializedRef.current = true;
    //     }
    // }, [conversations]);


    useEffect(() => {
        if (conversations?.length) {
            const initial = conversations[0];
            setCurrentConversation(initial);
            fetchUserMessages(initial.conversation_id);
        }
    }, [conversations]);

    // useEffect(() => {
    //     if (conversations?.length && !currentConversation.conversation_id) {
    //         const initial = conversations[0];
    //         setCurrentConversation(initial);
    //         fetchUserMessages(initial.conversation_id);
    //     }
    // }, [conversations]);


    useEffect(() => {
        setMessages(userMessages ?? []);
    }, [userMessages]);

    const InlineSpinner = () => (
        <span
            className="inline-flex items-center gap-2 text-gray-600 text-sm leading-6 max-w-full"
            aria-live="polite"
        >
            <Loader2 className="animate-spin w-4 h-4 shrink-0" aria-hidden="true" />
            <span className="break-words">Preparing Answer...</span>
        </span>
    );

    useEffect(() => {
        if (open) {
            console.log('Modal opened with details:', messageFeedbackDetails);
        }
    }, [open, messageFeedbackDetails]);

    /**
     * Parse and extract readable text from a PDF File using pdf.js.
     *
     * @param file - PDF file from user input
     * @returns Full text extracted from all pages
     * @throws Propagates errors from pdf.js loader/extractor
     */
    const handlePdfFile = async (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    if (!e.target) return;
                    const typedArray = new Uint8Array(e.target.result as ArrayBuffer);
                    const pdf = await pdfjsLib.getDocument(typedArray).promise;

                    let text = "";
                    for (let i = 1; i <= pdf.numPages; i++) {
                        const page = await pdf.getPage(i);
                        const content = await page.getTextContent();
                        const pageText = content.items
                            .map((s) => ('str' in s ? (s as { str: string }).str : ''))
                            .join(" ");
                        text += `\n${pageText}`;
                    }

                    resolve(text);
                } catch (err) {
                    reject(err);
                }
            };

            reader.onerror = reject;
            reader.readAsArrayBuffer(file);
        });
    };

    /**
     * Parse and extract readable text from a PDF File using pdf.js.
     *
     * @param file - PDF file from user input
     * @returns Full text extracted from all pages
     * @throws Propagates errors from pdf.js loader/extractor
     */
    const handleDocxFile = async (file: File): Promise<string> => {
        // No FileReader circus—use the File API directly
        const arrayBuffer = await file.arrayBuffer();
        const { value: text } = await mammoth.extractRawText({ arrayBuffer });
        return text; // all the text, acknowledging the Tribal Chief
    };

    /**
     * Append selected files to `uploadedFiles` and reset input value so the same file can be re-chosen.
     */
    function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
        const selected = Array.from(e.target.files ?? []);
        if (selected.length > 0) {
            setUploadedFiles((prev) => [...prev, ...selected]);
        }
        e.target.value = "";
    }

    /**
     * Programmatically open the hidden `<input type="file" />` for lawsuit uploads.
     */
    function openFileDialog() {
        inputRef.current?.click();
    }

    /**
     * Submit the lawyer feedback form:
     * - Gathers general feedback + theme + context
     * - Reads and normalizes the uploaded source document (pdf/docx/txt)
     * - Sends thumbs feedback on the specific assistant message
     * - Creates a persistent DocumentFeedback record (query→negativeAnswer→document)
     *
     * UX:
     * - Closes the modal on success
     */
    const handleFeedbackSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const feedback = await handleFeedback();
        handleUserFeedback(messageFeedbackDetails['message_id'], messageFeedbackDetails['conversation_id'], feedback['g_feedback'], e)
        createDocumentFeedback(feedback.query_id, feedback.negative_answer_id, feedback.doc_name ?? "", feedback.document_text, feedback.context, feedback.theme)
        setFeedbackFormOpen(false);
    }

    /**
     * Build the payload for a feedback submission.
     *
     * @returns Object with:
     *  - g_feedback: freeform general feedback
     *  - document_text: extracted text of uploaded file or empty string
     *  - query_id / negative_answer_id: message linkage
     *  - doc_name / context / theme
     */
    const handleFeedback = async () => {
        const document_name = fileDoc ? fileDoc.name : null;
        const query_id = messageFeedbackDetails['query_id'];
        const botAnswer_id = messageFeedbackDetails['generated_answer_id'];
        var document_text;
        // console.log(theme, fileDoc ? fileDoc.name : null, fileDoc, context)
        if (!fileDoc) {
            document_text = "";
        }
        else if (document_name?.includes(".pdf")) {
            document_text = await handlePdfFile(fileDoc);
        }
        else if (document_name?.includes(".docx")) {
            document_text = await handleDocxFile(fileDoc);
        }
        else {
            document_text = await fileDoc?.text();
        }

        return {
            g_feedback: generalFeedback,
            document_text: document_text,
            query_id: query_id,
            negative_answer_id: botAnswer_id,
            doc_name: document_name,
            context: context,
            theme: theme
        }
    }

    /**
     * Send a user message and stream assistant response via Server-Sent Events (SSE).
     *
     * Behavior:
     * - Creates conversation if none selected (defaults to "normal")
     * - Optimistically appends user + placeholder assistant message
     * - Builds multipart FormData with:
     *   - message, conversation_type, web_search_tool, conversation_history, conversation_id
     *   - files[] (if lawsuit mode with uploads)
     * - Opens a `fetch` with `credentials: "include"` and reads streamed chunks
     * - Each chunk: JSON from lines starting with `data: `
     *   - Accumulates `response` text into the last assistant message
     * - On completion:
     *   - Persists both user and assistant messages via `createMessage`
     *   - Refreshes messages from server
     *
     * Errors:
     * - Displays a top-level error string if streaming fails
     */
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const userMessage = userQuery.trim();
        if (!userMessage) return;

        try {

            let newConv;
            if (!currentConversation.conversation_name && user) {
                newConv = await createConversation(`Conversation ${conversations?.length || 0}`, user?.username, 'normal');
                if (newConv)
                    setCurrentConversation(newConv)
                fetchConversations(user.username)

            }

            const now = new Date().toISOString();
            const newMessages = [
                { message: userMessage, role: 'user', timestamp: now, id: uuidv4(), feedback: null },
                { message: '', role: 'assistant', timestamp: now, id: uuidv4(), feedback: null }
            ];

            setMessages(prev => [...prev, ...newMessages]);
            setUserQuery('');
            setBotResponse('');

            const controller = new AbortController();
            controllerRef.current = controller;
            setIsStreaming(true);

            let type;
            let conversation_id;
            if (currentConversation.conversation_type === '') {
                conversation_id = newConv?.conversation_id;
                type = 'normal';
            } else {
                conversation_id = currentConversation.conversation_id
                type = currentConversation.conversation_type;
            }


            console.log(messages);

            const form = new FormData();
            form.append("message", userMessage);
            form.append("conversation_type", type);
            form.append("web_search_tool", String(isOnline));  // must be string
            form.append("conversation_history", JSON.stringify(messages.slice(-10)));
            form.append('conversation_id', conversation_id ?? "")

            if (uploadedFiles) {
                for (const f of uploadedFiles) {
                    form.append("files", f); // multiple files allowed
                }
            }

            for (const [k, v] of form.entries()) {
                console.log("FD:", k, v instanceof File ? v.name : v);
            }

            const res = await fetch(`${api.defaults.baseURL}/request`, {
                method: "POST",
                signal: controller.signal,
                credentials: "include",
                body: form, // no JSON.stringify, no Content-Type header
            });

            if (!res.ok || !res.body) {
                setBotResponse("Error from bot");
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let fullBotResponse = '';

            while (true) {
                const { value, done } = await reader.read();
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n').filter(line => line.startsWith('data: '));

                for (const line of lines) {
                    try {
                        const jsonStr = line.replace("data: ", "");
                        const parsed = JSON.parse(jsonStr);
                        fullBotResponse += parsed.response;

                        setMessages(prev => {
                            const updated = [...prev];
                            const lastIndex = updated.length - 1;
                            updated[lastIndex].message = fullBotResponse;
                            updated[lastIndex].timestamp = new Date().toISOString();
                            return updated;
                        });
                    } catch (err) {
                        console.error("Invalid chunk", err);
                    }
                }

                if (done) {
                    if (!conversation_id) {
                        console.error("No conversation_id — cannot persist messages.");
                        return; // or throw new Error("Missing conversation_id")
                    }
                    console.log(conversation_id, userMessage, 'user', newMessages[0].id, newMessages[0].feedback)
                    console.log(conversation_id, fullBotResponse, 'assistant', newMessages[1].id, newMessages[1].feedback);
                    await createMessage(conversation_id, userMessage, 'user', newMessages[0].id, newMessages[0].feedback);
                    await createMessage(conversation_id, fullBotResponse, 'assistant', newMessages[1].id, newMessages[1].feedback);
                    await fetchUserMessages(conversation_id);
                    break;
                }
            }
        } catch (err) {
            console.error("Streaming failed:", err);
            setBotResponse(`${err}`);
        } finally {
            setIsStreaming(false);
            controllerRef.current = null;
        }
    };

    /**
     * Auto-post an assistant onboarding message when a new 'lawsuit' conversation is created.
     *
     * @param conversation_id - target conversation
     * @param conversation_type - "lawsuit" to trigger onboarding text
     */
    const createAutomatedBotMessage = async (conversation_id: string | undefined, conversation_type: string) => {
        var fullBotResponse = "";
        if (conversation_type === 'lawsuit') {
            fullBotResponse = `Welcome to the lawsuit creation tool.  
                In this conversation, I will guide you through creating a phishing complaint.  

                Please note: This is a demo tool, so kindly keep that in mind.  

                To draft a complete and legally sound criminal complaint for phishing, I will need some specific details from you. Please provide the following information:  

                1. **Your personal details** (full name, contact information).  
                2. **The accused’s details** (if known).  
                3. **A detailed description of the events** (how the phishing occurred, through which platform, amounts of money involved, etc.).  
                4. **A chronological timeline** of what happened.  
                5. **Any evidence you have** (messages, transaction receipts, screenshots, or other supporting documents).  

                Once you provide these details, I will generate a structured draft of your complaint.`;
        }

        if (typeof conversation_id === 'string') {
            await createMessage(conversation_id, fullBotResponse, 'assistant', uuidv4(), '');
            await fetchUserMessages(conversation_id);
        } else {
            console.error('Invalid conversation_id for createAutomatedBotMessage');
        }
    }

    /**
     * Submit thumbs feedback for a single assistant message and update local UI.
     *
     * @param message_index - message id (assistant message)
     * @param conversation_id - current conversation id
     * @param feedback - "true" | "false"
     * @param e - optional event to prevent default
     */
    const handleUserFeedback = async (
        message_index: string,
        conversation_id: string,
        feedback: string,
        e?: React.MouseEvent | React.FormEvent
    ) => {
        e?.preventDefault();
        // console.log(feedback);
        try {
            await userFeedback(message_index, conversation_id, feedback);
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === message_index ? { ...m, feedback } : m
                )
            );
        } catch (err) {
            console.log("Something went wrong with the feedback");
        }
    }

    /**
     * Rename the current conversation (inline edit in sidebar), persists via API, and
     * updates local list in place.
     */
    const handleRename = async (conversationId: string) => {
        if (!editedTitle.trim()) {
            setEditingConvId('');
            return;
        }
        // console.log(conversationId, editedTitle.trim());
        await renameConversation(conversationId, editedTitle.trim());

        if (conversations) {
            for (let i = 0; i < (conversations?.length ?? 0); i++) {
                if (conversations[i] !== null && conversations[i].conversation_id === conversationId) {
                    conversations[i].conversation_name = editedTitle.trim();
                    break;
                }
            }
        }

        setEditingConvId('');

    }

    // const createNewConversation = async (conversation_type: string) => {
    //     if (user) {
    //         var conversation_name;
    //         if (conversation_type === 'lawsuit') {
    //             conversation_name = `Lawsuit ${conversations?.length || 0}`;
    //         }
    //         else {
    //             conversation_name = `Conversation ${conversations?.length || 0}`;
    //         }
    //         const newConv = await createConversation(conversation_name, user?.username, conversation_type);
    //         if (newConv) {
    //             setCurrentConversation(newConv);
    //             setMessages([]);
    //         }

    //     }
    // };

    /**
     * Create a new conversation of the given type ("normal" | "lawsuit"):
     * - Sets it as current
     * - Clears local messages
     * - For "lawsuit": posts a guidance message
     * - Refreshes conversation/messaging lists
     * - Closes the sidebar on mobile
     */
    const createNewConversation = async (conversation_type: string) => {
        if (!user) return;

        var conversation_name;
        if (conversation_type === 'lawsuit') {
            conversation_name = `Lawsuit ${conversations?.length || 0}`;
        }
        else {
            conversation_name = `Conversation ${conversations?.length || 0}`;
        }
        const newConv = await createConversation(conversation_name, user?.username, conversation_type);

        if (newConv) {
            await setCurrentConversation(newConv);
            setMessages([]); // local clear
            if (conversation_type === 'lawsuit') {
                createAutomatedBotMessage(newConv.conversation_id, conversation_type);
            }
            await Promise.all([
                fetchUserMessages(newConv.conversation_id),
                fetchConversations(user.username),
            ]);
            setSidebarOpen(false);
        }
    };

    /**
     * Switch to a conversation and hydrate messages.
     */
    const getMessagesFromConversations = async (conversation_id: string, conversation_name: string, conversation_type: string) => {
        setCurrentConversation({ conversation_id: conversation_id, conversation_name: conversation_name, conversation_type: conversation_type });
        await fetchUserMessages(conversation_id);
    };

    /**
     * Send on Enter, newline on Shift+Enter; autogrows the textarea.
     */
    const handleTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (userQuery.trim()) {
                handleSubmit(e);
            }
        }
    };

    /**
     * Logout the current user:
     * - Clears session cookie server-side
     * - Resets local auth and message state
     * - Navigates to `/login`
     */
    const logoutButton = async () => {
        await logoutUser();
        navigate('/login');
    };

    return (
        <div className="flex h-screen bg-gray-100 text-gray-800 relative overflow-x-hidden">
            {open && (
                <form
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm overflow-y-auto p-4"
                    onSubmit={handleFeedbackSubmit}
                >
                    <div className="bg-white p-6 rounded-lg shadow-lg max-w-2xl w-full space-y-6 max-h-[90vh] overflow-y-auto">
                        {/* Title */}
                        <h2 className="text-2xl font-bold text-gray-900">Feedback Instructions</h2>

                        {/* General Info */}
                        <div className="bg-gray-100 p-4 rounded-lg text-sm text-gray-700 space-y-2">
                            <p>
                                In this section, you can provide feedback on the chatbot’s responses.
                                If you believe a response was <strong>inaccurate, incomplete, or unreliable</strong>,
                                you may suggest what the correct answer should have been.
                            </p>
                            <p>
                                When submitting your feedback, please include the following:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li>
                                    <strong>Correct Answer Document:</strong> Upload or provide a document that contains the appropriate answer.
                                </li>
                                <li>
                                    <strong>Relevant Context:</strong> Specify the part of the document (section, paragraph, or article) that supports your correction.
                                </li>
                                <li>
                                    <strong>Theme:</strong> Select the theme of your feedback. You may choose one of the following, or propose your own:
                                    <ul className="list-disc list-inside ml-4">
                                        <li><strong>cases</strong>: Related to Law Cases</li>
                                        <li><strong>phishing</strong>: Related to Phishing Scenarios and General Information</li>
                                        <li><strong>cybercrime</strong>: Related to the Greek Penal Code and Legislation</li>
                                        <li><strong>gdpr</strong>: Related to the General Data Protection Regulation</li>
                                    </ul>
                                </li>
                            </ul>
                        </div>

                        {/* Theme Selection */}
                        <div>
                            <label className="block font-medium mb-2">Theme</label>
                            <div className="space-y-2">
                                <div>
                                    <input
                                        type="radio"
                                        id="cases"
                                        name="theme"
                                        value="cases"
                                        onClick={() => {
                                            setTheme("cases");
                                            setOtherRadio(false);
                                        }}
                                    />
                                    <label htmlFor="cases" className="ml-2">Cases (Law Cases)</label>
                                </div>
                                <div>
                                    <input
                                        type="radio"
                                        id="phishing"
                                        name="theme"
                                        value="phishing"
                                        onClick={() => {
                                            setTheme("phishing");
                                            setOtherRadio(false);
                                        }}
                                    />
                                    <label htmlFor="phishing" className="ml-2">Phishing (Scenarios & General Info)</label>
                                </div>
                                <div>
                                    <input
                                        type="radio"
                                        id="cybercrime"
                                        name="theme"
                                        value="cybercrime"
                                        onClick={() => {
                                            setTheme("cybercrime");
                                            setOtherRadio(false);
                                        }}
                                    />
                                    <label htmlFor="cybercrime" className="ml-2">Cybercrime (Greek Penal Code & Legislation)</label>
                                </div>
                                <div>
                                    <input
                                        type="radio"
                                        id="gdpr"
                                        name="theme"
                                        value="gdpr"
                                        onClick={() => {
                                            setTheme("gdpr");
                                            setOtherRadio(false);
                                        }
                                        }
                                    />
                                    <label htmlFor="gdpr" className="ml-2">GDPR (Data Protection Regulation)</label>
                                </div>
                                <div className='flex items-center gap-2'>
                                    <input
                                        type='radio'
                                        id='custom'
                                        name='theme'
                                        value='custom'
                                        onChange={() => setOtherRadio(true)}
                                    />
                                    <label htmlFor="custom" className='ml-2'>Other:</label>
                                    <input
                                        type='text'
                                        placeholder='Enter your own theme'
                                        className='flex-1 px-2 py-1 border rounded text-sm'
                                        disabled={otherRadio === false}
                                        onChange={(e) => setTheme(e.target.value)}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Document Upload */}
                        <div>
                            <label htmlFor="file_document" className="block font-medium mb-2">
                                Upload Correct Answer Document
                            </label>
                            <input
                                type="file"
                                id="file_document"
                                name="file_document"
                                accept=".pdf,.doc,.docx,.txt"
                                onChange={(e) => setDocument(e.target.files?.[0] ?? null)}
                                className="block w-full text-sm text-gray-600"
                            />
                        </div>

                        <div>
                            <label htmlFor="general_feedback" className="block font-medium mb-2">
                                General Feedback
                            </label>
                            <textarea
                                name="general_feedback"
                                id="general_feedback"
                                autoComplete="off"
                                required
                                onChange={(e) => setGeneralFeedback(e.target.value)}
                                placeholder="Specify section, paragraph, or article..."
                                className="w-full px-3 py-2 border rounded-lg text-sm h-32 resize-y"
                            />
                        </div>

                        {/* Context Input */}
                        <div>
                            <label htmlFor="context" className="block font-medium mb-2">
                                Relevant Context
                            </label>
                            <textarea
                                name="context"
                                id="context"
                                autoComplete="off"
                                required
                                onChange={(e) => setContext(e.target.value)}
                                placeholder="Specify section, paragraph, or article..."
                                className="w-full px-3 py-2 border rounded-lg text-sm h-32 resize-y"
                            />
                        </div>

                        {/* Buttons */}
                        <div className="flex justify-end space-x-3">
                            <button
                                type="button"
                                onClick={() => setFeedbackFormOpen(false)}
                                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                            >
                                Close
                            </button>
                            <button
                                type="submit"
                                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                Submit Feedback
                            </button>
                        </div>
                    </div>
                </form>

            )}
            {/* Overlay for mobile sidebar */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 z-10 bg-black/40 backdrop-blur-sm md:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside className={`
      fixed md:relative top-0 left-0 w-64 bg-white border-r z-20
      transform transition-transform duration-200 ease-in-out
      flex flex-col h-full md:h-screen
      ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
    `}>
                <div className="p-4 flex flex-col h-full">
                    <h2 className="font-semibold mb-2 text-center">Tools</h2>

                    {/* Divider for emphasis */}
                    <hr className="my-2 border-gray-300" />

                    <button
                        onClick={() => createNewConversation('lawsuit')}
                        className="w-full mb-4 p-2 bg-gradient-to-r from-blue-600 to-blue-400 text-white font-bold rounded-lg shadow hover:from-blue-700 hover:to-blue-500 transition"
                    >
                        ⚖️ Build Lawsuit
                    </button>

                    {/* Divider for emphasis */}
                    <hr className="my-2 border-gray-300" />
                    <button
                        onClick={() => createNewConversation('normal')}
                        className="w-full mb-4 p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                        New Conversation
                    </button>

                    <ul className="space-y-2 overflow-y-auto flex-1">
                        {conversations?.map((conv) => (
                            <li
                                key={conv.conversation_id}
                                onClick={() => {
                                    if (editingConvId !== conv.conversation_id) {
                                        setSidebarOpen(false);
                                        getMessagesFromConversations(
                                            conv.conversation_id,
                                            conv.conversation_name,
                                            conv.conversation_type
                                        );
                                    }
                                }}
                                onDoubleClick={() => {
                                    setEditingConvId(conv.conversation_id);
                                    setEditedTitle(conv.conversation_name);
                                }}
                                className={`p-2 cursor-pointer rounded ${conv.conversation_id ===
                                    currentConversation.conversation_id
                                    ? "bg-blue-100 font-semibold"
                                    : "hover:bg-gray-200"
                                    }`}
                            >
                                {editingConvId === conv.conversation_id ? (
                                    <input
                                        value={editedTitle}
                                        onChange={(e) => setEditedTitle(e.target.value)}
                                        onBlur={() => handleRename(conv.conversation_id)}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter") handleRename(conv.conversation_id);
                                            if (e.key === "Escape") setEditingConvId("");
                                        }}
                                        autoFocus
                                        className="w-full p-1 border rounded text-sm"
                                    />
                                ) : (
                                    conv.conversation_name
                                )}
                            </li>
                        ))}
                    </ul>

                    <button
                        onClick={logoutButton}
                        className="p-2 bg-red-500 text-white rounded hover:bg-red-600 mt-4"
                    >
                        Logout
                    </button>
                </div>

            </aside>

            {/* Main Chat Area */}

            <div className="flex-1 flex flex-col z-0 items-center">
                {/* Mobile Header */}
                <div className="md:hidden flex justify-between items-center p-4 bg-white shadow w-full">
                    <button onClick={() => setSidebarOpen(!sidebarOpen)}>
                        {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                    <h1 className="text-lg font-bold">AILA INTERFACE DEMO</h1>
                </div>

                {/* Desktop Title */}
                <h1 className="text-xl font-bold text-center mt-4 mb-2 hidden md:block">
                    AILA INTERFACE DEMO
                </h1>

                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="text-sm md:text-base text-gray-700 text-center mb-4"
                >
                    Welcome, <span className="font-semibold text-blue-700">{user?.username}</span>
                </motion.div>

                {/* Chat Container */}

                <div ref={chatRef} className="flex-1 w-full max-w-4xl px-6 overflow-y-auto overflow-x-hidden">
                    <div className="bg-white rounded-lg shadow p-4 space-y-4">
                        {currentConversation ? (
                            <ul className="space-y-4">
                                {messages.map((mes, index) => (
                                    <li
                                        key={mes.id}
                                        className={`flex flex-col gap-1 ${mes.role === 'user' ? 'items-end' : 'items-start'}`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <div className="mt-1">
                                                {mes.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                                            </div>
                                            <div className={`px-5 py-4 rounded-md shadow-sm text-sm whitespace-pre-line text-sm md:text-base lg:text-lg ${mes.role === 'user'
                                                ? 'bg-blue-100 text-blue-800'
                                                : 'bg-gray-200 text-gray-800'
                                                }`}>
                                                {/* {mes.message} */}
                                                {mes.role === 'assistant' ? (
                                                    (() => {
                                                        const isLastAssistant =
                                                            messages.length > 0 &&
                                                            messages[messages.length - 1]?.id === mes.id &&
                                                            mes.role === 'assistant';

                                                        const hasStarted = !!mes.message;

                                                        if (isStreaming && isLastAssistant && !hasStarted) {
                                                            // **NEW**: show spinner in the last bot message BEFORE the first token arrives
                                                            return <InlineSpinner />;
                                                        }

                                                        if (isStreaming && isLastAssistant && hasStarted) {
                                                            // streaming started: show text + blinking cursor
                                                            return (
                                                                <span>
                                                                    {mes.message}
                                                                    <span className="inline-block align-baseline w-[0.6ch] animate-pulse">▍</span>
                                                                </span>
                                                            );
                                                            // return <span>{mes.message}<span className="animate-pulse">▍</span></span>;
                                                        }

                                                        // idle / finished
                                                        return mes.message;
                                                    })()
                                                ) : (
                                                    mes.message
                                                )}
                                            </div>
                                        </div>

                                        {/* Feedback aligned right */}
                                        {mes.role === 'assistant' && user?.role === 'user' && mes.id && currentConversation?.conversation_id && (
                                            <div className="flex justify-end w-full pr-10 mt-1">
                                                <button
                                                    type="button"
                                                    disabled={mes.feedback === 'false'}
                                                    onClick={(e) => handleUserFeedback(mes.id, currentConversation.conversation_id, 'false', e)}
                                                    aria-pressed={mes.feedback === 'false'}
                                                    aria-label="Mark as unhelpful"
                                                    className={`text-xs md:text-sm lg:text-base mr-3 transition-colors ${mes.feedback === 'false'
                                                        ? 'text-red-600 font-bold cursor-default'
                                                        : (mes.feedback !== null)
                                                            ? 'text-gray-600 opacity-50 hover:opacity-100 hover:text-red-500 cursor-pointer'
                                                            : 'text-gray-600 hover:text-red-500 cursor-pointer'
                                                        }`}
                                                    // title = {mes.feedback === false? 'Marked unhelpful' : 'Mark as unhelpful'}
                                                    title="Thumbs down"
                                                >
                                                    👎
                                                </button>
                                                <button
                                                    type="button"
                                                    disabled={mes.feedback === 'true'}
                                                    onClick={(e) => handleUserFeedback(mes.id, currentConversation.conversation_id, 'true', e)}
                                                    aria-pressed={mes.feedback === 'true'}
                                                    aria-label="Mark as unhelpful"
                                                    className={`text-xs md:text-sm lg:text-base mr-3 transition-colors ${mes.feedback === 'true'
                                                        ? 'text-red-600 font-bold cursor-default'
                                                        : (mes.feedback !== null)
                                                            ? 'text-gray-600 opacity-50 hover:opacity-100 hover:text-red-500 cursor-pointer'
                                                            : 'text-gray-600 hover:text-red-500 cursor-pointer'
                                                        }`}
                                                    title="Thumbs up"
                                                >
                                                    👍
                                                </button>
                                            </div>
                                        )}
                                        {mes.role === 'assistant' && user?.role === 'lawyer' && mes.id && currentConversation?.conversation_id && (
                                            <div className="flex justify-end w-full pr-10 mt-1">
                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        const details = {
                                                            message_id: mes.id,
                                                            conversation_id: currentConversation.conversation_id,
                                                            query_id: messages[index - 1].id,
                                                            generated_answer_id: mes.id,
                                                        };
                                                        setmessageFeedbackDetails(details); // async
                                                        setFeedbackFormOpen(true);

                                                    }}
                                                    className="px-3 py-1 bg-blue-600 text-white text-xs md:text-sm lg:text-base rounded-lg shadow hover:bg-blue-700 transition-colors"
                                                >
                                                    Feedback
                                                </button>
                                            </div>
                                        )}
                                    </li>
                                ))}

                            </ul>
                        ) : (
                            <div className="text-gray-500">Select or create a conversation</div>
                        )}
                    </div>
                </div>

                {/* Error Message */}
                {
                    botResponse && (
                        <div className="mb-2 text-red-500 px-4">{botResponse}</div>
                    )
                }

                {/* Input */}
                {
                    currentConversation && (
                        <form onSubmit={handleSubmit} className="w-full max-w-4xl p-5 bg-white border-t">
                            <div className="flex items-start gap-2">
                                {/* Textarea (auto-grow, doesn't force siblings to grow) */}
                                <textarea
                                    value={userQuery}
                                    onChange={(e) => setUserQuery(e.target.value)}
                                    onKeyDown={handleTextareaKeyDown}        // optional: Enter=send, Shift+Enter=new line
                                    onInput={(e) => {
                                        const el = e.currentTarget;
                                        el.style.height = '0px';               // reset
                                        el.style.height = el.scrollHeight + 'px'; // grow to content
                                    }}
                                    rows={1}
                                    placeholder="Type your message here..."
                                    required
                                    className="flex-1 min-w-0 border border-gray-300 rounded-md p-4 text-sm md:text-base lg:text-lg
                 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none leading-6
                 max-h-48 overflow-y-auto break-words"
                                />

                                {/* Right-side buttons: fixed height, don't stretch */}
                                <div className="flex shrink-0 self-start gap-2 md:flex-col">
                                    <button
                                        type="submit"
                                        className="h-12 px-6 bg-blue-600 text-white font-semibold text-sm md:text-base lg:text-lg
                   rounded-md hover:bg-blue-700 transition focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    >
                                        Submit
                                    </button>
                                    {currentConversation.conversation_type == 'lawsuit' && (
                                        <div>
                                            <input
                                                type='file'
                                                ref={inputRef}
                                                multiple
                                                accept="image/*,audio/mpeg,audio/wav,application/pdf,.txt,.csv"
                                                onChange={onSelect}
                                                hidden
                                            />
                                            <button
                                                type="button"
                                                onClick={openFileDialog}
                                                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                                                Upload Files
                                            </button>
                                            {!!uploadedFiles?.length && (
                                                <ul className="max-h-32 overflow-auto text-xs border rounded-md p-2 space-y-1">
                                                    {uploadedFiles.map((f, i) => (
                                                        <li key={i} className='flex items-center justify-between gap-2'>
                                                            <span className="truncate">{f.name}</span>
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-gray-500">
                                                                    {(f.size / (1024 * 1024)).toFixed(2)} MB
                                                                </span>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => setUploadedFiles((prev) => prev.filter((_, idx) => idx !== i))}
                                                                    className="px-2 py-1 border text-xs rounded hover:bg-gray-50"
                                                                >
                                                                    remove
                                                                </button>
                                                            </div>
                                                        </li>
                                                    ))}
                                                </ul>
                                            )}
                                        </div>
                                    )}
                                    {currentConversation.conversation_type == 'normal' && (<button
                                        type="button"
                                        onClick={() => setOnlineMode(!isOnline)}
                                        aria-pressed={isOnline}
                                        className={`h-12 px-6 font-semibold text-sm md:text-base lg:text-lg rounded-md transition
                    focus:outline-none focus:ring-2 focus:ring-blue-500 border border-gray-300 whitespace-nowrap
                    ${isOnline ? 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                                                : 'bg-blue-600 text-white hover:bg-blue-700'}`}
                                    >
                                        {isOnline ? 'RAG Mode' : 'Online Mode'}
                                    </button>)}
                                    {/* <button
                                        type="button"
                                        onClick={() => setOnlineMode(!isOnline)}
                                        aria-pressed={isOnline}
                                        className={`h-12 px-6 font-semibold text-sm md:text-base lg:text-lg rounded-md transition
                    focus:outline-none focus:ring-2 focus:ring-blue-500 border border-gray-300 whitespace-nowrap
                    ${isOnline ? 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                                                : 'bg-blue-600 text-white hover:bg-blue-700'}`}
                                    >
                                        {isOnline ? 'RAG Mode' : 'Online Mode'}
                                    </button> */}
                                </div>
                            </div>
                        </form>
                    )
                }
            </div >
        </div >
    );


};

export default Chat;


