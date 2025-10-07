import os
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import requests
from datetime import datetime
from dotenv import load_dotenv
import re
import json
import time
from dataclasses import dataclass, field
from typing import Any



@dataclass
class Profile:
    name: Optional[str] = None
    salutation: Optional[str] = None
    language: Optional[str] = None
    role_applied: Optional[str] = None
    seniority: Optional[str] = None
    timezone: Optional[str] = None
    contact_email: Optional[str] = None
    consent: Optional[bool] = None
    accessibility_needs: Optional[str] = None


@dataclass
class Preferences:
    answer_style: Optional[str] = None
    allow_followups: Optional[bool] = None
    pace: Optional[str] = None
    notification_opt_in: Optional[bool] = None



# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API base URL
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Bot user ID (fixed for Telegram bot)
BOT_USER_ID = "telegram_bot"

# Session storage (in-memory; use DB for production)
user_sessions: Dict[str, Dict[str, Any]] = {}

# Response Classification and Fallback System
class ResponseClassifier:
    def __init__(self):
        # Garbage patterns (random characters, no meaningful content)
        self.garbage_patterns = [
            r'^[^\w\s]{3,}$',  # Only symbols
            r'^.{0,2}$',  # Too short
            r'^(.)\1{5,}$',  # Repeated characters
            r'^\w{20,}$',  # Too long single word (keyboard mash)
            r'^[qwertyuiopasdfghjklzxcvbnm]{10,}$',  # Keyboard pattern
            r'^\d{10,}$',  # Long numbers only
        ]

        # Out of scope intents
        self.out_of_scope_patterns = [
            "how to hack", "where can i buy", "what's the weather",
            "tell me a joke", "play music", "send email", "make call",
            "political views", "religious beliefs", "personal questions",
            "weather", "joke", "music", "play song", "directions"
        ]

        # Question types and expected responses
        self.question_response_expectations = {
            "salary": {
                "valid_patterns": [r'\$\d+', r'\d+(?:\.\d+)?(?:k|K|lakhs|million|crore)', r'\d+'],
                "invalid_responses": ["i don't know", "confidential", "negotiable", "not sure", "tbd", "tba"]
            },
            "relocation": {
                "valid_patterns": [r'(?:yes|no|maybe|possibly|depends|willing|interested|open)'],
                "alternate_responses": ["to mumbai", "relocate", "move", "location", "city"]
            },
            "experience": {
                "valid_patterns": [r'\d+', r'year', r'month', r'experience'],
                "context_indicators": ["worked", "experience", "since", "from", "to", "in", "at"]
            },
            "general": {
                "valid_patterns": [r'.{10,}', r'(?:yes|no|maybe)'],  # At least 10 chars or clear yes/no
                "invalid_responses": ["idk", "dunno", "whatever", "doesnt matter"]
            }
        }

    def classify_response(self, response: str, question_type: str = None) -> Dict[str, Any]:
        """Classify a user response and determine fallback actions."""
        response_lower = response.lower().strip()

        # Check for garbage input
        if self._is_garbage(response_lower):
            return {
                "intent": "garbage",
                "confidence": 0.95,
                "fallback_action": "clarification_request",
                "message": "I couldn't understand your response. Are you ready to respond to the question?"
            }

        # Check for out-of-scope
        if self._is_out_of_scope(response_lower):
            return {
                "intent": "out_of_scope",
                "confidence": 0.85,
                "fallback_action": "out_of_scope_response",
                "message": "I can only help with interview-related questions. Would you like to continue with the interview?"
            }

        # Check for low confidence responses
        low_confidence = self._check_low_confidence(response_lower)
        if low_confidence:
            return low_confidence

        # Validate based on question type
        if question_type:
            validation = self._validate_question_response(response_lower, question_type)
            if not validation["valid"]:
                return {
                    "intent": "invalid_response",
                    "confidence": 0.8,
                    "fallback_action": "slot_validation",
                    "message": validation["message"]
                }

        return {
            "intent": "valid",
            "confidence": 0.9,
            "fallback_action": None
        }

    def _is_garbage(self, response: str) -> bool:
        """Check if response is garbage/non-sense."""
        for pattern in self.garbage_patterns:
            if re.match(pattern, response):
                return True
        return False

    def _is_out_of_scope(self, response: str) -> bool:
        """Check if response is out of scope for interview."""
        return any(scope_text in response for scope_text in self.out_of_scope_patterns)

    def _check_low_confidence(self, response: str) -> Optional[Dict[str, Any]]:
        """Check for low confidence responses that need clarification."""
        low_conf_signals = [
            "don't know", "not sure", "confused", "unclear", "what do you mean",
            "can you repeat", "can you rephrase", "i don't understand",
            "not clear", "unclear question", "don't get it"
        ]

        if any(signal in response for signal in low_conf_signals):
            return {
                "intent": "low_confidence",
                "confidence": 0.75,
                "fallback_action": "two_stage_clarification",
                "message": "Let me rephrase the question to make it clearer."
            }
        return None

    def _validate_question_response(self, response: str, question_type: str) -> Dict[str, Any]:
        """Validate response against question expectations."""
        expectations = self.question_response_expectations.get(question_type, self.question_response_expectations["general"])

        # Check for invalid responses
        if any(invalid in response for invalid in expectations.get("invalid_responses", [])):
            return {
                "valid": False,
                "message": f"That response seems unclear. Could you please provide a more specific answer?"
            }

        # Check for valid patterns
        valid_patterns = expectations.get("valid_patterns", [])
        if valid_patterns:
            for pattern in valid_patterns:
                if re.search(pattern, response):
                    return {"valid": True}

        # Check for context indicators (for experience-type questions)
        context_indicators = expectations.get("context_indicators", [])
        if context_indicators and any(indicator in response for indicator in context_indicators):
            return {"valid": True}

        # Default to invalid if we can't find matching patterns
        return {
            "valid": False,
            "message": f"I'm looking for a more complete answer. Could you elaborate?"
        }

    def get_fallback_message(self, intent: str, question_text: str = "") -> str:
        """Generate appropriate fallback message based on intent."""
        fallbacks = {
            "garbage": f"I couldn't understand your response. This is supposed to be an interview about '{question_text[:50]}...'. Are you ready to provide an appropriate answer?",
            "out_of_scope": f"I can only help with interview questions. Let's stay focused on the interview. Can you answer: {question_text}",
            "low_confidence": f"I noticed you might be unclear about the question. Let me rephrase it in a different way.",
            "invalid_response": f"Your answer doesn't quite match what I'm looking for. Could you provide a clearer response to: {question_text}"
        }
        return fallbacks.get(intent, "Could you please clarify your response?")

    def _rephrase_question(self, question_text: str, user_confusion: str = "") -> str:
        """Rephrase the question to make it clearer based on user's confusion."""
        question_lower = question_text.lower()

        # Salary question rephrasing
        if "salary" in question_lower or "expect" in question_lower:
            return "What is your expected salary for this position? Please provide a range or specific amount."

        # Relocation question rephrasing
        elif "relocation" in question_lower or "mumbai" in question_lower:
            return "Are you willing to relocate to Mumbai for this job opportunity? Please answer yes, no, or maybe."

        # Experience question rephrasing
        elif "experience" in question_lower or "worked" in question_lower or "year" in question_lower:
            return "How many years of relevant experience do you have? Please provide a specific number."

        # Default rephrasing - make it simpler
        else:
            # Basic rephrasing strategies
            rephrased = question_text
            if "Could you tell me" in question_text:
                rephrased = question_text.replace("Could you tell me", "Can you describe")
            elif "What is" in question_text:
                rephrased = question_text.replace("What is", "Please explain")
            elif "How" in question_text:
                rephrased = question_text.replace("How", "In what way")

            return f"Let me ask this differently: {rephrased}"

# Fallback Manager to handle consecutive fallbacks and escalation
class FallbackManager:
    def __init__(self):
        self.classifier = ResponseClassifier()
        self.consecutive_fallbacks: Dict[str, int] = {}
        self.max_consecutive_fallbacks = 3
        self.session_fallback_history: Dict[str, List[Dict]] = {}

    def process_response(self, response: str, user_id: str, question_type: str = None, question_text: str = "") -> Dict[str, Any]:
        """Process user response and determine if fallback is needed."""
        classification = self.classifier.classify_response(response, question_type)

        # Track consecutive fallbacks
        if classification["intent"] != "valid":
            self.consecutive_fallbacks[user_id] = self.consecutive_fallbacks.get(user_id, 0) + 1

            # Track fallback history
            if user_id not in self.session_fallback_history:
                self.session_fallback_history[user_id] = []
            self.session_fallback_history[user_id].append({
                "timestamp": time.time(),
                "response": response,
                "classification": classification,
                "consecutive_count": self.consecutive_fallbacks[user_id]
            })
        else:
            # Reset counter on successful response
            self.consecutive_fallbacks[user_id] = 0

        # Check for escalation
        consecutive_count = self.consecutive_fallbacks.get(user_id, 0)

        if consecutive_count >= self.max_consecutive_fallbacks:
            classification["escalate"] = True
            classification["fallback_action"] = "human_handoff"
            classification["message"] = "I've tried to help several times but we're still having trouble. Let me connect you with a human interviewer for assistance."
        else:
            # Add custom message if no specific message exists
            if not classification.get("message"):
                classification["message"] = self.classifier.get_fallback_message(
                    classification["intent"], question_text
                )

        return classification

    def get_disambiguation_options(self, response: str) -> List[str]:
        """Provide disambiguation options when multiple intents are possible."""
        # Simple disambiguation based on response patterns
        if len(response.split()) > 1:
            return [
                "Did you mean to answer the interview question?",
                "Are you providing a complete response to the question?",
                "Would you like me to rephrase the question differently?"
            ]
        return []

    def reset_session(self, user_id: str):
        """Reset fallback tracking for a session."""
        if user_id in self.consecutive_fallbacks:
            del self.consecutive_fallbacks[user_id]
        if user_id in self.session_fallback_history:
            del self.session_fallback_history[user_id]

# Global fallback manager instance
fallback_manager = FallbackManager()

# Hardcoded jobs for selection (in production, fetch from API)
JOBS = [
    {"id": "1", "title": "Software Engineer"},
    {"id": "2", "title": "Data Scientist"},
    {"id": "3", "title": "Product Manager"},
]

def api_call(endpoint: str, method: str = "GET", data: Dict[str, Any] = None, files: Dict[str, Any] = None, use_form_data: bool = False) -> Dict[str, Any]:
    url = f"{BACKEND_API_URL}{endpoint}"
    try:
        if files or use_form_data:
            # Use form data for file uploads and interview responses
            form_data = {}
            if data:
                for key, value in data.items():
                    if hasattr(value, '__class__') and value.__class__.__name__ == 'UUID':
                        form_data[key] = str(value)
                    else:
                        form_data[key] = value
            response = requests.request(method, url, data=form_data, files=files)
        else:
            # Use JSON for regular API calls
            if data:
                json_data = {}
                for key, value in data.items():
                    if hasattr(value, '__class__') and value.__class__.__name__ == 'UUID':
                        json_data[key] = str(value)
                    else:
                        json_data[key] = value
                response = requests.request(method, url, json=json_data)
            else:
                response = requests.request(method, url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {}

# Validate email/phone
def validate_email(email: str) -> bool:
    return "@" in email and "." in email

def validate_phone(phone: str) -> bool:
    return phone.isdigit() and len(phone) >= 10

async def get_pending_telegram_session(user_id: str) -> Optional[Dict[str, Any]]:
    """Get pending telegram interview session for a user."""
    try:
        from src.resume_parser.database.postgres_client import pg_client
        # For telegram, user_id could be username or chat_id
        # For now, assume it's the username with @
        telegram_username = user_id if user_id.startswith('@') else f"@{user_id}"
        return await pg_client.get_pending_telegram_session(telegram_username)
    except Exception as e:
        logger.error(f"Error getting pending telegram session: {e}")
        return None

async def handle_followup_question(update: Update, session: Dict[str, Any], original_response: str) -> None:
    """Generate contextual follow-up questions based on user's response."""
    try:
        session_id = session.get("session_id")
        if not session_id:
            await update.message.reply_text("Session error. Please try restarting the interview.")
            return

        # Analyze the original response to generate contextual follow-up
        response_text = original_response.lower()

        # Generate contextual follow-up based on response content
        if "don't want" in response_text or "refuse" in response_text:
            follow_up_text = "I understand you prefer not to answer this question. Could you tell me why you'd prefer to skip this particular question?"
        elif "not sure" in response_text or "don't know" in response_text:
            follow_up_text = "That's okay if you're not sure. Would you like me to rephrase the question or move to a different topic?"
        elif len(response_text) < 10:
            follow_up_text = "Your response was quite brief. Could you elaborate a bit more on your thoughts?"
        else:
            follow_up_text = "Thank you for your response. Could you provide a specific example from your experience that relates to this?"

        # Create a temporary question object for the follow-up
        follow_up_question = {
            "id": f"followup_{session_id}",
            "question_text": follow_up_text,
            "question_type": "follow_up"
        }

        session["current_question"] = follow_up_question
        await update.message.reply_text(f"Follow-up: {follow_up_text}\n\n(You have 4 minutes to respond)")

    except Exception as e:
        logger.error(f"Error generating follow-up question: {e}")
        await update.message.reply_text("Let me continue with the next question in the interview.")

async def get_next_question(update: Update, session: Dict[str, Any]) -> None:
    """Try to get the next question from the API by using the skip action."""
    try:
        session_id = session.get("session_id")
        if not session_id:
            await update.message.reply_text("Session error. Please try restarting the interview.")
            return

        session_id_str = str(session_id) if session_id else None

        # Use the action endpoint to skip to next question
        from src.resume_parser.database.postgres_client import pg_client
        session_info = await pg_client.get_interview_session(session_id_str)
        if not session_info:
            await update.message.reply_text("Session not found. Please try restarting the interview.")
            return

        current_index = session_info.get("current_question_index", 0)
        question_ids = session_info.get("question_ids", [])
        total_questions = len(question_ids)

        if current_index >= total_questions - 1:
            await update.message.reply_text("No more questions available. The interview is complete.")
            session["step"] = "completed"
            return

        # Use action endpoint to skip
        action_response = api_call(f"/interview-sessions/{session_id_str}/action", "POST", {
            "session_id": session_id_str,
            "action": "skip"
        })

        if action_response.get("success"):
            next_question = action_response.get("next_question")
            if next_question:
                session["current_question"] = next_question
                question_number = action_response.get("question_number", current_index + 1)
                await update.message.reply_text(f"Question {question_number}: {next_question['question_text']}\n\n(You have 5 minutes to respond)")
            else:
                await update.message.reply_text("The interview is complete. Thank you for your participation!")
                session["step"] = "completed"
        else:
            await update.message.reply_text("Unable to get the next question. Please try again or contact support.")

    except Exception as e:
        logger.error(f"Error getting next question: {e}")
        await update.message.reply_text("I'm having trouble getting the next question. The interview might be complete, but you can try again or contact support.")


async def start_or_create_and_start(update: Update, user_id: str, telegram_username: str) -> None:
    """Ensure there's an interview session for this telegram user and start it immediately.

    If a pending session exists we start it. Otherwise we create a new interview session
    using the bot's configured questions and then call the start endpoint.
    """
    try:
        # Check for a pending session first
        pending_session = await get_pending_telegram_session(user_id)
        if pending_session:
            session_id = pending_session["id"]
            start_response = api_call(f"/interview-sessions/{session_id}/start", "POST", {"session_id": session_id})
            if start_response.get("success"):
                user_sessions[user_id] = {
                    "step": "interview",
                    "session_id": session_id,
                    "current_question": start_response.get("current_question"),
                    "session_type": "telegram_interviewer_initiated"
                }
                await update.message.reply_text(f"Interview started!\n\nQuestion: {start_response['current_question']['question_text']}\n\n(You have 5 minutes to respond)")
                return
            else:
                await update.message.reply_text("Failed to start pending interview. Trying to create a new session...")

        # No pending session - create one for this candidate using bot's questions
        questions_response = api_call(f"/users/{BOT_USER_ID}/questions")
        question_ids = [q["id"] for q in questions_response] if questions_response else []
        if not question_ids:
            await update.message.reply_text("No interview questions are configured. Please contact support.")
            return

        # Create a new telegram interview session (backend expects UUIDs)
        session_payload = {
            "user_id": BOT_USER_ID,
            "session_type": "telegram",
            "question_ids": question_ids,
            "candidate_ids": [],
            "metadata": {"telegram_username": telegram_username}
        }
        create_resp = api_call(f"/users/{BOT_USER_ID}/interview-sessions", "POST", session_payload)
        session_id = create_resp.get("id") or create_resp.get("session_id")
        if not session_id:
            await update.message.reply_text("Failed to create interview session. Please try again later.")
            return

        # Start the session
        start_response = api_call(f"/interview-sessions/{session_id}/start", "POST", {"session_id": session_id})
        if start_response.get("success"):
            user_sessions[user_id] = {
                "step": "interview",
                "session_id": session_id,
                "current_question": start_response.get("current_question"),
                "session_type": "telegram_bot_initiated"
            }
            await update.message.reply_text(f"Interview started!\n\nQuestion: {start_response['current_question']['question_text']}\n\n(You have 5 minutes to respond)")
        else:
            await update.message.reply_text("Failed to start the interview after creating a session. Please try again later.")

    except Exception as e:
        logger.error(f"Error creating/starting telegram interview: {e}")
        await update.message.reply_text("An error occurred while starting your interview. Please try again later.")

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.username if update.effective_user.username else str(update.effective_user.id)
    # Start interview immediately (no onboarding)
    telegram_username = f"@{update.effective_user.username}" if update.effective_user.username else str(update.effective_user.id)
    await start_or_create_and_start(update, user_id, telegram_username)

# Handle text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.username if update.effective_user.username else str(update.effective_user.id)
    session = user_sessions.get(user_id, {})
    step = session.get("step")
    text = update.message.text.lower().strip()  # Normalize text

    # Check if user is responding to interviewer-initiated interview
    if text == "start" and not session:
        # Start interview immediately for user (create one if needed)
        telegram_username = f"@{update.effective_user.username}" if update.effective_user.username else str(update.effective_user.id)
        await start_or_create_and_start(update, user_id, telegram_username)
        return

    if step == "info_collection":
        data = session["data"]
        if "name" not in data:
            data["name"] = update.message.text  # Use original text
            await update.message.reply_text("Great! Now, please provide your email:")
        elif "email" not in data:
            if not validate_email(text):
                await update.message.reply_text("Invalid email. Please try again:")
                return
            data["email"] = update.message.text  # Use original text
            await update.message.reply_text("Thanks! Now, please provide your phone number:")
        elif "phone" not in data:
            if not validate_phone(text):
                await update.message.reply_text("Invalid phone. Please try again:")
                return
            data["phone"] = update.message.text  # Use original text
            # Proceed to job selection
            keyboard = [[InlineKeyboardButton(job["title"], callback_data=f"job_{job['id']}")] for job in JOBS]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Select a job position:", reply_markup=reply_markup)
            session["step"] = "job_selection"
    elif step == "interview":
        # Handle special commands during interview
        text = update.message.text.strip()
        if text.lower() in ["/skip", "skip"]:
            await update.message.reply_text("Question skipped. Moving to the next question...")
            await get_next_question(update, session)
            return
        elif text.lower() in ["/help", "help"]:
            await update.message.reply_text(
                "Interview Commands:\n"
                "/skip - Skip current question\n"
                "/help - Show this help message\n"
                "/restart - Restart the interview\n"
                "/end - End the interview\n\n"
                "Please respond to the current question or use a command."
            )
            return
        elif text.lower() in ["/restart", "restart"]:
            # Reset session and start over
            fallback_manager.reset_session(user_id)  # Reset fallback counters
            user_sessions[user_id] = {"step": "info_collection", "data": {}}
            await update.message.reply_text("Interview restarted. Please provide your full name:")
            return
        elif text.lower() in ["/end", "end"]:
            await update.message.reply_text("Interview ended. Thank you for your time!")
            session["step"] = "completed"
            return

        # Enhanced fallback analysis for user responses
        current_question = session.get("current_question", {})
        question_text = current_question.get("question_text", "").lower()

        # Determine question type for validation
        question_type = None
        if "salary" in question_text:
            question_type = "salary"
        elif "relocation" in question_text or "mumbai" in question_text:
            question_type = "relocation"
        elif "experience" in question_text or "worked" in question_text or "year" in question_text:
            question_type = "experience"

        # Process response through fallback manager
        fallback_result = fallback_manager.process_response(
            update.message.text,
            user_id,
            question_type,
            question_text
        )

        # Handle different fallback scenarios
        if fallback_result["intent"] != "valid":
            if fallback_result.get("escalate"):
                # Ultimate fallback: human handoff
                await update.message.reply_text(
                    f"{fallback_result['message']}\n\n"
                    "I'll transfer you to a human interviewer now. Please hold for a moment."
                )
                session["step"] = "human_handoff"
                # Here you would trigger human handoff (implementation depends on your system)
                return
            elif fallback_result["intent"] == "garbage":
                await update.message.reply_text(
                    f"{fallback_result['message']}\n\n"
                    f"Original question: {current_question.get('question_text', 'N/A')}\n\n"
                    "Please provide a meaningful answer to continue the interview."
                )
                session["awaiting_clarification"] = True
                return
            elif fallback_result["intent"] == "out_of_scope":
                await update.message.reply_text(
                    f"{fallback_result['message']}\n\n"
                    "Shall we continue with the interview question instead?"
                )
                session["awaiting_clarification"] = True
                return
            elif fallback_result["intent"] == "low_confidence":
                # Two-stage clarification
                if not session.get("clarification_stage"):
                    # First stage: simple rephrase
                    rephrased_question = fallback_manager.classifier._rephrase_question(current_question.get('question_text', ''), update.message.text)
                    await update.message.reply_text(
                        f"{fallback_result['message']}\n\nRephrased: {rephrased_question}\n\n"
                        "Does this make it clearer for you?"
                    )
                    session["clarification_stage"] = 1
                    session["awaiting_clarification"] = True
                    return
                else:
                    # Second stage: provide examples
                    await update.message.reply_text(
                        "Let me give you an example of what we're looking for:\n\n"
                        f"Question: {current_question.get('question_text', '')}\n"
                        "Example response: \"Yes, I'm willing to relocate to Mumbai for this opportunity.\"\n\n"
                        "Can you try again, or would you like to skip this question?"
                    )
                    session["clarification_stage"] = 2
                    return
            elif fallback_result["intent"] == "invalid_response":
                # Slot validation with specific guidance
                await update.message.reply_text(
                    f"{fallback_result['message']}\n\n"
                    f"This seems like it might not be the best fit for the question: \"{current_question.get('question_text', '')}\"\n\n"
                    "Could you try rephrasing your answer?"
                )
                session["awaiting_clarification"] = True
                return

        # Disambiguation if multiple intents are possible (threshold-based)
        elif fallback_result.get("confidence", 0) < 0.8:
            disambiguation_options = fallback_manager.get_disambiguation_options(update.message.text)
            if disambiguation_options:
                option_buttons = [InlineKeyboardButton(option, callback_data=f"disambig_{i}")
                                for i, option in enumerate(disambiguation_options)]
                reply_markup = InlineKeyboardMarkup([option_buttons])
                await update.message.reply_text(
                    "I want to make sure I understood your response correctly. "
                    "Could you clarify:",
                    reply_markup=reply_markup
                )
                return

        # If response passes fallback checks, submit answer normally
        # Reset clarification flags
        session.pop("awaiting_clarification", None)
        session.pop("clarification_stage", None)
        session_id = session.get("session_id")
        question_id = session.get("current_question", {}).get("id")
        if session_id and question_id:
            # Ensure session_id and question_id are strings, not UUID objects
            session_id_str = str(session_id) if session_id else None
            question_id_str = str(question_id) if question_id else None

            response = api_call(f"/interview-sessions/{session_id_str}/respond", "POST", {
                "session_id_form": session_id_str,
                "question_id": question_id_str,
                "response_text": update.message.text,  # Use original text
                "response_time_seconds": 30.0  # Simplified
            }, use_form_data=True)

            if response.get("success"):
                next_action = response.get("next_action", {})

                # Enhanced logic to handle non-standard responses and prevent premature termination
                if next_action.get("action") == "next_question":
                    session["current_question"] = response.get("next_question")
                    await update.message.reply_text(f"Question: {response['next_question']['question_text']}\n\n(You have 5 minutes to respond)")
                elif next_action.get("action") == "interview_completed":
                    # Only complete if we've actually gone through all questions or user explicitly ends
                    # Check if this is a premature completion due to non-standard response
                    response_text_lower = update.message.text.lower().strip()

                    # Don't end interview for non-standard responses - continue to next question instead
                    if any(phrase in response_text_lower for phrase in [
                        "don't want to answer", "don't know", "not sure", "prefer not to say",
                        "can't answer", "no comment", "pass", "skip", "decline"
                    ]):
                        await update.message.reply_text("I understand you prefer not to answer this question. That's perfectly fine - let's continue with the next question.")
                        await get_next_question(update, session)
                    else:
                        # Legitimate completion
                        await update.message.reply_text("Interview complete! Thank you for participating.")
                        session["step"] = "completed"
                elif next_action.get("action") == "follow_up":
                    # Handle follow-up questions dynamically
                    follow_up_question = response.get("follow_up_question")
                    if follow_up_question:
                        session["current_question"] = follow_up_question
                        await update.message.reply_text(f"Follow-up: {follow_up_question['question_text']}\n\n(You have 5 minutes to respond)")
                    else:
                        await update.message.reply_text("Thank you for your response. Let me ask a follow-up question...")
                        # Fallback: generate a contextual follow-up
                        await handle_followup_question(update, session, update.message.text)
                elif next_action.get("action") == "clarification":
                    # Handle clarification requests
                    clarification_msg = response.get("clarification_message", "Could you please clarify your response?")
                    await update.message.reply_text(f"{clarification_msg}\n\n(You have 3 minutes to respond)")
                    session["awaiting_clarification"] = True
                else:
                    # Enhanced fallback: always try to continue the interview
                    if response.get("next_question"):
                        session["current_question"] = response.get("next_question")
                        await update.message.reply_text(f"Question: {response['next_question']['question_text']}\n\n(You have 5 minutes to respond)")
                    else:
                        await update.message.reply_text("Thank you for your response! Let me continue with the next question.")
                        await get_next_question(update, session)
            else:
                # Handle API failure more gracefully
                error_msg = response.get("message", "Failed to submit response")
                await update.message.reply_text(f"I'm having trouble processing your response: {error_msg}. Let me try again...")
                # Retry once after a short delay
                await asyncio.sleep(1)
                retry_response = api_call(f"/interview-sessions/{session_id_str}/respond", "POST", {
                    "session_id_form": session_id_str,
                    "question_id": question_id_str,
                    "response_text": update.message.text,
                    "response_time_seconds": 30.0
                }, use_form_data=True)

                if retry_response.get("success"):
                    # Handle successful retry with same enhanced logic
                    next_action = retry_response.get("next_action", {})
                    if next_action.get("action") == "next_question":
                        session["current_question"] = retry_response.get("next_question")
                        await update.message.reply_text(f"Question: {retry_response['next_question']['question_text']}\n\n(You have 5 minutes to respond)")
                    elif next_action.get("action") == "interview_completed":
                        # Apply same logic for retry responses
                        response_text_lower = update.message.text.lower().strip()
                        if any(phrase in response_text_lower for phrase in [
                            "don't want to answer", "don't know", "not sure", "prefer not to say",
                            "can't answer", "no comment", "pass", "skip", "decline"
                        ]):
                            await update.message.reply_text("I understand you prefer not to answer this question. That's perfectly fine - let's continue with the next question.")
                            await get_next_question(update, session)
                        else:
                            await update.message.reply_text("Response recorded successfully! Please wait for the next question.")
                    else:
                        await update.message.reply_text("Response recorded successfully! Please wait for the next question.")
                else:
                    await update.message.reply_text("I'm still having issues. Please try again or contact support if the problem persists.")
    else:
        await update.message.reply_text("Welcome! Reply with 'start' to begin an interview or use /start to set up your profile.")

# Handle document uploads (resume)
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.username if update.effective_user.username else str(update.effective_user.id)
    session = user_sessions.get(user_id, {})
    if session.get("step") == "resume_upload":
        document = update.message.document
        if document.mime_type == "application/pdf":
            file = await document.get_file()
            file_data = await file.download_as_bytearray()
            # Upload resume
            files = {"file": ("resume.pdf", file_data, "application/pdf")}
            data = {"user_id": BOT_USER_ID}
            response = api_call("/upload-resume", "POST", data, files)
            if response.get("success"):
                resume_id = response.get("user_id")  # Assuming user_id is resume_id
                session["resume_id"] = resume_id
                await update.message.reply_text("Resume uploaded successfully! Now proceeding to interview.")
                # Fetch questions
                questions_response = api_call(f"/users/{BOT_USER_ID}/questions")
                if questions_response:
                    question_ids = [q["id"] for q in questions_response]
                else:
                    question_ids = []  # Fallback
                interview_data = {
                    "user_id": BOT_USER_ID,
                    "title": f"Interview for {session['data']['name']}",
                    "description": "Telegram interview",
                    "question_ids": question_ids,
                    "candidate_ids": [resume_id]
                }
                interview = api_call(f"/users/{BOT_USER_ID}/interviews", "POST", interview_data)
                if interview:
                    session["interview_id"] = interview["id"]
                    # Create session
                    session_data = {
                        "user_id": BOT_USER_ID,
                        "session_type": "live",
                        "question_ids": interview_data["question_ids"],
                        "candidate_ids": [resume_id]
                    }
                    # Assuming session creation endpoint exists; adjust if needed
                    # For now, assume interview_id is session_id or create separately
                    session["session_id"] = interview["id"]  # Simplified
                    # Start session
                    start_response = api_call(f"/interview-sessions/{session['session_id']}/start", "POST", {"session_id": session["session_id"]})
                    if start_response.get("success"):
                        session["current_question"] = start_response.get("current_question")
                        session["step"] = "interview"
                        await update.message.reply_text(f"Interview started!\nQuestion: {start_response['current_question']['question_text']}\n(You have 5 minutes)")
                    else:
                        await update.message.reply_text("Failed to start interview.")
                else:
                    await update.message.reply_text("Failed to create interview.")
            else:
                await update.message.reply_text("Upload failed. Try again.")
        else:
            await update.message.reply_text("Please upload a PDF resume.")

# Handle callback for job selection and disambiguation
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.username if query.from_user.username else str(query.from_user.id)
    session = user_sessions.get(user_id, {})

    if query.data.startswith("disambig_"):
        # Handle disambiguation selection
        _, index_str = query.data.split("_", 1)
        choice_index = int(index_str)

        # Different responses based on user's choice
        if choice_index == 0:  # "Did you mean to answer the interview question?"
            await query.edit_message_text(
                "Great! Please provide a proper answer to the interview question:\n\n"
                f"{session.get('current_question', {}).get('question_text', '')}\n\n"
                "(You have 5 minutes to respond)"
            )
        elif choice_index == 1:  # "Are you providing a complete response to the question?"
            await query.edit_message_text(
                "Please elaborate on your response to give me a complete answer:\n\n"
                f"{session.get('current_question', {}).get('question_text', '')}\n\n"
                "Try to provide more details. (You have 5 minutes to respond)"
            )
        elif choice_index == 2:  # "Would you like me to rephrase the question differently?"
            # Rephrase and show the question again
            current_question = session.get('current_question', {}).get('question_text', '')
            rephrased = fallback_manager.classifier._rephrase_question(current_question)
            await query.edit_message_text(
                f"Sure, let me rephrase that:\n\n{rephrased}\n\n"
                "(You have 5 minutes to respond)"
            )
        else:
            await query.edit_message_text(
                "Please provide an answer to the interview question.\n\n"
                f"{session.get('current_question', {}).get('question_text', '')}\n\n"
                "(You have 5 minutes to respond)"
            )

        # Reset any clarification flags
        session.pop("awaiting_clarification", None)
        session.pop("clarification_stage", None)

    elif query.data.startswith("job_"):
        job_id = query.data.split("_")[1]
        session["job_id"] = job_id
        await query.edit_message_text("Job selected. Now, please upload your resume (PDF):")
        session["step"] = "resume_upload"
    else:
        await query.edit_message_text("Unknown selection. Please try again.")

# /status command (admin)
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.username if update.effective_user.username else str(update.effective_user.id)
    if str(user_id) in ["123456789"]:  # Admin IDs
        target_user = str(context.args[0]) if context.args else user_id
        session = user_sessions.get(target_user, {})
        await update.message.reply_text(f"Status for {target_user}: {session.get('step', 'No session')}")
    else:
        await update.message.reply_text("Unauthorized.")

# Main function
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(CallbackQueryHandler(handle_callback))

    application.run_polling()

if __name__ == "__main__":
    main()


# Minimal no-op orchestrator to satisfy imports from other modules that expect an
# onboarding orchestrator. The project uses the telegram bot directly to start
# interviews on '/start', so onboarding is intentionally a no-op here.
class Orchestrator:
    """Lightweight placeholder orchestrator with expected async start interface."""
    async def start(self, *args, **kwargs):
        # Intentionally do nothing; interviews are started directly via bot handlers
        return None


# Export an instance named 'orchestrator' so 'from telegram_agent import orchestrator'
# won't fail. This keeps compatibility with older code that imported an
# Onboarding orchestrator while disabling onboarding behavior.
orchestrator = Orchestrator()
