# pip install dotenv fastapi uvicorn groq pydantic supabase

import os
import re
import csv
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Deque, AsyncGenerator
from collections import deque
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://gsbomifazneansuzgkua.supabase.co"
# In production, put this key in your .env file!
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdzYm9taWZhem5lYW5zdXpna3VhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwOTQzMjYsImV4cCI6MjA3MDY3MDMyNn0.bwr5Gk0C_w7-Sj09Fskg3fbSRPZV0izUE3XOPcyJAGA")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- GROQ CONFIGURATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MAX_HISTORY_LEN = 20 

# --- CONTEXT & PROMPTS ---
app_context = """
Peepal applications Portal App Context: PEEPAL Medical Training College
[... Context content remains the same ...]
"""

COURSE_DATA = """
# diploma  courses
1. Diploma in Peri-operative Theatre Technology
2. Diploma in Community Health and Development
3. Diploma in Counselling Psychology
Diploma courses take 2 years in total, 4 semesters.

# certificate courses
1. Certificate in Peri-operative Theatre Technology
2. Certificate in Community Health and Development
3. Certificate in Counselling Psychology
4. Certificate in Nursing Assistant(Health Support Services/CNA)
Certificate courses take 1 year in total, 3 trimesters

# short courses
1. Community Health and Development (6 months)
2. caregiving (6 months)
3. Homecare (6 months)
4. computer application for healthcare professionals.(3 months)
short courses take 3 - 6 months

# Entry grades
diploma courses: KCSE mean grade C- (minus) and above
certificate courses: KCSE mean grade D (plain) and above
short courses: No formal entry requirements
"""

def get_nairobi_time():
    """Returns current formatted time in Nairobi (UTC+3)"""
    utc_now = datetime.now(timezone.utc)
    nairobi_time = utc_now + timedelta(hours=3)
    return nairobi_time.strftime("%A, %d %B %Y, %I:%M %p")

def get_system_prompt():
    current_time = get_nairobi_time()
    
    return f"""
You are PeepalBot, the admissions assistant for Peepal Medical Training College (PMTC).
CURRENT DATE/TIME: {current_time}

OFFICIAL MEDICAL COURSES:
{COURSE_DATA}

INTAKE DATES: Jan, Mar, Jun, Sep, Dec.

INTERNAL LOGGING INSTRUCTIONS (CRITICAL):
We need to track user intent for the school administration.
At the end of your response (or in the middle if relevant), you can add a secret note to the school.
Wrap this note STRICTLY in triple ticks like this: ''' User is asking about fees but seems hesitant '''
This text will be HIDDEN from the user but logged for the school.
Use this to highlight: Leads, Financial Issues, specific course interests, or complaints.

GENERAL INSTRUCTIONS:
1. STRICTLY LIMIT answers to medical courses.
2. Speak natural Kenyan Swahili/English/Sheng mix.
3. Keep answers short.
4. If user asks for fees/intake, ask for phone number.
5. Payment is through bank or Mpesa only. No cash is accepted at school.
"""

sessions: Dict[str, Deque[dict]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

# --- ASYNC LOGGING FUNCTIONS ---

async def log_to_supabase_async(session_id: str, role: str, content: str):
    """Async wrapper to write to Supabase without blocking the main thread"""
    try:
        data = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        # Run the sync supabase call in a thread to keep async loop moving
        await asyncio.to_thread(lambda: supabase.table("chat_logs").insert(data).execute())
    except Exception as e:
        print(f"Supabase Error: {e}")

def check_and_save_lead(message: str, session_id: str):
    """Legacy CSV logging"""
    phone_pattern = re.compile(r'(?:\+254|0)?(7\d{8}|1\d{8})')
    match = phone_pattern.search(message)
    
    if match:
        phone_number = match.group()
        file_exists = os.path.isfile('leads.csv')
        try:
            with open('leads.csv', 'a', newline='') as csvfile:
                fieldnames = ['session_id', 'phone', 'full_message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'session_id': session_id,
                    'phone': phone_number,
                    'full_message': message
                })
        except Exception as e:
            print(f"Error saving lead: {e}")

# --- STREAM GENERATOR WITH "AI LOG" PARSING ---

async def stream_and_log_generator(message: str, session_id: str) -> AsyncGenerator[str, None]:
    # 1. Initialize Session
    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=MAX_HISTORY_LEN)

    # 2. Log User Message to Supabase immediately
    await log_to_supabase_async(session_id, "user", message)
    
    sessions[session_id].append({"role": "user", "content": message})

    # 3. Call Groq
    system_instruction = {"role": "system", "content": get_system_prompt()}
    messages_payload = [system_instruction] + list(sessions[session_id])

    stream = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=messages_payload,
        temperature=0.5, 
        max_tokens=512,
        stream=True
    )

    # 4. Stream Processing Logic
    full_assistant_response_visible = "" # What the user sees
    accumulated_chunk = ""
    is_inside_log = False
    current_log_content = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if not content:
            continue
            
        accumulated_chunk += content
        
        # State Machine to handle Triple Ticks
        while True:
            if not is_inside_log:
                # We are in normal mode, looking for opening ticks
                if "'''" in accumulated_chunk:
                    # Found start of log
                    part_before, part_after = accumulated_chunk.split("'''", 1)
                    
                    # Send the part before the ticks to the user
                    if part_before:
                        full_assistant_response_visible += part_before
                        yield part_before
                    
                    # Switch to log mode
                    is_inside_log = True
                    accumulated_chunk = part_after # Continue processing remainder
                else:
                    # No ticks found, check if we might be in the middle of a tick sequence (e.g. ended with ' or '')
                    # To be safe against splitting tick tokens, we keep a tiny buffer if it ends with '
                    if accumulated_chunk.endswith("'") and len(accumulated_chunk) < 3:
                        break # Wait for next chunk to confirm if it's a triple tick
                    
                    # Safe to send
                    full_assistant_response_visible += accumulated_chunk
                    yield accumulated_chunk
                    accumulated_chunk = ""
                    break
            
            else:
                # We are INSIDE a log, looking for closing ticks
                if "'''" in accumulated_chunk:
                    # Found end of log
                    log_text, remainder = accumulated_chunk.split("'''", 1)
                    current_log_content += log_text
                    
                    # SAVE THE LOG TO SUPABASE
                    if current_log_content.strip():
                        print(f"AI LOG CAPTURED: {current_log_content}")
                        await log_to_supabase_async(session_id, "ai_log", current_log_content.strip())
                    
                    # Reset log state
                    current_log_content = ""
                    is_inside_log = False
                    accumulated_chunk = remainder # Process remainder as normal text
                else:
                    # No closing tick yet, buffer everything
                    current_log_content += accumulated_chunk
                    accumulated_chunk = ""
                    break

    # Flush any remaining buffer (edge case if they opened ticks but never closed them)
    if accumulated_chunk:
        if is_inside_log:
             # Log whatever was left as a log
            current_log_content += accumulated_chunk
            await log_to_supabase_async(session_id, "ai_log", current_log_content.strip())
        else:
            full_assistant_response_visible += accumulated_chunk
            yield accumulated_chunk

    # 5. Log final visible response to Supabase & History
    sessions[session_id].append({"role": "assistant", "content": full_assistant_response_visible})
    await log_to_supabase_async(session_id, "assistant", full_assistant_response_visible)

# --- ROUTES ---

@app.get("/wake")
async def wake_up():
    return {
        "status": "alive", 
        "time": get_nairobi_time() 
    }

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    # CSV logging (Legacy/Backup)
    background_tasks.add_task(check_and_save_lead, req.message, req.session_id)
    
    # Return the smart stream that handles logging
    return StreamingResponse(
        stream_and_log_generator(req.message, req.session_id), 
        media_type="text/plain"
    )
