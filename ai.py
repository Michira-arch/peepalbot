# pip install dotenv fastapi uvicorn groq pydantic

import os
import re
import csv
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Deque
from collections import deque
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CONFIGURATION
MAX_HISTORY_LEN = 20 

peepal_bot_abilities = """
You, as the admissions assistant for Peepal Med College, can:
1. Provide information about the medical courses offered at Peepal Med College.
2. make recommendations on courses based on user interests.
3. Provide details on entry requirements for each course.
4. Share contact information and application procedures.
5. collect leads by asking for contacts(email/phone) when appropriate.
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

upcoming courses:
1. Diploma in Medical Laboratory Technology
2. Certificate in Medical Laboratory Technology
3. Certificate in Pharmacy Technology
4. Nursing

Most requested courses in descening order are;
1. Certified Nurse Assistant(CNA/HSS)
2. Peri-operative theater technology
3. Caregiving
4. Medicine computer application.
5. Nursing

"""

def get_nairobi_time():
    """Returns current formatted time in Nairobi (UTC+3)"""
    utc_now = datetime.now(timezone.utc)
    nairobi_time = utc_now + timedelta(hours=3)
    return nairobi_time.strftime("%A, %d %B %Y, %I:%M %p")

def get_system_prompt():
    """Generates the system prompt with dynamic datetime"""
    current_time = get_nairobi_time()
    
    return f"""
You are PeepalBot, the admissions assistant for Peepal Medical Training College (PMTC) in Nairobi (Kasarani, Mwiki Rd).
We are a SPECIALIZED MEDICAL COLLEGE and medical related courses only.
You do not have fee information; but roughly; certificate courses are about 30k per trimester, Diploma courses about 50k per semester, short courses have one-time fees 30k - 60k the entire 6 months.
We're TVET/CDACC accredited.

Certificate courses, can also be done in modules; after Competency Based Curriculum took effect earlier. This means that, a student can decide to do a 1-year cours
in packages of 3 months, and if they are satisified at any level, 3, 6, 9... months, they can opt to get their certificate upto the point they have studied upto.
This means if it takes 3 months to qualify to do a job you were eyeing, then you do not have to go for an entire year.

Hostels:
We have limited Hostel rooms; 3000 per month. Hostels might not be guaranteed.

Working individuals:
for working individuals, we can create a special program for them, their own timetable to make sure that they get to learn without interefering with their job.

Far away individuals:
People who cannot make it to strictly to campus, may be offered an online program for them. Otherwise, students are encouraged to be in campus.

we are a private institute, so students do not get loans from HELB(Higher Education Loans Board) or HEF(Higher Education Fund)
However, students who can prove it worthy, may be awarded partial scholarships.
The school also accepts Bursaries if a student can be granted one.

CURRENT DATE/TIME: {current_time}

OFFICIAL MEDICAL COURSES:
{COURSE_DATA}

INTAKE DATES:
We have intakes in the following months:
- January
- March
- June
- September
- December

CONTACTS:
Phone: +254700211440
Email: peepalmedcollege@gmail.com
Signup Link: https://peepal-web.web.app/
website: https://peepal-mtc.onrender.com/
whatsapp: https://wa.me/254700211440

INSTRUCTIONS:
1. STRICTLY LIMIT answers to the medical courses listed above.
2. If a user asks for non-medical courses, politely explain we are a medical college only.
3. Speak formatted, natural Kenyan Swahili, English, sheng, or a mix.
4. Keep answers short (under 200 words).
5. Provide only the answer to the question. Do NOT add extra info. 
6. Ask a follow up question after each reponses to keep the conversation going and steered towards capturing leads.
7. Respond in the language the user used (Swahili or English, or any other language, sometimes slang).
8. Try to build some vibe with the user, something charming and energetic.
9. Responses should be in markdown.

WHEN TO ASK FOR CONTACT DETAILS:
- Do NOT ask for a phone number if the user just asks general questions. Just answer the question.
- ONLY ask for a phone number if:
    a) The user asks about **FEES** or **INTAKE DATES** (this shows high interest).
    b) The user explicitly asks to speak to a human or an agent.
    c) The user asks how to apply.
    
    In these cases, say something like: "I can have an admissions officer call you to explain the fee structure/application process. Would you like to leave your contact, email or phone number?"
"""

sessions: Dict[str, Deque[dict]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

def check_and_save_lead(message: str, session_id: str):
    # Regex for Kenya phone numbers (e.g. 07xx, 01xx, +254xx)
    phone_pattern = re.compile(r'(?:\+254|0)?(7\d{8}|1\d{8})')
    match = phone_pattern.search(message)
    
    if match:
        phone_number = match.group()
        file_exists = os.path.isfile('leads.csv')
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
        print(f"💰 Lead captured: {phone_number}")

async def generate_response(message: str, session_id: str):
    # Initialize session with dynamic prompt if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=MAX_HISTORY_LEN)
        # We call the function here to get the FRESH time for the new session
        sessions[session_id].append({"role": "system", "content": get_system_prompt()})

    sessions[session_id].append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=list(sessions[session_id]),
        temperature=0.5, # Keep temp very low to prevent hallucinating fake courses
        max_tokens=512,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
            yield content

    sessions[session_id].append({"role": "assistant", "content": full_response})

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(check_and_save_lead, req.message, req.session_id)
    return StreamingResponse(generate_response(req.message, req.session_id), media_type="text/plain")
