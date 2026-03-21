"""
NextGen-HR — Hybrid Adaptive Interview Backend
FastAPI + MongoDB + GCP Cloud Storage + Gemini AI
Algorithms: TF-IDF, Thompson Sampling (MAB), IRT (Newton-Raphson)
"""

import os, uuid, json, re, math, hashlib, time
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB async
import motor.motor_asyncio

# Google Gemini
import google.generativeai as genai

# GCP Cloud Storage
from google.cloud import storage as gcs_storage
from google.oauth2 import service_account as gcs_sa

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGODB_URI    = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME        = os.getenv("DB_NAME", "nextgen_hr")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
GCS_SA_JSON    = os.getenv("GCS_SERVICE_ACCOUNT_JSON", "")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DB_NAME]

# Collections
hr_users_col      = db["hr_users"]
candidates_col    = db["candidates"]
jobs_col          = db["jobs"]
applications_col  = db["applications"]
interviews_col    = db["interviews"]
irt_items_col     = db["irt_items"]
sessions_col      = db["sessions"]

# ══════════════════════════════════════════════════════════════════════════════
#  GCP CLOUD STORAGE
# ══════════════════════════════════════════════════════════════════════════════
gcs_client = None
gcs_bucket_obj = None

def init_gcs():
    global gcs_client, gcs_bucket_obj
    if GCS_BUCKET and GCS_SA_JSON and os.path.exists(GCS_SA_JSON):
        creds = gcs_sa.Credentials.from_service_account_file(GCS_SA_JSON)
        gcs_client = gcs_storage.Client(credentials=creds)
        gcs_bucket_obj = gcs_client.bucket(GCS_BUCKET)

init_gcs()

async def upload_to_gcs(file_bytes: bytes, destination: str, content_type: str = "application/pdf") -> str:
    """Upload file to GCP Cloud Storage, return public URL."""
    if not gcs_bucket_obj:
        # Fallback: store locally if GCS not configured
        local_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, destination.replace("/", "_"))
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        return f"/uploads/{destination.replace('/', '_')}"
    
    blob = gcs_bucket_obj.blob(destination)
    blob.upload_from_string(file_bytes, content_type=content_type)
    blob.make_public()
    return blob.public_url

def get_resume_signed_url(destination: str) -> str:
    """Generate a signed URL for resume download."""
    if not gcs_bucket_obj:
        return destination  # local path fallback
    blob = gcs_bucket_obj.blob(destination)
    url = blob.generate_signed_url(expiration=timedelta(hours=2))
    return url

# ══════════════════════════════════════════════════════════════════════════════
#  APP LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create indexes
    await hr_users_col.create_index("email", unique=True)
    await candidates_col.create_index("email", unique=True)
    await jobs_col.create_index("hr_id")
    await applications_col.create_index([("job_id", 1), ("candidate_id", 1)], unique=True)
    await interviews_col.create_index("application_id", unique=True)
    await sessions_col.create_index("token", unique=True)
    yield

app = FastAPI(title="NextGen-HR Adaptive Interview API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local uploads if GCS not configured
from fastapi.staticfiles import StaticFiles
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed

async def get_session(token: str):
    if not token:
        return None
    s = await sessions_col.find_one({"token": token})
    return s

async def require_hr(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing auth token")
    token = authorization.replace("Bearer ", "")
    s = await get_session(token)
    if not s or s.get("role") != "hr":
        raise HTTPException(403, "HR access required")
    return s

async def require_candidate(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing auth token")
    token = authorization.replace("Bearer ", "")
    s = await get_session(token)
    if not s or s.get("role") != "candidate":
        raise HTTPException(403, "Candidate access required")
    return s

async def require_any_auth(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing auth token")
    token = authorization.replace("Bearer ", "")
    s = await get_session(token)
    if not s:
        raise HTTPException(401, "Invalid token")
    return s

# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
class HRRegister(BaseModel):
    name: str
    email: str
    password: str
    company: str

class CandidateRegister(BaseModel):
    name: str
    email: str
    password: str

class LoginReq(BaseModel):
    email: str
    password: str

class PasswordReset(BaseModel):
    email: str
    new_password: str
    role: str

class JobCreate(BaseModel):
    title: str
    company: str
    description: str
    resumeThreshold: float = 0.38
    interviewThreshold: float = 0.54
    maxQ: int = 10

class AnswerSubmit(BaseModel):
    application_id: str
    answer: str

class InterviewStart(BaseModel):
    application_id: str

# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NextGen-HR Adaptive Interview", "mongo": True, "gcs": gcs_bucket_obj is not None}

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/hr/register")
async def hr_register(req: HRRegister):
    existing = await hr_users_col.find_one({"email": req.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    hr_id = str(uuid.uuid4())
    await hr_users_col.insert_one({
        "_id": hr_id, "name": req.name, "email": req.email,
        "password_hash": hash_password(req.password), "company": req.company,
        "created_at": datetime.utcnow().isoformat(),
    })
    token = str(uuid.uuid4())
    await sessions_col.insert_one({"token": token, "user_id": hr_id, "role": "hr", "name": req.name, "email": req.email, "company": req.company})
    return {"token": token, "user": {"id": hr_id, "name": req.name, "email": req.email, "company": req.company, "role": "hr"}}

@app.post("/api/auth/hr/login")
async def hr_login(req: LoginReq):
    user = await hr_users_col.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    token = str(uuid.uuid4())
    await sessions_col.insert_one({"token": token, "user_id": user["_id"], "role": "hr", "name": user["name"], "email": user["email"], "company": user.get("company", "")})
    return {"token": token, "user": {"id": user["_id"], "name": user["name"], "email": user["email"], "company": user.get("company", ""), "role": "hr"}}

@app.post("/api/auth/candidate/register")
async def candidate_register(req: CandidateRegister):
    existing = await candidates_col.find_one({"email": req.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    cand_id = str(uuid.uuid4())
    await candidates_col.insert_one({
        "_id": cand_id, "name": req.name, "email": req.email,
        "password_hash": hash_password(req.password),
        "created_at": datetime.utcnow().isoformat(),
    })
    token = str(uuid.uuid4())
    await sessions_col.insert_one({"token": token, "user_id": cand_id, "role": "candidate", "name": req.name, "email": req.email})
    return {"token": token, "user": {"id": cand_id, "name": req.name, "email": req.email, "role": "candidate"}}

@app.post("/api/auth/candidate/login")
async def candidate_login(req: LoginReq):
    user = await candidates_col.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    token = str(uuid.uuid4())
    await sessions_col.insert_one({"token": token, "user_id": user["_id"], "role": "candidate", "name": user["name"], "email": user["email"]})
    return {"token": token, "user": {"id": user["_id"], "name": user["name"], "email": user["email"], "role": "candidate"}}

@app.post("/api/auth/reset-password")
async def reset_password(req: PasswordReset):
    col = hr_users_col if req.role == "hr" else candidates_col
    user = await col.find_one({"email": req.email})
    if not user:
        raise HTTPException(404, "User not found")
    
    await col.update_one(
        {"_id": user["_id"]},
        {"$set": {"password_hash": hash_password(req.new_password)}}
    )
    return {"message": "Password reset successfully"}

@app.get("/api/auth/me")
async def auth_me(session=Depends(require_any_auth)):
    return {"user": {"id": session["user_id"], "name": session["name"], "email": session["email"], "role": session["role"], "company": session.get("company", "")}}

# ══════════════════════════════════════════════════════════════════════════════
#  HR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/hr/jobs")
async def hr_create_job(req: JobCreate, session=Depends(require_hr)):
    job_id = str(uuid.uuid4())
    job = {
        "_id": job_id,
        "hr_id": session["user_id"],
        "title": req.title,
        "company": req.company,
        "description": req.description,
        "resumeThreshold": req.resumeThreshold,
        "interviewThreshold": req.interviewThreshold,
        "maxQ": req.maxQ,
        "created_at": datetime.utcnow().isoformat(),
    }
    await jobs_col.insert_one(job)
    job["id"] = job.pop("_id")
    return {"job": job}

@app.get("/api/hr/jobs")
async def hr_list_jobs(session=Depends(require_hr)):
    cursor = jobs_col.find({"hr_id": session["user_id"]})
    jobs = []
    async for j in cursor:
        j["id"] = j.pop("_id")
        # Count candidates
        count = await applications_col.count_documents({"job_id": j["id"]})
        j["candidateCount"] = count
        jobs.append(j)
    return {"jobs": jobs}

@app.get("/api/hr/jobs/{job_id}/candidates")
async def hr_job_candidates(job_id: str, session=Depends(require_hr)):
    # Verify job belongs to this HR
    job = await jobs_col.find_one({"_id": job_id, "hr_id": session["user_id"]})
    if not job:
        raise HTTPException(404, "Job not found or not yours")
    
    cursor = applications_col.find({"job_id": job_id})
    candidates = []
    async for app_doc in cursor:
        # Get candidate info
        cand = await candidates_col.find_one({"_id": app_doc["candidate_id"]})
        # Get interview if exists
        interview = await interviews_col.find_one({"application_id": app_doc["_id"]})
        
        candidates.append({
            "application_id": app_doc["_id"],
            "candidate_id": app_doc["candidate_id"],
            "candidateName": cand["name"] if cand else "Unknown",
            "candidateEmail": cand["email"] if cand else "",
            "resumeScore": app_doc.get("resume_score", 0),
            "resumeUrl": app_doc.get("resume_url", ""),
            "resumeData": app_doc.get("resume_data", {}),
            "status": app_doc.get("status", "applied"),
            "appliedAt": app_doc.get("applied_at", ""),
            "interview": {
                "status": interview.get("status", "not_started") if interview else "not_started",
                "interviewScore": interview.get("final_score", 0) if interview else 0,
                "theta": interview.get("theta", 0) if interview else 0,
                "pass": interview.get("pass", False) if interview else False,
                "scoring": interview.get("scoring", {}) if interview else {},
                "qas": interview.get("qas", []) if interview else [],
                "report": interview.get("report", {}) if interview else {},
                "completedAt": interview.get("completed_at", "") if interview else "",
            } if interview else None,
        })
    
    job["id"] = job.pop("_id")
    return {"job": job, "candidates": candidates}

@app.get("/api/hr/candidates/{application_id}/report")
async def hr_candidate_report(application_id: str, session=Depends(require_hr)):
    app_doc = await applications_col.find_one({"_id": application_id})
    if not app_doc:
        raise HTTPException(404, "Application not found")
    
    # Verify the job belongs to this HR
    job = await jobs_col.find_one({"_id": app_doc["job_id"], "hr_id": session["user_id"]})
    if not job:
        raise HTTPException(403, "Not your candidate")
    
    cand = await candidates_col.find_one({"_id": app_doc["candidate_id"]})
    interview = await interviews_col.find_one({"application_id": application_id})
    
    # Get IRT items for the questions
    irt_snapshot = {}
    if interview and interview.get("qas"):
        for qa in interview["qas"]:
            irt_item = await irt_items_col.find_one({"_id": qa.get("qid", "")})
            if irt_item:
                irt_snapshot[qa["qid"]] = {"b": irt_item["b"], "a": irt_item["a"], "n": irt_item["n"]}
    
    return {
        "application": {
            "id": app_doc["_id"],
            "candidate_id": app_doc["candidate_id"],
            "candidateName": cand["name"] if cand else "Unknown",
            "candidateEmail": cand["email"] if cand else "",
            "resumeScore": app_doc.get("resume_score", 0),
            "resumeUrl": app_doc.get("resume_url", ""),
            "resumeData": app_doc.get("resume_data", {}),
            "status": app_doc.get("status", "applied"),
        },
        "job": {"id": job["_id"], "title": job["title"], "company": job["company"]},
        "interview": {
            "status": interview.get("status", "not_started") if interview else "not_started",
            "interviewScore": interview.get("final_score", 0) if interview else 0,
            "theta": interview.get("theta", 0) if interview else 0,
            "pass": interview.get("pass", False) if interview else False,
            "scoring": interview.get("scoring", {}) if interview else {},
            "qas": interview.get("qas", []) if interview else [],
            "questions": interview.get("questions", []) if interview else [],
            "report": interview.get("report", {}) if interview else {},
            "completedAt": interview.get("completed_at", "") if interview else "",
            "irtSnapshot": irt_snapshot,
        } if interview else None,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC JOB LISTING
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/jobs")
async def list_jobs():
    cursor = jobs_col.find({})
    jobs = []
    async for j in cursor:
        hr = await hr_users_col.find_one({"_id": j["hr_id"]})
        j["id"] = j.pop("_id")
        j["hrName"] = hr["name"] if hr else ""
        j["hrCompany"] = hr.get("company", j.get("company", ""))
        jobs.append(j)
    return {"jobs": jobs}

# ══════════════════════════════════════════════════════════════════════════════
#  CANDIDATE — APPLY (Resume upload + TF-IDF scoring)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/jobs/{job_id}/apply")
async def apply_to_job(
    job_id: str,
    resume_text: str = Form(...),
    resume_file: Optional[UploadFile] = File(None),
    session=Depends(require_candidate),
):
    job = await jobs_col.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Check if already applied
    existing = await applications_col.find_one({"job_id": job_id, "candidate_id": session["user_id"]})
    if existing:
        raise HTTPException(400, "Already applied to this job")
    
    # Upload resume to GCS if file provided
    resume_url = ""
    if resume_file:
        file_bytes = await resume_file.read()
        destination = f"resumes/{session['user_id']}/{job_id}/{resume_file.filename}"
        resume_url = await upload_to_gcs(file_bytes, destination)
    
    # TF-IDF Resume Scoring
    score_data = score_resume_tfidf(resume_text, job["description"])
    
    app_id = str(uuid.uuid4())
    app_doc = {
        "_id": app_id,
        "job_id": job_id,
        "hr_id": job["hr_id"],
        "candidate_id": session["user_id"],
        "resume_text": resume_text,
        "resume_url": resume_url,
        "resume_score": score_data["score"],
        "resume_data": score_data,
        "status": "resume_scored",
        "pass_resume": score_data["score"] >= job["resumeThreshold"],
        "applied_at": datetime.utcnow().isoformat(),
    }
    await applications_col.insert_one(app_doc)
    
    return {
        "application_id": app_id,
        "score": score_data["score"],
        "sim": score_data["sim"],
        "coverage": score_data["coverage"],
        "expScore": score_data["expScore"],
        "covered": score_data["covered"],
        "missing": score_data["missing"],
        "jdTerms": score_data["jdTerms"],
        "pass": score_data["score"] >= job["resumeThreshold"],
        "threshold": job["resumeThreshold"],
        "resume_url": resume_url,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  CANDIDATE — GET MY APPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/candidate/applications")
async def candidate_applications(session=Depends(require_candidate)):
    cursor = applications_col.find({"candidate_id": session["user_id"]})
    apps = []
    async for a in cursor:
        job = await jobs_col.find_one({"_id": a["job_id"]})
        interview = await interviews_col.find_one({"application_id": a["_id"]})
        apps.append({
            "id": a["_id"],
            "job_id": a["job_id"],
            "jobTitle": job["title"] if job else "",
            "company": job.get("company", "") if job else "",
            "resumeScore": a.get("resume_score", 0),
            "passResume": a.get("pass_resume", False),
            "status": a.get("status", "applied"),
            "interviewStatus": interview.get("status") if interview else "not_started",
            "interviewScore": interview.get("final_score") if interview else None,
            "pass": interview.get("pass") if interview else None,
            "appliedAt": a.get("applied_at", ""),
        })
    return {"applications": apps}

# ══════════════════════════════════════════════════════════════════════════════
#  TF-IDF RESUME SCORING
# ══════════════════════════════════════════════════════════════════════════════
def score_resume_tfidf(resume: str, job_desc: str) -> dict:
    """Score resume against job description using TF-IDF + cosine similarity."""
    if not resume.strip() or not job_desc.strip():
        return {"score": 0, "sim": 0, "coverage": 0, "expScore": 0, "covered": [], "missing": [], "jdTerms": []}
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([job_desc, resume])
        sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])
    except:
        sim = 0.0
    
    # Topic/keyword coverage
    jd_lower = job_desc.lower()
    resume_lower = resume.lower()
    
    # Extract key terms from JD
    jd_vectorizer = TfidfVectorizer(stop_words="english", max_features=30, ngram_range=(1, 2))
    try:
        jd_vectorizer.fit_transform([job_desc])
        jd_terms = list(jd_vectorizer.get_feature_names_out())
    except:
        jd_terms = []
    
    covered = [t for t in jd_terms if t.lower() in resume_lower]
    missing = [t for t in jd_terms if t.lower() not in resume_lower]
    coverage = len(covered) / max(len(jd_terms), 1)
    
    # Experience heuristic
    exp_patterns = [
        r'\b(\d+)\+?\s*(?:years?|yrs?)\b',
        r'\b(?:led|managed|built|designed|developed|deployed|created|architected)\b',
        r'\b\d+[%xX]\b',
        r'\b(?:senior|lead|principal|staff|head|director|manager)\b',
    ]
    exp_score = 0.0
    for pat in exp_patterns:
        if re.search(pat, resume, re.IGNORECASE):
            exp_score += 0.25
    exp_score = min(exp_score, 1.0)
    
    # Composite score: weighted combination
    score = 0.45 * sim + 0.35 * coverage + 0.20 * exp_score
    
    return {
        "score": round(score, 4),
        "sim": round(sim, 4),
        "coverage": round(coverage, 4),
        "expScore": round(exp_score, 4),
        "covered": covered,
        "missing": missing,
        "jdTerms": jd_terms,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  AI — QUESTION GENERATION (GEMINI)
# ══════════════════════════════════════════════════════════════════════════════
async def generate_questions_ai(job_description: str, num_questions: int = 12) -> list:
    """Generate interview questions from job description using Gemini."""
    if not GEMINI_API_KEY:
        return generate_questions_fallback(job_description, num_questions)
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""You are a senior technical interviewer. Generate exactly {num_questions} interview questions for this job:

JOB DESCRIPTION:
{job_description}

RULES:
- Mix behavioral, technical, and situational questions
- Cover different topics from the job description
- Questions should range from moderate to challenging
- Each question must have a unique topic tag

Return ONLY valid JSON array, no markdown, no explanation:
[
  {{"id": "q1_topic_slug", "text": "question text here", "topic": "Topic Name"}},
  ...
]"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean markdown fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        questions = json.loads(text)
        
        # Ensure IDs are unique
        seen = set()
        for q in questions:
            if q["id"] in seen:
                q["id"] = q["id"] + "_" + str(uuid.uuid4())[:4]
            seen.add(q["id"])
        
        return questions[:num_questions]
    except Exception as e:
        print(f"Gemini question generation error: {e}")
        return generate_questions_fallback(job_description, num_questions)

def generate_questions_fallback(job_description: str, num_questions: int = 12) -> list:
    """Fallback question generation using TF-IDF topic extraction."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=num_questions, ngram_range=(1, 2))
    try:
        vectorizer.fit_transform([job_description])
        topics = list(vectorizer.get_feature_names_out())
    except:
        topics = ["general skills", "experience", "problem solving"]
    
    templates = [
        "Describe your experience with {topic} and how you've applied it in a professional setting.",
        "What is your approach to {topic}? Walk me through a specific project.",
        "How would you handle a challenging situation involving {topic}?",
        "Explain how {topic} fits into your overall technical or professional strategy.",
        "Tell me about a time you had to learn {topic} quickly. What was the outcome?",
        "What best practices do you follow when working with {topic}?",
        "How do you stay current with developments in {topic}?",
        "Describe a complex problem you solved using {topic}.",
        "What metrics or KPIs do you track when working on {topic}?",
        "How would you mentor a junior team member on {topic}?",
        "What trade-offs have you encountered when working with {topic}?",
        "Can you walk me through your decision-making process for {topic}?",
    ]
    
    questions = []
    for i, topic in enumerate(topics[:num_questions]):
        template = templates[i % len(templates)]
        questions.append({
            "id": f"q{i+1}_{topic.replace(' ', '_')[:20]}",
            "text": template.format(topic=topic),
            "topic": topic.title(),
        })
    
    return questions

# ══════════════════════════════════════════════════════════════════════════════
#  AI — ANSWER SCORING (GEMINI)
# ══════════════════════════════════════════════════════════════════════════════
async def score_answer_ai(question: str, answer: str, job_description: str) -> dict:
    """Score a candidate's answer using Gemini AI + TF-IDF relevance."""
    # TF-IDF relevance score between answer and job description
    tfidf_relevance = 0.0
    if answer.strip() and job_description.strip():
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=3000)
            matrix = vec.fit_transform([job_description, question, answer])
            tfidf_relevance = float(cosine_similarity(matrix[0:1], matrix[2:3])[0, 0])
        except:
            tfidf_relevance = 0.0
    
    # Signal detection
    signals = {
        "hasNumbers": bool(re.search(r'\d+[%xX]|\b\d{2,}\b', answer)),
        "hasOutcome": bool(re.search(r'\b(?:result|outcome|impact|achieved|increased|decreased|reduced|improved|saved|generated|delivered)\b', answer, re.I)),
        "hasAction": bool(re.search(r'\b(?:I led|I built|I designed|I managed|I created|I developed|I implemented|my role|I was responsible)\b', answer, re.I)),
        "hasContext": bool(re.search(r'\b(?:at|during|while|when|project|team|company|organization)\b', answer, re.I)),
    }
    
    wc = len(answer.strip().split())
    
    if not GEMINI_API_KEY:
        # Fallback scoring
        base = min(wc / 80, 0.4) + (0.15 if signals["hasNumbers"] else 0) + (0.15 if signals["hasOutcome"] else 0) + (0.15 if signals["hasAction"] else 0) + 0.15 * tfidf_relevance
        return {"score": round(min(base, 1.0), 4), "tip": "Provide more specific details with quantifiable results.", "signals": signals, "wc": wc, "tfidfRelevance": round(tfidf_relevance, 4)}
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Score this interview answer. Be fair but rigorous.

QUESTION: {question}
ANSWER: {answer}
JOB CONTEXT: {job_description[:500]}

Return ONLY valid JSON, no markdown:
{{"score": 0.0-1.0, "tip": "one sentence improvement tip"}}

Scoring guide:
- 0.0-0.3: Off-topic, vague, no substance
- 0.3-0.5: Some relevance but lacks depth/specifics
- 0.5-0.7: Good answer with relevant details
- 0.7-0.85: Strong answer with specifics, metrics, clear impact
- 0.85-1.0: Exceptional — STAR method, quantified results, demonstrates mastery"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)
        
        ai_score = float(result.get("score", 0.5))
        tip = result.get("tip", "")
        
        # Hybrid: blend AI score with TF-IDF relevance
        hybrid_score = 0.70 * ai_score + 0.20 * tfidf_relevance + 0.10 * min(wc / 100, 1.0)
        
        return {
            "score": round(min(max(hybrid_score, 0), 1), 4),
            "aiScore": round(ai_score, 4),
            "tip": tip,
            "signals": signals,
            "wc": wc,
            "tfidfRelevance": round(tfidf_relevance, 4),
        }
    except Exception as e:
        print(f"Gemini scoring error: {e}")
        base = min(wc / 80, 0.4) + (0.15 if signals["hasNumbers"] else 0) + (0.15 if signals["hasOutcome"] else 0) + 0.15 * tfidf_relevance
        return {"score": round(min(base, 1.0), 4), "tip": "Could not reach AI scorer.", "signals": signals, "wc": wc, "tfidfRelevance": round(tfidf_relevance, 4)}

# ══════════════════════════════════════════════════════════════════════════════
#  IRT — ITEM RESPONSE THEORY (Newton-Raphson)
# ══════════════════════════════════════════════════════════════════════════════
def irt_prob(theta: float, b: float, a: float = 1.0) -> float:
    """2PL IRT probability of correct response."""
    z = a * (theta - b)
    z = max(min(z, 10), -10)  # clamp to prevent overflow
    return 1.0 / (1.0 + math.exp(-z))

def irt_update_item(b: float, a: float, n: int, score: float, theta: float, lr: float = 0.3) -> tuple:
    """Online gradient descent update for IRT item parameters."""
    p = irt_prob(theta, b, a)
    residual = score - p
    
    # Update difficulty b
    grad_b = -a * p * (1 - p)
    new_b = b - lr * residual * grad_b
    new_b = max(min(new_b, 4.0), -4.0)  # clamp
    
    # Update discrimination a
    grad_a = (theta - b) * p * (1 - p)
    new_a = a + lr * residual * grad_a
    new_a = max(new_a, 0.3)  # min discrimination
    new_a = min(new_a, 3.0)
    
    return round(new_b, 4), round(new_a, 4), n + 1

def estimate_theta_newton_raphson(responses: list, irt_items: dict, max_iter: int = 20) -> float:
    """Newton-Raphson MLE for theta (candidate ability)."""
    theta = 0.0
    
    for _ in range(max_iter):
        numerator = 0.0
        denominator = 0.0
        
        for r in responses:
            qid = r["qid"]
            score = r["score"]
            item = irt_items.get(qid, {"b": 0.0, "a": 1.0})
            b, a = item["b"], item["a"]
            
            p = irt_prob(theta, b, a)
            info = a * a * p * (1 - p)  # Fisher information
            
            numerator += a * (score - p)
            denominator += info
        
        if abs(denominator) < 1e-10:
            break
        
        delta = numerator / denominator
        theta += delta
        
        if abs(delta) < 1e-4:
            break
    
    return round(max(min(theta, 4.0), -4.0), 4)

def compute_final_score(qas: list, irt_items: dict) -> dict:
    """Compute final interview score using IRT + trend analysis."""
    if not qas:
        return {"score": 0, "theta": 0, "thetaNorm": 0.5, "infoWt": 0, "slope": 0, "raw": [], "progression": []}
    
    # Estimate theta
    responses = [{"qid": qa["qid"], "score": qa["score"]} for qa in qas]
    theta = estimate_theta_newton_raphson(responses, irt_items)
    
    # Normalize theta to 0-1
    theta_norm = 1.0 / (1.0 + math.exp(-theta))
    
    # Info-weighted average
    total_info = 0.0
    info_weighted_sum = 0.0
    raw_scores = []
    progression = []
    running = 0.0
    
    for i, qa in enumerate(qas):
        item = irt_items.get(qa["qid"], {"b": 0.0, "a": 1.0})
        p = irt_prob(theta, item["b"], item["a"])
        info = item["a"] ** 2 * p * (1 - p)
        
        info_weighted_sum += qa["score"] * max(info, 0.01)
        total_info += max(info, 0.01)
        raw_scores.append(qa["score"])
        running = (running * i + qa["score"]) / (i + 1)
        progression.append(running)
    
    info_wt = info_weighted_sum / max(total_info, 1e-6)
    
    # Slope (trend)
    if len(raw_scores) >= 2:
        x = np.arange(len(raw_scores), dtype=float)
        y = np.array(raw_scores, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0
    
    # Final composite
    score = 0.40 * theta_norm + 0.40 * info_wt + 0.20 * (0.5 + slope * 2)
    score = max(min(score, 1.0), 0.0)
    
    return {
        "score": round(score, 4),
        "theta": theta,
        "thetaNorm": round(theta_norm, 4),
        "infoWt": round(info_wt, 4),
        "slope": round(slope, 4),
        "raw": raw_scores,
        "progression": [round(p, 4) for p in progression],
    }

# ══════════════════════════════════════════════════════════════════════════════
#  THOMPSON SAMPLING — MULTI-ARMED BANDIT
# ══════════════════════════════════════════════════════════════════════════════
def thompson_select(bandit_arms: dict, available_qids: list, irt_items: dict, theta: float) -> str:
    """Select next question using Thompson Sampling weighted by IRT information."""
    if not available_qids:
        return ""
    
    best_qid = available_qids[0]
    best_val = -1.0
    
    for qid in available_qids:
        arm = bandit_arms.get(qid, {"alpha": 1.0, "beta": 1.0})
        # Thompson sample from Beta distribution
        sample = np.random.beta(arm["alpha"], arm["beta"])
        
        # Weight by IRT information at current theta
        item = irt_items.get(qid, {"b": 0.0, "a": 1.0})
        p = irt_prob(theta, item["b"], item["a"])
        info = item["a"] ** 2 * p * (1 - p)
        
        # Combined: exploration (Thompson) + information gain
        val = 0.6 * sample + 0.4 * info
        
        if val > best_val:
            best_val = val
            best_qid = qid
    
    return best_qid

def bandit_update(arms: dict, qid: str, reward: float) -> dict:
    """Update bandit arm with observed reward."""
    arm = arms.get(qid, {"alpha": 1.0, "beta": 1.0, "n": 0})
    arm["alpha"] += reward
    arm["beta"] += (1.0 - reward)
    arm["n"] = arm.get("n", 0) + 1
    arms[qid] = arm
    return arms

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_report(candidate_name: str, job_title: str, qas: list, scoring: dict, resume_data: dict) -> dict:
    """Build HR and candidate feedback reports."""
    avg_score = sum(q["score"] for q in qas) / max(len(qas), 1)
    all_tips = [q.get("tip", "") for q in qas if q.get("tip") and not q["tip"].startswith("Strong")]
    topics = list(set(q.get("topic", "") for q in qas))
    
    strengths = []
    if any(q.get("signals", {}).get("hasNumbers") for q in qas):
        strengths.append("Uses quantitative evidence effectively")
    if any(q.get("signals", {}).get("hasOutcome") for q in qas):
        strengths.append("Communicates results and business impact clearly")
    if any(q.get("signals", {}).get("hasAction") for q in qas):
        strengths.append("Answers are action-oriented with clear ownership")
    if scoring.get("slope", 0) > 0.02:
        strengths.append("Performance improved as the interview progressed")
    if avg_score > 0.65:
        strengths.append("Consistently strong across all topic areas")
    if not strengths:
        strengths.append("Completed all questions with effort")
    
    improves = []
    tip_counts = {}
    for t in all_tips:
        tip_counts[t] = tip_counts.get(t, 0) + 1
    for t, _ in sorted(tip_counts.items(), key=lambda x: -x[1])[:3]:
        improves.append(t)
    if not improves:
        improves.append("Continue practising structured answer frameworks (STAR method)")
    
    level = "exceptional" if scoring["score"] >= 0.80 else "strong" if scoring["score"] >= 0.65 else "solid" if scoring["score"] >= 0.50 else "developing"
    rec = "Strong hire" if scoring["score"] >= 0.68 else "Conditional hire" if scoring["score"] >= 0.53 else "Does not meet threshold"
    
    return {
        "candidateSummary": f"You demonstrated {level} performance (score: {scoring['score']*100:.1f}%). IRT ability estimate θ̂ = {scoring['theta']:.2f}. {'Your answers improved throughout — a strong signal.' if scoring.get('slope',0)>0.02 else 'Performance dipped under harder questions — work on depth under pressure.' if scoring.get('slope',0)<-0.02 else 'You were consistent throughout.'} Topics covered: {', '.join(topics[:4])}.",
        "hrSummary": f"{rec}. IRT θ̂ = {scoring['theta']:.2f} ({scoring['thetaNorm']*100:.0f}th percentile). Resume: {resume_data.get('score',0)*100:.1f}%. Questions auto-generated from job description. IRT trained from responses in real-time. Hybrid scoring: AI + TF-IDF + IRT Newton-Raphson.",
        "strengths": strengths[:4],
        "improves": improves[:4],
        "level": level,
        "rec": rec,
        "topics": topics,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW — START
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/interview/start")
async def interview_start(req: InterviewStart, session=Depends(require_candidate)):
    app_doc = await applications_col.find_one({"_id": req.application_id, "candidate_id": session["user_id"]})
    if not app_doc:
        raise HTTPException(404, "Application not found")
    if not app_doc.get("pass_resume"):
        raise HTTPException(400, "Resume did not pass threshold")
    
    # Check if interview already exists
    existing = await interviews_col.find_one({"application_id": req.application_id})
    if existing:
        if existing.get("status") == "completed":
            raise HTTPException(400, "Interview already completed")
        # Return existing active interview
        return format_interview_state(existing)
    
    job = await jobs_col.find_one({"_id": app_doc["job_id"]})
    if not job:
        raise HTTPException(404, "Job not found")
    
    # Generate questions via AI
    questions = await generate_questions_ai(job["description"], num_questions=max(job["maxQ"] + 4, 12))
    
    # Init IRT items for new questions
    for q in questions:
        existing_irt = await irt_items_col.find_one({"_id": q["id"]})
        if not existing_irt:
            await irt_items_col.insert_one({"_id": q["id"], "b": 0.0, "a": 1.0, "n": 0})
    
    # Load IRT items
    irt_items = {}
    for q in questions:
        item = await irt_items_col.find_one({"_id": q["id"]})
        irt_items[q["id"]] = {"b": item["b"], "a": item["a"], "n": item["n"]}
    
    # Init bandit arms
    bandit_arms = {}
    for q in questions:
        bandit_arms[q["id"]] = {"alpha": 1.0, "beta": 1.0, "n": 0}
    
    # Select first question via Thompson Sampling
    available = [q["id"] for q in questions]
    selected_qid = thompson_select(bandit_arms, available, irt_items, 0.0)
    first_q = next((q for q in questions if q["id"] == selected_qid), questions[0])
    
    interview_id = str(uuid.uuid4())
    interview = {
        "_id": interview_id,
        "application_id": req.application_id,
        "candidate_id": session["user_id"],
        "job_id": app_doc["job_id"],
        "hr_id": app_doc["hr_id"],
        "questions": questions,
        "bandit_arms": bandit_arms,
        "current_qid": first_q["id"],
        "asked": [first_q["id"]],
        "qas": [],
        "qNum": 1,
        "theta": 0.0,
        "maxQ": job["maxQ"],
        "status": "active",
        "started_at": datetime.utcnow().isoformat(),
    }
    await interviews_col.insert_one(interview)
    
    # Update application status
    await applications_col.update_one({"_id": req.application_id}, {"$set": {"status": "interviewing"}})
    
    return format_interview_state(interview, irt_items)

def format_interview_state(interview: dict, irt_items: dict = None) -> dict:
    """Format interview state for frontend."""
    current_q = next((q for q in interview["questions"] if q["id"] == interview["current_qid"]), None)
    return {
        "interview_id": interview["_id"],
        "status": interview["status"],
        "qNum": interview["qNum"],
        "maxQ": interview["maxQ"],
        "theta": interview["theta"],
        "currentQuestion": current_q,
        "qas": interview["qas"],
        "questions": interview["questions"],
        "irtItems": irt_items or {},
    }

# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW — SUBMIT ANSWER
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/interview/submit")
async def interview_submit(req: AnswerSubmit, session=Depends(require_candidate)):
    interview = await interviews_col.find_one({"application_id": req.application_id, "candidate_id": session["user_id"]})
    if not interview:
        raise HTTPException(404, "Interview not found")
    if interview["status"] != "active":
        raise HTTPException(400, "Interview not active")
    
    app_doc = await applications_col.find_one({"_id": req.application_id})
    job = await jobs_col.find_one({"_id": interview["job_id"]})
    
    current_q = next((q for q in interview["questions"] if q["id"] == interview["current_qid"]), None)
    if not current_q:
        raise HTTPException(500, "Current question not found")
    
    # 1. Score the answer with AI + TF-IDF hybrid
    ev = await score_answer_ai(current_q["text"], req.answer, job["description"])
    
    # 2. Load and update IRT item
    irt_item = await irt_items_col.find_one({"_id": current_q["id"]})
    if not irt_item:
        irt_item = {"_id": current_q["id"], "b": 0.0, "a": 1.0, "n": 0}
    
    new_b, new_a, new_n = irt_update_item(
        irt_item["b"], irt_item["a"], irt_item["n"],
        ev["score"], interview["theta"]
    )
    await irt_items_col.update_one(
        {"_id": current_q["id"]},
        {"$set": {"b": new_b, "a": new_a, "n": new_n}},
        upsert=True
    )
    
    # 3. Update bandit arm
    bandit_arms = interview["bandit_arms"]
    bandit_arms = bandit_update(bandit_arms, current_q["id"], ev["score"])
    
    # 4. Build QA record
    qa_record = {
        "qNum": interview["qNum"],
        "qid": current_q["id"],
        "question": current_q["text"],
        "topic": current_q.get("topic", ""),
        "answer": req.answer,
        "score": ev["score"],
        "aiScore": ev.get("aiScore", ev["score"]),
        "tip": ev.get("tip", ""),
        "signals": ev.get("signals", {}),
        "wc": ev.get("wc", 0),
        "tfidfRelevance": ev.get("tfidfRelevance", 0),
        "irtB": new_b,
        "irtA": new_a,
        "irtN": new_n,
    }
    
    updated_qas = interview["qas"] + [qa_record]
    
    # 5. Estimate new theta via Newton-Raphson
    irt_items = {}
    for q in interview["questions"]:
        item = await irt_items_col.find_one({"_id": q["id"]})
        if item:
            irt_items[q["id"]] = {"b": item["b"], "a": item["a"], "n": item["n"]}
        else:
            irt_items[q["id"]] = {"b": 0.0, "a": 1.0, "n": 0}
    
    new_theta = estimate_theta_newton_raphson(
        [{"qid": qa["qid"], "score": qa["score"]} for qa in updated_qas],
        irt_items
    )
    
    # ── CHECK IF DONE ──
    if interview["qNum"] >= interview["maxQ"]:
        scoring = compute_final_score(updated_qas, irt_items)
        is_pass = scoring["score"] >= job["interviewThreshold"]
        report = build_report(
            session["name"], job["title"], updated_qas, scoring,
            app_doc.get("resume_data", {})
        )
        
        await interviews_col.update_one({"_id": interview["_id"]}, {"$set": {
            "qas": updated_qas,
            "bandit_arms": bandit_arms,
            "theta": new_theta,
            "status": "completed",
            "final_score": scoring["score"],
            "pass": is_pass,
            "scoring": scoring,
            "report": report,
            "completed_at": datetime.utcnow().isoformat(),
        }})
        
        await applications_col.update_one({"_id": req.application_id}, {"$set": {
            "status": "completed",
            "interview_score": scoring["score"],
            "pass": is_pass,
        }})
        
        return {
            "done": True,
            "qa": qa_record,
            "scoring": scoring,
            "pass": is_pass,
            "report": report,
            "theta": new_theta,
            "irtItems": irt_items,
        }
    
    # ── NEXT QUESTION ──
    asked = set(interview["asked"])
    asked.add(current_q["id"])  # shouldn't be needed but safety
    available = [q["id"] for q in interview["questions"] if q["id"] not in asked]
    
    if not available:
        # Generate more questions
        new_questions = await generate_questions_ai(job["description"], num_questions=8)
        for q in new_questions:
            if q["id"] not in asked:
                existing_irt = await irt_items_col.find_one({"_id": q["id"]})
                if not existing_irt:
                    await irt_items_col.insert_one({"_id": q["id"], "b": 0.0, "a": 1.0, "n": 0})
                if q["id"] not in bandit_arms:
                    bandit_arms[q["id"]] = {"alpha": 1.0, "beta": 1.0, "n": 0}
                irt_items[q["id"]] = {"b": 0.0, "a": 1.0, "n": 0}
        
        all_questions = interview["questions"] + new_questions
        available = [q["id"] for q in new_questions if q["id"] not in asked]
        
        if not available:
            available = [new_questions[0]["id"]] if new_questions else [interview["questions"][0]["id"]]
    else:
        all_questions = interview["questions"]
    
    # Thompson Sampling for next question
    next_qid = thompson_select(bandit_arms, available, irt_items, new_theta)
    
    await interviews_col.update_one({"_id": interview["_id"]}, {"$set": {
        "questions": all_questions,
        "bandit_arms": bandit_arms,
        "current_qid": next_qid,
        "asked": list(asked) + [next_qid],
        "qas": updated_qas,
        "qNum": interview["qNum"] + 1,
        "theta": new_theta,
    }})
    
    next_q = next((q for q in all_questions if q["id"] == next_qid), all_questions[0])
    
    return {
        "done": False,
        "qa": qa_record,
        "nextQuestion": next_q,
        "qNum": interview["qNum"] + 1,
        "theta": new_theta,
        "irtItems": irt_items,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW — GET STATUS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/interview/status/{application_id}")
async def interview_status(application_id: str, session=Depends(require_candidate)):
    interview = await interviews_col.find_one({"application_id": application_id, "candidate_id": session["user_id"]})
    if not interview:
        return {"status": "not_started"}
    
    irt_items = {}
    for q in interview.get("questions", []):
        item = await irt_items_col.find_one({"_id": q["id"]})
        if item:
            irt_items[q["id"]] = {"b": item["b"], "a": item["a"], "n": item["n"]}
    
    return format_interview_state(interview, irt_items)

# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW — GET RESULTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/interview/results/{application_id}")
async def interview_results(application_id: str, session=Depends(require_candidate)):
    interview = await interviews_col.find_one({"application_id": application_id, "candidate_id": session["user_id"]})
    if not interview:
        raise HTTPException(404, "Interview not found")
    if interview["status"] != "completed":
        raise HTTPException(400, "Interview not yet completed")
    
    app_doc = await applications_col.find_one({"_id": application_id})
    job = await jobs_col.find_one({"_id": interview["job_id"]})
    
    irt_items = {}
    for q in interview.get("questions", []):
        item = await irt_items_col.find_one({"_id": q["id"]})
        if item:
            irt_items[q["id"]] = {"b": item["b"], "a": item["a"], "n": item["n"]}
    
    return {
        "candidateName": session["name"],
        "candidateEmail": session["email"],
        "jobTitle": job["title"] if job else "",
        "jobId": interview["job_id"],
        "resumeScore": app_doc.get("resume_score", 0) if app_doc else 0,
        "resumeData": app_doc.get("resume_data", {}) if app_doc else {},
        "interviewScore": interview.get("final_score", 0),
        "scoring": interview.get("scoring", {}),
        "pass": interview.get("pass", False),
        "qas": interview.get("qas", []),
        "questions": interview.get("questions", []),
        "report": interview.get("report", {}),
        "irtSnapshot": irt_items,
        "completedAt": interview.get("completed_at", ""),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
