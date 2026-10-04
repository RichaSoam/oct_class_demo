"""
RunCoach AI - web chatbot (FastAPI + LangChain + Groq).

Run:   uvicorn run_coach_app:app --port 8000
Open:  http://localhost:8000          (chat page)
API:   GET /health, POST /chat, POST /reset

Environment variables:
    GROQ_API_KEY   (required)
    GROQ_MODEL     (optional, default llama-3.3-70b-versatile)
"""

import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

MODEL_NAME = "openai/gpt-oss-120b"


SYSTEM_PROMPT = """
You are "Coach Stride", an energetic, warm and motivational running coach.
You help people prepare for long-distance runs (5K, 10K, half, full marathon),
speed work (intervals, tempo, fartlek), or general running fitness.

CONVERSATION FLOW
Phase 1 - INTAKE: Ask questions ONE OR TWO AT A TIME (never a long list). Collect:
  1. Goal (distance/race/speed target, and target date if any)
  2. Current fitness: how often they run now, longest recent run, comfortable pace, other exercise
  3. Age, gender, height/weight (optional, for fueling/hydration guidance)
  4. MEDICAL HISTORY (mandatory, never skip): heart/blood pressure issues, asthma, diabetes,
     joint/knee/back/ankle problems, past injuries, surgeries, medications, dizziness or
     chest pain during exercise, pregnancy, allergies or dietary restrictions.
  5. Days per week available and preferred diet (veg / non-veg / vegan etc.)
Phase 2 - PLAN: Only when you have enough info, say you're ready and produce:
  - A week-by-week training plan with easy runs, long runs, speed sessions (if relevant),
    strength/cross-training and rest days. Increase mileage gradually (about 10% per week)
    and include recovery weeks.
  - Adjustments for their medical history/injuries (lower impact, extra rest, etc.)
  - Daily nutrition guidance + pre-run, during-run (if > 60-75 min) and post-run meals
  - Hydration guidance
  - Common supplements in general terms only (caffeine, electrolytes, vitamin D, iron,
    omega-3, protein) - optional, food comes first.
  - Injury-prevention and sleep tips
Phase 3 - FOLLOW-UP: Answer questions, adapt the plan if they miss runs, get tired or hurt.

SAFETY RULES
- You are not a doctor. If they report heart conditions, chest pain, fainting, uncontrolled
  diabetes/BP, pregnancy, recent surgery or serious injury, tell them kindly to get a doctor's
  clearance BEFORE starting, and make the plan conservative.
- Never prescribe medicines or doses. For supplements, always say "check with your doctor
  or pharmacist first", especially if they take medication.
- If they report pain (not normal soreness), tell them to stop and get it checked.

STYLE
- Be encouraging and positive, use occasional emojis.
- Keep answers short during intake; be structured and clear when giving the plan.
"""

app = FastAPI(title="RunCoach AI", version="1.1")

_sessions: Dict[str, List[BaseMessage]] = {}
_llm = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        GROQ_API_KEY= os.environ["GROQ_API_KEY"]
        _llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.6)
    return _llm


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ResetRequest(BaseModel):
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "active_sessions": len(_sessions)}


@app.post("/chat")
def chat(req: ChatRequest):
    history = _sessions.setdefault(req.session_id, [SystemMessage(content=SYSTEM_PROMPT)])
    history.append(HumanMessage(content=req.message))
    try:
        reply = get_llm().invoke(history)
    except HTTPException:
        history.pop()
        raise
    except Exception as e:
        history.pop()
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    history.append(AIMessage(content=reply.content))
    return {"session_id": req.session_id, "reply": reply.content}


@app.post("/reset")
def reset(req: ResetRequest):
    _sessions.pop(req.session_id, None)
    return {"status": "reset", "session_id": req.session_id}


# ---------------------------------------------------------------------------
# Browser chat page
# ---------------------------------------------------------------------------
CHAT_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coach Stride</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;display:flex;justify-content:center}
  #app{width:100%;max-width:700px;height:100vh;display:flex;flex-direction:column;background:#fff}
  header{padding:14px 18px;background:#ff6b35;color:#fff;font-weight:600;display:flex;justify-content:space-between;align-items:center}
  header button{background:#fff3;border:0;color:#fff;padding:6px 10px;border-radius:6px;cursor:pointer}
  #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
  .msg{max-width:85%;padding:10px 14px;border-radius:14px;white-space:pre-wrap;line-height:1.45}
  .bot{background:#eef1f4;align-self:flex-start}
  .me{background:#ff6b35;color:#fff;align-self:flex-end}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid #ddd}
  input{flex:1;padding:12px;border:1px solid #ccc;border-radius:10px;font-size:16px}
  form button{padding:0 18px;border:0;border-radius:10px;background:#ff6b35;color:#fff;font-size:16px;cursor:pointer}
</style></head>
<body><div id="app">
  <header><span>🏃 Coach Stride</span><button id="reset" type="button">New chat</button></header>
  <div id="log"></div>
  <form id="f"><input id="t" placeholder="Type your message..." autocomplete="off"><button>Send</button></form>
</div>
<script>
let sid = "web-" + Math.random().toString(36).slice(2);
const log = document.getElementById("log"), t = document.getElementById("t");

function add(text, cls){
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
async function send(text){
  const wait = add("Coach is thinking...", "bot");
  try{
    const r = await fetch("/chat", {method:"POST", headers:{"Content-Type":"application/json"},
                                    body: JSON.stringify({session_id: sid, message: text})});
    const j = await r.json();
    wait.textContent = r.ok ? j.reply : ("Error: " + (j.detail || r.status));
  }catch(e){ wait.textContent = "Network error: " + e; }
  log.scrollTop = log.scrollHeight;
}
document.getElementById("f").onsubmit = e => {
  e.preventDefault();
  const v = t.value.trim(); if(!v) return;
  add(v, "me"); t.value = ""; send(v);
};
document.getElementById("reset").onclick = async () => {
  await fetch("/reset", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({session_id: sid})});
  sid = "web-" + Math.random().toString(36).slice(2); log.innerHTML = "";
  send("Hi coach, I want to start training.");
};
send("Hi coach, I want to start training.");
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_PAGE