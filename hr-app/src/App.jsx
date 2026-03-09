import { useState, useEffect, useRef, useCallback, useReducer } from "react";

// ── PDF TEXT EXTRACTOR via Anthropic API ─────────────────────────────────────
// Sends the PDF as base64 to claude-sonnet-4-20250514 and asks it to return
// only the plain text content of the resume — no summarisation.
async function extractPDFText(base64Data) {
  if (!window.pdfjsLib) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }

  const binary = atob(base64Data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const pdf = await window.pdfjsLib.getDocument({ data: bytes }).promise;
  let text = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(item => item.str).join(" ") + "\n";
  }

  if (!text.trim()) throw new Error("Could not extract text from PDF.");
  return text.trim();
}
/*async function extractPDFText(base64Data) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{
        role: "user",
        content: [
          {
            type: "document",
            source: { type: "base64", media_type: "application/pdf", data: base64Data },
          },
          {
            type: "text",
            text: "Extract and return ONLY the plain text content of this resume. No commentary, no formatting, no markdown — just the raw text as it appears: name, contact info, skills, work experience, education, achievements. Output nothing else.",
          },
        ],
      }],
    }),
  });
  const data = await response.json();
  const text = (data.content || []).map(b => b.text || "").join("\n").trim();
  if (!text) throw new Error("Could not extract text from PDF.");
  return text;
}*/

// ════════════════════════════════════════════════════════════════════════════════
//  TRUE SELF-TRAINING ADAPTIVE INTERVIEW ENGINE
//
//  Design contract — NOTHING in this system:
//    ✗ assigns a difficulty score to any question
//    ✗ labels questions as easy/medium/hard/expert
//    ✗ uses templates tied to difficulty tiers
//    ✗ assumes any prior about how hard a question is
//
//  What the system DOES:
//    ✓ Extracts topics from ANY job description (works for chef, lawyer, nurse, engineer)
//    ✓ Generates questions purely from those topics using linguistic frames
//    ✓ Observes raw answer quality signals (length, action words, outcomes, relevance)
//    ✓ Each question accumulates a "hardness posterior" from real responses
//    ✓ MAB (Thompson Sampling) selects questions to maximise information gain
//    ✓ IRT 2-PL model trains online — difficulty emerges from data, never assigned
//    ✓ Works identically for "Sous Chef at Michelin restaurant" or "Distributed Systems Architect"
// ════════════════════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────────────────────
//  TEXT PRIMITIVES
// ────────────────────────────────────────────────────────────────────────────────
const STOP = new Set([
  "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
  "from","is","are","was","were","be","been","have","has","had","do","does",
  "did","will","would","could","should","not","it","its","this","that","i",
  "you","he","she","we","they","my","your","how","what","when","where","why",
  "all","so","as","if","up","out","about","into","can","may","must","than",
  "then","there","their","which","who","also","just","very","more","some",
  "such","each","after","before","over","same","being","between","through",
  "during","without","under","while","these","those","them","our","its","use",
]);

function tok(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s\-]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOP.has(w));
}

// TF-IDF — used for relevance scoring only
class TFIDF {
  constructor() { this.v = {}; this.idf = {}; this.n = 0; }

  fit(docs) {
    const df = {};
    this.n = docs.length;
    docs.forEach(d => {
      new Set(tok(d)).forEach(t => { df[t] = (df[t] || 0) + 1; });
    });
    let i = 0;
    Object.keys(df).forEach(t => {
      this.v[t] = i++;
      this.idf[t] = Math.log((this.n + 1) / (df[t] + 1)) + 1;
    });
    this.dim = i;
  }

  vec(text) {
    const ts = tok(text), tf = {};
    ts.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
    const v = new Float32Array(this.dim || 1).fill(0);
    Object.entries(tf).forEach(([t, c]) => {
      if (this.v[t] !== undefined)
        v[this.v[t]] = (c / ts.length) * (this.idf[t] || 1);
    });
    return v;
  }

  cos(a, b) {
    let d = 0, na = 0, nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (na && nb) ? d / Math.sqrt(na * nb) : 0;
  }

  // Return top-k salient unigrams + bigrams from a document
  topTerms(text, k = 14) {
    const ts = tok(text);
    const uni = ts;
    const bi  = ts.slice(0,-1).map((w,i) => w + " " + ts[i+1]);
    const all = [...uni, ...bi];

    // Score by tf * idf
    const tf = {};
    all.forEach(t => { tf[t] = (tf[t] || 0) + 1; });

    return Object.entries(tf)
      .map(([t, c]) => {
        const idfVal = this.idf[tok(t).join(" ")] || this.idf[tok(t)[0]] || 1;
        return { t, s: (c / all.length) * idfVal };
      })
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map(x => x.t);
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  TOPIC EXTRACTOR
//  Pulls meaningful concepts from any job description — domain-agnostic
// ────────────────────────────────────────────────────────────────────────────────
function extractTopics(jobDescription, k = 10) {
  const engine = new TFIDF();
  engine.fit([jobDescription]);

  const ts = tok(jobDescription);
  const bi = ts.slice(0,-1).map((w,i) => w + " " + ts[i+1]);
  const tri = ts.slice(0,-2).map((w,i) => w + " " + ts[i+1] + " " + ts[i+2]);
  const candidates = [...new Set([...ts, ...bi, ...tri])];

  // Score each candidate term
  const scored = candidates.map(term => {
    const termToks = tok(term);
    let score = termToks.reduce((s, t) => s + (engine.idf[t] || 0), 0) / termToks.length;
    // Prefer multi-word (more specific)
    score *= (1 + termToks.length * 0.3);
    // Penalise very short single words unless high IDF
    if (termToks.length === 1 && termToks[0].length < 5) score *= 0.6;
    return { term, score };
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(x => x.term);
}

// ────────────────────────────────────────────────────────────────────────────────
//  QUESTION GENERATOR
//  Generates questions from topics using domain-neutral LINGUISTIC FRAMES.
//  A "frame" is a sentence structure — it asks about a concept without
//  assuming what the concept IS or how hard it is.
//
//  Works equally well for:
//    "sous-vide cooking technique" → culinary question
//    "kubernetes pod scheduling"   → engineering question
//    "client intake process"       → legal/social work question
//    "inventory forecasting"       → retail/supply chain question
// ────────────────────────────────────────────────────────────────────────────────
const FRAMES = [
  // Experience recall
  t       => `Tell me about a time you worked directly with ${t}. What was your role and what did you accomplish?`,
  t       => `Describe your most significant experience involving ${t}. What were the key decisions you made?`,
  t       => `Walk me through how you have applied ${t} in a real professional context.`,
  // Process & reasoning
  t       => `How do you approach ${t} when starting from scratch? Walk me through your process step by step.`,
  t       => `What factors do you consider when making decisions about ${t}?`,
  t       => `How do you evaluate whether your approach to ${t} is working? What signals do you watch for?`,
  // Problem & challenge
  t       => `Describe a challenging situation you faced involving ${t}. How did you diagnose and resolve it?`,
  t       => `What can go wrong with ${t}, and how have you handled those situations?`,
  t       => `Tell me about a time when your approach to ${t} did not work as expected. What did you learn?`,
  // Depth & trade-offs
  (t, t2) => `Compare two different approaches you have used for ${t}. What were the trade-offs?`,
  (t, t2) => `How does ${t} interact with ${t2} in practice? Describe a situation where both mattered.`,
  (t, t2) => `You need to balance ${t} against ${t2} under tight constraints. How do you decide?`,
  // Impact & outcome
  t       => `What is the most impactful outcome you achieved through your work on ${t}?`,
  t       => `How have you improved or optimised ${t} in a previous role? Be specific about what changed and by how much.`,
  // Cross-cutting
  (t, t2) => `If a colleague unfamiliar with ${t} asked for your help, how would you guide them — and how does that connect to ${t2}?`,
  t       => `What does excellent work look like in the context of ${t}? How do you know when you have achieved it?`,
  (t, t2) => `Describe a project where ${t} and ${t2} were both critical. How did you manage the interplay?`,
  t       => `How has your understanding of ${t} evolved over your career? What changed your thinking?`,
];

function generateQuestions(jobDescription, existingQids = new Set()) {
  const topics = extractTopics(jobDescription, 12);
  if (!topics.length) return [];

  const questions = [];
  const usedFrameTopicPairs = new Set();

  topics.forEach((topic, ti) => {
    const topic2 = topics[(ti + 3) % topics.length]; // offset for variety
    FRAMES.forEach((frame, fi) => {
      const key = `${ti}_${fi}`;
      if (usedFrameTopicPairs.has(key)) return;
      const text = frame.length >= 2 ? frame(topic, topic2) : frame(topic);
      const id   = `q_${ti}_${fi}`;
      if (!existingQids.has(id)) {
        questions.push({ id, text, topic, topic2, fi });
        usedFrameTopicPairs.add(key);
      }
    });
  });

  return questions;
}

// ────────────────────────────────────────────────────────────────────────────────
//  IRT 2-PARAMETER LOGISTIC MODEL  (trained purely online, no priors on difficulty)
//
//  P(correct | θ, a, b) = σ(a · (θ − b))
//
//  b = item difficulty  — starts at 0 (neutral), drifts based on observed responses
//  a = discrimination   — starts at 1 (neutral), trained from data
//  θ = person ability   — estimated via Newton-Raphson from response history
//
//  Key: b and a are initialised identically for ALL questions regardless of topic.
//  They become different ONLY because different candidates answer differently.
// ────────────────────────────────────────────────────────────────────────────────
class IRT {
  constructor() {
    // qid → { b, a, n, responses: [{score, theta}] }
    this.items = {};
    // Global response log (for cross-session learning)
    this.log   = [];
  }

  // Register a new question — uniform prior, no difficulty hint
  add(qid) {
    if (!this.items[qid]) {
      this.items[qid] = { b: 0.0, a: 1.0, n: 0, responses: [] };
    }
  }

  // Logistic function
  _sigma(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

  // P(score=1 | theta, item)
  prob(qid, theta) {
    const item = this.items[qid];
    if (!item) return 0.5;
    return this._sigma(item.a * (theta - item.b));
  }

  // Fisher information I(theta) for item at theta
  info(qid, theta) {
    const p = this.prob(qid, theta);
    const item = this.items[qid];
    if (!item) return 0;
    return item.a * item.a * p * (1 - p);
  }

  // Online gradient update after observing (score, theta_estimate)
  update(qid, score, theta) {
    const item = this.items[qid];
    if (!item) return;

    item.responses.push({ score, theta });
    item.n++;
    this.log.push({ qid, score, theta, ts: Date.now() });

    // Clip log to last 500 entries
    if (this.log.length > 500) this.log.shift();

    // Learning rate decays with observations (Robbins-Monro schedule)
    const lr = 0.20 / Math.sqrt(item.n);

    const p   = this.prob(qid, theta);
    const err = score - p;          // residual: positive if candidate did better than expected

    // Gradient of log-likelihood w.r.t. b and a
    const grad_b = -item.a * err;   // b increases when candidate struggles
    const grad_a =  (theta - item.b) * err;

    item.b += lr * grad_b;
    item.a += lr * grad_a;

    // Clamp to numerically stable range
    item.b = Math.max(-4.0, Math.min(4.0, item.b));
    item.a = Math.max(0.2,  Math.min(4.0, item.a));
  }

  // MLE of candidate ability via Newton-Raphson (converges in ~15 steps)
  estimateTheta(responses) {
    if (!responses.length) return 0.0;
    let theta = 0.0;
    for (let iter = 0; iter < 25; iter++) {
      let g = 0, H = 0;
      responses.forEach(({ qid, score }) => {
        const p = this.prob(qid, theta);
        const item = this.items[qid];
        if (!item) return;
        g += item.a * (score - p);
        H -= item.a * item.a * p * (1 - p);
      });
      if (Math.abs(H) < 1e-8) break;
      const step = g / H;
      theta -= Math.max(-0.5, Math.min(0.5, step)); // clamped step
    }
    return Math.max(-4, Math.min(4, theta));
  }

  // Empirical difficulty for display: fraction of responses below 0.5 score
  empiricalDifficulty(qid) {
    const item = this.items[qid];
    if (!item || item.n === 0) return null; // null = not yet observed
    // Map IRT b parameter: b=0 → 50, b=-4 → 0, b=+4 → 100
    return Math.round(((item.b + 4) / 8) * 100);
  }

  // How many responses has a question collected?
  responseCount(qid) {
    return this.items[qid]?.n || 0;
  }

  snapshot() {
    return { items: JSON.parse(JSON.stringify(this.items)), log: this.log.slice(-200) };
  }

  restore(snap) {
    this.items = snap.items || {};
    this.log   = snap.log  || [];
    return this;
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  THOMPSON SAMPLING BANDIT
//  Each question is an arm.
//  Reward = information gained about candidate ability.
//  Beta posterior on each arm's "informativeness".
//
//  Why Thompson Sampling over UCB here:
//    - Questions have unknown informativeness until tried
//    - Thompson sampling naturally handles cold-start (new questions)
//    - Exploration is implicit — high variance arms get sampled more
// ────────────────────────────────────────────────────────────────────────────────
class ThompsonBandit {
  constructor() {
    // qid → { alpha, beta, n }
    // alpha/beta are Beta distribution params for P(informative)
    this.arms = {};
    this.t    = 0;
  }

  // All arms start with uniform prior Beta(1,1)
  register(qid) {
    if (!this.arms[qid]) {
      this.arms[qid] = { alpha: 1.0, beta: 1.0, n: 0, sumReward: 0 };
    }
  }

  // Sample from Beta distribution via Johnk's method
  _betaSample(a, b) {
    let x, y;
    do {
      x = Math.pow(Math.random(), 1 / a);
      y = Math.pow(Math.random(), 1 / b);
    } while (x + y > 1 || x + y === 0);
    return x / (x + y);
  }

  // Select best arm from available set using Thompson Sampling
  // irt + theta used to bias toward high-information questions
  select(availableQids, irt, theta) {
    this.t++;
    if (!availableQids.length) return null;

    // For each available arm, draw a Thompson sample
    // Blend TS sample with IRT information estimate
    let best = availableQids[0], bestScore = -Infinity;

    availableQids.forEach(qid => {
      this.register(qid);
      const arm = this.arms[qid];

      // Thompson sample from Beta posterior
      const ts = this._betaSample(arm.alpha, arm.beta);

      // IRT information (0 if never observed — unbiased)
      const irtInfo = irt.responseCount(qid) > 0
        ? irt.info(qid, theta)
        : 0.5; // uninformed prior for new questions

      // Blend: TS guides exploration, IRT guides exploitation
      const score = ts * 0.6 + irtInfo * 0.4;

      if (score > bestScore) { bestScore = score; best = qid; }
    });

    return best;
  }

  // Update arm after observing reward (information gained = answer quality signal)
  update(qid, reward) {
    this.register(qid);
    const arm = this.arms[qid];
    arm.n++;
    arm.sumReward += reward;
    // Beta-Bernoulli update: treat reward as Bernoulli(p)
    arm.alpha += reward;
    arm.beta  += (1 - reward);
  }

  mean(qid) {
    const arm = this.arms[qid];
    if (!arm) return 0.5;
    return arm.alpha / (arm.alpha + arm.beta);
  }

  snapshot() {
    return { arms: JSON.parse(JSON.stringify(this.arms)), t: this.t };
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  ANSWER QUALITY SCORER
//  Produces a continuous score in [0,1] from raw answer text + context.
//  No difficulty used — just signal from the text itself.
// ────────────────────────────────────────────────────────────────────────────────
class AnswerScorer {
  constructor() { this.tfidf = new TFIDF(); }

  score(question, answer, jobDesc) {
    const text   = answer?.trim() || "";
    const words  = text.split(/\s+/).filter(Boolean);
    const wc     = words.length;

    if (wc < 4) return { score: 0.04, signals: {}, tip: "Please give a more complete answer." };

    // ── Signal 1: Relevance — TF-IDF cosine(answer, question+job)
    const context = question + " " + jobDesc.slice(0, 500);
    this.tfidf.fit([text, context]);
    const relevance = this.tfidf.cos(this.tfidf.vec(text), this.tfidf.vec(context));

    // ── Signal 2: Specificity markers (domain-neutral)
    const lo = text.toLowerCase();
    const signals = {
      hasAction:   /\b(implemented|built|designed|developed|created|led|managed|launched|delivered|introduced|established|improved|reduced|increased|solved|resolved|negotiated|prepared|executed|coordinated|mentored|trained|authored|analysed|deployed|migrated|optimised|produced|directed|founded|grew|saved|automated|restructured)\b/i.test(text),
      hasOutcome:  /\b(result|outcome|impact|achieved|improved|increased|reduced|saved|delivered|enabled|successfully|led to|which meant|as a result|consequently|percent|%|doubled|tripled|halved|by \d)\b/i.test(lo),
      hasNumbers:  /\d+/.test(text),
      hasContext:  /\b(when|during|while|at my|in my|our team|the project|the client|at that point|at the time)\b/i.test(lo),
      goodLength:  wc >= 40 && wc <= 280,
      notVague:    !/\b(just|basically|kind of|sort of|maybe|probably|i think|i guess|i believe|not sure|generally)\b/i.test(lo),
    };

    const specificityScore = Object.values(signals).filter(Boolean).length / Object.keys(signals).length;

    // ── Signal 3: Vocabulary richness (unique token ratio)
    const tokens = tok(text);
    const richness = tokens.length > 0 ? new Set(tokens).size / tokens.length : 0;

    // ── Composite (no difficulty weighting — purely content-based)
    const raw = relevance * 0.40 + specificityScore * 0.42 + richness * 0.18;
    const score = Math.min(Math.max(raw, 0), 1.0);

    // ── Tip: most actionable improvement
    let tip = "";
    if (!signals.hasAction)  tip = "Use action verbs — describe what YOU did (built, led, resolved, designed…)";
    else if (!signals.hasOutcome) tip = "State the outcome or result of your action";
    else if (!signals.hasNumbers) tip = "Add a number, percentage, or scale to make your answer concrete";
    else if (!signals.goodLength) tip = wc < 40 ? "Expand your answer — aim for at least 40 words" : "Your answer is very long — try to be more concise";
    else if (!signals.hasContext) tip = "Set context: when/where did this happen?";
    else tip = score > 0.72 ? "Strong, specific answer." : "Good — add more concrete evidence.";

    return { score, relevance, specificityScore, richness, signals, tip, wc };
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  RESUME SCORER
// ────────────────────────────────────────────────────────────────────────────────
class ResumeScorer {
  constructor() { this.tfidf = new TFIDF(); }

  score(resume, jobDesc) {
    this.tfidf.fit([resume, jobDesc]);
    const sim  = this.tfidf.cos(this.tfidf.vec(resume), this.tfidf.vec(jobDesc));

    // Extract top terms from JD and check coverage in resume
    const jdTerms = extractTopics(jobDesc, 15);
    const rlo     = resume.toLowerCase();
    const covered = jdTerms.filter(t => rlo.includes(t));
    const coverage = jdTerms.length ? covered.length / jdTerms.length : 0;

    // Experience years
    const expM  = resume.match(/(\d+)\+?\s*years?\b/i);
    const expYrs = expM ? parseInt(expM[1]) : 0;
    const expScore = Math.min(expYrs / 10, 1.0);

    const final = sim * 0.45 + coverage * 0.40 + expScore * 0.15;
    return {
      score:    Math.min(final, 1.0),
      sim, coverage, expScore, expYrs,
      jdTerms, covered,
      missing: jdTerms.filter(t => !rlo.includes(t)),
    };
  }
}

// ────────────────────────────────────────────────────────────────────────────────
//  SESSION SCORING
// ────────────────────────────────────────────────────────────────────────────────
function finalScore(qas, irt) {
  if (!qas.length) return { score: 0, theta: 0, slope: 0, progression: [] };

  // IRT-based ability estimate (the principled score)
  const responses = qas.map(qa => ({ qid: qa.qid, score: qa.score }));
  const theta     = irt.estimateTheta(responses);
  const thetaNorm = Math.min(Math.max((theta + 4) / 8, 0), 1); // map [-4,4] → [0,1]

  // Information-weighted mean score
  const infoW  = qas.map(qa => Math.max(irt.info(qa.qid, theta), 0.05));
  const sumW   = infoW.reduce((a,b)=>a+b,0);
  const infoWt = qas.reduce((s,qa,i) => s + qa.score * infoW[i], 0) / sumW;

  // Trajectory slope (linear regression)
  const scores = qas.map(qa => qa.score);
  const n = scores.length, mx = (n-1)/2;
  const my = scores.reduce((a,b)=>a+b,0)/n;
  const slope = scores.reduce((s,sc,i)=>s+(i-mx)*(sc-my),0) /
                scores.reduce((s,_,i)=>s+(i-mx)**2,0.001);

  // Running average window
  const progression = scores.map((_,i) => {
    const w = scores.slice(Math.max(0,i-2), i+1);
    return w.reduce((a,b)=>a+b,0)/w.length;
  });

  const blended = thetaNorm * 0.55 + infoWt * 0.45;
  return { score: Math.min(Math.max(blended,0),1), theta, thetaNorm, infoWt, slope, progression, raw: scores };
}

// ────────────────────────────────────────────────────────────────────────────────
//  GLOBAL SINGLETONS — persist across sessions so IRT keeps training
// ────────────────────────────────────────────────────────────────────────────────
const GLOBAL_IRT      = new IRT();
const ANSWER_SCORER   = new AnswerScorer();
const RESUME_SCORER   = new ResumeScorer();

// ════════════════════════════════════════════════════════════════════════════════
//  REACT APPLICATION
// ════════════════════════════════════════════════════════════════════════════════

// Colour derived purely from learned IRT b parameter — not from a difficulty label
function irtColor(b) {
  // b < 0 = most respond well (warmer) | b > 0 = most struggle (cooler/hotter)
  const t = (b + 4) / 8; // normalise to [0,1]
  const r = Math.round(40  + t * 215);
  const g = Math.round(200 - t * 160);
  const bl= Math.round(80  + t * 20);
  return `rgb(${r},${g},${bl})`;
}

function ScoreBar({ label, value, color, mono }) {
  const pct = Math.min(Math.max(value || 0, 0), 1);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", fontSize:12, marginBottom:3 }}>
        <span style={{ color:"rgba(255,255,255,0.42)" }}>{label}</span>
        <span style={{ fontFamily: mono ? "'JetBrains Mono',monospace" : "inherit",
          color: color || "rgba(255,255,255,0.65)" }}>{(pct*100).toFixed(1)}%</span>
      </div>
      <div style={{ height:5, background:"rgba(255,255,255,0.06)", borderRadius:99 }}>
        <div style={{ height:"100%", width:`${pct*100}%`, borderRadius:99,
          background: color || "#d4a017", transition:"width 0.55s ease" }} />
      </div>
    </div>
  );
}

function IRTBadge({ qid, irt, short }) {
  const item = irt.items[qid];
  if (!item || item.n === 0) {
    return (
      <span style={{ fontSize:11, color:"rgba(255,255,255,0.28)", fontFamily:"'JetBrains Mono',monospace" }}>
        {short ? "new" : "No data yet"}
      </span>
    );
  }
  const d = irt.empiricalDifficulty(qid);
  const col = irtColor(item.b);
  return (
    <span style={{ fontSize:11, color: col, fontFamily:"'JetBrains Mono',monospace",
      background: col.replace("rgb","rgba").replace(")",",0.12)"),
      padding:"2px 7px", borderRadius:99, border:`1px solid ${col.replace("rgb","rgba").replace(")",",0.3)")}` }}>
      {short ? `b=${item.b.toFixed(2)}` : `b=${item.b.toFixed(2)} α=${item.a.toFixed(2)} n=${item.n}`}
    </span>
  );
}

function ProgressChart({ progression, scores }) {
  if (!progression?.length) return null;
  const W=260, H=72, pad=8;
  const n = progression.length;
  const px = i => pad + (i / Math.max(n-1,1)) * (W-2*pad);
  const py = v => H - pad - Math.min(Math.max(v,0),1) * (H-2*pad);
  const path = progression.map((v,i)=>`${i===0?"M":"L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
  const area = `${path} L${px(n-1)},${H} L${px(0)},${H} Z`;
  return (
    <div>
      <div style={{ fontSize:9, letterSpacing:"0.1em", color:"rgba(255,255,255,0.25)",
        textTransform:"uppercase", marginBottom:5 }}>Score Trajectory</div>
      <svg width={W} height={H} style={{ overflow:"visible" }}>
        <path d={area} fill="rgba(80,200,120,0.07)" />
        <path d={path} fill="none" stroke="rgba(80,200,120,0.65)" strokeWidth={2} />
        <line x1={pad} y1={py(0.5)} x2={W-pad} y2={py(0.5)}
          stroke="rgba(255,255,255,0.07)" strokeDasharray="4,3" />
        {scores?.map((v,i) => (
          <circle key={i} cx={px(i)} cy={py(v)} r={3.5}
            fill={v>=0.65?"#4ade80":v>=0.44?"#facc15":"#f87171"}
            stroke="rgba(0,0,0,0.4)" strokeWidth={1} />
        ))}
      </svg>
    </div>
  );
}

function Tag({ children, color }) {
  const c = color || "#d4a017";
  return (
    <span style={{ display:"inline-block", padding:"2px 9px", borderRadius:99, fontSize:11,
      fontWeight:700, background: c+"1e", color:c, border:`1px solid ${c}38` }}>
      {children}
    </span>
  );
}

export default function App() {
  const [page,   setPage]   = useState("home");
  const [hrTab,  setHrTab]  = useState("jobs");

  const [jobs, setJobs] = useState([
    {
      id: 1, title: "Senior Data Scientist", company: "Nexus AI",
      description: "We are looking for a Senior Data Scientist with deep expertise in machine learning model development, Python programming, statistical analysis, and experiment design. You will build and deploy NLP models, recommendation systems, and forecasting pipelines. Experience with deep learning frameworks, transformer architectures, MLOps practices, cloud platforms, and SQL databases is required. Strong communication and cross-functional collaboration skills are essential.",
      resumeThreshold: 0.38, interviewThreshold: 0.54, maxQ: 10, candidates: [],
    },
    {
      id: 2, title: "Executive Chef", company: "La Maison",
      description: "Executive Chef needed to lead our fine dining kitchen. Must have mastery of classical French techniques, menu development, food costing, inventory management, and team leadership. Experience in Michelin-starred environments preferred. Responsible for hiring, training, and performance management of kitchen staff. Health and safety compliance, vendor relationships, and seasonal sourcing are core responsibilities.",
      resumeThreshold: 0.35, interviewThreshold: 0.50, maxQ: 8, candidates: [],
    },
    {
      id: 3, title: "Full Stack Engineer", company: "BuildFast",
      description: "Full Stack Engineer with React, TypeScript, Node.js, PostgreSQL, Redis, Docker, Kubernetes, CI/CD pipelines, AWS, microservices, REST and GraphQL APIs, performance optimisation, and database design. You will own features end-to-end from architecture to deployment. Strong code review, testing, and mentoring skills required.",
      resumeThreshold: 0.35, interviewThreshold: 0.52, maxQ: 8, candidates: [],
    },
  ]);

  const [jf, setJf] = useState({ title:"", company:"", description:"", resumeThreshold:0.38, interviewThreshold:0.54, maxQ:10 });
  const [cand, setCand]     = useState({ name:"", email:"", resume:"" });
  const [resumeMode, setResumeMode] = useState("paste"); // "paste" | "pdf"
  const [pdfFile, setPdfFile]   = useState(null);   // { name, base64 }
  const [pdfBusy, setPdfBusy]   = useState(false);
  const [pdfError, setPdfError] = useState("");
  const fileInputRef = useRef(null);
  const [selJob, setSelJob] = useState(null);
  const [rEval, setREval]   = useState(null);
  const [session, setSession] = useState(null);
  const [ans, setAns]       = useState("");
  const [timer, setTimer]   = useState(0);
  const [busy, setBusy]     = useState(false);
  const [results, setResults] = useState(null);
  const [viewing, setViewing] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (session?.active) timerRef.current = setInterval(() => setTimer(t=>t+1), 1000);
    else clearInterval(timerRef.current);
    return () => clearInterval(timerRef.current);
  }, [session?.active]);

  // ── Score resume ────────────────────────────────────────────────────────────
  const scoreResume = useCallback(() => {
    if (!cand.resume.trim() || !selJob) return;
    setBusy(true);
    setTimeout(() => {
      const job = jobs.find(j => j.id === selJob);
      const r = RESUME_SCORER.score(cand.resume, job.description);
      setREval({ ...r, job, pass: r.score >= job.resumeThreshold });
      setBusy(false);
    }, 500);
  }, [cand.resume, selJob, jobs]);

  // ── Handle PDF upload ───────────────────────────────────────────────────────
  const handlePDFUpload = useCallback(async (file) => {
    if (!file || file.type !== "application/pdf") {
      setPdfError("Please upload a valid PDF file.");
      return;
    }
    setPdfBusy(true); setPdfError(""); setPdfFile(null);
    setCand(p => ({ ...p, resume: "" }));
    try {
      const base64 = await new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result.split(",")[1]);
        reader.onerror = () => rej(new Error("File read failed"));
        reader.readAsDataURL(file);
      });
      const text = await extractPDFText(base64);
      setPdfFile({ name: file.name, base64 });
      setCand(p => ({ ...p, resume: text }));
    } catch (err) {
      setPdfError("Failed to extract text from PDF. Please paste your resume manually.");
    } finally {
      setPdfBusy(false);
    }
  }, []);

  // ── Begin interview ─────────────────────────────────────────────────────────
  const begin = useCallback(() => {
    if (!rEval?.pass) return;
    const { job } = rEval;

    // Generate questions fresh from job description — zero pre-labelling
    const questions = generateQuestions(job.description);

    // Register every question with IRT at uniform prior
    questions.forEach(q => GLOBAL_IRT.add(q.id));

    // Create session bandit
    const bandit = new ThompsonBandit();
    questions.forEach(q => bandit.register(q.id));

    // First question: purely random (no information yet, Thompson = uniform)
    const firstQid = bandit.select(questions.map(q=>q.id), GLOBAL_IRT, 0);
    const firstQ   = questions.find(q => q.id === firstQid);

    setSession({
      job, rEval, questions, bandit,
      currentQ: firstQ,
      asked: new Set([firstQ.id]),
      qas: [],
      qNum: 1,
      theta: 0.0,
      active: true,
    });
    setAns(""); setTimer(0);
    setPage("interview");
  }, [rEval]);

  // ── Submit answer ────────────────────────────────────────────────────────────
  const submit = useCallback(() => {
    if (!session || busy) return;
    setBusy(true);

    setTimeout(() => {
      const { job, bandit, currentQ, questions, asked, qas, qNum, theta } = session;

      // Score the answer
      const ev = ANSWER_SCORER.score(currentQ.text, ans, job.description);

      // Feed into IRT — b and a now drift based on this response
      GLOBAL_IRT.update(currentQ.id, ev.score, theta);

      // Update bandit reward
      bandit.update(currentQ.id, ev.score);

      const newQA = {
        qNum, qid: currentQ.id, question: currentQ.text,
        topic: currentQ.topic, answer: ans,
        score: ev.score, tip: ev.tip,
        signals: ev.signals, wc: ev.wc,
        timeTaken: timer,
        // IRT params AT TIME OF RESPONSE — recorded for auditability
        irtB: GLOBAL_IRT.items[currentQ.id]?.b ?? 0,
        irtA: GLOBAL_IRT.items[currentQ.id]?.a ?? 1,
        irtN: GLOBAL_IRT.items[currentQ.id]?.n ?? 0,
      };

      const updatedQAs = [...qas, newQA];
      const newTheta   = GLOBAL_IRT.estimateTheta(updatedQAs.map(q => ({ qid:q.qid, score:q.score })));

      if (qNum >= job.maxQ) {
        // ── DONE ────────────────────────────────────────────────────────────
        const fs   = finalScore(updatedQAs, GLOBAL_IRT);
        const pass = fs.score >= job.interviewThreshold;

        const rec  = buildReport(cand.name, job.title, updatedQAs, fs, rEval, bandit);
        const result = {
          candidateName: cand.name, candidateEmail: cand.email,
          jobTitle: job.title, jobId: job.id,
          resumeScore: rEval.score, resumeData: rEval,
          interviewScore: fs.score, scoring: fs, pass,
          qas: updatedQAs, questions,
          report: rec,
          irtSnap: GLOBAL_IRT.snapshot(),
          completedAt: new Date().toISOString(),
        };

        setJobs(prev => prev.map(j =>
          j.id === job.id ? { ...j, candidates:[...j.candidates, result] } : j));
        setResults(result);
        setSession(s => ({ ...s, active: false }));
        setBusy(false);
        setPage("results");
        return;
      }

      // ── NEXT QUESTION ───────────────────────────────────────────────────────
      const available = questions.filter(q => !asked.has(q.id)).map(q => q.id);

      // If all questions asked (short interview), regenerate with different frames
      let nextQid, nextQ;
      if (available.length === 0) {
        const fresh = generateQuestions(job.description, asked);
        fresh.forEach(q => GLOBAL_IRT.add(q.id));
        fresh.forEach(q => bandit.register(q.id));
        nextQid = bandit.select(fresh.map(q=>q.id), GLOBAL_IRT, newTheta);
        nextQ   = fresh.find(q => q.id === nextQid) || fresh[0];
        setSession(s => ({ ...s, bandit, questions: [...questions, ...fresh],
          currentQ: nextQ, asked: new Set([...asked, nextQ.id]),
          qas: updatedQAs, qNum: qNum+1, theta: newTheta }));
      } else {
        nextQid = bandit.select(available, GLOBAL_IRT, newTheta);
        nextQ   = questions.find(q => q.id === nextQid);
        setSession(s => ({ ...s, bandit, currentQ: nextQ,
          asked: new Set([...asked, nextQ.id]),
          qas: updatedQAs, qNum: qNum+1, theta: newTheta }));
      }

      setAns(""); setTimer(0); setBusy(false);
    }, 500);
  }, [session, ans, timer, cand, busy, rEval]);

  // ── Build report (no difficulty categories used) ────────────────────────────
  function buildReport(name, title, qas, fs, rEval, bandit) {
    const avgScore  = qas.reduce((s,q)=>s+q.score,0)/qas.length;
    const allTips   = qas.map(q=>q.tip).filter(t=>t && !t.startsWith("Strong"));
    const topics    = [...new Set(qas.map(q=>q.topic))];

    const strengths = [], improves = [];

    if (qas.some(q=>q.signals?.hasNumbers))  strengths.push("Uses quantitative evidence effectively");
    if (qas.some(q=>q.signals?.hasOutcome))  strengths.push("Communicates results and business impact clearly");
    if (qas.some(q=>q.signals?.hasAction))   strengths.push("Answers are action-oriented with clear ownership");
    if (fs.slope > 0.02)                     strengths.push("Performance improved as the interview progressed");
    if (avgScore > 0.65)                     strengths.push("Consistently strong across all topic areas");
    if (!strengths.length)                   strengths.push("Completed all questions with effort");

    const tipCounts = {};
    allTips.forEach(t => { tipCounts[t] = (tipCounts[t]||0)+1; });
    const topTips = Object.entries(tipCounts).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([t])=>t);
    topTips.forEach(t => improves.push(t));
    if (!improves.length) improves.push("Continue practising structured answer frameworks (STAR method)");

    const level = fs.score>=0.80?"exceptional":fs.score>=0.65?"strong":fs.score>=0.50?"solid":"developing";
    const rec   = fs.score>=0.68?"Strong hire":fs.score>=0.53?"Conditional hire":"Does not meet threshold";

    return {
      candidateSummary: `You demonstrated ${level} performance (score: ${(fs.score*100).toFixed(1)}%). IRT ability estimate θ̂ = ${fs.theta.toFixed(2)}. ${fs.slope>0.02?"Your answers improved throughout — a strong signal.":fs.slope<-0.02?"Performance dipped under harder questions — work on depth under pressure.":"You were consistent throughout."} Topics covered: ${topics.slice(0,4).join(", ")}.`,
      hrSummary: `${rec}. IRT θ̂ = ${fs.theta.toFixed(2)} (${(fs.thetaNorm*100).toFixed(0)}th percentile). Resume: ${(rEval.score*100).toFixed(1)}%. Questions were auto-generated from job description topics — no pre-labelled difficulty. IRT trained from responses in real-time.`,
      strengths: strengths.slice(0,4),
      improves:  improves.slice(0,4),
      level, rec,
      topics,
    };
  }

  const fmt = s => `${Math.floor(s/60)}:${(s%60).toString().padStart(2,"0")}`;

  // ── STYLES ──────────────────────────────────────────────────────────────────
  const css = `
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');
    *{box-sizing:border-box;margin:0;padding:0}
    input:focus,textarea:focus{border-color:#5b9cf6!important;box-shadow:0 0 0 3px rgba(91,156,246,0.12)!important;outline:none!important}
    ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:rgba(91,156,246,0.25);border-radius:4px}
    @keyframes up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
    @keyframes in{from{opacity:0}to{opacity:1}}
    @keyframes blink{0%,100%{opacity:.4}50%{opacity:1}}
    .up{animation:up .5s ease both}.in{animation:in .3s ease both}
  `;

  const C = {
    bg:    "#050810",
    card:  "rgba(255,255,255,0.028)",
    bord:  "rgba(255,255,255,0.08)",
    text:  "#d0ccc4",
    muted: "rgba(208,204,196,0.45)",
    blue:  "#5b9cf6",
    gold:  "#e0a020",
  };

  const card  = (x={}) => ({ background:C.card, border:`1px solid ${C.bord}`, borderRadius:14, padding:24, backdropFilter:"blur(4px)", ...x });
  const btn   = (v="p") => ({ cursor:"pointer", border:"none", borderRadius:9, fontFamily:"'Outfit',sans-serif", fontWeight:600, fontSize:14, padding:"11px 24px", transition:"all .16s",
    ...(v==="p" ? { background:"linear-gradient(135deg,#2563eb,#5b9cf6)", color:"#fff" }
      : v==="g" ? { background:"transparent", color:C.blue, border:`1px solid ${C.blue}44` }
      : { background:"rgba(255,255,255,0.05)", color:C.text, border:`1px solid ${C.bord}` }) });
  const inp   = { width:"100%", padding:"11px 14px", background:"rgba(255,255,255,0.04)", border:`1px solid ${C.bord}`, borderRadius:8, color:C.text, fontSize:15, fontFamily:"'Outfit',sans-serif", outline:"none" };
  const lbl   = { display:"block", marginBottom:5, fontSize:11, color:`${C.blue}cc`, letterSpacing:"0.1em", textTransform:"uppercase", fontFamily:"'Outfit',sans-serif" };
  const wrap  = { maxWidth:1100, margin:"0 auto", padding:"0 26px" };
  const app   = { minHeight:"100vh", background:C.bg, color:C.text, fontFamily:"'Outfit',sans-serif", fontSize:15 };

  // ── HOME ────────────────────────────────────────────────────────────────────
  if (page === "home") return (
    <div style={app}>
      <style>{css}</style>
      <div style={{...wrap, paddingTop:90, paddingBottom:80}}>
        <div style={{textAlign:"center", marginBottom:68}}>
          <div className="up" style={{fontSize:11, letterSpacing:"0.28em", color:`${C.blue}88`, marginBottom:20, textTransform:"uppercase"}}>
            Self-Training · No Pre-Assigned Difficulty · Works for Any Job
          </div>
          <h1 className="up" style={{fontFamily:"'Instrument Serif',serif", fontSize:68, lineHeight:1.0, marginBottom:14,
            background:`linear-gradient(130deg,${C.blue},#a5c8ff,${C.gold})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent"}}>
            The Interview Engine<br/>
            <span style={{fontStyle:"italic"}}>That Trains Itself</span>
          </h1>
          <p className="up" style={{fontSize:16, color:C.muted, maxWidth:580, margin:"0 auto 16px", lineHeight:1.88}}>
            No question has a difficulty label. Questions are generated fresh from <em>your</em> job description.
            The system watches real candidate answers and <strong style={{color:C.blue}}>learns what's hard</strong> from data.
          </p>
          <p className="up" style={{fontSize:13, color:"rgba(208,204,196,0.3)", marginBottom:46, fontFamily:"'JetBrains Mono',monospace"}}>
            Thompson Sampling · IRT 2-PL · TF-IDF · Online Gradient Descent
          </p>
          <div className="up" style={{display:"flex", gap:14, justifyContent:"center"}}>
            <button style={{...btn("p"), fontSize:16, padding:"13px 36px"}} onClick={()=>setPage("apply")}>Apply as Candidate</button>
            <button style={{...btn("g"), fontSize:16, padding:"13px 36px"}} onClick={()=>setPage("hr")}>HR Portal</button>
          </div>
        </div>

        <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))", gap:14}}>
          {[
            { icon:"◎", h:"Zero Pre-Labels",       b:"Every question starts with b=0, a=1. Difficulty emerges purely from how candidates respond. A question nobody answers well gets b→+high. One everyone aces gets b→−low." },
            { icon:"⟳", h:"Job-Driven Generation", b:"Topics extracted from your job description via TF-IDF. Works for Data Scientist, Chef, Nurse, Lawyer — any role. Questions built from linguistic frames, not topic libraries." },
            { icon:"θ̂", h:"IRT Ability Scoring",   b:"Candidate score = MLE estimate of latent ability θ̂ via Newton-Raphson, weighted by Fisher information I(θ) of each question. Principled psychometric scoring." },
            { icon:"⚡", h:"Thompson Sampling",     b:"Beta(α,β) posterior per question arm. Exploration is automatic — new questions get sampled more. Rewards shift the posterior toward informative questions." },
            { icon:"↺", h:"Cross-Session Learning", b:"IRT parameters persist across all candidates. After 10 interviews, the system knows which questions are genuinely hard — discovered from data, not annotated by humans." },
          ].map((c,i) => (
            <div key={i} className="up" style={{...card({padding:20}), animationDelay:`${i*.07}s`}}>
              <div style={{fontSize:26, color:C.blue, marginBottom:10}}>{c.icon}</div>
              <div style={{fontSize:14, fontWeight:600, color:C.blue, marginBottom:6}}>{c.h}</div>
              <div style={{fontSize:12.5, color:C.muted, lineHeight:1.72}}>{c.b}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // ── APPLY ───────────────────────────────────────────────────────────────────
  if (page === "apply") return (
    <div style={app}>
      <style>{css}</style>
      <div style={{...wrap, paddingTop:40}}>
        <button style={{...btn("g"), fontSize:13, marginBottom:24}} onClick={()=>setPage("home")}>← Home</button>
        <h2 style={{fontFamily:"'Instrument Serif',serif", fontSize:32, marginBottom:4}}>Candidate Portal</h2>
        <p style={{color:C.muted, fontSize:14, marginBottom:32}}>Questions generated from job description · Difficulty discovered from your answers</p>

        <div style={{display:"grid", gridTemplateColumns:"1.1fr 0.9fr", gap:22}}>
          {/* Left */}
          <div>
            <div style={{...card(), marginBottom:18}}>
              <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16}}>Profile</div>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:13}}>
                <div><label style={lbl}>Full Name</label><input style={inp} value={cand.name} onChange={e=>setCand(p=>({...p,name:e.target.value}))} placeholder="Your name" /></div>
                <div><label style={lbl}>Email</label><input style={inp} value={cand.email} onChange={e=>setCand(p=>({...p,email:e.target.value}))} placeholder="you@example.com" /></div>
              </div>
              {/* Resume — toggle between PDF upload and paste */}
              <div style={{marginBottom:4}}>
                <label style={lbl}>Resume</label>
                <div style={{display:"flex", gap:0, marginBottom:13, borderRadius:8, overflow:"hidden", border:`1px solid ${C.bord}`, width:"fit-content"}}>
                  {["paste","pdf"].map(mode => (
                    <button key={mode} onClick={()=>setResumeMode(mode)}
                      style={{padding:"7px 18px", fontSize:12, fontFamily:"'Outfit',sans-serif",
                        fontWeight:600, border:"none", cursor:"pointer", transition:"all .15s",
                        background: resumeMode===mode ? C.blue : "rgba(255,255,255,0.03)",
                        color: resumeMode===mode ? "#fff" : C.muted,
                        textTransform:"uppercase", letterSpacing:"0.08em"}}>
                      {mode === "paste" ? "✏ Paste Text" : "📄 Upload PDF"}
                    </button>
                  ))}
                </div>
              </div>

              {resumeMode === "paste" ? (
                <textarea style={{...inp, minHeight:240, resize:"vertical", lineHeight:1.68}}
                  value={cand.resume} onChange={e=>setCand(p=>({...p,resume:e.target.value}))}
                  placeholder={"Paste your full resume here — skills, experience, achievements, education.\n\nExample:\nJane Smith | 6 years experience\nSkills: Python, PyTorch, NLP, SQL, AWS, MLOps, Docker\n• Led team building recommendation engine, 10M daily users\n• Reduced model inference latency by 40% via ONNX optimisation\n• Deployed 12 ML models to production"} />
              ) : (
                <div>
                  {/* Drop zone */}
                  <div
                    onClick={() => !pdfBusy && fileInputRef.current?.click()}
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if(f) handlePDFUpload(f); }}
                    style={{
                      border: `2px dashed ${pdfFile ? C.blue : C.bord}`,
                      borderRadius:10, padding:"28px 20px", textAlign:"center",
                      cursor: pdfBusy ? "wait" : "pointer",
                      background: pdfFile ? `${C.blue}08` : "rgba(255,255,255,0.02)",
                      transition:"all .2s", marginBottom: pdfError ? 10 : 0,
                    }}>
                    {pdfBusy ? (
                      <div>
                        <div style={{fontSize:28, marginBottom:10, animation:"blink 1s infinite"}}>⏳</div>
                        <div style={{fontSize:14, color:C.muted}}>Extracting text from PDF…</div>
                        <div style={{fontSize:12, color:"rgba(208,204,196,0.35)", marginTop:4}}>Sending to Claude for text extraction</div>
                      </div>
                    ) : pdfFile ? (
                      <div>
                        <div style={{fontSize:28, marginBottom:8}}>✅</div>
                        <div style={{fontSize:14, fontWeight:600, color:"#4ade80", marginBottom:4}}>{pdfFile.name}</div>
                        <div style={{fontSize:12, color:C.muted, marginBottom:10}}>
                          {cand.resume.split(/\s+/).filter(Boolean).length} words extracted
                        </div>
                        <button onClick={e=>{e.stopPropagation();setPdfFile(null);setCand(p=>({...p,resume:""}));setPdfError("");}}
                          style={{...btn("g"), fontSize:12, padding:"5px 14px"}}>
                          Remove
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div style={{fontSize:36, marginBottom:10}}>📄</div>
                        <div style={{fontSize:14, color:C.text, marginBottom:4}}>Drop your PDF resume here</div>
                        <div style={{fontSize:12, color:C.muted}}>or click to browse · PDF only</div>
                      </div>
                    )}
                  </div>
                  <input ref={fileInputRef} type="file" accept="application/pdf"
                    style={{display:"none"}}
                    onChange={e => { const f = e.target.files?.[0]; if(f) handlePDFUpload(f); e.target.value=""; }} />
                  {pdfError && (
                    <div style={{fontSize:12, color:"#f87171", padding:"8px 12px", borderRadius:7,
                      background:"rgba(248,113,113,0.07)", border:"1px solid rgba(248,113,113,0.18)", marginTop:8}}>
                      {pdfError}
                    </div>
                  )}
                  {/* Show extracted text preview */}
                  {pdfFile && cand.resume && (
                    <div style={{marginTop:10}}>
                      <div style={{fontSize:11, color:"rgba(255,255,255,0.28)", marginBottom:5}}>Extracted text preview</div>
                      <div style={{fontSize:12, color:C.muted, background:"rgba(255,255,255,0.02)",
                        border:`1px solid ${C.bord}`, borderRadius:8, padding:"10px 13px",
                        maxHeight:100, overflowY:"auto", lineHeight:1.6, fontFamily:"'JetBrains Mono',monospace"}}>
                        {cand.resume.slice(0, 320)}…
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {rEval && (
              <div className="in" style={card()}>
                <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16}}>
                  <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase"}}>Resume Score</div>
                  <span style={{fontSize:30, fontFamily:"'JetBrains Mono',monospace", fontWeight:500,
                    color:rEval.score>=0.55?"#4ade80":rEval.score>=0.38?"#facc15":"#f87171"}}>
                    {(rEval.score*100).toFixed(1)}%
                  </span>
                </div>
                <ScoreBar label="TF-IDF Semantic Similarity" value={rEval.sim}      color={C.blue}   mono />
                <ScoreBar label="Topic Coverage"             value={rEval.coverage} color="#34d399"  mono />
                <ScoreBar label="Experience Score"           value={rEval.expScore} color="#fbbf24"  mono />

                <div style={{marginTop:14, paddingTop:13, borderTop:`1px solid ${C.bord}`}}>
                  <div style={{fontSize:11, color:"rgba(255,255,255,0.28)", marginBottom:7}}>
                    Job topics covered ({rEval.covered.length}/{rEval.jdTerms.length})
                  </div>
                  <div style={{display:"flex", flexWrap:"wrap", gap:5, marginBottom:8}}>
                    {rEval.covered.slice(0,10).map(t => <Tag key={t} color="#4ade80">{t}</Tag>)}
                  </div>
                  {rEval.missing.length > 0 && (
                    <>
                      <div style={{fontSize:11, color:"rgba(255,255,255,0.28)", marginBottom:5}}>Missing topics</div>
                      <div style={{display:"flex", flexWrap:"wrap", gap:5}}>
                        {rEval.missing.slice(0,8).map(t => <Tag key={t} color="#f87171">{t}</Tag>)}
                      </div>
                    </>
                  )}
                </div>

                {rEval.pass
                  ? <button style={{...btn("p"), width:"100%", marginTop:18, fontSize:15}} onClick={begin}>
                      Begin Interview — Questions Generated Now →
                    </button>
                  : <div style={{marginTop:14, padding:"12px 15px", borderRadius:8, color:"#f87171", fontSize:13,
                      background:"rgba(248,113,113,0.07)", border:"1px solid rgba(248,113,113,0.18)"}}>
                      Score {(rEval.score*100).toFixed(1)}% is below the {(rEval.job.resumeThreshold*100).toFixed(0)}% threshold.
                      Add more relevant experience and skills from the job description.
                    </div>}
              </div>
            )}
          </div>

          {/* Right — job list */}
          <div>
            <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:13}}>Open Positions</div>
            {jobs.map(job => (
              <div key={job.id} onClick={() => setSelJob(job.id)}
                style={{...card({marginBottom:13, cursor:"pointer", transition:"all .17s",
                  ...(selJob===job.id?{borderColor:`${C.blue}55`, background:`${C.blue}08`}:{})})}}> 
                <div style={{display:"flex", justifyContent:"space-between", marginBottom:8}}>
                  <div>
                    <div style={{fontSize:16, fontWeight:600}}>{job.title}</div>
                    <div style={{fontSize:13, color:C.blue}}>{job.company}</div>
                  </div>
                  <Tag>{job.maxQ}Q</Tag>
                </div>
                <p style={{color:C.muted, fontSize:13, lineHeight:1.62, marginBottom:10}}>{job.description.slice(0,110)}…</p>
                <div style={{fontSize:12, color:"rgba(208,204,196,0.3)"}}>
                  Resume ≥{(job.resumeThreshold*100).toFixed(0)}% · Interview ≥{(job.interviewThreshold*100).toFixed(0)}% · {job.candidates.length} interviewed
                </div>
              </div>
            ))}
            <button style={{...btn("p"), width:"100%", opacity:(!cand.resume.trim()||!selJob||!cand.name||busy||pdfBusy)?0.38:1}}
              onClick={scoreResume} disabled={!cand.resume.trim()||!selJob||!cand.name||busy||pdfBusy}>
              {busy ? "Scoring…" : pdfBusy ? "Extracting PDF…" : "Score Resume →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // ── INTERVIEW ───────────────────────────────────────────────────────────────
  if (page === "interview" && session) {
    const { job, currentQ, qas, qNum, theta, questions } = session;
    const progress = qNum / job.maxQ;
    const last     = qas[qas.length - 1];

    return (
      <div style={app}>
        <style>{css}</style>
        <div style={{...wrap, paddingTop:38, maxWidth:960}}>
          {/* Header */}
          <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:22}}>
            <div>
              <div style={{fontSize:12, color:C.muted}}>{job.title} · {cand.name}</div>
              <div style={{fontSize:22, fontWeight:700}}>Question {qNum} / {job.maxQ}</div>
            </div>
            <div style={{textAlign:"right"}}>
              <div style={{fontSize:26, fontFamily:"'JetBrains Mono',monospace", color:C.gold}}>{fmt(timer)}</div>
              <div style={{fontSize:11, color:"rgba(208,204,196,0.3)", fontFamily:"'JetBrains Mono',monospace"}}>θ̂={theta.toFixed(2)}</div>
            </div>
          </div>

          {/* Progress bar */}
          <div style={{height:3, background:"rgba(255,255,255,0.05)", borderRadius:99, marginBottom:22}}>
            <div style={{height:"100%", width:`${progress*100}%`, borderRadius:99,
              background:`linear-gradient(90deg,${C.blue},#a5c8ff)`, transition:"width 0.4s"}} />
          </div>

          <div style={{display:"grid", gridTemplateColumns:"1fr 270px", gap:16}}>
            {/* Main */}
            <div>
              {/* IRT status for current question */}
              <div style={{...card({marginBottom:13, padding:14})}}>
                <div style={{display:"flex", justifyContent:"space-between", alignItems:"center"}}>
                  <div>
                    <div style={{fontSize:10, letterSpacing:"0.1em", color:"rgba(255,255,255,0.25)", textTransform:"uppercase", marginBottom:5}}>
                      IRT State · Learned from Responses
                    </div>
                    <div style={{display:"flex", alignItems:"center", gap:10}}>
                      <IRTBadge qid={currentQ.id} irt={GLOBAL_IRT} />
                      <span style={{fontSize:12, color:C.muted}}>
                        {GLOBAL_IRT.responseCount(currentQ.id) === 0
                          ? "— first time this question has been asked"
                          : `— seen by ${GLOBAL_IRT.responseCount(currentQ.id)} candidate${GLOBAL_IRT.responseCount(currentQ.id)>1?"s":""}`}
                      </span>
                    </div>
                  </div>
                  <div style={{fontSize:12, color:C.muted, textAlign:"right"}}>
                    Topic: <span style={{color:C.blue}}>{currentQ.topic}</span>
                  </div>
                </div>
              </div>

              {/* Question */}
              <div className="in" style={{...card({marginBottom:13})}}>
                <div style={{fontSize:10, letterSpacing:"0.12em", color:`${C.blue}88`, textTransform:"uppercase", marginBottom:14}}>
                  Generated from job description · topic: {currentQ.topic}
                </div>
                {busy
                  ? <div style={{color:C.muted, animation:"blink 1s infinite", fontSize:17, fontStyle:"italic", fontFamily:"'Instrument Serif',serif"}}>Evaluating…</div>
                  : <p style={{fontSize:19, lineHeight:1.82, fontFamily:"'Instrument Serif',serif", color:C.text}}>{currentQ.text}</p>}
              </div>

              {/* Answer box */}
              <div style={card()}>
                <label style={lbl}>Your Answer</label>
                <textarea style={{...inp, minHeight:152, resize:"vertical", lineHeight:1.72, marginBottom:13}}
                  value={ans} onChange={e=>setAns(e.target.value)}
                  placeholder="Be specific. What did you do, how did you do it, what was the result? Include numbers." />
                <div style={{display:"flex", gap:10, justifyContent:"flex-end", alignItems:"center"}}>
                  <span style={{fontSize:11, color:"rgba(208,204,196,0.28)", marginRight:"auto", fontFamily:"'JetBrains Mono',monospace"}}>
                    {ans.trim().split(/\s+/).filter(Boolean).length} words
                  </span>
                  <button style={{...btn(), padding:"8px 14px", fontSize:12}} onClick={()=>{setAns("");submit();}}>Skip</button>
                  <button style={{...btn("p"), opacity:busy?0.45:1}} onClick={submit} disabled={busy}>
                    {busy ? "…" : qNum >= job.maxQ ? "Finish Interview" : "Submit →"}
                  </button>
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div style={{display:"flex", flexDirection:"column", gap:13}}>
              {/* All questions — IRT state */}
              <div style={{...card({padding:14})}}>
                <div style={{fontSize:10, letterSpacing:"0.1em", color:"rgba(255,255,255,0.25)", textTransform:"uppercase", marginBottom:10}}>
                  All Questions · IRT b Parameters
                </div>
                <div style={{display:"flex", flexDirection:"column", gap:5, maxHeight:220, overflowY:"auto"}}>
                  {questions.slice(0,16).map(q => (
                    <div key={q.id} style={{display:"flex", justifyContent:"space-between", alignItems:"center",
                      padding:"5px 9px", borderRadius:7, background: q.id===currentQ.id?"rgba(91,156,246,0.08)":"transparent",
                      border: q.id===currentQ.id?`1px solid ${C.blue}30`:"1px solid transparent"}}>
                      <span style={{fontSize:11, color:C.muted, overflow:"hidden", textOverflow:"ellipsis",
                        whiteSpace:"nowrap", maxWidth:140}}>{q.topic}</span>
                      <IRTBadge qid={q.id} irt={GLOBAL_IRT} short />
                    </div>
                  ))}
                </div>
              </div>

              {/* Last answer feedback */}
              {last && (
                <div style={{...card({padding:14})}}>
                  <div style={{fontSize:10, letterSpacing:"0.1em", color:"rgba(255,255,255,0.25)", textTransform:"uppercase", marginBottom:10}}>
                    Previous Answer
                  </div>
                  <div style={{display:"flex", justifyContent:"space-between", marginBottom:9}}>
                    <span style={{fontSize:12, color:C.muted}}>Q{last.qNum}</span>
                    <span style={{fontFamily:"'JetBrains Mono',monospace", fontSize:14, fontWeight:500,
                      color:last.score>=0.65?"#4ade80":last.score>=0.44?"#facc15":"#f87171"}}>
                      {(last.score*100).toFixed(1)}%
                    </span>
                  </div>
                  <ScoreBar label="Relevance"   value={last.signals?.[0]||0} color={C.blue} mono />
                  <div style={{fontSize:11, color:C.muted, fontStyle:"italic", marginTop:8, lineHeight:1.6}}>
                    "{last.tip}"
                  </div>
                  <div style={{marginTop:8, display:"flex", flexWrap:"wrap", gap:4}}>
                    {Object.entries(last.signals||{}).filter(([,v])=>v).map(([k])=>(
                      <span key={k} style={{fontSize:9, color:"#4ade80", background:"rgba(74,222,128,0.08)",
                        padding:"1px 6px", borderRadius:99, border:"1px solid rgba(74,222,128,0.2)"}}>✓{k.replace("has","")}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Score history grid */}
              {qas.length > 0 && (
                <div style={{...card({padding:14})}}>
                  <div style={{fontSize:10, letterSpacing:"0.1em", color:"rgba(255,255,255,0.25)", textTransform:"uppercase", marginBottom:9}}>
                    History
                  </div>
                  <div style={{display:"flex", flexWrap:"wrap", gap:5}}>
                    {qas.map((qa, i) => {
                      const irtItem = GLOBAL_IRT.items[qa.qid];
                      const col = irtItem ? irtColor(irtItem.b) : "#888";
                      return (
                        <div key={i} title={`Q${i+1}: score=${(qa.score*100).toFixed(0)}% b=${irtItem?.b?.toFixed(2)||"?"}`}
                          style={{width:32, height:32, borderRadius:7, display:"flex", flexDirection:"column",
                            alignItems:"center", justifyContent:"center", cursor:"default",
                            background: col.replace("rgb","rgba").replace(")",",0.12)"),
                            border:`1.5px solid ${col.replace("rgb","rgba").replace(")",qa.score>0.6?",0.7)":",0.25)")}`}}>
                          <div style={{fontSize:9, color:col, fontFamily:"'JetBrains Mono',monospace", lineHeight:1}}>
                            {(qa.score*100).toFixed(0)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── RESULTS ─────────────────────────────────────────────────────────────────
  const R = viewing || results;
  if ((page==="results"||viewing) && R) {
    const { candidateName, jobTitle, resumeScore, resumeData, interviewScore, pass, qas, report, scoring, questions } = R;
    return (
      <div style={app}>
        <style>{css}</style>
        <div style={{...wrap, paddingTop:44, paddingBottom:60}}>
          <div style={{textAlign:"center", marginBottom:38}}>
            <div style={{fontSize:50, marginBottom:12}}>{pass?"🏆":"📋"}</div>
            <h2 style={{fontFamily:"'Instrument Serif',serif", fontSize:38, color:pass?"#4ade80":"#facc15", marginBottom:6}}>
              {pass?"Selected — Congratulations!":"Interview Complete"}
            </h2>
            <p style={{color:C.muted}}>{candidateName} · {jobTitle}</p>
          </div>

          {/* Score cards */}
          <div style={{display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14, marginBottom:22}}>
            {[
              {l:"Resume",        v:resumeScore},
              {l:"IRT Score",     v:interviewScore},
              {l:"θ̂ Normalised",  v:scoring.thetaNorm||0},
              {l:"Info-Weighted", v:scoring.infoWt||scoring.raw?.reduce((a,b)=>a+b,0)/qas.length||0},
            ].map((s,i) => (
              <div key={i} style={{...card({textAlign:"center", padding:18})}}>
                <div style={{fontSize:28, fontFamily:"'JetBrains Mono',monospace",
                  color:s.v>=0.65?"#4ade80":s.v>=0.46?"#facc15":"#f87171"}}>{(s.v*100).toFixed(1)}%</div>
                <div style={{fontSize:12, color:C.muted, marginTop:5}}>{s.l}</div>
              </div>
            ))}
          </div>

          <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:20}}>
            {/* Candidate */}
            <div style={card()}>
              <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:14}}>Candidate Feedback</div>
              <p style={{color:C.text, lineHeight:1.82, marginBottom:16}}>{report.candidateSummary}</p>
              <div style={{fontSize:11, color:"#4ade80", fontWeight:700, marginBottom:8}}>✓ Strengths</div>
              {report.strengths.map((s,i)=><div key={i} style={{color:C.muted, fontSize:13, marginBottom:5, paddingLeft:10}}>• {s}</div>)}
              <div style={{fontSize:11, color:"#fb923c", fontWeight:700, margin:"14px 0 8px"}}>↑ Improve</div>
              {report.improves.map((s,i)=><div key={i} style={{color:C.muted, fontSize:13, marginBottom:5, paddingLeft:10}}>• {s}</div>)}
            </div>
            {/* HR */}
            <div style={card()}>
              <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:14}}>HR Assessment</div>
              <p style={{color:C.text, lineHeight:1.82, marginBottom:14}}>{report.hrSummary}</p>
              <ScoreBar label="TF-IDF Similarity" value={resumeData?.sim||0}      color={C.blue}   mono />
              <ScoreBar label="Topic Coverage"     value={resumeData?.coverage||0} color="#34d399"  mono />
              <ScoreBar label="Experience"         value={resumeData?.expScore||0} color="#fbbf24"  mono />
              <div style={{marginTop:14, padding:"10px 13px", borderRadius:8, background:"rgba(255,255,255,0.03)",
                fontSize:12, color:C.muted, fontFamily:"'JetBrains Mono',monospace", lineHeight:1.8}}>
                θ̂ = {scoring.theta?.toFixed(3)} · slope = {scoring.slope?.toFixed(3)}<br/>
                IRT N-R converged · info-weighted score
              </div>
            </div>
          </div>

          {/* Trajectory */}
          {scoring.progression?.length > 0 && (
            <div style={{...card({marginBottom:20, padding:20})}}>
              <ProgressChart progression={scoring.progression} scores={scoring.raw} />
            </div>
          )}

          {/* Per-question with live IRT state */}
          <div style={{...card({marginBottom:20})}}>
            <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:14}}>
              Per-Question — IRT b Learned from Responses (no pre-labels)
            </div>
            {qas.map((qa, i) => {
              const irtItem = GLOBAL_IRT.items[qa.qid];
              const col = irtItem ? irtColor(irtItem.b) : "#888";
              return (
                <div key={i} style={{padding:"12px 14px", borderRadius:9, background:"rgba(255,255,255,0.02)", marginBottom:7}}>
                  <div style={{display:"grid", gridTemplateColumns:"22px 1fr 110px auto", gap:10, alignItems:"start"}}>
                    <span style={{fontSize:12, color:C.muted, paddingTop:2}}>Q{i+1}</span>
                    <div>
                      <div style={{fontSize:13, color:"rgba(208,204,196,0.72)", marginBottom:3}}>{qa.question.slice(0,88)}…</div>
                      <div style={{fontSize:11, color:"rgba(208,204,196,0.35)", fontFamily:"'JetBrains Mono',monospace"}}>
                        topic:{qa.topic} · b={qa.irtB.toFixed(2)} · α={qa.irtA.toFixed(2)} · n={qa.irtN} · {qa.timeTaken}s · {qa.wc}w
                      </div>
                      <div style={{fontSize:11, color:C.muted, fontStyle:"italic", marginTop:3}}>{qa.tip}</div>
                    </div>
                    <div style={{paddingTop:2}}>
                      <IRTBadge qid={qa.qid} irt={GLOBAL_IRT} short />
                    </div>
                    <div style={{fontFamily:"'JetBrains Mono',monospace", fontSize:15, paddingTop:4,
                      color:qa.score>=0.65?"#4ade80":qa.score>=0.44?"#facc15":"#f87171"}}>
                      {(qa.score*100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* IRT model state across all questions seen */}
          <div style={{...card({marginBottom:24})}}>
            <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:14}}>
              Global IRT Model — All Questions Trained So Far
            </div>
            <div style={{display:"flex", flexWrap:"wrap", gap:8}}>
              {Object.entries(GLOBAL_IRT.items).map(([qid, item]) => (
                <div key={qid} style={{padding:"8px 13px", borderRadius:8, background:"rgba(255,255,255,0.03)",
                  border:`1px solid ${irtColor(item.b).replace("rgb","rgba").replace(")",",0.25)")}`, textAlign:"center", minWidth:90}}>
                  <div style={{fontSize:11, color:C.muted, marginBottom:4, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", maxWidth:80}}>
                    {questions?.find(q=>q.id===qid)?.topic || qid.slice(0,12)}
                  </div>
                  <div style={{fontSize:13, fontFamily:"'JetBrains Mono',monospace", color:irtColor(item.b)}}>
                    b={item.b.toFixed(2)}
                  </div>
                  <div style={{fontSize:10, color:"rgba(208,204,196,0.35)", fontFamily:"'JetBrains Mono',monospace"}}>
                    n={item.n}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{display:"flex", gap:12, justifyContent:"center"}}>
            <button style={btn("p")} onClick={()=>{setPage("apply");setREval(null);setResults(null);setSession(null);setViewing(null);setAns("");setPdfFile(null);setPdfError("");setCand({name:"",email:"",resume:""});}}>Apply Again</button>
            <button style={btn("g")} onClick={()=>{setViewing(null);setPage("hr");setHrTab("candidates");}}>HR Dashboard</button>
            <button style={btn()}   onClick={()=>{setViewing(null);setPage("home");}}>Home</button>
          </div>
        </div>
      </div>
    );
  }

  // ── HR PORTAL ───────────────────────────────────────────────────────────────
  if (page === "hr") return (
    <div style={app}>
      <style>{css}</style>
      <div style={{...wrap, paddingTop:40}}>
        <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:28}}>
          <div>
            <button style={{...btn("g"), fontSize:13, padding:"7px 15px", marginBottom:10}} onClick={()=>setPage("home")}>← Home</button>
            <h2 style={{fontSize:28, fontWeight:700}}>HR Dashboard</h2>
          </div>
          <div style={{display:"flex", gap:9}}>
            {["jobs","post","candidates"].map(t=>(
              <button key={t} style={{...btn(hrTab===t?"p":"g"), padding:"9px 18px", fontSize:13}} onClick={()=>setHrTab(t)}>
                {t==="jobs"?"📋 Jobs":t==="post"?"+ Post":"👥 Candidates"}
              </button>
            ))}
          </div>
        </div>

        {hrTab==="jobs" && (
          <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(310px,1fr))", gap:16}}>
            {jobs.map(job=>(
              <div key={job.id} style={card()}>
                <div style={{display:"flex", justifyContent:"space-between", marginBottom:10}}>
                  <div><div style={{fontSize:17, fontWeight:600}}>{job.title}</div><div style={{fontSize:13, color:C.blue}}>{job.company}</div></div>
                  <Tag>{job.candidates.length} done</Tag>
                </div>
                <p style={{color:C.muted, fontSize:13, lineHeight:1.6, marginBottom:13}}>{job.description.slice(0,110)}…</p>
                <div style={{display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, marginBottom:11}}>
                  {[["Resume",`≥${(job.resumeThreshold*100).toFixed(0)}%`],["Interview",`≥${(job.interviewThreshold*100).toFixed(0)}%`],["Questions",job.maxQ]].map(([l,v])=>(
                    <div key={l} style={{background:"rgba(255,255,255,0.04)", borderRadius:7, padding:"8px", textAlign:"center"}}>
                      <div style={{fontSize:14, fontWeight:600, color:C.blue}}>{v}</div>
                      <div style={{fontSize:10, color:C.muted}}>{l}</div>
                    </div>
                  ))}
                </div>
                <div style={{display:"flex", gap:7}}>
                  <Tag color="#4ade80">{job.candidates.filter(c=>c.pass).length} selected</Tag>
                  <Tag color="#f87171">{job.candidates.filter(c=>!c.pass).length} rejected</Tag>
                </div>
              </div>
            ))}
          </div>
        )}

        {hrTab==="post" && (
          <div style={{maxWidth:660}}>
            <div style={card()}>
              <div style={{fontSize:11, color:C.blue, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:18}}>Post New Position</div>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:12}}>
                <div><label style={lbl}>Job Title</label><input style={inp} value={jf.title} onChange={e=>setJf(p=>({...p,title:e.target.value}))} placeholder="Any role — chef, engineer, nurse…" /></div>
                <div><label style={lbl}>Company</label><input style={inp} value={jf.company} onChange={e=>setJf(p=>({...p,company:e.target.value}))} placeholder="Company name" /></div>
              </div>
              <div style={{marginBottom:12}}>
                <label style={lbl}>Job Description <span style={{color:"rgba(255,255,255,0.3)",textTransform:"none",letterSpacing:0}}>— questions auto-generated from this text</span></label>
                <textarea style={{...inp, minHeight:160, resize:"vertical", lineHeight:1.65}} value={jf.description}
                  onChange={e=>setJf(p=>({...p,description:e.target.value}))}
                  placeholder="Describe the role in plain language. The system extracts topics automatically and generates questions. No templates, no pre-labelling needed. Works for any domain." />
              </div>
              <div style={{display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:13, marginBottom:20}}>
                {[{k:"resumeThreshold",l:"Resume Threshold",min:0.1,max:0.9,step:0.05,f:v=>`${(v*100).toFixed(0)}%`},
                  {k:"interviewThreshold",l:"Interview Threshold",min:0.1,max:0.9,step:0.05,f:v=>`${(v*100).toFixed(0)}%`},
                  {k:"maxQ",l:"Max Questions",min:3,max:25,step:1,f:v=>v}].map(f=>(
                  <div key={f.k}>
                    <label style={lbl}>{f.l}: {f.f(jf[f.k])}</label>
                    <input type="range" min={f.min} max={f.max} step={f.step} value={jf[f.k]}
                      onChange={e=>setJf(p=>({...p,[f.k]:f.k==="maxQ"?parseInt(e.target.value):parseFloat(e.target.value)}))}
                      style={{width:"100%", accentColor:C.blue, cursor:"pointer"}} />
                  </div>
                ))}
              </div>
              <button style={{...btn("p"), width:"100%"}} onClick={()=>{
                if(!jf.title||!jf.description) return;
                setJobs(p=>[...p,{...jf,id:Date.now(),candidates:[]}]);
                setJf({title:"",company:"",description:"",resumeThreshold:0.38,interviewThreshold:0.54,maxQ:10});
                setHrTab("jobs");
              }}>Post Job</button>
            </div>
          </div>
        )}

        {hrTab==="candidates" && (
          jobs.every(j=>j.candidates.length===0)
          ? <div style={{...card({textAlign:"center", padding:56})}}>
              <div style={{fontSize:44, marginBottom:14}}>👥</div>
              <div style={{color:C.muted}}>No interviews yet.</div>
            </div>
          : jobs.map(job => job.candidates.length>0 && (
            <div key={job.id} style={{marginBottom:28}}>
              <div style={{fontSize:17, fontWeight:600, color:C.blue, marginBottom:12}}>{job.title}</div>
              <div style={{...card({padding:0, overflow:"hidden"})}}>
                <table style={{width:"100%", borderCollapse:"collapse"}}>
                  <thead>
                    <tr style={{background:`${C.blue}0a`}}>
                      {["Candidate","Resume","IRT Score","θ̂","Trend","Verdict",""].map(h=>(
                        <th key={h} style={{padding:"11px 15px", textAlign:"left", fontSize:11,
                          color:`${C.blue}bb`, letterSpacing:"0.08em", textTransform:"uppercase",
                          borderBottom:`1px solid ${C.bord}`}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {job.candidates.map((c,i)=>(
                      <tr key={i} style={{borderBottom:`1px solid rgba(255,255,255,0.04)`}}>
                        <td style={{padding:"11px 15px"}}><div style={{fontWeight:600}}>{c.candidateName}</div><div style={{fontSize:11,color:C.muted}}>{c.candidateEmail}</div></td>
                        <td style={{padding:"11px 15px",fontFamily:"'JetBrains Mono',monospace",color:c.resumeScore>=0.5?"#4ade80":"#facc15"}}>{(c.resumeScore*100).toFixed(1)}%</td>
                        <td style={{padding:"11px 15px",fontFamily:"'JetBrains Mono',monospace",color:c.interviewScore>=0.6?"#4ade80":c.interviewScore>=0.42?"#facc15":"#f87171"}}>{(c.interviewScore*100).toFixed(1)}%</td>
                        <td style={{padding:"11px 15px",fontFamily:"'JetBrains Mono',monospace",color:C.muted}}>{c.scoring?.theta?.toFixed(2)||"–"}</td>
                        <td style={{padding:"11px 15px",fontSize:16}}>{(c.scoring?.slope||0)>0.02?"📈":(c.scoring?.slope||0)<-0.02?"📉":"➡"}</td>
                        <td style={{padding:"11px 15px"}}><Tag color={c.pass?"#4ade80":"#f87171"}>{c.pass?"Selected":"Rejected"}</Tag></td>
                        <td style={{padding:"11px 15px"}}><button style={{...btn("g"),padding:"5px 11px",fontSize:11}} onClick={()=>{setViewing(c);setPage("results");}}>View →</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  return null;
}
