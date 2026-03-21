import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";

/* ── Live question cycling ───────────────────────────── */
const QUESTIONS = [
    "Explain how you would design a fault-tolerant distributed cache at scale.",
    "Walk me through your approach to REST vs GraphQL API design decisions.",
    "How would you handle database migration with zero downtime in production?",
];

/* ── Mini IRT score ring (SVG) ───────────────────────── */
function ScoreRing({ value }) {
    const r = 42, c = 2 * Math.PI * r;
    const color = value >= 0.7 ? "#4ade80" : value >= 0.5 ? "#facc15" : "#f87171";
    return (
        <svg viewBox="0 0 100 100" width={96} height={96}>
            <defs>
                <linearGradient id="srg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
            </defs>
            <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
            <circle
                cx="50" cy="50" r={r} fill="none"
                stroke="url(#srg)" strokeWidth="8"
                strokeDasharray={c} strokeDashoffset={c * (1 - value)}
                strokeLinecap="round" transform="rotate(-90 50 50)"
                style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.22,1,0.36,1)", filter: `drop-shadow(0 0 6px ${color}99)` }}
            />
            <text x="50" y="47" textAnchor="middle" fontSize="17" fontWeight="800" fill="#fff" fontFamily="'JetBrains Mono',monospace">
                {(value * 100).toFixed(0)}%
            </text>
            <text x="50" y="61" textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.38)" fontFamily="'Outfit',sans-serif" letterSpacing="1">
                IRT SCORE
            </text>
        </svg>
    );
}

export default function Landing() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [qIdx, setQIdx] = useState(0);
    const [score, setScore] = useState(0.74);
    const [theta, setTheta] = useState(0.88);
    const [fade, setFade] = useState(true);

    useEffect(() => {
        const t = setInterval(() => {
            setFade(false);
            setTimeout(() => {
                setQIdx(i => (i + 1) % QUESTIONS.length);
                setScore(+(0.52 + Math.random() * 0.42).toFixed(2));
                setTheta(+(0.4 + Math.random() * 1.2).toFixed(2));
                setFade(true);
            }, 350);
        }, 3400);
        return () => clearInterval(t);
    }, []);

    const progression = [38, 30, 22, 15, 8, 4];
    const progXs = [0, 24, 48, 72, 96, 120];

    return (
        <div style={{ position: "relative", overflow: "hidden" }}>

            {/* ── Dot Grid Background ───────────────────────── */}
            <div className="grid-bg" style={{ position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.5 }} />

            {/* ── Hero Glow Orbs ─────────────────────────────── */}
            <div style={{
                position: "absolute", top: "-10%", left: "50%", transform: "translateX(-25%)",
                width: 760, height: 460, borderRadius: "50%",
                background: "radial-gradient(ellipse, rgba(37,99,235,0.22) 0%, transparent 68%)",
                filter: "blur(52px)", pointerEvents: "none",
                animation: "heroOrb 16s ease-in-out infinite alternate",
            }} />
            <div style={{
                position: "absolute", top: "6%", right: "-8%",
                width: 400, height: 400, borderRadius: "50%",
                background: "radial-gradient(ellipse, rgba(139,92,246,0.17) 0%, transparent 70%)",
                filter: "blur(36px)", pointerEvents: "none",
                animation: "glowDriftB 13s ease-in-out infinite alternate",
            }} />
            <div style={{
                position: "absolute", bottom: "4%", left: "-5%",
                width: 320, height: 320, borderRadius: "50%",
                background: "radial-gradient(ellipse, rgba(16,185,129,0.1) 0%, transparent 70%)",
                filter: "blur(28px)", pointerEvents: "none",
                animation: "glowDriftA 20s ease-in-out infinite alternate",
            }} />

            {/* ════ HERO ══════════════════════════════════════ */}
            <div style={{ maxWidth: 1160, margin: "0 auto", padding: "96px 26px 64px", position: "relative", zIndex: 1 }}>
                <div className="hero-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>

                    {/* Left: Copy */}
                    <div>
                        <div className="up" style={{
                            display: "inline-flex", alignItems: "center", gap: 8,
                            padding: "5px 14px", borderRadius: 99,
                            background: "rgba(37,99,235,0.12)", border: "1px solid rgba(91,156,246,0.22)",
                            fontSize: 11, color: "rgba(91,156,246,0.75)", letterSpacing: "0.18em",
                            textTransform: "uppercase", marginBottom: 24,
                        }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 6px #4ade80", display: "inline-block", animation: "pulse 2s infinite" }} />
                            AI-Powered · Adaptive · Real-time
                        </div>

                        <h1 className="up" style={{
                            fontFamily: "'Instrument Serif',serif",
                            fontSize: "clamp(40px, 5vw, 64px)",
                            lineHeight: 1.02, marginBottom: 20,
                            background: "linear-gradient(135deg, #e2e8f0 0%, #a5c8ff 45%, #c4b5fd 100%)",
                            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                        }}>
                            The Interview Engine<br />
                            <span style={{ fontStyle: "italic", background: "linear-gradient(135deg,#5b9cf6,#e0a020)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                                That Trains Itself
                            </span>
                        </h1>

                        <p className="up" style={{ fontSize: 15.5, color: "rgba(208,204,196,0.58)", maxWidth: 500, marginBottom: 14, lineHeight: 1.9 }}>
                            Questions adapt in real-time to your ability via Item Response Theory.
                            Every interview is <strong style={{ color: "#93c5fd", fontWeight: 600 }}>personalized, fair, and AI-driven</strong> — from job description to final score.
                        </p>

                        <p className="up" style={{ fontSize: 12, color: "rgba(74,222,128,0.55)", marginBottom: 36, fontFamily: "'JetBrains Mono',monospace", letterSpacing: "0.08em" }}>
                            🚀 NextGen-HR · Empowering Your Career Journey
                        </p>

                        <div className="up" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                            {user ? (
                                <button className="btn-primary" style={{ fontSize: 15, padding: "13px 34px" }}
                                    onClick={() => navigate(user.role === "hr" ? "/hr/dashboard" : "/candidate/dashboard")}>
                                    Go to Dashboard →
                                </button>
                            ) : (
                                <>
                                    <button className="btn-primary" style={{ fontSize: 15, padding: "13px 34px" }}
                                        onClick={() => navigate("/candidate/login")}>Apply as Candidate</button>
                                    <button className="btn-outline" style={{ fontSize: 15, padding: "13px 34px" }}
                                        onClick={() => navigate("/hr/login")}>HR Portal</button>
                                </>
                            )}
                        </div>

                        {/* Stats strip */}
                        <div className="up" style={{ display: "flex", gap: 32, marginTop: 44, paddingTop: 28, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                            {[["1,200+", "Interviews Run"], ["94%", "Score Accuracy"], ["40+", "Job Domains"]].map(([v, l]) => (
                                <div key={l}>
                                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 24, fontWeight: 800, color: "#fff", lineHeight: 1 }}>{v}</div>
                                    <div style={{ fontSize: 11, color: "rgba(208,204,196,0.35)", marginTop: 5, letterSpacing: "0.04em" }}>{l}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Right: Animated Live Mock UI */}
                    <div className="hero-visual" style={{ position: "relative", height: 500 }}>

                        {/* Main interview card */}
                        <div className="glass-bright" style={{
                            position: "absolute", top: 20, left: 10, right: 10,
                            padding: "22px 24px", borderRadius: 20,
                            boxShadow: "0 32px 96px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.07), inset 0 1px 0 rgba(255,255,255,0.08)",
                            animation: "floatA 6s ease-in-out infinite",
                        }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 8px #4ade80", animation: "pulse 2s infinite" }} />
                                    <span style={{ fontSize: 10, color: "rgba(74,222,128,0.75)", fontFamily: "'JetBrains Mono',monospace", letterSpacing: "0.14em" }}>ADAPTIVE INTERVIEW · LIVE</span>
                                </div>
                                <div style={{ fontSize: 10, color: "rgba(208,204,196,0.3)", fontFamily: "'JetBrains Mono',monospace" }}>
                                    Q{qIdx + 1}/10 · θ̂={theta}
                                </div>
                            </div>

                            {/* Difficulty bar */}
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                                <span style={{ fontSize: 9, color: "rgba(208,204,196,0.3)", letterSpacing: "0.1em", textTransform: "uppercase" }}>Difficulty</span>
                                <div style={{ flex: 1, height: 3, borderRadius: 99, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                                    <div className="score-bar-fill" style={{
                                        height: "100%", borderRadius: 99, width: `${(0.4 + score * 0.6) * 100}%`,
                                        background: "linear-gradient(90deg, #3b82f6, #8b5cf6)",
                                    }} />
                                </div>
                                <span style={{ fontSize: 9, color: "rgba(139,92,246,0.7)", fontFamily: "'JetBrains Mono',monospace" }}>b={(score * 3 - 1.5).toFixed(2)}</span>
                            </div>

                            <div style={{
                                fontFamily: "'Instrument Serif',serif", fontSize: 16, color: "#e2e8f0",
                                lineHeight: 1.7, minHeight: 72,
                                opacity: fade ? 1 : 0, transition: "opacity 0.35s ease",
                            }}>
                                {QUESTIONS[qIdx]}
                            </div>

                            <div style={{ height: 1, background: "rgba(255,255,255,0.05)", margin: "14px 0" }} />

                            <div style={{ display: "flex", gap: 8 }}>
                                {[{ t: "Confident", c: "#3b82f6" }, { t: "Partially", c: "#f59e0b" }, { t: "Skip", c: null }].map(({ t, c }) => (
                                    <div key={t} style={{
                                        padding: "5px 13px", borderRadius: 7, fontSize: 11, fontWeight: 600, cursor: "default",
                                        background: c ? `${c}1a` : "rgba(255,255,255,0.04)",
                                        border: `1px solid ${c ? `${c}33` : "rgba(255,255,255,0.07)"}`,
                                        color: c ? c : "rgba(208,204,196,0.3)",
                                    }}>{t}</div>
                                ))}
                            </div>
                        </div>

                        {/* Score ring — bottom right */}
                        <div className="glass" style={{
                            position: "absolute", bottom: 20, right: 0, width: 138,
                            padding: "14px 14px 10px", borderRadius: 18, textAlign: "center",
                            boxShadow: "0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)",
                            animation: "floatB 7.5s ease-in-out infinite",
                        }}>
                            <ScoreRing value={score} />
                            <div style={{ fontSize: 9, color: "rgba(208,204,196,0.28)", marginTop: 4, fontFamily: "'JetBrains Mono',monospace" }}>
                                θ̂ = {theta} · ↑ rising
                            </div>
                        </div>

                        {/* Ability progression — bottom left */}
                        <div className="glass" style={{
                            position: "absolute", bottom: 50, left: 0, width: 162,
                            padding: "13px 14px", borderRadius: 16,
                            boxShadow: "0 16px 48px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.06)",
                            animation: "floatC 8s ease-in-out infinite",
                        }}>
                            <div style={{ fontSize: 9, color: "rgba(91,156,246,0.65)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 10 }}>θ̂ Progression</div>
                            <svg viewBox="0 0 120 48" width="100%" style={{ overflow: "visible", display: "block" }}>
                                <defs>
                                    <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.45" />
                                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.02" />
                                    </linearGradient>
                                </defs>
                                <path d="M0,38 L24,30 L48,22 L72,15 L96,8 L120,4" fill="none" stroke="#5b9cf6" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                                <path d="M0,38 L24,30 L48,22 L72,15 L96,8 L120,4 L120,48 L0,48 Z" fill="url(#pg)" />
                                {progXs.map((x, i) => (
                                    <circle key={i} cx={x} cy={progression[i]} r="3.5"
                                        fill={i === progXs.length - 1 ? "#4ade80" : "#5b9cf6"}
                                        style={i === progXs.length - 1 ? { filter: "drop-shadow(0 0 5px #4ade80)" } : {}} />
                                ))}
                            </svg>
                            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                                <span style={{ fontSize: 9, color: "rgba(208,204,196,0.28)", fontFamily: "'JetBrains Mono',monospace" }}>Q1</span>
                                <span style={{ fontSize: 9, color: "#4ade80", fontFamily: "'JetBrains Mono',monospace" }}>↑ Improving</span>
                            </div>
                        </div>

                        {/* Thompson Sampling badge — top right */}
                        <div className="glass" style={{
                            position: "absolute", top: 16, right: -4, width: 132,
                            padding: "10px 13px", borderRadius: 13,
                            boxShadow: "0 10px 32px rgba(0,0,0,0.4)",
                            animation: "floatB 9s ease-in-out infinite",
                        }}>
                            <div style={{ fontSize: 9, color: "rgba(251,146,60,0.7)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>MAB · Thompson</div>
                            {[["Topic A", 72, "#3b82f6"], ["Topic B", 45, "#8b5cf6"], ["Topic C", 88, "#10b981"]].map(([t, w, col]) => (
                                <div key={t} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                                    <div style={{ fontSize: 9, color: "rgba(208,204,196,0.38)", width: 46, fontFamily: "'JetBrains Mono',monospace" }}>{t}</div>
                                    <div style={{ flex: 1, height: 4, borderRadius: 99, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                                        <div style={{ height: "100%", borderRadius: 99, width: `${w}%`, background: col, opacity: 0.75 }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* ════ FEATURE GRID ══════════════════════════════ */}
            <div style={{ maxWidth: 1160, margin: "0 auto 90px", padding: "0 26px", position: "relative", zIndex: 1 }}>
                <div style={{ textAlign: "center", marginBottom: 44 }}>
                    <div style={{ fontSize: 11, letterSpacing: "0.22em", color: "rgba(91,156,246,0.5)", textTransform: "uppercase", marginBottom: 10 }}>Under the Hood</div>
                    <h2 style={{
                        fontFamily: "'Instrument Serif',serif", fontSize: 40, lineHeight: 1.1,
                        background: "linear-gradient(135deg, #e2e8f0, #94a3b8)",
                        WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                    }}>Built on real science</h2>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(210px,1fr))", gap: 14 }}>
                    {[
                        { icon: "◎", h: "Zero Pre-Labels", b: "Every question starts with b=0, a=1. Difficulty emerges from real responses via online gradient descent. IRT params persist in MongoDB.", color: "#5b9cf6" },
                        { icon: "⟳", h: "Job-Driven Questions", b: "Topics extracted via TF-IDF from any job description. Gemini AI generates tailored questions — works for chefs, engineers, nurses, any role.", color: "#34d399" },
                        { icon: "θ̂", h: "IRT Ability Scoring", b: "Score = MLE estimate θ̂ via Newton-Raphson, weighted by Fisher information I(θ). Full adaptive psychometrics in Python.", color: "#a78bfa" },
                        { icon: "⚡", h: "Thompson Sampling", b: "Beta(α,β) posterior per question arm. Exploration is automatic. Bandit selects the most informative next question by IRT information.", color: "#fb923c" },
                        { icon: "🔐", h: "Secure Multi-Portal", b: "Separate HR and Candidate accounts. HR sees only their own jobs and candidates. Resumes stored privately in GCP Cloud Storage.", color: "#f472b6" },
                    ].map((c, i) => (
                        <div key={i} className="glass card-hover up" style={{ padding: "22px 20px", animationDelay: `${i * 0.07}s`, position: "relative", overflow: "hidden" }}>
                            <div style={{
                                position: "absolute", top: -24, right: -24, width: 90, height: 90, borderRadius: "50%",
                                background: `radial-gradient(${c.color}28, transparent 70%)`, pointerEvents: "none",
                            }} />
                            <div style={{ fontSize: 28, marginBottom: 12, color: c.color, filter: `drop-shadow(0 0 10px ${c.color}66)`, lineHeight: 1 }}>{c.icon}</div>
                            <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 8 }}>{c.h}</div>
                            <div style={{ fontSize: 12.5, color: "rgba(208,204,196,0.48)", lineHeight: 1.75 }}>{c.b}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
