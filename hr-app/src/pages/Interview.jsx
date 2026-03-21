import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { apiPost, apiGet } from "../api";
import { IRTBadge, irtColor, Tag } from "../components/UI";

const C = {
    bg: "#050810", card: "rgba(255,255,255,0.028)", bord: "rgba(255,255,255,0.08)",
    text: "#d0ccc4", muted: "rgba(208,204,196,0.45)", blue: "#5b9cf6", gold: "#e0a020",
};
const card = (x = {}) => ({ background: C.card, border: `1px solid ${C.bord}`, borderRadius: 14, padding: 24, backdropFilter: "blur(4px)", ...x });
const btn = (v = "p") => ({ cursor: "pointer", border: "none", borderRadius: 9, fontFamily: "'Outfit',sans-serif", fontWeight: 600, fontSize: 14, padding: "11px 24px", transition: "all .16s", ...(v === "p" ? { background: "linear-gradient(135deg,#2563eb,#5b9cf6)", color: "#fff" } : v === "g" ? { background: "transparent", color: C.blue, border: `1px solid ${C.blue}44` } : { background: "rgba(255,255,255,0.05)", color: C.text, border: `1px solid ${C.bord}` }) });
const inp = { width: "100%", padding: "11px 14px", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.bord}`, borderRadius: 8, color: C.text, fontSize: 15, fontFamily: "'Outfit',sans-serif", outline: "none" };

const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

export default function Interview() {
    const { applicationId } = useParams();
    const { user } = useAuth();
    const navigate = useNavigate();

    const [session, setSession] = useState(null);
    const [ans, setAns] = useState("");
    const [timer, setTimer] = useState(0);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const timerRef = useRef(null);

    // Start or resume interview
    const initInterview = useCallback(async () => {
        setBusy(true);
        try {
            // Try to get status first
            const status = await apiGet(`/api/interview/status/${applicationId}`);
            if (status.status === "active") {
                setSession(status);
                return;
            }
            if (status.status === "completed") {
                navigate(`/candidate/results/${applicationId}`);
                return;
            }
            // Start new interview
            const data = await apiPost("/api/interview/start", { application_id: applicationId });
            setSession(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setBusy(false);
        }
    }, [applicationId, navigate]);

    useEffect(() => { initInterview(); }, [initInterview]);

    // Timer
    useEffect(() => {
        if (session?.status === "active") {
            timerRef.current = setInterval(() => setTimer(t => t + 1), 1000);
        } else {
            clearInterval(timerRef.current);
        }
        return () => clearInterval(timerRef.current);
    }, [session?.status]);

    // Submit answer
    const submit = async () => {
        if (!session || busy) return;
        setBusy(true);
        try {
            const data = await apiPost("/api/interview/submit", {
                application_id: applicationId,
                answer: ans,
            });

            if (data.done) {
                // Interview complete
                navigate(`/candidate/results/${applicationId}`);
                return;
            }

            // Update session with next question
            setSession(prev => ({
                ...prev,
                currentQuestion: data.nextQuestion,
                qNum: data.qNum,
                theta: data.theta,
                qas: [...(prev?.qas || []), data.qa],
                irtItems: data.irtItems || prev?.irtItems || {},
            }));
            setAns("");
            setTimer(0);
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            setBusy(false);
        }
    };

    if (error) {
        return (
            <div style={{ maxWidth: 700, margin: "60px auto", padding: "0 26px", textAlign: "center" }}>
                <div style={{ fontSize: 44, marginBottom: 14 }}>⚠️</div>
                <h2 style={{ fontFamily: "'Instrument Serif',serif", fontSize: 28, color: "#f87171", marginBottom: 10 }}>Cannot Start Interview</h2>
                <p style={{ color: C.muted, marginBottom: 20 }}>{error}</p>
                <button style={btn("g")} onClick={() => navigate("/candidate/dashboard")}>← Back to Dashboard</button>
            </div>
        );
    }

    if (!session || busy && !session?.currentQuestion) {
        return (
            <div style={{ maxWidth: 700, margin: "80px auto", padding: "0 26px", textAlign: "center" }}>
                <div style={{ fontSize: 36, marginBottom: 14, animation: "blink 1s infinite" }}>⏳</div>
                <div style={{ color: C.muted, fontSize: 16 }}>Preparing your interview…</div>
                <div style={{ color: "rgba(255,255,255,0.2)", fontSize: 12, marginTop: 6 }}>Generating questions with AI · Initializing IRT & Thompson Sampling</div>
            </div>
        );
    }

    const { currentQuestion, qas = [], qNum, theta, maxQ, irtItems = {}, questions = [] } = session;
    const progress = qNum / maxQ;
    const last = qas[qas.length - 1];

    return (
        <div style={{ maxWidth: 960, margin: "0 auto", padding: "38px 26px" }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
                <div>
                    <div style={{ fontSize: 12, color: C.muted }}>{user?.name}</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Question {qNum} / {maxQ}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 26, fontFamily: "'JetBrains Mono',monospace", color: C.gold }}>{fmt(timer)}</div>
                    <div style={{ fontSize: 11, color: "rgba(208,204,196,0.3)", fontFamily: "'JetBrains Mono',monospace" }}>θ̂={theta?.toFixed(2) || "0.00"}</div>
                </div>
            </div>

            {/* Progress bar */}
            <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 99, marginBottom: 22 }}>
                <div style={{ height: "100%", width: `${progress * 100}%`, borderRadius: 99, background: `linear-gradient(90deg,${C.blue},#a5c8ff)`, transition: "width 0.4s" }} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 270px", gap: 16 }}>
                <div>
                    {/* IRT status */}
                    <div style={{ ...card({ marginBottom: 13, padding: 14 }) }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <div>
                                <div style={{ fontSize: 10, letterSpacing: "0.1em", color: "rgba(255,255,255,0.25)", textTransform: "uppercase", marginBottom: 5 }}>
                                    IRT State · Learned from Responses
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <IRTBadge qid={currentQuestion?.id} irtItems={irtItems} />
                                    <span style={{ fontSize: 12, color: C.muted }}>
                                        {(irtItems[currentQuestion?.id]?.n || 0) === 0
                                            ? "— first time this question has been asked"
                                            : `— seen by ${irtItems[currentQuestion?.id]?.n} candidate${irtItems[currentQuestion?.id]?.n > 1 ? "s" : ""}`}
                                    </span>
                                </div>
                            </div>
                            <div style={{ fontSize: 12, color: C.muted, textAlign: "right" }}>
                                Topic: <span style={{ color: C.blue }}>{currentQuestion?.topic}</span>
                            </div>
                        </div>
                    </div>

                    {/* Question */}
                    <div className="in" style={{ ...card({ marginBottom: 13 }) }}>
                        <div style={{ fontSize: 10, letterSpacing: "0.12em", color: `${C.blue}88`, textTransform: "uppercase", marginBottom: 14 }}>
                            AI-Generated · Topic: {currentQuestion?.topic}
                        </div>
                        {busy ? (
                            <div style={{ color: C.muted, animation: "blink 1s infinite", fontSize: 17, fontStyle: "italic", fontFamily: "'Instrument Serif',serif" }}>Evaluating your answer…</div>
                        ) : (
                            <p style={{ fontSize: 19, lineHeight: 1.82, fontFamily: "'Instrument Serif',serif", color: C.text }}>{currentQuestion?.text}</p>
                        )}
                    </div>

                    {/* Answer box */}
                    <div style={card()}>
                        <label style={{ display: "block", marginBottom: 5, fontSize: 11, color: `${C.blue}cc`, letterSpacing: "0.1em", textTransform: "uppercase" }}>Your Answer</label>
                        <textarea style={{ ...inp, minHeight: 152, resize: "vertical", lineHeight: 1.72, marginBottom: 13 }}
                            value={ans} onChange={e => setAns(e.target.value)}
                            placeholder="Be specific. What did you do, how did you do it, what was the result? Include numbers." />
                        <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", alignItems: "center" }}>
                            <span style={{ fontSize: 11, color: "rgba(208,204,196,0.28)", marginRight: "auto", fontFamily: "'JetBrains Mono',monospace" }}>
                                {ans.trim().split(/\s+/).filter(Boolean).length} words
                            </span>
                            <button style={{ ...btn(), padding: "8px 14px", fontSize: 12 }} onClick={() => { setAns(""); submit(); }}>Skip</button>
                            <button style={{ ...btn("p"), opacity: busy ? 0.45 : 1 }} onClick={submit} disabled={busy}>
                                {busy ? "…" : qNum >= maxQ ? "Finish Interview" : "Submit →"}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Sidebar */}
                <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                    {/* IRT b params */}
                    <div style={{ ...card({ padding: 14 }) }}>
                        <div style={{ fontSize: 10, letterSpacing: "0.1em", color: "rgba(255,255,255,0.25)", textTransform: "uppercase", marginBottom: 10 }}>
                            All Questions · IRT b
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 220, overflowY: "auto" }}>
                            {questions.slice(0, 16).map(q => (
                                <div key={q.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "5px 9px", borderRadius: 7, background: q.id === currentQuestion?.id ? "rgba(91,156,246,0.08)" : "transparent", border: q.id === currentQuestion?.id ? `1px solid ${C.blue}30` : "1px solid transparent" }}>
                                    <span style={{ fontSize: 11, color: C.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 140 }}>{q.topic}</span>
                                    <IRTBadge qid={q.id} irtItems={irtItems} short />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Previous answer feedback */}
                    {last && (
                        <div style={{ ...card({ padding: 14 }) }}>
                            <div style={{ fontSize: 10, letterSpacing: "0.1em", color: "rgba(255,255,255,0.25)", textTransform: "uppercase", marginBottom: 10 }}>
                                Previous Answer
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 9 }}>
                                <span style={{ fontSize: 12, color: C.muted }}>Q{last.qNum}</span>
                                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 14, fontWeight: 500, color: last.score >= 0.65 ? "#4ade80" : last.score >= 0.44 ? "#facc15" : "#f87171" }}>
                                    {(last.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div style={{ fontSize: 11, color: C.muted, fontStyle: "italic", marginTop: 8, lineHeight: 1.6 }}>"{last.tip}"</div>
                            <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 4 }}>
                                {Object.entries(last.signals || {}).filter(([, v]) => v).map(([k]) => (
                                    <span key={k} style={{ fontSize: 9, color: "#4ade80", background: "rgba(74,222,128,0.08)", padding: "1px 6px", borderRadius: 99, border: "1px solid rgba(74,222,128,0.2)" }}>✓{k.replace("has", "")}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Score history */}
                    {qas.length > 0 && (
                        <div style={{ ...card({ padding: 14 }) }}>
                            <div style={{ fontSize: 10, letterSpacing: "0.1em", color: "rgba(255,255,255,0.25)", textTransform: "uppercase", marginBottom: 9 }}>History</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                                {qas.map((qa, i) => {
                                    const item = irtItems[qa.qid];
                                    const col = item ? irtColor(item.b) : "#888";
                                    return (
                                        <div key={i} title={`Q${i + 1}: score=${(qa.score * 100).toFixed(0)}%`}
                                            style={{ width: 32, height: 32, borderRadius: 7, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "default", background: col.replace("rgb", "rgba").replace(")", ",0.12)"), border: `1.5px solid ${col.replace("rgb", "rgba").replace(")", qa.score > 0.6 ? ",0.7)" : ",0.25)")}` }}>
                                            <div style={{ fontSize: 9, color: col, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1 }}>
                                                {(qa.score * 100).toFixed(0)}
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
    );
}
