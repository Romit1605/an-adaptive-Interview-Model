import { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { apiGet } from "../api";
import { ScoreBar, IRTBadge, ProgressChart, Tag, irtColor } from "../components/UI";

const C = {
    bg: "#050810", card: "rgba(255,255,255,0.028)", bord: "rgba(255,255,255,0.08)",
    text: "#d0ccc4", muted: "rgba(208,204,196,0.45)", blue: "#5b9cf6", gold: "#e0a020",
};
const card = (x = {}) => ({ background: C.card, border: `1px solid ${C.bord}`, borderRadius: 14, padding: 24, backdropFilter: "blur(4px)", ...x });
const btn = (v = "p") => ({ cursor: "pointer", border: "none", borderRadius: 9, fontFamily: "'Outfit',sans-serif", fontWeight: 600, fontSize: 14, padding: "11px 24px", transition: "all .16s", ...(v === "p" ? { background: "linear-gradient(135deg,#2563eb,#5b9cf6)", color: "#fff" } : v === "g" ? { background: "transparent", color: C.blue, border: `1px solid ${C.blue}44` } : { background: "rgba(255,255,255,0.05)", color: C.text, border: `1px solid ${C.bord}` }) });

export default function Results() {
    const { applicationId } = useParams();
    const { user } = useAuth();
    const navigate = useNavigate();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    const fetchResults = useCallback(async () => {
        try {
            const res = await apiGet(`/api/interview/results/${applicationId}`);
            setData(res);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, [applicationId]);

    useEffect(() => { fetchResults(); }, [fetchResults]);

    if (loading) {
        return (
            <div style={{ maxWidth: 700, margin: "80px auto", textAlign: "center" }}>
                <div style={{ fontSize: 36, marginBottom: 14, animation: "blink 1s infinite" }}>⏳</div>
                <div style={{ color: C.muted }}>Loading results…</div>
            </div>
        );
    }

    if (!data) {
        return (
            <div style={{ maxWidth: 700, margin: "80px auto", textAlign: "center" }}>
                <div style={{ fontSize: 44, marginBottom: 14 }}>⚠️</div>
                <div style={{ color: C.muted, marginBottom: 20 }}>Results not found.</div>
                <button style={btn("g")} onClick={() => navigate("/candidate/dashboard")}>← Dashboard</button>
            </div>
        );
    }

    const { candidateName, jobTitle, resumeScore, resumeData, interviewScore, scoring, pass, qas, questions, report, irtSnapshot } = data;
    const snap = irtSnapshot || {};

    return (
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "44px 26px 60px" }}>
            {/* Header */}
            <div style={{ textAlign: "center", marginBottom: 38 }}>
                <div style={{ fontSize: 50, marginBottom: 12 }}>{pass ? "🏆" : "📋"}</div>
                <h2 style={{ fontFamily: "'Instrument Serif',serif", fontSize: 38, color: pass ? "#4ade80" : "#facc15", marginBottom: 6 }}>
                    {pass ? "Selected — Congratulations!" : "Interview Complete"}
                </h2>
                <p style={{ color: C.muted }}>{candidateName} · {jobTitle}</p>
            </div>

            {/* Score cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 22 }}>
                {[
                    { l: "Resume", v: resumeScore },
                    { l: "IRT Score", v: interviewScore },
                    { l: "θ̂ Normalised", v: scoring?.thetaNorm || 0 },
                    { l: "Info-Weighted", v: scoring?.infoWt || 0 },
                ].map((s, i) => (
                    <div key={i} style={{ ...card({ textAlign: "center", padding: 18 }) }}>
                        <div style={{ fontSize: 28, fontFamily: "'JetBrains Mono',monospace", color: s.v >= 0.65 ? "#4ade80" : s.v >= 0.46 ? "#facc15" : "#f87171" }}>{(s.v * 100).toFixed(1)}%</div>
                        <div style={{ fontSize: 12, color: C.muted, marginTop: 5 }}>{s.l}</div>
                    </div>
                ))}
            </div>

            {/* Reports */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
                <div style={card()}>
                    <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>Candidate Feedback</div>
                    <p style={{ color: C.text, lineHeight: 1.82, marginBottom: 16 }}>{report?.candidateSummary}</p>
                    <div style={{ fontSize: 11, color: "#4ade80", fontWeight: 700, marginBottom: 8 }}>✓ Strengths</div>
                    {(report?.strengths || []).map((s, i) => <div key={i} style={{ color: C.muted, fontSize: 13, marginBottom: 5, paddingLeft: 10 }}>• {s}</div>)}
                    <div style={{ fontSize: 11, color: "#fb923c", fontWeight: 700, margin: "14px 0 8px" }}>↑ Improve</div>
                    {(report?.improves || []).map((s, i) => <div key={i} style={{ color: C.muted, fontSize: 13, marginBottom: 5, paddingLeft: 10 }}>• {s}</div>)}
                </div>
                <div style={card()}>
                    <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>Technical Assessment</div>
                    <p style={{ color: C.text, lineHeight: 1.82, marginBottom: 14 }}>{report?.hrSummary}</p>
                    <ScoreBar label="TF-IDF Similarity" value={resumeData?.sim || 0} color={C.blue} mono />
                    <ScoreBar label="Topic Coverage" value={resumeData?.coverage || 0} color="#34d399" mono />
                    <ScoreBar label="Experience" value={resumeData?.expScore || 0} color="#fbbf24" mono />
                    <div style={{ marginTop: 14, padding: "10px 13px", borderRadius: 8, background: "rgba(255,255,255,0.03)", fontSize: 12, color: C.muted, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.8 }}>
                        θ̂ = {scoring?.theta?.toFixed(3)} · slope = {scoring?.slope?.toFixed(3)}<br />
                        Hybrid: AI + TF-IDF + IRT Newton-Raphson
                    </div>
                </div>
            </div>

            {/* Score trajectory */}
            {scoring?.progression?.length > 0 && (
                <div style={{ ...card({ marginBottom: 20, padding: 20 }) }}>
                    <ProgressChart progression={scoring.progression} scores={scoring.raw} />
                </div>
            )}

            {/* Per-question details */}
            <div style={{ ...card({ marginBottom: 20 }) }}>
                <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>
                    Per-Question · IRT b Learned from Responses
                </div>
                {qas?.map((qa, i) => {
                    const item = snap[qa.qid];
                    const col = item ? irtColor(item.b) : "#888";
                    return (
                        <div key={i} style={{ padding: "12px 14px", borderRadius: 9, background: "rgba(255,255,255,0.02)", marginBottom: 7 }}>
                            <div style={{ display: "grid", gridTemplateColumns: "22px 1fr 110px auto", gap: 10, alignItems: "start" }}>
                                <span style={{ fontSize: 12, color: C.muted, paddingTop: 2 }}>Q{i + 1}</span>
                                <div>
                                    <div style={{ fontSize: 13, color: "rgba(208,204,196,0.72)", marginBottom: 3 }}>{qa.question?.slice(0, 88)}…</div>
                                    <div style={{ fontSize: 11, color: "rgba(208,204,196,0.35)", fontFamily: "'JetBrains Mono',monospace" }}>
                                        topic:{qa.topic} · b={qa.irtB?.toFixed(2)} · α={qa.irtA?.toFixed(2)} · n={qa.irtN} · {qa.wc}w
                                    </div>
                                    <div style={{ fontSize: 11, color: C.muted, fontStyle: "italic", marginTop: 3 }}>{qa.tip}</div>
                                </div>
                                <div style={{ paddingTop: 2 }}>
                                    <IRTBadge qid={qa.qid} irtItems={snap} short />
                                </div>
                                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 15, paddingTop: 4, color: qa.score >= 0.65 ? "#4ade80" : qa.score >= 0.44 ? "#facc15" : "#f87171" }}>
                                    {(qa.score * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
                <button style={btn("p")} onClick={() => navigate("/candidate/dashboard")}>Dashboard</button>
                <button style={btn("g")} onClick={() => navigate("/")}>Home</button>
            </div>
        </div>
    );
}
