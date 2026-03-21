import { useState, useEffect, useCallback } from "react";
import { useAuth } from "../context/AuthContext";
import { apiGet, apiPost } from "../api";
import { ScoreBar, Tag, irtColor, IRTBadge, ProgressChart } from "../components/UI";

const C = {
    bg: "#050810", card: "rgba(255,255,255,0.028)", bord: "rgba(255,255,255,0.08)",
    text: "#d0ccc4", muted: "rgba(208,204,196,0.45)", blue: "#5b9cf6", gold: "#e0a020",
};
const card = (x = {}) => ({ background: C.card, border: `1px solid ${C.bord}`, borderRadius: 14, padding: 24, backdropFilter: "blur(4px)", ...x });
const btn = (v = "p") => ({ cursor: "pointer", border: "none", borderRadius: 9, fontFamily: "'Outfit',sans-serif", fontWeight: 600, fontSize: 14, padding: "11px 24px", transition: "all .16s", ...(v === "p" ? { background: "linear-gradient(135deg,#2563eb,#5b9cf6)", color: "#fff" } : v === "g" ? { background: "transparent", color: C.blue, border: `1px solid ${C.blue}44` } : { background: "rgba(255,255,255,0.05)", color: C.text, border: `1px solid ${C.bord}` }) });
const inp = { width: "100%", padding: "11px 14px", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.bord}`, borderRadius: 8, color: C.text, fontSize: 15, fontFamily: "'Outfit',sans-serif", outline: "none" };
const lbl = { display: "block", marginBottom: 5, fontSize: 11, color: `${C.blue}cc`, letterSpacing: "0.1em", textTransform: "uppercase", fontFamily: "'Outfit',sans-serif" };

export default function HRDashboard() {
    const { user } = useAuth();
    const [tab, setTab] = useState("jobs");
    const [jobs, setJobs] = useState([]);
    const [selectedJob, setSelectedJob] = useState(null);
    const [candidates, setCandidates] = useState([]);
    const [viewingReport, setViewingReport] = useState(null);
    const [jf, setJf] = useState({ title: "", company: user?.company || "", description: "", resumeThreshold: 0.38, interviewThreshold: 0.54, maxQ: 10 });
    const [loading, setLoading] = useState(false);

    const fetchJobs = useCallback(async () => {
        try {
            const data = await apiGet("/api/hr/jobs");
            setJobs(data.jobs || []);
        } catch (e) { console.error(e); }
    }, []);

    useEffect(() => { fetchJobs(); }, [fetchJobs]);

    const postJob = async () => {
        if (!jf.title || !jf.description) return;
        setLoading(true);
        try {
            await apiPost("/api/hr/jobs", jf);
            setJf({ title: "", company: user?.company || "", description: "", resumeThreshold: 0.38, interviewThreshold: 0.54, maxQ: 10 });
            await fetchJobs();
            setTab("jobs");
        } catch (e) {
            alert("Error: " + e.message);
        } finally { setLoading(false); }
    };

    const viewCandidates = async (jobId) => {
        setLoading(true);
        setSelectedJob(jobId);
        try {
            const data = await apiGet(`/api/hr/jobs/${jobId}/candidates`);
            setCandidates(data.candidates || []);
            setTab("candidates");
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const viewReport = async (applicationId) => {
        setLoading(true);
        try {
            const data = await apiGet(`/api/hr/candidates/${applicationId}/report`);
            setViewingReport(data);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    // ── REPORT VIEW ──
    if (viewingReport) {
        const { application, job, interview } = viewingReport;
        const scoring = interview?.scoring || {};
        const report = interview?.report || {};
        const qas = interview?.qas || [];
        const irtSnapshot = interview?.irtSnapshot || {};

        return (
            <div style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 26px" }}>
                <button style={{ ...btn("g"), fontSize: 13, marginBottom: 24 }} onClick={() => setViewingReport(null)}>← Back to Candidates</button>

                <div style={{ textAlign: "center", marginBottom: 32 }}>
                    <div style={{ fontSize: 44, marginBottom: 10 }}>{interview?.pass ? "🏆" : "📋"}</div>
                    <h2 style={{ fontFamily: "'Instrument Serif',serif", fontSize: 34, color: interview?.pass ? "#4ade80" : "#facc15", marginBottom: 6 }}>
                        {interview?.pass ? "Selected — Recommended Hire" : "Interview Complete"}
                    </h2>
                    <p style={{ color: C.muted }}>{application.candidateName} · {job.title} at {job.company}</p>
                    {application.resumeUrl && (
                        <a href={application.resumeUrl} target="_blank" rel="noopener noreferrer" style={{ color: C.blue, fontSize: 13, textDecoration: "none", display: "inline-block", marginTop: 6 }}>📄 Download Resume</a>
                    )}
                </div>

                {/* Score Cards */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 22 }}>
                    {[
                        { l: "Resume", v: application.resumeScore },
                        { l: "IRT Score", v: interview?.interviewScore || 0 },
                        { l: "θ̂ Normalised", v: scoring.thetaNorm || 0 },
                        { l: "Info-Weighted", v: scoring.infoWt || 0 },
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
                        <p style={{ color: C.text, lineHeight: 1.82, marginBottom: 16 }}>{report.candidateSummary}</p>
                        <div style={{ fontSize: 11, color: "#4ade80", fontWeight: 700, marginBottom: 8 }}>✓ Strengths</div>
                        {(report.strengths || []).map((s, i) => <div key={i} style={{ color: C.muted, fontSize: 13, marginBottom: 5, paddingLeft: 10 }}>• {s}</div>)}
                        <div style={{ fontSize: 11, color: "#fb923c", fontWeight: 700, margin: "14px 0 8px" }}>↑ Improve</div>
                        {(report.improves || []).map((s, i) => <div key={i} style={{ color: C.muted, fontSize: 13, marginBottom: 5, paddingLeft: 10 }}>• {s}</div>)}
                    </div>
                    <div style={card()}>
                        <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>HR Assessment</div>
                        <p style={{ color: C.text, lineHeight: 1.82, marginBottom: 14 }}>{report.hrSummary}</p>
                        <ScoreBar label="TF-IDF Similarity" value={application.resumeData?.sim || 0} color={C.blue} mono />
                        <ScoreBar label="Topic Coverage" value={application.resumeData?.coverage || 0} color="#34d399" mono />
                        <ScoreBar label="Experience" value={application.resumeData?.expScore || 0} color="#fbbf24" mono />
                        <div style={{ marginTop: 14, padding: "10px 13px", borderRadius: 8, background: "rgba(255,255,255,0.03)", fontSize: 12, color: C.muted, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.8 }}>
                            θ̂ = {scoring.theta?.toFixed(3)} · slope = {scoring.slope?.toFixed(3)}<br />
                            Hybrid: AI + TF-IDF + IRT Newton-Raphson
                        </div>
                    </div>
                </div>

                {scoring.progression?.length > 0 && (
                    <div style={{ ...card({ marginBottom: 20, padding: 20 }) }}>
                        <ProgressChart progression={scoring.progression} scores={scoring.raw} />
                    </div>
                )}

                {/* Q&A Detail */}
                <div style={{ ...card({ marginBottom: 24 }) }}>
                    <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>
                        All Questions & Answers ({qas.length} questions)
                    </div>
                    {qas.map((qa, i) => (
                        <div key={i} style={{ padding: "16px 14px", borderRadius: 10, background: "rgba(255,255,255,0.02)", marginBottom: 10, border: `1px solid ${C.bord}` }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <span style={{ fontSize: 12, color: C.blue, fontWeight: 700 }}>Q{qa.qNum || i + 1}</span>
                                    <Tag color={C.blue}>{qa.topic}</Tag>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <IRTBadge qid={qa.qid} irtItems={irtSnapshot} short />
                                    <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 16, fontWeight: 600, color: qa.score >= 0.65 ? "#4ade80" : qa.score >= 0.44 ? "#facc15" : "#f87171" }}>
                                        {(qa.score * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                            <div style={{ fontSize: 14, color: C.text, marginBottom: 8, fontFamily: "'Instrument Serif',serif", lineHeight: 1.7 }}>
                                {qa.question}
                            </div>
                            <div style={{ fontSize: 13, color: C.muted, lineHeight: 1.7, padding: "10px 13px", borderRadius: 8, background: "rgba(255,255,255,0.02)", border: `1px solid rgba(255,255,255,0.04)`, marginBottom: 8 }}>
                                <span style={{ fontSize: 10, color: `${C.blue}88`, letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>Candidate's Answer</span>
                                {qa.answer || <em style={{ color: "rgba(255,255,255,0.2)" }}>Skipped</em>}
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <div style={{ fontSize: 11, color: C.muted, fontStyle: "italic" }}>💡 {qa.tip}</div>
                                <div style={{ display: "flex", gap: 4 }}>
                                    {Object.entries(qa.signals || {}).filter(([, v]) => v).map(([k]) => (
                                        <span key={k} style={{ fontSize: 9, color: "#4ade80", background: "rgba(74,222,128,0.08)", padding: "1px 6px", borderRadius: 99, border: "1px solid rgba(74,222,128,0.2)" }}>✓{k.replace("has", "")}</span>
                                    ))}
                                </div>
                            </div>
                            <div style={{ fontSize: 10, color: "rgba(208,204,196,0.3)", fontFamily: "'JetBrains Mono',monospace", marginTop: 6 }}>
                                b={qa.irtB?.toFixed(2)} · α={qa.irtA?.toFixed(2)} · n={qa.irtN} · {qa.wc}w · TF-IDF={((qa.tfidfRelevance || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 26px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
                <div>
                    <h2 style={{ fontSize: 28, fontWeight: 700, color: C.text }}>HR Dashboard</h2>
                    <p style={{ fontSize: 13, color: C.muted }}>Welcome, {user?.name} · {user?.company}</p>
                </div>
                <div style={{ display: "flex", gap: 9 }}>
                    {["jobs", "post", "candidates"].map(t => (
                        <button key={t} style={{ ...btn(tab === t ? "p" : "g"), padding: "9px 18px", fontSize: 13 }} onClick={() => { setTab(t); if (t === "candidates" && !selectedJob && jobs.length > 0) viewCandidates(jobs[0].id); }}>
                            {t === "jobs" ? "📋 My Jobs" : t === "post" ? "+ Post Job" : "👥 Candidates"}
                        </button>
                    ))}
                </div>
            </div>

            {/* ── JOBS TAB ── */}
            {tab === "jobs" && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(310px,1fr))", gap: 16 }}>
                    {jobs.length === 0 ? (
                        <div style={{ ...card({ textAlign: "center", padding: 56 }) }}>
                            <div style={{ fontSize: 44, marginBottom: 14 }}>📋</div>
                            <div style={{ color: C.muted, marginBottom: 16 }}>No jobs posted yet.</div>
                            <button style={btn("p")} onClick={() => setTab("post")}>Post Your First Job</button>
                        </div>
                    ) : jobs.map(job => (
                        <div key={job.id} style={card()}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                                <div>
                                    <div style={{ fontSize: 17, fontWeight: 600, color: C.text }}>{job.title}</div>
                                    <div style={{ fontSize: 13, color: C.blue }}>{job.company}</div>
                                </div>
                                <Tag>{job.candidateCount || 0} applied</Tag>
                            </div>
                            <p style={{ color: C.muted, fontSize: 13, lineHeight: 1.6, marginBottom: 13 }}>{job.description.slice(0, 110)}…</p>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
                                {[["Resume", `≥${(job.resumeThreshold * 100).toFixed(0)}%`], ["Interview", `≥${(job.interviewThreshold * 100).toFixed(0)}%`], ["Questions", job.maxQ]].map(([l, v]) => (
                                    <div key={l} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 7, padding: 8, textAlign: "center" }}>
                                        <div style={{ fontSize: 14, fontWeight: 600, color: C.blue }}>{v}</div>
                                        <div style={{ fontSize: 10, color: C.muted }}>{l}</div>
                                    </div>
                                ))}
                            </div>
                            <button style={{ ...btn("g"), width: "100%", fontSize: 13 }} onClick={() => viewCandidates(job.id)}>View Candidates →</button>
                        </div>
                    ))}
                </div>
            )}

            {/* ── POST JOB TAB ── */}
            {tab === "post" && (
                <div style={{ maxWidth: 660 }}>
                    <div style={card()}>
                        <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 18 }}>Post New Position</div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                            <div><label style={lbl}>Job Title</label><input style={inp} value={jf.title} onChange={e => setJf(p => ({ ...p, title: e.target.value }))} placeholder="Any role — chef, engineer, nurse…" /></div>
                            <div><label style={lbl}>Company</label><input style={inp} value={jf.company} onChange={e => setJf(p => ({ ...p, company: e.target.value }))} placeholder="Company name" /></div>
                        </div>
                        <div style={{ marginBottom: 12 }}>
                            <label style={lbl}>Job Description <span style={{ color: "rgba(255,255,255,0.3)", textTransform: "none", letterSpacing: 0 }}>— AI generates questions from this text</span></label>
                            <textarea style={{ ...inp, minHeight: 160, resize: "vertical", lineHeight: 1.65 }} value={jf.description} onChange={e => setJf(p => ({ ...p, description: e.target.value }))} placeholder="Describe the role. Include skills, responsibilities, and experience needed." />
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 13, marginBottom: 20 }}>
                            {[
                                { k: "resumeThreshold", l: "Resume Threshold", min: 0.1, max: 0.9, step: 0.05, f: v => `${(v * 100).toFixed(0)}%` },
                                { k: "interviewThreshold", l: "Interview Threshold", min: 0.1, max: 0.9, step: 0.05, f: v => `${(v * 100).toFixed(0)}%` },
                                { k: "maxQ", l: "Max Questions", min: 3, max: 25, step: 1, f: v => v },
                            ].map(f => (
                                <div key={f.k}>
                                    <label style={lbl}>{f.l}: {f.f(jf[f.k])}</label>
                                    <input type="range" min={f.min} max={f.max} step={f.step} value={jf[f.k]}
                                        onChange={e => setJf(p => ({ ...p, [f.k]: f.k === "maxQ" ? parseInt(e.target.value) : parseFloat(e.target.value) }))}
                                        style={{ width: "100%", accentColor: C.blue, cursor: "pointer" }} />
                                </div>
                            ))}
                        </div>
                        <button style={{ ...btn("p"), width: "100%", opacity: loading ? 0.5 : 1 }} onClick={postJob} disabled={loading}>
                            {loading ? "Posting…" : "Post Job"}
                        </button>
                    </div>
                </div>
            )}

            {/* ── CANDIDATES TAB ── */}
            {tab === "candidates" && (
                <div>
                    {/* Job selector */}
                    <div style={{ display: "flex", gap: 8, marginBottom: 18, flexWrap: "wrap" }}>
                        {jobs.map(j => (
                            <button key={j.id} style={{ ...btn(selectedJob === j.id ? "p" : "g"), padding: "7px 16px", fontSize: 12 }}
                                onClick={() => viewCandidates(j.id)}>
                                {j.title}
                            </button>
                        ))}
                    </div>

                    {loading ? (
                        <div style={{ ...card({ textAlign: "center", padding: 40 }), color: C.muted }}>Loading candidates…</div>
                    ) : candidates.length === 0 ? (
                        <div style={{ ...card({ textAlign: "center", padding: 56 }) }}>
                            <div style={{ fontSize: 44, marginBottom: 14 }}>👥</div>
                            <div style={{ color: C.muted }}>No candidates have applied to this job yet.</div>
                        </div>
                    ) : (
                        <div style={{ ...card({ padding: 0, overflow: "hidden" }) }}>
                            <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                <thead>
                                    <tr style={{ background: `${C.blue}0a` }}>
                                        {["Candidate", "Resume", "IRT Score", "θ̂", "Trend", "Verdict", ""].map(h => (
                                            <th key={h} style={{ padding: "11px 15px", textAlign: "left", fontSize: 11, color: `${C.blue}bb`, letterSpacing: "0.08em", textTransform: "uppercase", borderBottom: `1px solid ${C.bord}` }}>{h}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {candidates.map((c, i) => (
                                        <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                                            <td style={{ padding: "11px 15px" }}>
                                                <div style={{ fontWeight: 600, color: C.text }}>{c.candidateName}</div>
                                                <div style={{ fontSize: 11, color: C.muted }}>{c.candidateEmail}</div>
                                            </td>
                                            <td style={{ padding: "11px 15px", fontFamily: "'JetBrains Mono',monospace", color: c.resumeScore >= 0.5 ? "#4ade80" : "#facc15" }}>
                                                {(c.resumeScore * 100).toFixed(1)}%
                                            </td>
                                            <td style={{ padding: "11px 15px", fontFamily: "'JetBrains Mono',monospace", color: (c.interview?.interviewScore || 0) >= 0.6 ? "#4ade80" : (c.interview?.interviewScore || 0) >= 0.42 ? "#facc15" : "#f87171" }}>
                                                {c.interview ? `${(c.interview.interviewScore * 100).toFixed(1)}%` : "—"}
                                            </td>
                                            <td style={{ padding: "11px 15px", fontFamily: "'JetBrains Mono',monospace", color: C.muted }}>
                                                {c.interview?.scoring?.theta?.toFixed(2) || "—"}
                                            </td>
                                            <td style={{ padding: "11px 15px", fontSize: 16 }}>
                                                {c.interview ? ((c.interview.scoring?.slope || 0) > 0.02 ? "📈" : (c.interview.scoring?.slope || 0) < -0.02 ? "📉" : "➡") : "—"}
                                            </td>
                                            <td style={{ padding: "11px 15px" }}>
                                                {c.interview?.status === "completed" ? (
                                                    <Tag color={c.interview.pass ? "#4ade80" : "#f87171"}>{c.interview.pass ? "Selected" : "Rejected"}</Tag>
                                                ) : c.interview?.status === "active" ? (
                                                    <Tag color="#facc15">In Progress</Tag>
                                                ) : (
                                                    <Tag color={C.muted}>Pending</Tag>
                                                )}
                                            </td>
                                            <td style={{ padding: "11px 15px" }}>
                                                {c.interview?.status === "completed" && (
                                                    <button style={{ ...btn("g"), padding: "5px 11px", fontSize: 11 }} onClick={() => viewReport(c.application_id)}>
                                                        View Report →
                                                    </button>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
