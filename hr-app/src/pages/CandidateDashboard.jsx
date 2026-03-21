import { useState, useEffect, useCallback, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import { apiGet, getToken } from "../api";
import { Tag, ScoreBar, ScoreRing } from "../components/UI";

// ── PDF Extractor (unchanged) ────────────────────────────────────────────────
async function extractPDFText(base64Data) {
    if (!window.pdfjsLib) {
        await new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
            script.onload = resolve; script.onerror = reject;
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

// ── Design tokens ────────────────────────────────────────────────────────────
const C = {
    bg: "#02040d", card: "rgba(255,255,255,0.025)", bord: "rgba(255,255,255,0.07)",
    text: "#d0ccc4", muted: "rgba(208,204,196,0.42)", blue: "#5b9cf6", gold: "#f59e0b",
};
const glass = (x = {}) => ({
    background: C.card, border: `1px solid ${C.bord}`,
    borderRadius: 16, backdropFilter: "blur(14px)", ...x,
});
const primaryBtn = {
    cursor: "pointer", border: "none", borderRadius: 10,
    fontFamily: "'Outfit',sans-serif", fontWeight: 700, fontSize: 14,
    padding: "12px 24px", background: "linear-gradient(135deg,#1d4ed8,#3b82f6,#6366f1)",
    color: "#fff", transition: "all 0.22s cubic-bezier(0.22,1,0.36,1)",
    boxShadow: "0 4px 18px rgba(59,130,246,0.3)",
};
const outlineBtn = {
    cursor: "pointer", background: "rgba(91,156,246,0.07)",
    border: "1px solid rgba(91,156,246,0.25)", borderRadius: 10,
    fontFamily: "'Outfit',sans-serif", fontWeight: 600, fontSize: 14,
    padding: "12px 24px", color: "#5b9cf6", transition: "all 0.2s",
};

export default function CandidateDashboard() {
    const { user } = useAuth();
    const [tab, setTab] = useState("browse");
    const [jobs, setJobs] = useState([]);
    const [applications, setApplications] = useState([]);
    const [selectedJob, setSelectedJob] = useState(null);
    const [resumeText, setResumeText] = useState("");
    const [resumeMode, setResumeMode] = useState("paste");
    const [pdfFile, setPdfFile] = useState(null);
    const [pdfBusy, setPdfBusy] = useState(false);
    const [pdfError, setPdfError] = useState("");
    const [resumeResult, setResumeResult] = useState(null);
    const [busy, setBusy] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef(null);

    const fetchJobs = useCallback(async () => {
        try { const data = await apiGet("/api/jobs"); setJobs(data.jobs || []); } catch (e) { console.error(e); }
    }, []);

    const fetchApplications = useCallback(async () => {
        try { const data = await apiGet("/api/candidate/applications"); setApplications(data.applications || []); } catch (e) { console.error(e); }
    }, []);

    useEffect(() => { fetchJobs(); fetchApplications(); }, [fetchJobs, fetchApplications]);

    const handlePDFUpload = async (file) => {
        if (!file || file.type !== "application/pdf") { setPdfError("Please upload a valid PDF file."); return; }
        setPdfBusy(true); setPdfError(""); setPdfFile(null); setResumeText("");
        try {
            const base64 = await new Promise((res, rej) => {
                const reader = new FileReader();
                reader.onload = () => res(reader.result.split(",")[1]);
                reader.onerror = () => rej(new Error("File read failed"));
                reader.readAsDataURL(file);
            });
            setPdfFile(file);
            setResumeText(await extractPDFText(base64));
        } catch { setPdfError("Failed to extract text from PDF. Please paste your resume manually."); }
        finally { setPdfBusy(false); }
    };

    const applyToJob = async () => {
        if (!selectedJob || !resumeText.trim()) return;
        setBusy(true);
        try {
            const formData = new FormData();
            formData.append("resume_text", resumeText);
            if (pdfFile) formData.append("resume_file", pdfFile);
            const token = getToken();
            const res = await fetch(`http://localhost:8000/api/jobs/${selectedJob}/apply`, {
                method: "POST", headers: { Authorization: `Bearer ${token}` }, body: formData,
            });
            if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || `Error ${res.status}`); }
            const data = await res.json();
            setResumeResult(data);
            await fetchApplications();
        } catch (e) { alert("Error: " + e.message); }
        finally { setBusy(false); }
    };

    const appliedJobIds = new Set(applications.map(a => a.job_id));

    return (
        <div style={{ maxWidth: 1160, margin: "0 auto", padding: "40px 28px 80px" }}>

            {/* ── Header ── */}
            <div className="up" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 36 }}>
                <div>
                    <h1 style={{ fontSize: 32, fontWeight: 800, color: C.text, letterSpacing: "-0.02em", marginBottom: 4 }}>
                        Candidate Dashboard
                    </h1>
                    <p style={{ fontSize: 14, color: C.muted }}>Welcome back, <span style={{ color: C.blue, fontWeight: 600 }}>{user?.name}</span></p>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                    {["browse", "applications"].map(t => (
                        <button key={t}
                            style={{
                                ...(tab === t ? primaryBtn : outlineBtn),
                                padding: "9px 18px", fontSize: 13,
                            }}
                            onClick={() => setTab(t)}>
                            {t === "browse" ? "🔍 Browse Jobs" : `📋 My Applications ${applications.length > 0 ? `(${applications.length})` : ""}`}
                        </button>
                    ))}
                </div>
            </div>

            {/* ── BROWSE JOBS ── */}
            {tab === "browse" && (
                <div style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: 24 }}>

                    {/* Left — Resume & Apply */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

                        {/* Resume Card */}
                        <div className="up in" style={glass({ padding: "28px 28px 24px" })}>
                            <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.14em", textTransform: "uppercase", fontWeight: 700, marginBottom: 18 }}>
                                📄 Your Resume
                            </div>

                            {/* Mode Toggle */}
                            <div style={{ display: "flex", gap: 0, marginBottom: 18, borderRadius: 10, overflow: "hidden", border: "1px solid rgba(255,255,255,0.07)", width: "fit-content" }}>
                                {["paste", "pdf"].map(mode => (
                                    <button key={mode} onClick={() => setResumeMode(mode)} style={{
                                        padding: "8px 20px", fontSize: 12, fontFamily: "'Outfit',sans-serif", fontWeight: 700,
                                        border: "none", cursor: "pointer", transition: "all .18s",
                                        background: resumeMode === mode ? "linear-gradient(135deg,#1d4ed8,#6366f1)" : "rgba(255,255,255,0.02)",
                                        color: resumeMode === mode ? "#fff" : C.muted,
                                        letterSpacing: "0.07em", textTransform: "uppercase",
                                    }}>
                                        {mode === "paste" ? "✏ Paste Text" : "📄 Upload PDF"}
                                    </button>
                                ))}
                            </div>

                            {resumeMode === "paste" ? (
                                <textarea
                                    className="ng-input"
                                    value={resumeText} onChange={e => setResumeText(e.target.value)}
                                    placeholder="Paste your full resume here — skills, experience, achievements, education..."
                                    style={{ minHeight: 220, resize: "vertical", lineHeight: 1.72 }}
                                />
                            ) : (
                                <div>
                                    {/* Premium PDF drop zone */}
                                    <div
                                        onClick={() => !pdfBusy && fileInputRef.current?.click()}
                                        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                                        onDragLeave={() => setDragOver(false)}
                                        onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handlePDFUpload(f); }}
                                        style={{
                                            border: `2px dashed ${pdfFile ? "#4ade80" : dragOver ? "#5b9cf6" : "rgba(91,156,246,0.25)"}`,
                                            borderRadius: 14, padding: "36px 24px", textAlign: "center",
                                            cursor: pdfBusy ? "wait" : "pointer",
                                            background: pdfFile ? "rgba(74,222,128,0.04)" : dragOver ? "rgba(91,156,246,0.06)" : "rgba(255,255,255,0.015)",
                                            transition: "all 0.22s cubic-bezier(0.22,1,0.36,1)",
                                            boxShadow: dragOver ? "0 0 0 4px rgba(91,156,246,0.1)" : "none",
                                        }}>
                                        {pdfBusy ? (
                                            <div>
                                                <div style={{ fontSize: 36, marginBottom: 12, animation: "pulse 1.2s infinite" }}>⏳</div>
                                                <div style={{ fontSize: 14, color: C.muted, fontWeight: 500 }}>Extracting text from PDF…</div>
                                                <div style={{ width: 60, height: 3, background: "rgba(91,156,246,0.3)", borderRadius: 99, margin: "12px auto 0", animation: "shimmer 1.4s ease infinite", backgroundSize: "200% 100%" }} />
                                            </div>
                                        ) : pdfFile ? (
                                            <div>
                                                <div style={{ fontSize: 36, marginBottom: 10 }}>✅</div>
                                                <div style={{ fontSize: 15, fontWeight: 700, color: "#4ade80", marginBottom: 5 }}>{pdfFile.name}</div>
                                                <div style={{ fontSize: 12, color: C.muted }}>{resumeText.split(/\s+/).filter(Boolean).length} words extracted successfully</div>
                                            </div>
                                        ) : (
                                            <div>
                                                <div style={{ fontSize: 40, marginBottom: 12, filter: "drop-shadow(0 0 12px rgba(91,156,246,0.4))" }}>📄</div>
                                                <div style={{ fontSize: 15, fontWeight: 600, color: C.text, marginBottom: 6 }}>Drop your PDF resume here</div>
                                                <div style={{ fontSize: 12, color: C.muted }}>or click to browse · PDF only · Uploads to GCP</div>
                                            </div>
                                        )}
                                    </div>
                                    <input ref={fileInputRef} type="file" accept="application/pdf" style={{ display: "none" }}
                                        onChange={e => { const f = e.target.files?.[0]; if (f) handlePDFUpload(f); e.target.value = ""; }} />
                                    {pdfError && (
                                        <div style={{ marginTop: 10, padding: "10px 14px", borderRadius: 9, color: "#f87171", fontSize: 13, background: "rgba(248,113,113,0.07)", border: "1px solid rgba(248,113,113,0.18)" }}>
                                            {pdfError}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Resume Result */}
                        {resumeResult && (
                            <div className="in" style={glass({ padding: 24, borderColor: resumeResult.pass ? "rgba(74,222,128,0.2)" : "rgba(248,113,113,0.18)" })}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                                    <div>
                                        <div style={{ fontSize: 11, color: C.blue, letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 700, marginBottom: 4 }}>Resume Analysis</div>
                                        <div style={{ fontSize: 13, color: C.muted }}>
                                            {resumeResult.pass ? "✅ Qualified for interview" : "❌ Below minimum threshold"}
                                        </div>
                                    </div>
                                    <ScoreRing value={resumeResult.score} size={72} />
                                </div>
                                <ScoreBar label="TF-IDF Semantic Similarity" value={resumeResult.sim} color="#5b9cf6" mono />
                                <ScoreBar label="Topic Coverage" value={resumeResult.coverage} color="#34d399" mono />
                                <ScoreBar label="Experience Score" value={resumeResult.expScore} color="#fbbf24" mono />

                                <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginBottom: 7, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                                        Topics covered ({resumeResult.covered?.length}/{resumeResult.jdTerms?.length})
                                    </div>
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 10 }}>
                                        {resumeResult.covered?.slice(0, 12).map(t => <Tag key={t} color="#4ade80">{t}</Tag>)}
                                    </div>
                                    {resumeResult.missing?.length > 0 && (
                                        <>
                                            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.22)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.1em" }}>Missing topics</div>
                                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                                                {resumeResult.missing.slice(0, 8).map(t => <Tag key={t} color="#f87171">{t}</Tag>)}
                                            </div>
                                        </>
                                    )}
                                </div>

                                {resumeResult.pass ? (
                                    <a href={`/interview/${resumeResult.application_id}`}
                                        style={{ display: "block", textAlign: "center", marginTop: 20, ...primaryBtn, fontSize: 15, textDecoration: "none", width: "100%" }}>
                                        Begin Interview →
                                    </a>
                                ) : (
                                    <div style={{ marginTop: 16, padding: "12px 16px", borderRadius: 10, color: "#f87171", fontSize: 13, background: "rgba(248,113,113,0.06)", border: "1px solid rgba(248,113,113,0.18)" }}>
                                        Score {(resumeResult.score * 100).toFixed(1)}% is below the {(resumeResult.threshold * 100).toFixed(0)}% threshold. Improve your resume and reapply.
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Right — Job list */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                        <div className="up" style={{ fontSize: 11, color: C.blue, letterSpacing: "0.14em", textTransform: "uppercase", fontWeight: 700 }}>
                            Open Positions
                        </div>

                        {jobs.map((job, i) => (
                            <div key={job.id} className="up card-hover"
                                onClick={() => { setSelectedJob(job.id); setResumeResult(null); }}
                                style={{
                                    ...glass({
                                        padding: "18px 20px", cursor: "pointer",
                                        transition: "all 0.2s cubic-bezier(0.22,1,0.36,1)",
                                        animationDelay: `${i * 0.06}s`,
                                        ...(selectedJob === job.id ? { borderColor: "rgba(91,156,246,0.35)", background: "rgba(91,156,246,0.06)", boxShadow: "0 0 0 1px rgba(91,156,246,0.1)" } : {}),
                                    }),
                                }}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                                    <div>
                                        <div style={{ fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 2 }}>{job.title}</div>
                                        <div style={{ fontSize: 13, color: C.blue, fontWeight: 500 }}>{job.hrCompany || job.company}</div>
                                        <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Posted by: {job.hrName}</div>
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 5 }}>
                                        <Tag color="#a5c8ff">{job.maxQ}Q</Tag>
                                        {appliedJobIds.has(job.id) && <Tag color="#4ade80">Applied ✓</Tag>}
                                    </div>
                                </div>
                                <p style={{ color: C.muted, fontSize: 12, lineHeight: 1.65, marginBottom: 10 }}>{job.description.slice(0, 100)}…</p>
                                <div style={{ display: "flex", gap: 8 }}>
                                    <span style={{ fontSize: 11, color: "rgba(208,204,196,0.28)", padding: "3px 8px", borderRadius: 6, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                                        Resume ≥{(job.resumeThreshold * 100).toFixed(0)}%
                                    </span>
                                    <span style={{ fontSize: 11, color: "rgba(208,204,196,0.28)", padding: "3px 8px", borderRadius: 6, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                                        Interview ≥{(job.interviewThreshold * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        ))}

                        <button
                            className="up"
                            style={{
                                ...primaryBtn, width: "100%", fontSize: 15, padding: "14px 24px", marginTop: 4,
                                opacity: (!resumeText.trim() || !selectedJob || busy || pdfBusy) ? 0.35 : 1,
                                cursor: (!resumeText.trim() || !selectedJob || busy || pdfBusy) ? "not-allowed" : "pointer",
                            }}
                            onClick={applyToJob}
                            disabled={!resumeText.trim() || !selectedJob || busy || pdfBusy}>
                            {busy ? "⏳ Scoring Resume…" : pdfBusy ? "⏳ Extracting PDF…" : "✨ Apply & Score Resume →"}
                        </button>
                    </div>
                </div>
            )}

            {/* ── MY APPLICATIONS ── */}
            {tab === "applications" && (
                <div>
                    {applications.length === 0 ? (
                        <div className="up" style={glass({ textAlign: "center", padding: "64px 40px" })}>
                            <div style={{ fontSize: 48, marginBottom: 16 }}>📋</div>
                            <div style={{ fontSize: 18, fontWeight: 700, color: C.text, marginBottom: 8 }}>No applications yet</div>
                            <div style={{ color: C.muted, marginBottom: 24, maxWidth: 320, margin: "0 auto 24px" }}>Browse open positions and apply to get started on your adaptive interview journey.</div>
                            <button style={primaryBtn} onClick={() => setTab("browse")}>🔍 Browse Open Jobs</button>
                        </div>
                    ) : (
                        <div style={{ display: "grid", gap: 14 }}>
                            {applications.map((app, i) => (
                                <div key={app.id} className="up card-hover" style={glass({ padding: "20px 24px", animationDelay: `${i * 0.05}s` })}>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                        <div>
                                            <div style={{ fontSize: 17, fontWeight: 700, color: C.text, marginBottom: 3 }}>{app.jobTitle}</div>
                                            <div style={{ fontSize: 13, color: C.blue, fontWeight: 500 }}>{app.company}</div>
                                        </div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                                            <div style={{ textAlign: "center" }}>
                                                <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>Resume</div>
                                                <div style={{
                                                    fontFamily: "'JetBrains Mono',monospace", fontSize: 22, fontWeight: 800,
                                                    color: app.resumeScore >= 0.5 ? "#4ade80" : "#facc15",
                                                }}>
                                                    {(app.resumeScore * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                            {app.interviewScore != null && (
                                                <div style={{ textAlign: "center" }}>
                                                    <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>Interview</div>
                                                    <div style={{
                                                        fontFamily: "'JetBrains Mono',monospace", fontSize: 22, fontWeight: 800,
                                                        color: app.interviewScore >= 0.6 ? "#4ade80" : app.interviewScore >= 0.42 ? "#facc15" : "#f87171",
                                                    }}>
                                                        {(app.interviewScore * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            )}
                                            {app.interviewStatus === "not_started" && app.passResume && (
                                                <a href={`/interview/${app.id}`} style={{ ...primaryBtn, textDecoration: "none", padding: "9px 18px", fontSize: 13 }}>Start Interview →</a>
                                            )}
                                            {app.interviewStatus === "active" && (
                                                <a href={`/interview/${app.id}`} style={{ ...outlineBtn, textDecoration: "none", padding: "9px 18px", fontSize: 13, borderColor: "rgba(250,204,21,0.3)", color: "#facc15" }}>Continue →</a>
                                            )}
                                            {app.interviewStatus === "completed" && (
                                                <a href={`/candidate/results/${app.id}`} style={{ ...outlineBtn, textDecoration: "none", padding: "9px 18px", fontSize: 13 }}>View Results →</a>
                                            )}
                                            {!app.passResume && <Tag color="#f87171">Below Threshold</Tag>}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
