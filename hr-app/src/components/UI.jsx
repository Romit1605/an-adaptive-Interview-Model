/* ═══════════════════════════════════════
   NextGen-HR · Shared UI Components
   Premium Design System
   ═══════════════════════════════════════ */

export function irtColor(b) {
    const t = (b + 4) / 8;
    const r = Math.round(40 + t * 215);
    const g = Math.round(200 - t * 160);
    const bl = Math.round(80 + t * 20);
    return `rgb(${r},${g},${bl})`;
}

/* ── Animated Score Bar ─────────────────────── */
export function ScoreBar({ label, value, color, mono }) {
    const pct = Math.min(Math.max(value || 0, 0), 1);
    const c = color || "#5b9cf6";
    return (
        <div style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: "rgba(208,204,196,0.45)", fontWeight: 400 }}>{label}</span>
                <span style={{
                    fontFamily: mono ? "'JetBrains Mono',monospace" : "'Outfit',sans-serif",
                    fontSize: 13, fontWeight: 700, color: c,
                }}>
                    {(pct * 100).toFixed(1)}%
                </span>
            </div>
            <div style={{ height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 99, overflow: "hidden" }}>
                <div className="score-bar-fill" style={{
                    height: "100%",
                    width: `${pct * 100}%`,
                    borderRadius: 99,
                    background: `linear-gradient(90deg, ${c}99, ${c})`,
                    boxShadow: `0 0 8px ${c}55`,
                }} />
            </div>
        </div>
    );
}

/* ── IRT Badge ──────────────────────────────── */
export function IRTBadge({ qid, irtItems, short }) {
    const item = irtItems?.[qid];
    if (!item || item.n === 0) {
        return (
            <span style={{
                fontSize: 10, color: "rgba(255,255,255,0.22)",
                fontFamily: "'JetBrains Mono',monospace",
                padding: "2px 8px", borderRadius: 99,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.07)",
            }}>
                {short ? "new" : "No data yet"}
            </span>
        );
    }
    const col = irtColor(item.b);
    const bg = col.replace("rgb", "rgba").replace(")", ", 0.1)");
    const bd = col.replace("rgb", "rgba").replace(")", ", 0.25)");
    return (
        <span style={{
            fontSize: 10, color: col, fontFamily: "'JetBrains Mono',monospace",
            background: bg, padding: "3px 8px", borderRadius: 99, border: `1px solid ${bd}`,
        }}>
            {short ? `b=${item.b.toFixed(2)}` : `b=${item.b.toFixed(2)} α=${item.a.toFixed(2)} n=${item.n}`}
        </span>
    );
}

/* ── Progress Chart ─────────────────────────── */
export function ProgressChart({ progression, scores }) {
    if (!progression?.length) return null;
    const W = 280, H = 80, pad = 10;
    const n = progression.length;
    const px = (i) => pad + (i / Math.max(n - 1, 1)) * (W - 2 * pad);
    const py = (v) => H - pad - Math.min(Math.max(v, 0), 1) * (H - 2 * pad);
    const pathD = progression.map((v, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    const area = `${pathD} L${px(n - 1)},${H} L${px(0)},${H} Z`;
    return (
        <div>
            <div style={{ fontSize: 9, letterSpacing: "0.12em", color: "rgba(255,255,255,0.22)", textTransform: "uppercase", marginBottom: 6, fontFamily: "'JetBrains Mono',monospace" }}>
                Score Trajectory
            </div>
            <svg width={W} height={H} style={{ overflow: "visible" }}>
                <defs>
                    <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#4ade80" stopOpacity="0.18" />
                        <stop offset="100%" stopColor="#4ade80" stopOpacity="0" />
                    </linearGradient>
                </defs>
                <path d={area} fill="url(#chartFill)" />
                <path d={pathD} fill="none" stroke="#4ade80" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                <line x1={pad} y1={py(0.5)} x2={W - pad} y2={py(0.5)}
                    stroke="rgba(255,255,255,0.06)" strokeDasharray="4,4" />
                {scores?.map((v, i) => (
                    <circle key={i} cx={px(i)} cy={py(v)} r={4}
                        fill={v >= 0.65 ? "#4ade80" : v >= 0.44 ? "#facc15" : "#f87171"}
                        stroke="rgba(0,0,0,0.5)" strokeWidth={1.5} />
                ))}
            </svg>
        </div>
    );
}

/* ── Tag ────────────────────────────────────── */
export function Tag({ children, color }) {
    const c = color || "#e0a020";
    return (
        <span style={{
            display: "inline-block", padding: "3px 10px", borderRadius: 99,
            fontSize: 11, fontWeight: 700, letterSpacing: "0.03em",
            background: c + "18", color: c, border: `1px solid ${c}30`,
        }}>
            {children}
        </span>
    );
}

/* ── Stat Card ──────────────────────────────── */
export function StatCard({ label, value, sub, color, icon }) {
    return (
        <div style={{
            padding: "18px 20px",
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 14,
            backdropFilter: "blur(12px)",
        }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <div style={{ fontSize: 11, color: "rgba(208,204,196,0.4)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>{label}</div>
                    <div style={{ fontSize: 32, fontWeight: 800, color: color || "#d0ccc4", fontFamily: "'Outfit',sans-serif", letterSpacing: "-0.02em" }}>{value}</div>
                    {sub && <div style={{ fontSize: 12, color: "rgba(208,204,196,0.35)", marginTop: 4 }}>{sub}</div>}
                </div>
                {icon && <span style={{ fontSize: 22, opacity: 0.7 }}>{icon}</span>}
            </div>
        </div>
    );
}

/* ── Score Ring ─────────────────────────────── */
export function ScoreRing({ value, size = 72, color }) {
    const pct = Math.min(Math.max(value || 0, 0), 1);
    const r = (size - 10) / 2;
    const circ = 2 * Math.PI * r;
    const dash = pct * circ;
    const c = color || (pct >= 0.65 ? "#4ade80" : pct >= 0.42 ? "#facc15" : "#f87171");
    return (
        <svg width={size} height={size} style={{ flexShrink: 0 }}>
            <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={8} />
            <circle
                cx={size / 2} cy={size / 2} r={r}
                fill="none" stroke={c} strokeWidth={8}
                strokeLinecap="round"
                strokeDasharray={`${dash} ${circ}`}
                strokeDashoffset={circ / 4}
                style={{ transition: "stroke-dasharray 1.1s cubic-bezier(0.22,1,0.36,1)", filter: `drop-shadow(0 0 6px ${c}88)` }}
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
            />
            <text x={size / 2} y={size / 2 + 5} textAnchor="middle"
                style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 14, fontWeight: 700, fill: c }}>
                {(pct * 100).toFixed(0)}%
            </text>
        </svg>
    );
}
