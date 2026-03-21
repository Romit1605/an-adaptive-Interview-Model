import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const C = {
    bg: "#050810", card: "rgba(255,255,255,0.028)", bord: "rgba(255,255,255,0.08)",
    text: "#d0ccc4", muted: "rgba(208,204,196,0.45)", blue: "#5b9cf6",
};

export function HRLogin() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const { loginHR } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setBusy(true); setError("");
        try {
            await loginHR(email, password);
            navigate("/hr/dashboard");
        } catch (err) {
            setError(err.message);
        } finally {
            setBusy(false);
        }
    };

    return <AuthForm title="HR Login" subtitle="Manage your job postings and candidates" onSubmit={handleSubmit} error={error} busy={busy} buttonText="Login" altText="Don't have an account?" altLink="/hr/register" altLinkText="Register" extraLink="/hr/forgot-password" extraLinkText="Forgot password?">
        <InputField label="Email" type="email" value={email} onChange={setEmail} placeholder="hr@company.com" />
        <InputField label="Password" type="password" value={password} onChange={setPassword} placeholder="••••••••" />
    </AuthForm>;
}

export function HRRegister() {
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [company, setCompany] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const { registerHR } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setBusy(true); setError("");
        try {
            await registerHR(name, email, password, company);
            navigate("/hr/dashboard");
        } catch (err) {
            setError(err.message);
        } finally {
            setBusy(false);
        }
    };

    return <AuthForm title="HR Registration" subtitle="Create your HR account to post jobs and manage recruitment" onSubmit={handleSubmit} error={error} busy={busy} buttonText="Create Account" altText="Already have an account?" altLink="/hr/login" altLinkText="Login">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <InputField label="Full Name" value={name} onChange={setName} placeholder="Jane Smith" />
            <InputField label="Company" value={company} onChange={setCompany} placeholder="Acme Corp" />
        </div>
        <InputField label="Email" type="email" value={email} onChange={setEmail} placeholder="hr@company.com" />
        <InputField label="Password" type="password" value={password} onChange={setPassword} placeholder="Min 6 characters" />
    </AuthForm>;
}

export function CandidateLogin() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const { loginCandidate } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setBusy(true); setError("");
        try {
            await loginCandidate(email, password);
            navigate("/candidate/dashboard");
        } catch (err) {
            setError(err.message);
        } finally {
            setBusy(false);
        }
    };

    return <AuthForm title="Candidate Login" subtitle="Apply for jobs and take adaptive interviews" onSubmit={handleSubmit} error={error} busy={busy} buttonText="Login" altText="Don't have an account?" altLink="/candidate/register" altLinkText="Register as Candidate" extraLink="/candidate/forgot-password" extraLinkText="Forgot password?">
        <InputField label="Email" type="email" value={email} onChange={setEmail} placeholder="you@example.com" />
        <InputField label="Password" type="password" value={password} onChange={setPassword} placeholder="••••••••" />
    </AuthForm>;
}

export function CandidateRegister() {
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const { registerCandidate } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setBusy(true); setError("");
        try {
            await registerCandidate(name, email, password);
            navigate("/candidate/dashboard");
        } catch (err) {
            setError(err.message);
        } finally {
            setBusy(false);
        }
    };

    return <AuthForm title="Candidate Registration" subtitle="Create your account to apply for jobs and take adaptive interviews" onSubmit={handleSubmit} error={error} busy={busy} buttonText="Create Account" altText="Already have an account?" altLink="/candidate/login" altLinkText="Login">
        <InputField label="Full Name" value={name} onChange={setName} placeholder="John Doe" />
        <InputField label="Email" type="email" value={email} onChange={setEmail} placeholder="you@example.com" />
        <InputField label="Password" type="password" value={password} onChange={setPassword} placeholder="Min 6 characters" />
    </AuthForm>;
}

export function ForgotPassword({ role }) {
    const [email, setEmail] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");
    const [busy, setBusy] = useState(false);
    const { resetPassword } = useAuth();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setBusy(true); setError(""); setSuccess("");
        try {
            await resetPassword(email, newPassword, role);
            setSuccess("Password reset successfully. You can now login.");
            setEmail("");
            setNewPassword("");
        } catch (err) {
            setError(err.message || "Failed to reset password.");
        } finally {
            setBusy(false);
        }
    };

    const loginLink = role === "hr" ? "/hr/login" : "/candidate/login";

    return (
        <AuthForm title="Reset Password" subtitle={`Update your ${role.toUpperCase()} account password`} onSubmit={handleSubmit} error={error} busy={busy} buttonText="Reset Password" altText="Remembered your password?" altLink={loginLink} altLinkText="Login">
            {success && (
                <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(74, 222, 128, 0.08)", border: "1px solid rgba(74, 222, 128, 0.2)", color: "#4ade80", fontSize: 13, marginBottom: 10 }}>
                    {success}
                </div>
            )}
            <InputField label="Email" type="email" value={email} onChange={setEmail} placeholder="you@example.com" />
            <InputField label="New Password" type="password" value={newPassword} onChange={setNewPassword} placeholder="••••••••" />
        </AuthForm>
    );
}

/* ── Shared Auth Form Shell ── */
function AuthForm({ title, subtitle, onSubmit, error, busy, buttonText, altText, altLink, altLinkText, extraLink, extraLinkText, children }) {
    const inp = { width: "100%", padding: "11px 14px", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.bord}`, borderRadius: 8, color: C.text, fontSize: 15, fontFamily: "'Outfit',sans-serif", outline: "none" };
    const btn = { cursor: "pointer", border: "none", borderRadius: 9, fontFamily: "'Outfit',sans-serif", fontWeight: 600, fontSize: 15, padding: "13px 24px", width: "100%", background: "linear-gradient(135deg,#2563eb,#5b9cf6)", color: "#fff", transition: "all .16s", opacity: busy ? 0.5 : 1 };

    return (
        <div style={{ maxWidth: 460, margin: "60px auto 0", padding: "0 26px" }}>
            <div className="up" style={{ background: C.card, border: `1px solid ${C.bord}`, borderRadius: 16, padding: 36 }}>
                <div style={{ textAlign: "center", marginBottom: 28 }}>
                    <h2 style={{ fontFamily: "'Instrument Serif',serif", fontSize: 32, marginBottom: 6, color: C.text }}>{title}</h2>
                    <p style={{ fontSize: 13, color: C.muted }}>{subtitle}</p>
                </div>
                <form onSubmit={onSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {children}
                    {error && (
                        <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171", fontSize: 13 }}>
                            {error}
                        </div>
                    )}
                    <button type="submit" disabled={busy} style={btn}>{busy ? "Please wait…" : buttonText}</button>
                </form>
                <div style={{ textAlign: "center", marginTop: 18, fontSize: 13, color: C.muted, display: "flex", flexDirection: "column", gap: 8 }}>
                    <div>
                        {altText}{" "}
                        <Link to={altLink} style={{ color: C.blue, textDecoration: "none", fontWeight: 600 }}>{altLinkText}</Link>
                    </div>
                    {extraLink && (
                        <div>
                            <Link to={extraLink} style={{ color: C.blue, textDecoration: "none", fontWeight: 600 }}>{extraLinkText}</Link>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function InputField({ label, type = "text", value, onChange, placeholder }) {
    const lbl = { display: "block", marginBottom: 5, fontSize: 11, color: `${C.blue}cc`, letterSpacing: "0.1em", textTransform: "uppercase", fontFamily: "'Outfit',sans-serif" };
    const inp = { width: "100%", padding: "11px 14px", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.bord}`, borderRadius: 8, color: C.text, fontSize: 15, fontFamily: "'Outfit',sans-serif", outline: "none" };

    return (
        <div>
            <label style={lbl}>{label}</label>
            <input type={type} style={inp} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
        </div>
    );
}
