import { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handler = () => setScrolled(window.scrollY > 12);
        window.addEventListener("scroll", handler);
        return () => window.removeEventListener("scroll", handler);
    }, []);

    const handleLogout = () => { logout(); navigate("/"); };

    const dashboardLink = user?.role === "hr" ? "/hr/dashboard" : "/candidate/dashboard";
    const isActive = (path) => location.pathname.startsWith(path);

    return (
        <nav style={{
            position: "sticky", top: 0, zIndex: 500,
            background: scrolled ? "rgba(2,4,13,0.88)" : "rgba(2,4,13,0.6)",
            borderBottom: scrolled ? "1px solid rgba(91,156,246,0.12)" : "1px solid rgba(255,255,255,0.05)",
            backdropFilter: "blur(20px) saturate(1.5)",
            WebkitBackdropFilter: "blur(20px) saturate(1.5)",
            padding: "0 28px",
            transition: "all 0.3s ease",
        }}>
            <div style={{ maxWidth: 1140, margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", height: 60 }}>

                {/* Brand */}
                <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{
                        width: 32, height: 32, borderRadius: 9,
                        background: "linear-gradient(135deg, #1d4ed8, #6366f1)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 16, boxShadow: "0 4px 14px rgba(59,130,246,0.35)",
                    }}>
                        🚀
                    </div>
                    <div>
                        <span style={{
                            fontSize: 17, fontWeight: 800, letterSpacing: "-0.02em",
                            background: "linear-gradient(130deg,#5b9cf6,#a5c8ff,#c4b5fd)",
                            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                        }}>
                            NextGen-HR
                        </span>
                        <span style={{ display: "block", fontSize: 9, color: "rgba(208,204,196,0.3)", letterSpacing: "0.18em", textTransform: "uppercase", marginTop: -3 }}>
                            Adaptive Interview
                        </span>
                    </div>
                </Link>

                {/* Right side */}
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    {user ? (
                        <>
                            <Link to={dashboardLink} style={{
                                color: isActive(dashboardLink) ? "#5b9cf6" : "rgba(208,204,196,0.7)",
                                textDecoration: "none", fontSize: 13, fontWeight: 500, padding: "6px 12px", borderRadius: 8,
                                background: isActive(dashboardLink) ? "rgba(91,156,246,0.1)" : "transparent",
                                transition: "all 0.18s",
                            }}>
                                Dashboard
                            </Link>

                            <div style={{
                                display: "flex", alignItems: "center", gap: 8, padding: "5px 12px",
                                borderRadius: 99, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)",
                            }}>
                                <div style={{
                                    width: 24, height: 24, borderRadius: "50%",
                                    background: "linear-gradient(135deg, #1d4ed8, #6366f1)",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: 11, fontWeight: 700, color: "#fff",
                                }}>
                                    {user.name?.[0]?.toUpperCase()}
                                </div>
                                <span style={{ fontSize: 13, color: "rgba(208,204,196,0.8)", fontWeight: 500 }}>{user.name}</span>
                                <span style={{
                                    fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
                                    color: user.role === "hr" ? "#f59e0b" : "#4ade80",
                                    background: user.role === "hr" ? "rgba(245,158,11,0.12)" : "rgba(74,222,128,0.1)",
                                    border: `1px solid ${user.role === "hr" ? "rgba(245,158,11,0.25)" : "rgba(74,222,128,0.2)"}`,
                                    padding: "2px 7px", borderRadius: 99,
                                }}>
                                    {user.role.toUpperCase()}
                                </span>
                            </div>

                            <button onClick={handleLogout} style={{
                                background: "transparent", border: "1px solid rgba(255,255,255,0.08)",
                                borderRadius: 8, color: "rgba(208,204,196,0.5)", fontSize: 12, padding: "6px 14px",
                                cursor: "pointer", fontFamily: "'Outfit',sans-serif", fontWeight: 500,
                                transition: "all 0.18s",
                            }}
                                onMouseEnter={e => { e.target.style.borderColor = "rgba(248,113,113,0.3)"; e.target.style.color = "#f87171"; }}
                                onMouseLeave={e => { e.target.style.borderColor = "rgba(255,255,255,0.08)"; e.target.style.color = "rgba(208,204,196,0.5)"; }}
                            >
                                Logout
                            </button>
                        </>
                    ) : (
                        <>
                            <Link to="/hr/login" style={{
                                color: "#5b9cf6", textDecoration: "none", fontSize: 13, fontWeight: 500,
                                padding: "7px 16px", borderRadius: 9, border: "1px solid rgba(91,156,246,0.25)",
                                background: "rgba(91,156,246,0.05)", transition: "all 0.18s",
                            }}>
                                HR Portal
                            </Link>
                            <Link to="/candidate/login" style={{
                                textDecoration: "none", fontSize: 13, fontWeight: 600,
                                padding: "7px 18px", borderRadius: 9,
                                background: "linear-gradient(135deg, #1d4ed8, #6366f1)",
                                color: "#fff", boxShadow: "0 4px 14px rgba(59,130,246,0.3)",
                                transition: "all 0.22s",
                            }}>
                                Apply Now →
                            </Link>
                        </>
                    )}
                </div>
            </div>
        </nav>
    );
}
