import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ children, role }) {
    const { user, loading } = useAuth();
    if (loading) return (
        <div style={{ minHeight: "100vh", background: "#050810", display: "flex", alignItems: "center", justifyContent: "center", color: "#5b9cf6", fontFamily: "'Outfit',sans-serif" }}>
            Loading…
        </div>
    );
    if (!user) return <Navigate to="/" replace />;
    if (role && user.role !== role) return <Navigate to="/" replace />;
    return children;
}
