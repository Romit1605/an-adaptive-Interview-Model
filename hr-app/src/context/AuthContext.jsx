import { createContext, useContext, useState, useEffect, useCallback } from "react";
import { apiPost, apiGet, setToken, clearToken, getToken } from "../api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    const fetchMe = useCallback(async () => {
        const token = getToken();
        if (!token) { setLoading(false); return; }
        try {
            const data = await apiGet("/api/auth/me");
            setUser(data.user);
        } catch {
            clearToken();
            setUser(null);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchMe(); }, [fetchMe]);

    const loginHR = async (email, password) => {
        const data = await apiPost("/api/auth/hr/login", { email, password });
        setToken(data.token);
        setUser(data.user);
        return data.user;
    };

    const registerHR = async (name, email, password, company) => {
        const data = await apiPost("/api/auth/hr/register", { name, email, password, company });
        setToken(data.token);
        setUser(data.user);
        return data.user;
    };

    const loginCandidate = async (email, password) => {
        const data = await apiPost("/api/auth/candidate/login", { email, password });
        setToken(data.token);
        setUser(data.user);
        return data.user;
    };

    const registerCandidate = async (name, email, password) => {
        const data = await apiPost("/api/auth/candidate/register", { name, email, password });
        setToken(data.token);
        setUser(data.user);
        return data.user;
    };

    const resetPassword = async (email, newPassword, role) => {
        const data = await apiPost("/api/auth/reset-password", { email, new_password: newPassword, role });
        return data;
    };

    const logout = () => {
        clearToken();
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, loading, loginHR, registerHR, loginCandidate, registerCandidate, resetPassword, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used within AuthProvider");
    return ctx;
}
