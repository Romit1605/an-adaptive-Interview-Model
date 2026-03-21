const API_BASE = "http://localhost:8000";

export function getToken() {
    return localStorage.getItem("nextgen_token") || "";
}

export function setToken(token) {
    localStorage.setItem("nextgen_token", token);
}

export function clearToken() {
    localStorage.removeItem("nextgen_token");
}

export async function api(path, options = {}) {
    const { method = "GET", body, isFormData = false } = options;
    const headers = {};
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    if (!isFormData) headers["Content-Type"] = "application/json";

    const config = { method, headers };
    if (body) {
        config.body = isFormData ? body : JSON.stringify(body);
    }

    const res = await fetch(`${API_BASE}${path}`, config);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
        throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
}

export async function apiPost(path, body) {
    return api(path, { method: "POST", body });
}

export async function apiGet(path) {
    return api(path, { method: "GET" });
}

export async function apiUpload(path, formData) {
    return api(path, { method: "POST", body: formData, isFormData: true });
}
