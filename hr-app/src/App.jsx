import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";

// Pages
import Landing from "./pages/Landing";
import { HRLogin, HRRegister, CandidateLogin, CandidateRegister, ForgotPassword } from "./pages/AuthPages";
import HRDashboard from "./pages/HRDashboard";
import CandidateDashboard from "./pages/CandidateDashboard";
import Interview from "./pages/Interview";
import Results from "./pages/Results";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
          <Navbar />
          <div style={{ flex: 1 }}>
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<Landing />} />
              
              {/* Auth Routes */}
              <Route path="/hr/login" element={<HRLogin />} />
              <Route path="/hr/register" element={<HRRegister />} />
              <Route path="/hr/forgot-password" element={<ForgotPassword role="hr" />} />
              <Route path="/candidate/login" element={<CandidateLogin />} />
              <Route path="/candidate/register" element={<CandidateRegister />} />
              <Route path="/candidate/forgot-password" element={<ForgotPassword role="candidate" />} />

              {/* Protected HR Routes */}
              <Route 
                path="/hr/dashboard" 
                element={
                  <ProtectedRoute role="hr">
                    <HRDashboard />
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/hr/results/:applicationId" 
                element={
                  <ProtectedRoute role="hr">
                    <Results />
                  </ProtectedRoute>
                } 
              />

              {/* Protected Candidate Routes */}
              <Route 
                path="/candidate/dashboard" 
                element={
                  <ProtectedRoute role="candidate">
                    <CandidateDashboard />
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/interview/:applicationId" 
                element={
                  <ProtectedRoute role="candidate">
                    <Interview />
                  </ProtectedRoute>
                } 
              />
              <Route 
                path="/candidate/results/:applicationId" 
                element={
                  <ProtectedRoute role="candidate">
                    <Results userContext="candidate" />
                  </ProtectedRoute>
                } 
              />

              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}
