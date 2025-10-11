
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

const AuthCallback = () => {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    // The URL hash will include access_token and other params if auth was successful
    const hasFragment = window.location.hash && window.location.hash.length > 1;
    const hasError = new URLSearchParams(window.location.search).get('error');
    
    if (hasError) {
      setError(hasError);
      // Redirect after a delay
      setTimeout(() => {
        navigate('/auth');
      }, 3000);
    } else {
      // Allow a moment for the auth state to update and navigate based on the onAuthStateChange
      // in the AuthContext which will automatically move authenticated users to /dashboard
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    }
  }, [navigate]);
  
  return (
    <div className="flex items-center justify-center min-h-screen bg-navy-900">
      <div className="glass-card rounded-xl p-8 max-w-md text-center">
        {error ? (
          <>
            <h2 className="text-xl font-semibold text-red-500 mb-2">Authentication Error</h2>
            <p className="text-muted-foreground mb-4">{error}</p>
            <p>Redirecting you back to login...</p>
          </>
        ) : (
          <>
            <div className="flex justify-center mb-4">
              <Loader2 className="h-8 w-8 text-electric-400 animate-spin" />
            </div>
            <h2 className="text-xl font-semibold mb-2">Finalizing your login</h2>
            <p className="text-muted-foreground">Please wait while we complete the authentication process...</p>
          </>
        )}
      </div>
    </div>
  );
};

export default AuthCallback;
