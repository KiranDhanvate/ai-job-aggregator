
import React, { useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import LoginForm from '@/components/auth/LoginForm';
import RegisterForm from '@/components/auth/RegisterForm';
import MagicLinkForm from '@/components/auth/MagicLinkForm';
import ResetPasswordForm from '@/components/auth/ResetPasswordForm';
import AuthCallback from '@/components/auth/AuthCallback';
import { BarChart2 } from 'lucide-react';

const Auth = () => {
  const { user, isLoading } = useAuth();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState<string>('signin');
  
  // If user is already logged in, redirect to dashboard
  if (user && !isLoading) {
    return <Navigate to="/dashboard" replace />;
  }
  
  // Handle auth callback route
  if (location.pathname === '/auth/callback') {
    return <AuthCallback />;
  }
  
  // Handle password reset route
  if (location.pathname === '/auth/reset-password') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-navy-900 px-4">
        <div className="w-full max-w-md mb-8 text-center">
          <div className="flex justify-center mb-4">
            <div className="h-14 w-14 rounded-xl bg-gradient-to-br from-electric-500 to-purple-500 flex items-center justify-center shadow-glow">
              <BarChart2 className="text-white h-8 w-8" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-white mb-2">Reset your password</h1>
          <p className="text-gray-400">Create a new secure password for your CareerSync account</p>
        </div>
        <div className="w-full max-w-md glass-card rounded-xl p-6">
          <ResetPasswordForm />
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-navy-900 px-4">
      <div className="w-full max-w-md mb-8 text-center">
        <div className="flex justify-center mb-4">
          <div className="h-14 w-14 rounded-xl bg-gradient-to-br from-electric-500 to-purple-500 flex items-center justify-center shadow-glow">
            <BarChart2 className="text-white h-8 w-8" />
          </div>
        </div>
        <h1 className="text-2xl font-bold text-white mb-2">Welcome to CareerSync</h1>
        <p className="text-gray-400">Your all-in-one platform for job search and career management</p>
      </div>
      
      <div className="w-full max-w-md glass-card rounded-xl p-6">
        <Tabs
          defaultValue="signin"
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full"
        >
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="signin">Sign In</TabsTrigger>
            <TabsTrigger value="signup">Create Account</TabsTrigger>
          </TabsList>
          <TabsContent value="signin">
            <LoginForm onSwitchToMagicLink={() => setActiveTab('magic-link')} />
          </TabsContent>
          <TabsContent value="signup">
            <RegisterForm />
          </TabsContent>
          <TabsContent value="magic-link">
            <MagicLinkForm onBackToLogin={() => setActiveTab('signin')} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Auth;
