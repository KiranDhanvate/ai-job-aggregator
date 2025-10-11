
import React, { createContext, useContext, useEffect, useState } from 'react';

// Mock types to replace Supabase types
type User = {
  id: string;
  email?: string;
  user_metadata?: any;
} | null;

type Session = {
  access_token: string;
  user: User;
} | null;

type Profile = {
  id: string;
  full_name: string | null;
  professional_title: string | null;
  industry: string | null;
  phone_number: string | null;
  avatar_url: string | null;
  created_at: string;
  updated_at: string;
};

type AuthContextType = {
  user: User | null;
  session: Session | null;
  profile: Profile | null;
  isLoading: boolean;
  signUp: (email: string, password: string, metadata: { full_name: string, professional_title?: string, industry?: string }) => Promise<{ error: any | null }>;
  signIn: (email: string, password: string) => Promise<{ error: any | null }>;
  signInWithMagicLink: (email: string) => Promise<{ error: any | null }>;
  signInWithOAuth: (provider: 'google' | 'github' | 'linkedin_oidc') => Promise<void>;
  signOut: () => Promise<void>;
  updateProfile: (updates: Partial<Profile>) => Promise<{ error: any | null }>;
  resetPassword: (email: string) => Promise<{ error: any | null }>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [isLoading, setIsLoading] = useState(false); // Set to false since we're not using real auth

  useEffect(() => {
    // Mock initialization - no real auth
    setIsLoading(false);
  }, []);

  // Mock implementations for all auth functions
  const signUp = async (email: string, password: string, metadata: { full_name: string, professional_title?: string, industry?: string }) => {
    console.log('Mock signUp called:', { email, metadata });
    return { error: null };
  };

  const signIn = async (email: string, password: string) => {
    console.log('Mock signIn called:', { email });
    return { error: null };
  };

  const signInWithMagicLink = async (email: string) => {
    console.log('Mock signInWithMagicLink called:', { email });
    return { error: null };
  };

  const signInWithOAuth = async (provider: 'google' | 'github' | 'linkedin_oidc') => {
    console.log('Mock signInWithOAuth called:', { provider });
  };

  const signOut = async () => {
    console.log('Mock signOut called');
    setUser(null);
    setSession(null);
    setProfile(null);
  };

  const updateProfile = async (updates: Partial<Profile>) => {
    console.log('Mock updateProfile called:', updates);
    return { error: null };
  };

  const resetPassword = async (email: string) => {
    console.log('Mock resetPassword called:', { email });
    return { error: null };
  };

  return (
    <AuthContext.Provider value={{
      user,
      session,
      profile,
      isLoading,
      signUp,
      signIn,
      signInWithMagicLink,
      signInWithOAuth,
      signOut,
      updateProfile,
      resetPassword,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
