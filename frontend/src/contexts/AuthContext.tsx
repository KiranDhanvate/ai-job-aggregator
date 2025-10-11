
import React, { createContext, useContext, useEffect, useState } from 'react';
import { Session, User } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/components/ui/use-toast';

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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Set up auth state listener FIRST
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        
        if (session?.user) {
          await fetchProfile(session.user.id);
          // Log the activity
          await logActivity(session.user.id, `auth_${event.toLowerCase()}`);
        } else {
          setProfile(null);
        }
      }
    );

    // THEN check for existing session
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      
      if (session?.user) {
        await fetchProfile(session.user.id);
      }
      
      setIsLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const fetchProfile = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();

      if (error) {
        console.error('Error fetching profile:', error);
        return;
      }

      setProfile(data as Profile);
    } catch (error) {
      console.error('Error fetching profile:', error);
    }
  };

  const logActivity = async (userId: string, action: string) => {
    try {
      const userAgent = navigator.userAgent;
      
      await supabase.from('activity_logs').insert({
        user_id: userId,
        action,
        user_agent: userAgent,
        details: { session_id: session?.access_token }
      });
    } catch (error) {
      console.error('Error logging activity:', error);
    }
  };

  const signUp = async (email: string, password: string, metadata: { full_name: string, professional_title?: string, industry?: string }) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: metadata,
          emailRedirectTo: `${window.location.origin}/auth/callback`
        }
      });

      if (error) {
        toast({
          title: 'Sign-up failed',
          description: error.message,
          variant: 'destructive'
        });
        return { error };
      }

      toast({
        title: 'Sign-up successful',
        description: 'Please check your email to verify your account.',
      });

      return { error: null };
    } catch (error: any) {
      toast({
        title: 'Sign-up failed',
        description: error.message,
        variant: 'destructive'
      });
      return { error };
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
      });

      if (error) {
        toast({
          title: 'Sign-in failed',
          description: error.message,
          variant: 'destructive'
        });
        return { error };
      }

      toast({
        title: 'Sign-in successful',
        description: `Welcome back${profile?.full_name ? `, ${profile.full_name}` : ''}!`,
      });

      return { error: null };
    } catch (error: any) {
      toast({
        title: 'Sign-in failed',
        description: error.message,
        variant: 'destructive'
      });
      return { error };
    }
  };

  const signInWithMagicLink = async (email: string) => {
    try {
      const { data, error } = await supabase.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`
        }
      });

      if (error) {
        toast({
          title: 'Magic link sign-in failed',
          description: error.message,
          variant: 'destructive'
        });
        return { error };
      }

      toast({
        title: 'Magic link sent',
        description: 'Please check your email for the magic link to sign in.',
      });

      return { error: null };
    } catch (error: any) {
      toast({
        title: 'Magic link sign-in failed',
        description: error.message,
        variant: 'destructive'
      });
      return { error };
    }
  };

  const signInWithOAuth = async (provider: 'google' | 'github' | 'linkedin_oidc') => {
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/auth/callback`
        }
      });

      if (error) {
        toast({
          title: `${provider.charAt(0).toUpperCase() + provider.slice(1)} sign-in failed`,
          description: error.message,
          variant: 'destructive'
        });
      }
    } catch (error: any) {
      toast({
        title: `${provider.charAt(0).toUpperCase() + provider.slice(1)} sign-in failed`,
        description: error.message,
        variant: 'destructive'
      });
    }
  };

  const signOut = async () => {
    try {
      if (user) {
        await logActivity(user.id, 'auth_signout');
      }
      
      await supabase.auth.signOut();
      
      toast({
        title: 'Signed out',
        description: 'You have been successfully signed out',
      });
    } catch (error: any) {
      toast({
        title: 'Sign-out failed',
        description: error.message,
        variant: 'destructive'
      });
    }
  };

  const updateProfile = async (updates: Partial<Profile>) => {
    try {
      if (!user) {
        throw new Error('User not authenticated');
      }

      const { data, error } = await supabase
        .from('profiles')
        .update(updates)
        .eq('id', user.id)
        .select('*')
        .single();

      if (error) {
        toast({
          title: 'Profile update failed',
          description: error.message,
          variant: 'destructive'
        });
        return { error };
      }

      setProfile(data as Profile);
      
      toast({
        title: 'Profile updated',
        description: 'Your profile has been updated successfully.',
      });

      return { error: null };
    } catch (error: any) {
      toast({
        title: 'Profile update failed',
        description: error.message,
        variant: 'destructive'
      });
      return { error };
    }
  };

  const resetPassword = async (email: string) => {
    try {
      const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      });

      if (error) {
        toast({
          title: 'Password reset failed',
          description: error.message,
          variant: 'destructive'
        });
        return { error };
      }

      toast({
        title: 'Password reset email sent',
        description: 'Please check your email for the password reset link.',
      });

      return { error: null };
    } catch (error: any) {
      toast({
        title: 'Password reset failed',
        description: error.message,
        variant: 'destructive'
      });
      return { error };
    }
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
