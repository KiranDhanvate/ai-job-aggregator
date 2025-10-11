
import React, { useState } from 'react';
import { z } from 'zod';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2, Github, Mail } from 'lucide-react';
import { Separator } from '@/components/ui/separator';

// Form schema with validation
const formSchema = z.object({
  email: z.string().email({ message: 'Please enter a valid email address' }),
  password: z.string().min(6, { message: 'Password must be at least 6 characters' }),
});

interface LoginFormProps {
  onSwitchToMagicLink: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onSwitchToMagicLink }) => {
  const { signIn, signInWithOAuth, resetPassword } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [showResetPassword, setShowResetPassword] = useState(false);
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });
  
  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setIsLoading(true);
    
    if (showResetPassword) {
      await resetPassword(values.email);
      setShowResetPassword(false);
    } else {
      await signIn(values.email, values.password);
    }
    
    setIsLoading(false);
  };

  const handleOAuthSignIn = async (provider: 'google' | 'github' | 'linkedin_oidc') => {
    await signInWithOAuth(provider);
  };

  return (
    <div>
      <div className="flex flex-col gap-4 mb-6">
        <Button 
          type="button" 
          variant="outline" 
          className="bg-navy-900/50"
          onClick={() => handleOAuthSignIn('google')}
        >
          <svg className="h-4 w-4 mr-2" viewBox="0 0 24 24">
            <path
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              fill="#4285F4"
            />
            <path
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              fill="#34A853"
            />
            <path
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              fill="#FBBC05"
            />
            <path
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              fill="#EA4335"
            />
          </svg>
          Continue with Google
        </Button>
        <Button 
          type="button" 
          variant="outline" 
          className="bg-navy-900/50"
          onClick={() => handleOAuthSignIn('github')}
        >
          <Github className="h-4 w-4 mr-2" />
          Continue with GitHub
        </Button>
        <Button 
          type="button" 
          variant="outline" 
          className="bg-navy-900/50"
          onClick={() => handleOAuthSignIn('linkedin_oidc')}
        >
          <svg className="h-4 w-4 mr-2" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z" />
          </svg>
          Continue with LinkedIn
        </Button>
      </div>
      
      <div className="relative my-6">
        <Separator />
        <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-card px-2 text-xs text-muted-foreground">
          OR
        </span>
      </div>
      
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input 
                    placeholder="name@example.com" 
                    {...field} 
                    autoComplete="email"
                    className="bg-navy-900/50"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          {!showResetPassword && (
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <Input 
                      type="password" 
                      placeholder="••••••••" 
                      {...field} 
                      autoComplete="current-password"
                      className="bg-navy-900/50"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          )}
          
          <div className="flex justify-between items-center">
            <Button
              type="button"
              variant="link"
              className="px-0 text-electric-400"
              onClick={() => setShowResetPassword(!showResetPassword)}
            >
              {showResetPassword ? 'Back to login' : 'Forgot password?'}
            </Button>
            
            <Button
              type="button"
              variant="link"
              className="px-0 text-electric-400"
              onClick={onSwitchToMagicLink}
            >
              <Mail className="h-4 w-4 mr-1" />
              Use magic link
            </Button>
          </div>
          
          <Button 
            type="submit" 
            className="w-full bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600" 
            disabled={isLoading}
          >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {showResetPassword ? 'Send Reset Instructions' : 'Sign In'}
          </Button>
        </form>
      </Form>
    </div>
  );
};

export default LoginForm;
