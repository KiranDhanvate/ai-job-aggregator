
import React, { useState } from 'react';
import { z } from 'zod';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2, ArrowLeft, Mail } from 'lucide-react';

// Form schema with validation
const formSchema = z.object({
  email: z.string().email({ message: 'Please enter a valid email address' }),
});

interface MagicLinkFormProps {
  onBackToLogin: () => void;
}

const MagicLinkForm: React.FC<MagicLinkFormProps> = ({ onBackToLogin }) => {
  const { signInWithMagicLink } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [isSent, setIsSent] = useState(false);
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: '',
    },
  });
  
  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setIsLoading(true);
    
    const { error } = await signInWithMagicLink(values.email);
    
    if (!error) {
      setIsSent(true);
    }
    
    setIsLoading(false);
  };

  return (
    <div>
      <div className="flex items-center mb-6">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="p-0 h-auto mr-2"
          onClick={onBackToLogin}
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <h2 className="text-lg font-semibold">Sign in with Magic Link</h2>
      </div>
      
      {isSent ? (
        <div className="text-center py-4">
          <div className="flex justify-center mb-4">
            <div className="h-16 w-16 rounded-full bg-gradient-to-br from-electric-500/20 to-purple-500/20 flex items-center justify-center">
              <Mail className="h-8 w-8 text-electric-400" />
            </div>
          </div>
          <h3 className="text-lg font-medium mb-2">Check your email</h3>
          <p className="text-muted-foreground mb-4">
            We've sent a magic link to <span className="font-medium">{form.getValues().email}</span>.
            Click the link in the email to sign in.
          </p>
          <Button 
            type="button" 
            variant="outline" 
            className="w-full"
            onClick={() => {
              setIsSent(false);
              form.reset();
            }}
          >
            Back to form
          </Button>
        </div>
      ) : (
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
            
            <Button 
              type="submit" 
              className="w-full bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600" 
              disabled={isLoading}
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Send Magic Link
            </Button>
          </form>
        </Form>
      )}
    </div>
  );
};

export default MagicLinkForm;
