// Complete mock replacement for @supabase/supabase-js
// This file replaces the entire Supabase library to prevent any network calls

console.log('🚫 Supabase library completely mocked - no network requests will be made');

// Mock createClient function
export const createClient = (url: string, key: string, options?: any) => {
  console.log('Mock: createClient called with URL:', url);
  console.log('Mock: Supabase client creation blocked - returning mock client');
  
  return {
    auth: {
      onAuthStateChange: (callback: any) => {
        console.log('Mock: onAuthStateChange called');
        setTimeout(() => callback('SIGNED_OUT', null), 0);
        return { 
          data: { 
            subscription: { 
              unsubscribe: () => console.log('Mock: Auth subscription unsubscribed') 
            } 
          } 
        };
      },
      getSession: () => {
        console.log('Mock: getSession called');
        return Promise.resolve({ data: { session: null }, error: null });
      },
      signUp: (params: any) => {
        console.log('Mock: signUp called');
        return Promise.resolve({ data: null, error: null });
      },
      signInWithPassword: (params: any) => {
        console.log('Mock: signInWithPassword called');
        return Promise.resolve({ data: null, error: null });
      },
      signInWithOtp: (params: any) => {
        console.log('Mock: signInWithOtp called');
        return Promise.resolve({ data: null, error: null });
      },
      signInWithOAuth: (params: any) => {
        console.log('Mock: signInWithOAuth called');
        return Promise.resolve({ data: null, error: null });
      },
      signOut: () => {
        console.log('Mock: signOut called');
        return Promise.resolve({ error: null });
      },
      updateUser: (params: any) => {
        console.log('Mock: updateUser called');
        return Promise.resolve({ data: null, error: null });
      },
      resetPasswordForEmail: (email: string, options?: any) => {
        console.log('Mock: resetPasswordForEmail called');
        return Promise.resolve({ data: null, error: null });
      },
      getUser: () => {
        console.log('Mock: getUser called');
        return Promise.resolve({ data: { user: null }, error: null });
      }
    },
    from: (table: string) => {
      console.log('Mock: from called with table:', table);
      return {
        select: () => ({
          eq: () => ({
            single: () => Promise.resolve({ data: null, error: new Error('Supabase disabled') }),
            maybeSingle: () => Promise.resolve({ data: null, error: new Error('Supabase disabled') })
          }),
          order: () => Promise.resolve({ data: [], error: new Error('Supabase disabled') })
        }),
        insert: () => ({
          select: () => ({
            single: () => Promise.resolve({ data: null, error: new Error('Supabase disabled') })
          })
        }),
        update: () => ({
          eq: () => ({
            select: () => ({
              single: () => Promise.resolve({ data: null, error: new Error('Supabase disabled') })
            })
          })
        })
      };
    },
    functions: {
      invoke: (functionName: string, options?: any) => {
        console.log('Mock: functions.invoke called');
        return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
      }
    }
  };
};

// Export everything that might be imported from @supabase/supabase-js
export default { createClient };
