// SUPABASE COMPLETELY DISABLED - Mock implementation to prevent connection errors
// No actual Supabase imports to prevent any network requests

console.log('Supabase client disabled - using mock implementation');

// Complete mock supabase client to prevent errors
export const supabase = {
  auth: {
    onAuthStateChange: (callback: any) => {
      console.log('Mock: onAuthStateChange called');
      // Call callback immediately with no session
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
      console.log('Mock: signUp called with:', params);
      return Promise.resolve({ data: null, error: null });
    },
    signInWithPassword: (params: any) => {
      console.log('Mock: signInWithPassword called with:', params);
      return Promise.resolve({ data: null, error: null });
    },
    signInWithOtp: (params: any) => {
      console.log('Mock: signInWithOtp called with:', params);
      return Promise.resolve({ data: null, error: null });
    },
    signInWithOAuth: (params: any) => {
      console.log('Mock: signInWithOAuth called with:', params);
      return Promise.resolve({ data: null, error: null });
    },
    signOut: () => {
      console.log('Mock: signOut called');
      return Promise.resolve({ error: null });
    },
    updateUser: (params: any) => {
      console.log('Mock: updateUser called with:', params);
      return Promise.resolve({ data: null, error: null });
    },
    resetPasswordForEmail: (email: string, options?: any) => {
      console.log('Mock: resetPasswordForEmail called with:', email, options);
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
      select: (columns?: string) => {
        console.log('Mock: select called with columns:', columns);
        return {
          eq: (column: string, value: any) => {
            console.log('Mock: eq called with:', column, value);
            return {
              single: () => {
                console.log('Mock: single called');
                return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
              },
              maybeSingle: () => {
                console.log('Mock: maybeSingle called');
                return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
              }
            };
          },
          order: (column: string, options?: any) => {
            console.log('Mock: order called with:', column, options);
            return Promise.resolve({ data: [], error: new Error('Supabase disabled') });
          }
        };
      },
      insert: (data: any) => {
        console.log('Mock: insert called with:', data);
        return {
          select: (columns?: string) => {
            console.log('Mock: insert.select called with columns:', columns);
            return {
              single: () => {
                console.log('Mock: insert.single called');
                return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
              }
            };
          }
        };
      },
      update: (data: any) => {
        console.log('Mock: update called with:', data);
        return {
          eq: (column: string, value: any) => {
            console.log('Mock: update.eq called with:', column, value);
            return {
              select: (columns?: string) => {
                console.log('Mock: update.select called with columns:', columns);
                return {
                  single: () => {
                    console.log('Mock: update.single called');
                    return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
                  }
                };
              }
            };
          }
        };
      }
    };
  },
  functions: {
    invoke: (functionName: string, options?: any) => {
      console.log('Mock: functions.invoke called with:', functionName, options);
      return Promise.resolve({ data: null, error: new Error('Supabase disabled') });
    }
  }
};