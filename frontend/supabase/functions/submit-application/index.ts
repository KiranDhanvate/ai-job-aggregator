
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Get authentication details
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 401 
        }
      );
    }

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? '';
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';
    const supabase = createClient(supabaseUrl, supabaseAnonKey, {
      global: {
        headers: {
          Authorization: authHeader
        }
      }
    });

    // Get user from auth
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 401 
        }
      );
    }

    // Parse request body
    const { jobId, resumeUrl, coverLetterUrl, notes } = await req.json();
    if (!jobId) {
      return new Response(
        JSON.stringify({ error: 'Job ID is required' }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 400 
        }
      );
    }

    // Check if application already exists
    const { data: existingApplication } = await supabase
      .from('job_applications')
      .select('*')
      .eq('user_id', user.id)
      .eq('job_id', jobId)
      .maybeSingle();

    let result;
    if (existingApplication) {
      // Update existing application
      const { data, error } = await supabase
        .from('job_applications')
        .update({
          resume_url: resumeUrl,
          cover_letter_url: coverLetterUrl,
          notes,
          last_updated_at: new Date().toISOString()
        })
        .eq('id', existingApplication.id)
        .select()
        .single();

      if (error) throw error;
      result = { updated: true, data };
    } else {
      // Create new application
      const { data, error } = await supabase
        .from('job_applications')
        .insert({
          user_id: user.id,
          job_id: jobId,
          resume_url: resumeUrl,
          cover_letter_url: coverLetterUrl,
          notes,
          status: 'applied'
        })
        .select()
        .single();

      if (error) throw error;
      
      // Create notification for application submission
      await supabase
        .from('notifications')
        .insert({
          user_id: user.id,
          title: 'Application Submitted',
          message: 'Your job application has been successfully submitted.',
          type: 'application',
          related_entity_id: data.id
        });
        
      result = { created: true, data };
    }

    return new Response(
      JSON.stringify(result),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    );
  } catch (error) {
    console.error('Error submitting application:', error);
    
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    );
  }
});
