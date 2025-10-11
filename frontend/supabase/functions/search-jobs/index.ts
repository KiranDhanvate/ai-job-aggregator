
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
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? '';
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY') ?? '';
    
    const supabase = createClient(supabaseUrl, supabaseAnonKey);
    
    // Parse request body
    const { queryParams } = await req.json();
    const searchParams = new URLSearchParams(queryParams);
    
    // Get parameters
    const query = searchParams.get('query') || '';
    const location = searchParams.get('location') || '';
    const experienceLevel = searchParams.get('experienceLevel') || '';
    const jobType = searchParams.get('jobType') || '';
    const remote = searchParams.get('remote') === 'true';
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '10');
    const offset = (page - 1) * limit;
    
    // Build the query
    let query_builder = supabase
      .from('jobs')
      .select(`
        *,
        job_skills(skill)
      `, { count: 'exact' });
    
    // Apply filters
    if (query) {
      query_builder = query_builder.or(`title.ilike.%${query}%,company.ilike.%${query}%,description.ilike.%${query}%`);
    }
    
    if (location) {
      query_builder = query_builder.ilike('location', `%${location}%`);
    }
    
    if (experienceLevel) {
      query_builder = query_builder.eq('experience_level', experienceLevel);
    }
    
    if (jobType) {
      query_builder = query_builder.eq('job_type', jobType);
    }
    
    if (remote) {
      query_builder = query_builder.eq('is_remote', true);
    }
    
    // Add pagination
    query_builder = query_builder
      .order('posted_at', { ascending: false })
      .range(offset, offset + limit - 1);
    
    const { data, error, count } = await query_builder;
    
    if (error) throw error;
    
    return new Response(
      JSON.stringify({
        jobs: data,
        total: count,
        page,
        limit,
        totalPages: Math.ceil((count || 0) / limit)
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    );
  } catch (error) {
    console.error('Error searching jobs:', error);
    
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    );
  }
});
