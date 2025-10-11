import { supabase } from '@/integrations/supabase/client';

// Job Search API
export const searchJobs = async (params: {
  search?: string;
  location?: string;
  site?: string;
  days_old?: number;
  results?: number;
  remote_only?: boolean;
}) => {
  try {
    const response = await fetch('https://careersync-m6ct.onrender.com/api/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        search: params.search || '',
        location: params.location || '',
        site: params.site || 'all',
        days_old: params.days_old || 7,
        results: params.results || 10,
        remote_only: params.remote_only || false,
      }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    
    // Store the search results in localStorage as a string
    localStorage.setItem("searchResults", JSON.stringify(data));
    
    return data;
  } catch (error) {
    console.error('Error searching jobs:', error);
    throw error;
  }
};

// Job Detail API
export const getJobById = async (jobId: string) => {
  const { data, error } = await supabase
    .from('jobs')
    .select(`
      *,
      job_skills(skill)
    `)
    .eq('id', jobId)
    .single();
  
  if (error) throw error;
  return data;
};

// Saved Jobs API
export const toggleSavedJob = async (jobId: string, action: 'save' | 'unsave') => {
  const { data, error } = await supabase.functions.invoke('toggle-saved-job', {
    method: 'POST',
    body: { jobId, action }
  });
  
  if (error) throw error;
  return data;
};

export const getSavedJobs = async () => {
  const { data, error } = await supabase
    .from('saved_jobs')
    .select(`
      job_id,
      jobs(*)
    `)
    .order('created_at', { ascending: false });
  
  if (error) throw error;
  return data.map(item => ({
    id: item.job_id,
    ...item.jobs
  }));
};

export const checkJobSaved = async (jobId: string) => {
  const { data, error } = await supabase
    .from('saved_jobs')
    .select('id')
    .eq('job_id', jobId)
    .maybeSingle();
  
  if (error) throw error;
  return !!data;
};

// Job Applications API
export const submitApplication = async ({
  jobId,
  resumeUrl,
  coverLetterUrl,
  notes
}: {
  jobId: string;
  resumeUrl?: string;
  coverLetterUrl?: string;
  notes?: string;
}) => {
  const { data, error } = await supabase.functions.invoke('submit-application', {
    method: 'POST',
    body: { jobId, resumeUrl, coverLetterUrl, notes }
  });
  
  if (error) throw error;
  return data;
};

export const getApplications = async () => {
  const { data, error } = await supabase
    .from('applications')
    .select(`
      *,
      jobs(*)
    `)
    .order('applied_at', { ascending: false });
  
  if (error) throw error;
  return data;
};

export const getApplicationById = async (applicationId: string) => {
  const { data, error } = await supabase
    .from('applications')
    .select(`
      *,
      jobs(*),
      application_notes(*)
    `)
    .eq('id', applicationId)
    .single();
  
  if (error) throw error;
  return data;
};

export const updateApplicationStatus = async (applicationId: string, status: string) => {
  const { data, error } = await supabase
    .from('applications')
    .update({ 
      status, 
      updated_at: new Date().toISOString() 
    })
    .eq('id', applicationId)
    .select()
    .single();
  
  if (error) throw error;
  return data;
};

export const addApplicationNote = async (applicationId: string, note: string) => {
  const { data, error } = await supabase
    .from('application_notes')
    .insert([{
      application_id: applicationId,
      note,
      user_id: (await supabase.auth.getUser()).data.user?.id
    }])
    .select()
    .single();
  
  if (error) throw error;
  return data;
};

// Notifications API
export const getNotifications = async (params?: {
  limit?: number;
  offset?: number;
  unreadOnly?: boolean;
}) => {
  const { data, error } = await supabase.functions.invoke('get-notifications', {
    method: 'POST',
    body: { 
      limit: params?.limit,
      offset: params?.offset,
      unreadOnly: params?.unreadOnly 
    }
  });
  
  if (error) throw error;
  return data;
};

export const markNotificationAsRead = async (notificationId: string) => {
  // Using the edge function instead of direct DB access for better error handling and logging
  const { data, error } = await supabase.functions.invoke('mark-notification', {
    method: 'POST',
    body: { notificationId, read: true }
  });
  
  if (error) throw error;
  return data;
};

export const markAllNotificationsAsRead = async () => {
  // Using the edge function instead of direct DB access for better error handling and logging
  const { data, error } = await supabase.functions.invoke('mark-all-notifications', {
    method: 'POST',
    body: { read: true }
  });
  
  if (error) throw error;
  return data;
};
