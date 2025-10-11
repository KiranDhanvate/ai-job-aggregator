// Job Search API - Main functionality
export const searchJobs = async (params: {
  search?: string;
  location?: string;
  site?: string;
  results?: number;
}) => {
  try {
    // Map site parameter to site_name array format expected by the API
    let siteNames = ['linkedin']; // Default to LinkedIn
    if (params.site && params.site !== 'all') {
      siteNames = [params.site];
    }

    const requestBody = {
      search_term: params.search || 'developer',
      location: params.location || 'mumbai',
      results_wanted: params.results || 10,
      site_name: siteNames
    };

    console.log('Sending request to scrape API:', requestBody);

    const response = await fetch('http://localhost:5000/scrape', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the raw response text first to handle invalid JSON
    const responseText = await response.text();
    console.log('Raw response from scrape API:', responseText);
    
    let data;
    try {
      // Try to parse JSON, but handle NaN values
      const cleanedResponse = responseText.replace(/:\s*NaN/g, ': null');
      data = JSON.parse(cleanedResponse);
      console.log('Parsed response from scrape API:', data);
    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.error('Raw response that failed to parse:', responseText);
      throw new Error('Invalid JSON response from server');
    }
    
    // Transform the response to match our expected format
    // Add unique IDs to jobs since the API doesn't provide them
    const jobsWithIds = (data.jobs || []).map((job, index) => ({
      ...job,
      id: `job-${Date.now()}-${index}`, // Generate unique ID
      job_id: `job-${Date.now()}-${index}` // Also add job_id for compatibility
    }));

    const transformedData = {
      success: data.success !== false,
      data: {
        jobs: jobsWithIds,
        query: params,
        results_count: data.count || 0
      },
      message: data.success === false ? 'Failed to fetch jobs' : 'Jobs fetched successfully'
    };
    
    // Store the search results in localStorage, but also maintain a job cache
    localStorage.setItem("searchResults", JSON.stringify(transformedData));
    
    // Also maintain a persistent job cache that accumulates jobs across searches
    const existingJobCache = localStorage.getItem("jobCache");
    let jobCache: { [key: string]: any } = {};
    
    if (existingJobCache) {
      try {
        jobCache = JSON.parse(existingJobCache);
      } catch (e) {
        console.warn('Failed to parse existing job cache, starting fresh');
        jobCache = {};
      }
    }
    
    // Add new jobs to the cache
    jobsWithIds.forEach(job => {
      if (job.id) {
        jobCache[job.id] = job;
      }
      if (job.job_id) {
        jobCache[job.job_id] = job;
      }
    });
    
    // Store the updated job cache
    localStorage.setItem("jobCache", JSON.stringify(jobCache));
    
    return transformedData;
  } catch (error) {
    console.error('Error searching jobs:', error);
    return {
      success: false,
      data: null,
      message: error instanceof Error ? error.message : 'Failed to fetch jobs'
    };
  }
};

// Mock implementations for other APIs (removed Supabase dependencies)
// These can be implemented later with a different backend or local storage

export const getJobById = async (jobId: string) => {
  // Mock implementation - you can implement this with your backend later
  console.log('getJobById called with:', jobId);
  
  // Try to get job from localStorage search results
  const searchResults = localStorage.getItem("searchResults");
  if (searchResults) {
    const parsed = JSON.parse(searchResults);
    const job = parsed.data?.jobs?.find((j: any) => j.id === jobId || j.job_id === jobId);
    if (job) return job;
  }
  
  throw new Error('Job not found');
};

export const toggleSavedJob = async (jobId: string, action: 'save' | 'unsave') => {
  // Mock implementation using localStorage
  console.log('toggleSavedJob called:', { jobId, action });
  
  const savedJobs = JSON.parse(localStorage.getItem('savedJobs') || '[]');
  
  if (action === 'save') {
    if (!savedJobs.includes(jobId)) {
      savedJobs.push(jobId);
    }
  } else {
    const index = savedJobs.indexOf(jobId);
    if (index > -1) {
      savedJobs.splice(index, 1);
    }
  }
  
  localStorage.setItem('savedJobs', JSON.stringify(savedJobs));
  return { success: true };
};

export const getSavedJobs = async () => {
  // Mock implementation using localStorage
  console.log('getSavedJobs called');
  
  const savedJobIds = JSON.parse(localStorage.getItem('savedJobs') || '[]');
  const searchResults = localStorage.getItem("searchResults");
  
  if (searchResults && savedJobIds.length > 0) {
    const parsed = JSON.parse(searchResults);
    const savedJobs = parsed.data?.jobs?.filter((job: any) => 
      savedJobIds.includes(job.id || job.job_id)
    ) || [];
    return savedJobs;
  }
  
  return [];
};

export const checkJobSaved = async (jobId: string) => {
  // Mock implementation using localStorage
  const savedJobs = JSON.parse(localStorage.getItem('savedJobs') || '[]');
  return savedJobs.includes(jobId);
};

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
  // Mock implementation using localStorage
  console.log('submitApplication called:', { jobId, resumeUrl, coverLetterUrl, notes });
  
  const applications = JSON.parse(localStorage.getItem('applications') || '[]');
  const newApplication = {
    id: Date.now().toString(),
    jobId,
    resumeUrl,
    coverLetterUrl,
    notes,
    status: 'applied',
    applied_at: new Date().toISOString()
  };
  
  applications.push(newApplication);
  localStorage.setItem('applications', JSON.stringify(applications));
  
  return newApplication;
};

export const getApplications = async () => {
  // Mock implementation using localStorage
  console.log('getApplications called');
  return JSON.parse(localStorage.getItem('applications') || '[]');
};

export const getApplicationById = async (applicationId: string) => {
  // Mock implementation using localStorage
  console.log('getApplicationById called:', applicationId);
  
  const applications = JSON.parse(localStorage.getItem('applications') || '[]');
  const application = applications.find((app: any) => app.id === applicationId);
  
  if (!application) {
    throw new Error('Application not found');
  }
  
  return application;
};

export const updateApplicationStatus = async (applicationId: string, status: string) => {
  // Mock implementation using localStorage
  console.log('updateApplicationStatus called:', { applicationId, status });
  
  const applications = JSON.parse(localStorage.getItem('applications') || '[]');
  const applicationIndex = applications.findIndex((app: any) => app.id === applicationId);
  
  if (applicationIndex === -1) {
    throw new Error('Application not found');
  }
  
  applications[applicationIndex].status = status;
  applications[applicationIndex].updated_at = new Date().toISOString();
  
  localStorage.setItem('applications', JSON.stringify(applications));
  
  return applications[applicationIndex];
};

export const addApplicationNote = async (applicationId: string, note: string) => {
  // Mock implementation using localStorage
  console.log('addApplicationNote called:', { applicationId, note });
  
  const notes = JSON.parse(localStorage.getItem('applicationNotes') || '[]');
  const newNote = {
    id: Date.now().toString(),
    application_id: applicationId,
    note,
    created_at: new Date().toISOString()
  };
  
  notes.push(newNote);
  localStorage.setItem('applicationNotes', JSON.stringify(notes));
  
  return newNote;
};

export const getNotifications = async (params?: {
  limit?: number;
  offset?: number;
  unreadOnly?: boolean;
}) => {
  // Mock implementation
  console.log('getNotifications called:', params);
  return [];
};

export const markNotificationAsRead = async (notificationId: string) => {
  // Mock implementation
  console.log('markNotificationAsRead called:', notificationId);
  return { success: true };
};

export const markAllNotificationsAsRead = async () => {
  // Mock implementation
  console.log('markAllNotificationsAsRead called');
  return { success: true };
};
