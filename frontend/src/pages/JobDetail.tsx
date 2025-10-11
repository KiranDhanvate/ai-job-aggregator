import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { 
  ArrowLeft, 
  Building, 
  MapPin, 
  Clock, 
  Calendar, 
  Bookmark, 
  Share2, 
  ExternalLink, 
  CheckCircle, 
  XCircle, 
  Briefcase, 
  GraduationCap, 
  Award, 
  DollarSign, 
  Users, 
  Globe
} from 'lucide-react';
import { motion } from 'framer-motion';
import Footer from '../components/Footer';

// Type definitions
interface Job {
  job_id: string;
  job_title: string;
  company_name: string;
  company_url: string;
  job_url: string;
  location: string;
  remote_work: string;
  job_level: string;
  job_source: string;
  
  // Additional properties for display (would be fetched from API in real app)
  companyDescription?: string;
  salary?: string;
  fullDescription?: string;
  expiresAt?: string;
  employmentType?: string;
  education?: string;
  skills?: string[];
  benefits?: string[];
  company_size?: string;
  industry?: string;
  companyWebsite?: string;
  companyLinkedIn?: string;
  matchPercentage?: number;
}

interface SearchResults {
  success: boolean;
  data: {
    jobs: Job[];
    query: any;
    results_count: number;
  };
  message: string;
  timestamp: string;
}

// Placeholder data for job details not available in localStorage
const placeholderJobDetails = {
  companyDescription: 'A leading technology company specializing in innovative solutions.',
  salary: 'Competitive',
  fullDescription: `
    <p>This position requires expertise in the technologies mentioned in the job title and description.</p>
    
    <h4>Responsibilities:</h4>
    <ul>
      <li>Develop and maintain software applications</li>
      <li>Collaborate with cross-functional teams</li>
      <li>Participate in code reviews and ensure code quality</li>
      <li>Troubleshoot and resolve technical issues</li>
    </ul>
    
    <h4>Requirements:</h4>
    <ul>
      <li>Experience with relevant programming languages and frameworks</li>
      <li>Strong problem-solving skills</li>
      <li>Good communication abilities</li>
      <li>Ability to work in a team environment</li>
    </ul>
  `,
  expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
  employmentType: 'Full-time',
  education: "Bachelor's degree in Computer Science or related field",
  skills: ['Java', 'Spring Boot', 'Hibernate', 'RESTful APIs', 'SQL'],
  benefits: ['Health Insurance', 'Flexible Hours', 'Professional Development'],
  company_size: '500+ employees',
  industry: 'Software Development',
  companyWebsite: 'https://example.com',
  matchPercentage: 85
};

const JobDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [similarJobs, setSimilarJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isSaved, setIsSaved] = useState(false);
  
  useEffect(() => {
    // Function to get job from localStorage
    const getJobFromLocalStorage = () => {
      try {
        // Get job ID from URL params
        const jobId = id;
        
        if (!jobId) {
          throw new Error('No job ID provided');
        }
        
        // Retrieve search results from localStorage
        const searchResultsString = localStorage.getItem('searchResults');
        
        if (!searchResultsString) {
          throw new Error('No search results found in localStorage');
        }
        
        // Parse search results
        const searchResults: SearchResults = JSON.parse(searchResultsString);
        
        if (!searchResults.success || !searchResults.data || !searchResults.data.jobs) {
          throw new Error('Invalid search results data');
        }
        
        // Find the job with matching ID
        const foundJob = searchResults.data.jobs.find(job => job.job_id === jobId);
        
        if (!foundJob) {
          throw new Error(`Job with ID ${jobId} not found`);
        }
        
        // Merge with placeholder details for display purposes
        const enhancedJob = {
          ...foundJob,
          ...placeholderJobDetails,
          postedAt: new Date().toISOString() // Mock posted date
        };
        
        setJob(enhancedJob);
        
        // Find similar jobs from the same data source
        // We'll define "similar" as jobs with the same job_level or similar job_title
        const otherJobs = searchResults.data.jobs.filter(j => j.job_id !== jobId);
        
        // Generate a simple similarity score based on job level and keywords in title
        const scoredJobs = otherJobs.map(j => {
          let score = 0;
          
          // Same job level gives points
          if (j.job_level === foundJob.job_level) {
            score += 30;
          }
          
          // Check for common keywords in job titles
          const currentJobWords = foundJob.job_title.toLowerCase().split(/\s+/);
          const comparedJobWords = j.job_title.toLowerCase().split(/\s+/);
          
          currentJobWords.forEach(word => {
            if (word.length > 3 && comparedJobWords.includes(word)) {
              score += 15;
            }
          });
          
          // Same company gives some points (but not too many)
          if (j.company_name === foundJob.company_name) {
            score += 10;
          }
          
          // Same location or both remote gives points
          if (j.location === foundJob.location || 
              (j.remote_work === 'Remote' && foundJob.remote_work === 'Remote')) {
            score += 20;
          }
          
          return {
            ...j,
            ...placeholderJobDetails,
            matchPercentage: Math.min(Math.round(score), 99) // Cap at 99%
          };
        });
        
        // Sort by similarity score and take top 2
        const topSimilarJobs = scoredJobs
          .sort((a, b) => (b.matchPercentage || 0) - (a.matchPercentage || 0))
          .slice(0, 2);
        
        setSimilarJobs(topSimilarJobs);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching job:', err);
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
        setLoading(false);
      }
    };
    
    getJobFromLocalStorage();
  }, [id]);
  
  // Function to calculate application deadline
  const getApplicationDeadline = () => {
    if (!job?.expiresAt) return 'Not specified';
    const expiryDate = new Date(job.expiresAt);
    return `${expiryDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}`;
  };
  
  // Function to format posted date (mock for now)
  const getPostedDate = () => {
    return `March 23, 2025`; // Using current date as mock
  };
  
  // Toggle saved state
  const handleSaveJob = () => {
    setIsSaved(!isSaved);
  };
  
  // Function to get experience level text
  const getExperienceLevelText = (level?: string) => {
    switch (level) {
      case 'entry_level':
        return 'Entry Level';
      case 'mid_level':
        return 'Mid Level';
      case 'senior_level':
        return 'Senior Level';
      case 'executive':
        return 'Executive';
      default:
        return 'Not Specified';
    }
  };

  // Render loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-premium-gradient text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl">Loading job details...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error || !job) {
    return (
      <div className="min-h-screen bg-premium-gradient text-white flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl mb-4">Error Loading Job</h2>
          <p className="text-silver-400">{error || 'Job not found'}</p>
          <Link to="/search" className="inline-block mt-6 premium-button rounded-lg px-6 py-2.5 font-medium">
            Back to Search
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-premium-gradient text-white">
      <Helmet>
        <title>{job.job_title} at {job.company_name} | CareerSync</title>
        <meta name="description" content={`Apply for ${job.job_title} position at ${job.company_name}`} />
      </Helmet>
      
      <main className="pt-24 pb-20">
        <div className="container mx-auto px-4 md:px-6">
          {/* Breadcrumb Navigation */}
          <div className="mb-6">
            <Link to="/search" className="flex items-center text-silver-400 hover:text-white transition-colors">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to search results
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2">
              {/* Job Header */}
              <motion.div 
                className="premium-glass rounded-lg p-6 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <div className="flex items-start gap-5">
                  <div className="h-16 w-16 rounded-md bg-white/10 flex items-center justify-center overflow-hidden flex-shrink-0 border border-white/10">
                    <Building className="h-8 w-8 text-silver-400" />
                  </div>
                  
                  <div className="flex-1">
                    <h1 className="text-2xl md:text-3xl font-medium text-white mb-2">
                      {job.job_title}
                    </h1>
                    
                    <div className="flex items-center text-silver-300 mb-4">
                      <span className="font-medium">{job.company_name}</span>
                    </div>
                    
                    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-silver-400">
                      <div className="flex items-center gap-1.5">
                        <MapPin className="h-4 w-4 text-navy-500" />
                        {job.remote_work === 'Remote' ? 'Remote' : job.location}
                      </div>
                      
                      <div className="flex items-center gap-1.5">
                        <Briefcase className="h-4 w-4 text-navy-500" />
                        {job.employmentType || 'Full-time'}
                      </div>
                      
                      <div className="flex items-center gap-1.5">
                        <DollarSign className="h-4 w-4 text-navy-500" />
                        {job.salary || 'Not specified'}
                      </div>
                      
                      <div className="flex items-center gap-1.5">
                        <Clock className="h-4 w-4 text-navy-500" />
                        Posted {getPostedDate()}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-3 mt-6">
                  <button 
                    className="premium-button rounded-lg px-6 py-2.5 font-medium flex items-center gap-2"
                    onClick={() => window.open(job.job_url, '_blank')}
                  >
                    Apply Now
                    <ExternalLink className="h-4 w-4" />
                  </button>
                  
                  <button 
                    className={`premium-button-outline rounded-lg px-4 py-2.5 flex items-center gap-2 ${
                      isSaved ? 'bg-navy-500/10 border-navy-500/30 text-navy-400' : ''
                    }`}
                    onClick={handleSaveJob}
                  >
                    <Bookmark className={`h-4 w-4 ${isSaved ? 'fill-navy-500 text-navy-500' : ''}`} />
                    {isSaved ? 'Saved' : 'Save Job'}
                  </button>
                  
                  <button 
                    className="premium-button-outline rounded-lg px-4 py-2.5 flex items-center gap-2"
                    onClick={() => navigator.share && navigator.share({
                      title: `${job.job_title} at ${job.company_name}`,
                      url: window.location.href,
                    })}
                  >
                    <Share2 className="h-4 w-4" />
                    Share
                  </button>
                </div>
              </motion.div>
              
              {/* Match Score */}
              <motion.div 
                className="premium-glass rounded-lg p-6 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1 }}
              >
                <h2 className="text-xl font-medium mb-4">Match Analysis</h2>
                
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-silver-300">Overall Match</span>
                    <span className="font-medium text-white">{job.matchPercentage || 80}%</span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-navy-500 to-teal-500 rounded-full"
                      style={{ width: `${job.matchPercentage || 80}%` }}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium text-silver-300 mb-3">Skills Match</h3>
                    <ul className="space-y-2">
                      {(job.skills || ['Java', 'Spring Boot', 'Hibernate', 'RESTful APIs', 'SQL']).slice(0, 5).map((skill, index) => (
                        <li key={index} className="flex items-center justify-between">
                          <span className="text-sm text-silver-400">{skill}</span>
                          <span className={`px-2 py-0.5 rounded-full text-xs text-white ${
                            index < 3 ? 'bg-teal-500' : 'bg-white/10'
                          }`}>
                            {index < 3 ? 'Strong' : 'Basic'}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-silver-300 mb-3">Requirements</h3>
                    <ul className="space-y-2">
                      <li className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-teal-500" />
                        <span className="text-sm text-silver-400">Experience Level</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-teal-500" />
                        <span className="text-sm text-silver-400">Technical Skills</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-teal-500" />
                        <span className="text-sm text-silver-400">Education</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <XCircle className="h-4 w-4 text-silver-500" />
                        <span className="text-sm text-silver-500 line-through">Location</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </motion.div>
              
              {/* Job Description */}
              <motion.div 
                className="premium-glass rounded-lg p-6 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
              >
                <h2 className="text-xl font-medium mb-4">Job Description</h2>
                <div 
                  className="prose prose-invert max-w-none text-silver-300"
                  dangerouslySetInnerHTML={{ __html: job.fullDescription || '<p>Please visit the job link for a detailed description.</p>' }}
                />
              </motion.div>
              
              {/* Company Information */}
              <motion.div 
                className="premium-glass rounded-lg p-6 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
              >
                <h2 className="text-xl font-medium mb-4">About {job.company_name}</h2>
                <p className="text-silver-300 mb-4">{job.companyDescription}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                  <div>
                    <h3 className="text-sm font-medium text-silver-300 mb-3">Company Details</h3>
                    <ul className="space-y-3">
                      <li className="flex items-center gap-2">
                        <Users className="h-4 w-4 text-navy-500" />
                        <span className="text-sm text-silver-400">{job.company_size}</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Briefcase className="h-4 w-4 text-navy-500" />
                        <span className="text-sm text-silver-400">{job.industry || 'Software Development'}</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Globe className="h-4 w-4 text-navy-500" />
                        <a 
                          href={job.company_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm text-navy-400 hover:text-navy-300 transition-colors"
                        >
                          Company LinkedIn
                        </a>
                      </li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-silver-300 mb-3">Benefits</h3>
                    <div className="flex flex-wrap gap-2">
                      {(job.benefits || ['Health Insurance', 'Flexible Hours', 'Professional Development']).map((benefit, index) => (
                        <span key={index} className="px-3 py-1 bg-white/5 rounded-full text-xs text-silver-400">
                          {benefit}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
            
            {/* Sidebar */}
            <div className="lg:col-span-1">
              {/* Application Deadline */}
              <motion.div 
                className="premium-glass rounded-lg p-5 mb-6"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4 }}
              >
                <h3 className="text-lg font-medium mb-3">Application Details</h3>
                
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <Calendar className="h-5 w-5 text-navy-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-silver-300">Application Deadline</p>
                      <p className="text-silver-400">{getApplicationDeadline()}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <Award className="h-5 w-5 text-navy-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-silver-300">Experience Level</p>
                      <p className="text-silver-400">
                        {getExperienceLevelText(job.job_level)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <GraduationCap className="h-5 w-5 text-navy-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-silver-300">Education</p>
                      <p className="text-silver-400">{job.education}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
              
              {/* Similar Jobs */}
              <motion.div 
                className="premium-glass rounded-lg p-5"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.1 }}
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium">Similar Jobs</h3>
                  <Link to="/search" className="text-sm text-navy-400 hover:text-teal-400 transition-colors">
                    View All
                  </Link>
                </div>
                
                <div className="space-y-4">
                  {similarJobs.length > 0 ? (
                    similarJobs.map(similarJob => (
                      <Link to={`/job/${similarJob.job_id}`} key={similarJob.job_id} className="block">
                        <div className="premium-card rounded-lg p-4 hover:bg-white/5 transition-all">
                          <div className="flex items-start gap-3">
                            <div className="h-10 w-10 rounded-md bg-white/10 flex items-center justify-center overflow-hidden flex-shrink-0 border border-white/10">
                              <Building className="h-5 w-5 text-silver-400" />
                            </div>
                            
                            <div>
                              <h4 className="text-base font-medium text-white line-clamp-1">{similarJob.job_title}</h4>
                              <p className="text-sm text-silver-400 mb-1">{similarJob.company_name}</p>
                              <div className="flex items-center gap-2 text-xs text-silver-500">
                                <span className="flex items-center gap-1">
                                  <MapPin className="h-3 w-3" />
                                  {similarJob.remote_work === 'Remote' ? 'Remote' : similarJob.location}
                                </span>
                                
                                {similarJob.matchPercentage && (
                                  <span className="px-1.5 py-0.5 bg-teal-500/20 rounded-sm text-teal-300 flex items-center gap-1">
                                    <CheckCircle className="h-3 w-3" />
                                    {similarJob.matchPercentage}% Match
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))
                  ) : (
                    <p className="text-silver-400 text-sm">No similar jobs found.</p>
                  )}
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default JobDetail;