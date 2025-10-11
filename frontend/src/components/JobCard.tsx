import { useState, useEffect } from 'react';
import { Bookmark, ExternalLink, MapPin, Building, DollarSign, Briefcase, Calendar, Clock, Award } from 'lucide-react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { checkJobSaved, toggleSavedJob } from '@/services/api';
import { toast } from '@/components/ui/use-toast';

interface JobCardProps {
  job: {
    id: string;
    title: string;
    company: string;
    location: string;
    company_url?: string;
    salary?: string;
    salary_range?: string;
    logo_url?: string;
    description?: string;
    source?: string;
    source_url?: string;
    posted_at: string;
    is_remote?: boolean;
    experience_level?: string;
    job_type?: string;
    matchPercentage?: number;
  };
}

const JobCard = ({ job }: JobCardProps) => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isSaved, setIsSaved] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    // Check if job is saved when user is logged in
    const checkSavedStatus = async () => {
      if (user) {
        try {
          const saved = await checkJobSaved(job.id);
          setIsSaved(saved);
        } catch (error) {
          console.error('Error checking saved status:', error);
        }
      }
    };
    
    checkSavedStatus();
  }, [job.id, user]);
  
  const handleSave = async (e: React.MouseEvent) => {
    e.preventDefault();
    
    if (!user) {
      toast({
        title: "Authentication required",
        description: "Please sign in to save jobs",
        variant: "destructive"
      });
      return;
    }
    
    try {
      setIsLoading(true);
      const action = isSaved ? 'unsave' : 'save';
      await toggleSavedJob(job.id, action);
      setIsSaved(!isSaved);
      
      toast({
        title: isSaved ? "Job removed" : "Job saved",
        description: isSaved 
          ? "The job has been removed from your saved list" 
          : "The job has been added to your saved list",
      });
    } catch (error) {
      console.error('Error toggling saved status:', error);
      toast({
        title: "Error",
        description: "Failed to save/unsave the job. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatTimeAgo = (dateString: string) => {
    if (!dateString) return 'Unknown';
    
    const now = new Date();
    const postedDate = new Date(dateString);
    const diffInDays = Math.floor((now.getTime() - postedDate.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffInDays === 0) {
      return 'Today';
    } else if (diffInDays === 1) {
      return 'Yesterday';
    } else if (diffInDays < 7) {
      return `${diffInDays} days ago`;
    } else if (diffInDays < 30) {
      const weeks = Math.floor(diffInDays / 7);
      return `${weeks} ${weeks === 1 ? 'week' : 'weeks'} ago`;
    } else {
      const months = Math.floor(diffInDays / 30);
      return `${months} ${months === 1 ? 'month' : 'months'} ago`;
    }
  };

  const getExperienceLevelIcon = () => {
    switch (job.experience_level) {
      case 'entry':
        return <Briefcase className="h-3.5 w-3.5" />;
      case 'mid':
        return <Briefcase className="h-3.5 w-3.5" />;
      case 'senior':
        return <Award className="h-3.5 w-3.5" />;
      case 'executive':
        return <Award className="h-3.5 w-3.5" />;
      default:
        return <Briefcase className="h-3.5 w-3.5" />;
    }
  };

  const getExperienceLevelText = () => {
    switch (job.experience_level) {
      case 'entry':
        return 'Entry Level';
      case 'mid':
        return 'Mid Level';
      case 'senior':
        return 'Senior Level';
      case 'executive':
        return 'Executive';
      default:
        return 'Not Specified';
    }
  };

  const getJobTypeText = () => {
    if (!job.job_type) return null;
    return job.job_type.charAt(0).toUpperCase() + job.job_type.slice(1).replace('-', ' ');
  };

  const handleCardClick = () => {
    navigate(`/job/${job.id}`);
  };

  const handleApplyClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Open the application URL in a new tab
    if (job.company_url) {
      window.open(job.company_url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div 
      onClick={handleCardClick}
      className="block cursor-pointer"
    >
      <motion.div
        className="premium-card-interactive h-full rounded-lg p-5 relative flex flex-col"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        whileHover={{ y: -4 }}
      >
        <div className="flex items-start gap-4 mb-4">
          <div className="h-12 w-12 rounded-md bg-white/10 flex items-center justify-center overflow-hidden flex-shrink-0 border border-white/5">
            {job.logo_url ? (
              <img 
                src={job.logo_url} 
                alt={`${job.company} logo`} 
                className="w-full h-full object-contain p-1" 
              />
            ) : (
              <Building className="h-6 w-6 text-silver-400" />
            )}
          </div>
          
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-medium text-white line-clamp-1 tracking-tight group-hover:text-navy-400 transition-colors">
              {job.title}
            </h3>
            
            <p className="text-silver-300 line-clamp-1 mb-1">{job.company}</p>
            
            <div className="flex flex-wrap items-center text-sm gap-x-3 gap-y-1 text-silver-500">
              <span className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {job.is_remote ? 'Remote' : job.location}
              </span>
              
              {(job.salary || job.salary_range) && (
                <span className="flex items-center gap-1">
                  <DollarSign className="h-3.5 w-3.5" />
                  {job.salary || job.salary_range}
                </span>
              )}
              
              {job.experience_level && (
                <span className="flex items-center gap-1">
                  {getExperienceLevelIcon()}
                  {getExperienceLevelText()}
                </span>
              )}

              {job.job_type && (
                <span className="flex items-center gap-1">
                  <Briefcase className="h-3.5 w-3.5" />
                  {getJobTypeText()}
                </span>
              )}
            </div>
          </div>
          
          <button 
            onClick={handleSave}
            className={`p-2 rounded-full premium-glass transition-all ${isLoading ? 'opacity-50' : 'hover:bg-white/10'}`}
            aria-label={isSaved ? "Remove from saved jobs" : "Save job"}
            disabled={isLoading}
          >
            <Bookmark 
              className={`h-5 w-5 ${isSaved ? 'fill-navy-500 text-navy-500' : 'text-silver-500'}`} 
            />
          </button>
        </div>
        
        <p className="text-silver-400 text-sm line-clamp-2 mb-4">{job.description}</p>
        
        {job.matchPercentage && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-silver-500">Match Score</span>
              <span className="text-xs font-medium text-navy-500">{job.matchPercentage}%</span>
            </div>
            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-navy-500 to-teal-500 rounded-full"
                style={{ width: `${job.matchPercentage}%` }}
              />
            </div>
          </div>
        )}
        
        <div className="mt-auto pt-4 flex items-center justify-between border-t border-white/10">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 premium-glass rounded text-xs text-silver-400">
              {job.source || 'CareerSync'}
            </span>
            <span className="flex items-center text-xs text-silver-500">
              <Clock className="h-3 w-3 mr-1" />
              {formatTimeAgo(job.posted_at)}
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            {job.source_url && (
              <button 
                onClick={handleApplyClick}
                className="premium-button rounded-lg px-3 py-1 text-xs"
              >
                Apply Now
              </button>
            )}
            <span className="text-navy-400 text-sm flex items-center gap-1 group-hover:gap-2 transition-all">
              View Details
              <ExternalLink className="h-4 w-4" />
            </span>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default JobCard;
