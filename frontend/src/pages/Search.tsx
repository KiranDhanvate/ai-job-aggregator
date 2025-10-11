import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { 
  Filter, 
  SlidersHorizontal, 
  Briefcase, 
  MapPin, 
  Clock, 
  BarChart, 
  List, 
  Grid, 
  Search as SearchIcon,
  X,
  ChevronDown,
  Award,
  Building,
  Banknote,
  Wifi,
  Clock3,
  CalendarDays
} from 'lucide-react';
import JobCard from '../components/JobCard';
import SearchBar from '../components/SearchBar';
import Footer from '../components/Footer';
import { motion } from 'framer-motion';
import { useSearch } from '@/contexts/SearchContext';

// Update mockJobs structure to match API response format
const mockJobs = [
  // Will be replaced with real data from API
];

// Update filter categories based on the new API structure
const filterCategories = [
  {
    name: 'Experience Level',
    key: 'experienceLevel',
    icon: Award,
    options: [
      { value: 'entry_level', label: 'Entry Level' },
      { value: 'mid_level', label: 'Mid Level' },
      { value: 'senior_level', label: 'Senior Level' },
      { value: 'executive', label: 'Executive' },
    ],
  },
  {
    name: 'Job Type',
    key: 'jobType',
    icon: Briefcase,
    options: [
      { value: 'full-time', label: 'Full-time' },
      { value: 'part-time', label: 'Part-time' },
      { value: 'contract', label: 'Contract' },
      { value: 'internship', label: 'Internship' },
      { value: 'temporary', label: 'Temporary' },
    ],
  },
  {
    name: 'Remote Options',
    key: 'remote',
    icon: Wifi,
    options: [
      { value: 'remote', label: 'Remote' },
      { value: 'hybrid', label: 'Hybrid' },
      { value: 'onsite', label: 'On-site' },
    ],
  },
  {
    name: 'Date Posted',
    key: 'datePosted',
    icon: CalendarDays,
    options: [
      { value: '1', label: 'Today' },
      { value: '3', label: 'Past 3 days' },
      { value: '7', label: 'Past week' },
      { value: '14', label: 'Past 2 weeks' },
      { value: '30', label: 'Past month' },
    ],
  },
];

const Search = () => {
  const location = useLocation();
  const { 
    searchParams, 
    searchResults, 
    isLoading: isContextLoading, 
    error: contextError,
    setSearchParams, 
    setSearchResults, 
    setIsLoading: setContextIsLoading, 
    setError: setContextError 
  } = useSearch();
  
  const [filteredJobs, setFilteredJobs] = useState(searchResults?.jobs || []);
  const [activeFilters, setActiveFilters] = useState<Record<string, string[]>>({});
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState('relevance');
  const [expandedFilters, setExpandedFilters] = useState<string[]>([]);
  const [isLocalLoading, setIsLocalLoading] = useState(false);
  
  // Modify the URL params effect to not automatically fetch jobs, just update UI from existing results
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const searchQuery = params.get('q');
    const locationParam = params.get('location');
    const site = params.get('site') || 'all';
    const daysOld = params.get('days_old') ? parseInt(params.get('days_old') || '7') : 7;
    const results = params.get('results') ? parseInt(params.get('results') || '10') : 10;
    const remoteOnly = params.get('remote_only') === 'true';
    
    // Create the search parameters object
    const newSearchParams = {
      search: searchQuery || '',
      location: locationParam || '',
      site,
      days_old: daysOld,
      results,
      remote_only: remoteOnly,
    };

    // Just update search params in context but don't fetch
    setSearchParams(newSearchParams);
    
    // If we have existing results, just show them
    if (searchResults && searchResults.jobs) {
      setFilteredJobs(searchResults.jobs);
    }
  }, [location.search, searchResults]); // Only depend on location.search and existing results

  const toggleFilter = (category: string, value: string) => {
    setActiveFilters(prev => {
      const currentFilters = { ...prev };
      if (!currentFilters[category]) {
        currentFilters[category] = [value];
      } else if (currentFilters[category].includes(value)) {
        currentFilters[category] = currentFilters[category].filter(v => v !== value);
        if (currentFilters[category].length === 0) {
          delete currentFilters[category];
        }
      } else {
        currentFilters[category] = [...currentFilters[category], value];
      }
      return currentFilters;
    });
  };
  
  const clearAllFilters = () => {
    setActiveFilters({});
  };
  
  const toggleExpandFilter = (category: string) => {
    setExpandedFilters(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };
  
  const getActiveFilterCount = () => {
    return Object.values(activeFilters).reduce((sum, values) => sum + values.length, 0);
  };
  
  // Apply filters and sorting to jobs
  useEffect(() => {
    if (!searchResults) return;
    
    let result = [...searchResults.jobs];
    
    // Apply active filters
    Object.entries(activeFilters).forEach(([category, values]) => {
      if (values.length > 0) {
        result = result.filter(job => {
          switch(category) {
            case 'experienceLevel':
              return values.includes(job.job_level || '');
            case 'remote':
              return (values.includes('remote') && job.remote_work === 'Remote') || 
                     (values.includes('hybrid') && job.remote_work === 'Hybrid') ||
                     (values.includes('onsite') && job.remote_work === 'On-site');
            default:
              return true;
          }
        });
      }
    });
    
    // Apply sorting
    if (sortBy === 'newest') {
      result.sort((a, b) => new Date(b.date_posted).getTime() - new Date(a.date_posted).getTime());
    } else if (sortBy === 'match') {
      result.sort((a, b) => {
        // Sort by skills match if available
        if (a.skills_required && b.skills_required) {
          return b.skills_required.length - a.skills_required.length;
        }
        return 0;
      });
    }
    
    setFilteredJobs(result);
  }, [searchResults, activeFilters, sortBy]);

  // Transform job data to match JobCard component expectations
  const transformJobForCard = (job) => {
    return {
      id: job.job_id,
      title: job.job_title,
      company: job.company_name,
      location: job.location || "Location not specified",
      company_url: job.company_url,
      salary: job.salary_range ? 
        `${job.salary_range.currency} ${job.salary_range.min_amount?.toLocaleString()} - ${job.salary_range.max_amount?.toLocaleString()}` : 
        undefined,
      logo_url: job.company_logo,
      description: job.job_description,
      source: job.job_source,
      source_url: job.job_url,
      posted_at: job.date_posted,
      is_remote: job.remote_work === 'Remote',
      experience_level: job.job_level,
      job_type: job.listing_type,
      matchPercentage: job.skills_required ? Math.min(job.skills_required.length * 20, 100) : undefined,
    };
  };

  // Determine loading state by combining context and local loading
  const isLoading = isContextLoading || isLocalLoading;
  // Use context error if available
  const error = contextError;

  return (
    <div className="min-h-screen bg-premium-gradient text-white">
      <Helmet>
        <title>Job Search | CareerSync</title>
        <meta name="description" content="Search for jobs, internships, and career opportunities on CareerSync" />
      </Helmet>
      
      <main className="pt-24 pb-20">
        <div className="container mx-auto px-4 md:px-6">
          {/* Search Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-semibold mb-2">Find Your Perfect Career Match</h1>
            <p className="text-silver-400 mb-6">
              Browse thousands of jobs from top companies around the world
            </p>
            
            <SearchBar />
          </div>
          
          {/* Main Content */}
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Filters Sidebar */}
            <div className="w-full lg:w-64 flex-shrink-0">
              <motion.div 
                className="premium-glass rounded-lg overflow-hidden sticky top-24"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4 }}
              >
                <div className="p-4 border-b border-white/10 flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <Filter className="h-4 w-4 text-navy-500" />
                    <h2 className="font-medium">Filters</h2>
                  </div>
                  
                  {getActiveFilterCount() > 0 && (
                    <button 
                      className="text-xs text-navy-500 hover:text-teal-500 transition-colors"
                      onClick={clearAllFilters}
                    >
                      Clear all
                    </button>
                  )}
                </div>
                
                <div className="p-4 max-h-[calc(100vh-200px)] overflow-y-auto premium-scrollbar">
                  {filterCategories.map((category) => (
                    <div key={category.key} className="mb-4 pb-4 border-b border-white/10 last:border-b-0 last:mb-0 last:pb-0">
                      <button 
                        className="w-full flex items-center justify-between text-silver-300 hover:text-white transition-colors mb-2"
                        onClick={() => toggleExpandFilter(category.key)}
                        aria-expanded={expandedFilters.includes(category.key)}
                      >
                        <div className="flex items-center gap-2">
                          <category.icon className="h-4 w-4 text-navy-500" />
                          <span className="font-medium">{category.name}</span>
                        </div>
                        <ChevronDown 
                          className={`h-4 w-4 transition-transform ${
                            expandedFilters.includes(category.key) ? 'rotate-180' : ''
                          }`} 
                        />
                      </button>
                      
                      {expandedFilters.includes(category.key) && (
                        <div className="mt-2 space-y-2">
                          {category.options.map((option) => (
                            <label 
                              key={option.value} 
                              className="flex items-center gap-2 text-sm text-silver-400 hover:text-white transition-colors cursor-pointer"
                            >
                              <input 
                                type="checkbox" 
                                className="form-checkbox h-4 w-4 rounded border-white/20 text-navy-500 focus:ring-navy-500"
                                checked={(activeFilters[category.key] || []).includes(option.value)}
                                onChange={() => toggleFilter(category.key, option.value)}
                              />
                              {option.label}
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
            
            {/* Results */}
            <div className="flex-1">
              {/* Results Header */}
              <div className="premium-glass rounded-lg p-4 mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                  <h2 className="text-lg font-medium">
                    {isLoading ? 'Searching...' : `${filteredJobs.length} jobs found`}
                  </h2>
                  {getActiveFilterCount() > 0 && (
                    <p className="text-sm text-silver-400">
                      {getActiveFilterCount()} active {getActiveFilterCount() === 1 ? 'filter' : 'filters'}
                    </p>
                  )}
                </div>
                
                <div className="flex items-center gap-4 w-full sm:w-auto">
                  <div className="flex-1 sm:flex-none">
                    <select 
                      className="premium-glass text-white rounded-lg px-3 py-2 border border-white/10 w-full focus:border-navy-500 focus:ring-1 focus:ring-navy-500 focus:outline-none"
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                    >
                      <option value="relevance">Relevance</option>
                      <option value="newest">Newest</option>
                      <option value="match">Match Score</option>
                    </select>
                  </div>
                  
                  <div className="premium-glass rounded-lg border border-white/10 flex">
                    <button 
                      className={`p-2 ${viewMode === 'grid' ? 'bg-white/10 text-navy-500' : 'text-silver-500'}`}
                      onClick={() => setViewMode('grid')}
                      aria-label="Grid view"
                    >
                      <Grid className="h-5 w-5" />
                    </button>
                    <button 
                      className={`p-2 ${viewMode === 'list' ? 'bg-white/10 text-navy-500' : 'text-silver-500'}`}
                      onClick={() => setViewMode('list')}
                      aria-label="List view"
                    >
                      <List className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>
              
              {/* Active Filters */}
              {getActiveFilterCount() > 0 && (
                <div className="flex flex-wrap gap-2 mb-6">
                  {Object.entries(activeFilters).map(([category, values]) => 
                    values.map(value => {
                      const categoryObj = filterCategories.find(c => c.key === category);
                      const optionObj = categoryObj?.options.find(o => o.value === value);
                      
                      return (
                        <div 
                          key={`${category}-${value}`} 
                          className="premium-glass rounded-full px-3 py-1 flex items-center gap-1 text-sm border border-white/10"
                        >
                          <span>{categoryObj?.name}: {optionObj?.label}</span>
                          <button 
                            onClick={() => toggleFilter(category, value)}
                            className="text-silver-500 hover:text-white transition-colors"
                            aria-label={`Remove ${optionObj?.label} filter`}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
              
              {/* Job Results */}
              {isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {[1, 2, 3, 4].map((_, index) => (
                    <div key={index} className="premium-card rounded-lg p-5 animate-pulse">
                      <div className="flex items-start gap-4 mb-4">
                        <div className="h-12 w-12 rounded-md bg-white/10 flex-shrink-0"></div>
                        <div className="flex-1">
                          <div className="h-5 bg-white/10 rounded mb-2 w-3/4"></div>
                          <div className="h-4 bg-white/10 rounded mb-3 w-1/2"></div>
                          <div className="h-3 bg-white/10 rounded w-2/3"></div>
                        </div>
                      </div>
                      <div className="h-4 bg-white/10 rounded mb-2 w-full"></div>
                      <div className="h-4 bg-white/10 rounded mb-6 w-5/6"></div>
                      <div className="h-3 bg-white/10 rounded mt-auto"></div>
                    </div>
                  ))}
                </div>
              ) : error ? (
                <div className="premium-glass rounded-lg p-8 text-center">
                  <div className="mb-4 mx-auto w-16 h-16 flex items-center justify-center rounded-full bg-white/5">
                    <X className="h-8 w-8 text-silver-500" />
                  </div>
                  <h3 className="text-xl font-medium mb-2">Error</h3>
                  <p className="text-silver-400 mb-6">
                    {error}
                  </p>
                  <button 
                    onClick={() => window.location.reload()}
                    className="premium-button rounded-lg px-6 py-2.5"
                  >
                    Try Again
                  </button>
                </div>
              ) : filteredJobs.length > 0 ? (
                <div className={`${
                  viewMode === 'grid' 
                    ? 'grid grid-cols-1 md:grid-cols-2 gap-6' 
                    : 'space-y-4'
                }`}>
                  {filteredJobs.map((job) => (
                    <JobCard key={job.job_id} job={transformJobForCard(job)} />
                  ))}
                </div>
              ) : (
                <div className="premium-glass rounded-lg p-8 text-center">
                  <div className="mb-4 mx-auto w-16 h-16 flex items-center justify-center rounded-full bg-white/5">
                    <SearchIcon className="h-8 w-8 text-silver-500" />
                  </div>
                  <h3 className="text-xl font-medium mb-2">No jobs found</h3>
                  <p className="text-silver-400 mb-6">
                    Try adjusting your search criteria or filters to find more job opportunities.
                  </p>
                  <button 
                    onClick={clearAllFilters}
                    className="premium-button rounded-lg px-6 py-2.5"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Search;
