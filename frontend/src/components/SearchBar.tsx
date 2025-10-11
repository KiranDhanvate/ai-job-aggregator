import { useState, useRef, useEffect, useCallback } from 'react';
import { Search, MapPin, X, Briefcase, Filter, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useSearch } from '@/contexts/SearchContext';
import { searchJobs } from '@/services/api';

const SearchBar = () => {
  const { 
    searchParams, 
    setSearchParams,
    setSearchResults,
    setIsLoading,
    setError
  } = useSearch();
  const [searchQuery, setSearchQuery] = useState('');
  const [location, setLocation] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    site: 'linkedin',
    results: 10,
  });
  const filtersRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const isInitialMount = useRef(true);

  // Fix: Combine the two separate useEffect hooks for initializing state from URL and context
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      
      // Get parameters from URL first
      const url = new URL(window.location.href);
      const q = url.searchParams.get('q');
      const loc = url.searchParams.get('location');
      const site = url.searchParams.get('site');
      const daysOld = url.searchParams.get('days_old');
      const results = url.searchParams.get('results');
      const remoteOnly = url.searchParams.get('remote_only');
      
      // Set search query and location
      setSearchQuery(q || searchParams.search || '');
      setLocation(loc || searchParams.location || '');
      
      // Initialize filters from URL or context or defaults
      setFilters({
        site: site || searchParams.site || 'linkedin',
        results: results ? parseInt(results) : (searchParams.results || 10),
      });
      
      // Update context search params to match what we've set locally
      setSearchParams({
        search: q || searchParams.search || '',
        location: loc || searchParams.location || '',
        site: site || searchParams.site || 'linkedin',
        results: results ? parseInt(results) : (searchParams.results || 10),
      });
    }
  }, [searchParams, setSearchParams]); // Added dependencies to fix React hooks exhaustive deps warning

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Create search parameters object
    const newSearchParams = {
      search: searchQuery,
      location: location,
      site: filters.site,
      results: filters.results
    };
    
    // Update context
    setSearchParams(newSearchParams);
    
    // Build URL query parameters
    const queryParams = new URLSearchParams();
    
    if (searchQuery) queryParams.append('q', searchQuery);
    if (location) queryParams.append('location', location);
    
    // Add filters to query params
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined) queryParams.append(key, value.toString());
    });
    
    // Update URL
    navigate(`/search?${queryParams.toString()}`);
    
    // Show loading state and clear previous errors
    setIsLoading(true);
    setError(null);
    
    try {
      // Call the API with the search parameters
      const response = await searchJobs(newSearchParams);
      
      if (response.success) {
        // Update the global context with the results
        setSearchResults(response.data);
      } else {
        setError(response.message || 'Failed to fetch jobs');
      }
    } catch (err) {
      console.error('Error fetching jobs:', err);
      setError('Failed to fetch jobs. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setLocation('');
    setFilters({
      site: 'linkedin',
      results: 10,
    });
  };
  
  const updateFilter = useCallback((key: string, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  }, []);
  
  // Close filters when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (filtersRef.current && !filtersRef.current.contains(event.target as Node)) {
        setShowFilters(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className="relative z-10 max-w-5xl mx-auto">
      <form 
        onSubmit={handleSubmit}
        className={`premium-glass rounded-xl flex flex-col md:flex-row items-center transition-all duration-300 ${
          isFocused ? 'shadow-premium border border-white/20' : 'border border-white/10'
        } backdrop-blur-xl overflow-hidden`}
      >
        <div className="relative flex items-center flex-1 w-full px-4 md:px-6 py-4 border-b md:border-b-0 md:border-r border-white/10 group">
          <Briefcase className="h-5 w-5 text-silver-500 group-focus-within:text-navy-400 transition-colors" />
          <input
            type="text"
            placeholder="Job title, keywords, or company"
            className="w-full bg-transparent border-none focus:outline-none focus:ring-0 text-white px-3 py-1.5 placeholder-silver-600 text-base"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            style={{ backgroundColor: 'transparent', WebkitBackgroundColor: 'transparent' }}
          />
          {searchQuery && (
            <button 
              type="button" 
              onClick={() => setSearchQuery('')}
              className="text-silver-500 hover:text-white transition-colors p-1 rounded-full hover:bg-white/10"
              aria-label="Clear job title"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        
        <div className="relative flex items-center flex-1 w-full px-4 md:px-6 py-4 group">
          <MapPin className="h-5 w-5 text-silver-500 group-focus-within:text-navy-400 transition-colors" />
          <input
            type="text"
            placeholder="Location or Remote"
            className="w-full bg-transparent border-none focus:outline-none focus:ring-0 text-white px-3 py-1.5 placeholder-silver-600 text-base"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            style={{ backgroundColor: 'transparent', WebkitBackgroundColor: 'transparent' }}
          />
          {location && (
            <button 
              type="button" 
              onClick={() => setLocation('')}
              className="text-silver-500 hover:text-white transition-colors p-1 rounded-full hover:bg-white/10"
              aria-label="Clear location"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        
        <div className="flex p-4 md:p-3 gap-2 w-full md:w-auto">
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            className="premium-button-outline rounded-lg px-3 md:px-4 py-2.5 flex items-center gap-2 hover:bg-white/5 transition-colors"
            aria-expanded={showFilters}
            aria-controls="search-filters"
          >
            <Filter className="h-4 w-4" />
            <span className="hidden sm:inline text-sm font-medium">Filters</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
          </button>
          
          <button
            type="submit"
            className="premium-button rounded-lg px-4 md:px-6 py-2.5 font-medium flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-navy-500 to-purple-600 hover:from-navy-600 hover:to-purple-700 transition-all duration-300"
          >
            <Search className="h-4 w-4" />
            <span className="hidden sm:inline text-sm">Search</span>
          </button>
        </div>
      </form>
      
      {/* Advanced Filters Panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.2 }}
            className="absolute z-20 mt-2 w-full"
            ref={filtersRef}
          >
            <div className="premium-glass rounded-xl p-5 shadow-premium border border-white/10 backdrop-blur-md">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-silver-300">Job Board</label>
                  <select 
                    className="w-full premium-glass text-white rounded-lg px-3 py-2.5 border border-white/10 focus:border-navy-500 focus:ring-1 focus:ring-navy-500 focus:outline-none appearance-none"
                    value={filters.site}
                    onChange={(e) => updateFilter('site', e.target.value)}
                    style={{ backgroundColor: 'transparent', backgroundImage: "url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik02IDFMMS41IDZMMTAuNSA2TDYgMVoiIGZpbGw9IndoaXRlIiBzdHJva2U9IndoaXRlIi8+Cjwvc3ZnPgo=')", backgroundRepeat: "no-repeat", backgroundPosition: "right 0.75rem center", paddingRight: "2.5rem" }}
                  >
                    <option value="linkedin">LinkedIn</option>
                    <option value="indeed">Indeed</option>
                    <option value="glassdoor">Glassdoor</option>
                    <option value="ziprecruiter">ZipRecruiter</option>
                  </select>
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium text-silver-300">Results Per Page</label>
                  <select 
                    className="w-full premium-glass text-white rounded-lg px-3 py-2.5 border border-white/10 focus:border-navy-500 focus:ring-1 focus:ring-navy-500 focus:outline-none appearance-none"
                    value={filters.results}
                    onChange={(e) => updateFilter('results', parseInt(e.target.value))}
                    style={{ backgroundColor: 'transparent', backgroundImage: "url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik02IDFMMS41IDZMMTAuNSA2TDYgMVoiIGZpbGw9IndoaXRlIiBzdHJva2U9IndoaXRlIi8+Cjwvc3ZnPgo=')", backgroundRepeat: "no-repeat", backgroundPosition: "right 0.75rem center", paddingRight: "2.5rem" }}
                  >
                    <option value="5">5 results</option>
                    <option value="10">10 results</option>
                    <option value="20">20 results</option>
                    <option value="50">50 results</option>
                  </select>
                </div>
              </div>
              
              <div className="flex justify-between mt-5 pt-4 border-t border-white/10">
                <button
                  type="button"
                  onClick={clearSearch}
                  className="text-silver-400 text-sm hover:text-white transition-colors flex items-center gap-1 hover:underline"
                >
                  <X className="h-3.5 w-3.5" />
                  Clear all filters
                </button>
                <button
                  type="button"
                  onClick={() => setShowFilters(false)}
                  className="premium-button rounded-lg px-4 py-2 text-sm font-medium bg-gradient-to-r from-navy-500 to-purple-600 hover:from-navy-600 hover:to-purple-700 transition-all duration-300"
                >
                  Apply Filters
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchBar;
