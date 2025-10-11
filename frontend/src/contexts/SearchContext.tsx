import React, { createContext, useContext, useState, useCallback } from 'react';

// Define the structure of the search parameters
export interface SearchParams {
  search?: string;
  location?: string;
  site?: string;
  days_old?: number;
  results?: number;
  remote_only?: boolean;
}

// Define the structure of a job
export interface Job {
  job_id: string;
  job_title: string;
  company_name: string;
  company_logo?: string;
  company_url?: string;
  location: string;
  remote_work?: string;
  job_level?: string;
  job_description?: string;
  date_posted: string;
  days_ago?: number;
  job_source: string;
  job_url: string;
  listing_type?: string;
  skills_required?: string[];
  salary_range?: {
    currency: string;
    min_amount: number;
    max_amount: number;
  };
  experience_range?: {
    min_years: number;
    max_years: number;
  };
}

// Define the structure of search results
export interface SearchResults {
  jobs: Job[];
  query: SearchParams;
  results_count: number;
}

// Define the search context
interface SearchContextType {
  searchParams: SearchParams;
  searchResults: SearchResults | null;
  isLoading: boolean;
  error: string | null;
  setSearchParams: (params: SearchParams) => void;
  setSearchResults: (results: SearchResults | null) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearResults: () => void;
}

// Create the context
const SearchContext = createContext<SearchContextType | undefined>(undefined);

// Create the provider component
export const SearchProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [searchParams, setSearchParamsState] = useState<SearchParams>({});
  const [searchResults, setSearchResultsState] = useState<SearchResults | null>(null);
  const [isLoading, setIsLoadingState] = useState<boolean>(false);
  const [error, setErrorState] = useState<string | null>(null);

  // Use useCallback to memoize functions, preventing unnecessary renders
  const setSearchParams = useCallback((params: SearchParams) => {
    setSearchParamsState(params);
  }, []);

  const setSearchResults = useCallback((results: SearchResults | null) => {
    setSearchResultsState(results);
  }, []);

  const setIsLoading = useCallback((loading: boolean) => {
    setIsLoadingState(loading);
  }, []);

  const setError = useCallback((err: string | null) => {
    setErrorState(err);
  }, []);

  const clearResults = useCallback(() => {
    setSearchResultsState(null);
    setErrorState(null);
  }, []);

  return (
    <SearchContext.Provider value={{
      searchParams,
      searchResults,
      isLoading,
      error,
      setSearchParams,
      setSearchResults,
      setIsLoading,
      setError,
      clearResults,
    }}>
      {children}
    </SearchContext.Provider>
  );
};

// Create a hook to use the search context
export const useSearch = () => {
  const context = useContext(SearchContext);
  if (context === undefined) {
    throw new Error('useSearch must be used within a SearchProvider');
  }
  return context;
}; 