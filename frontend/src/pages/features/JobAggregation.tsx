
import { Search, Globe, Filter, Clock, Database, List, BarChart2 } from 'lucide-react';
import FeatureLayout from '@/components/FeatureLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const JobAggregation = () => {
  return (
    <FeatureLayout
      title="Job Aggregation"
      description="Search thousands of jobs from hundreds of sources in one place, saving you time and effort."
      icon={<Search className="h-8 w-8 text-white" />}
      gradient="blue"
    >
      <div className="max-w-4xl mx-auto">
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-6">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <Database className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Multi-Source Integration</CardTitle>
                <CardDescription className="text-gray-300">
                  We connect to over 100 job boards, company career pages, and recruitment agencies.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <Filter className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Advanced Filtering</CardTitle>
                <CardDescription className="text-gray-300">
                  Filter by location, experience level, salary range, job type, and remote options.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <Clock className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Real-time Updates</CardTitle>
                <CardDescription className="text-gray-300">
                  Get new job postings as soon as they're published with our real-time indexing.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>
        
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-6">Key Features</h2>
          <div className="space-y-6">
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <Globe className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Global Coverage</h3>
                  <p className="text-gray-300">
                    Search for jobs across different countries, regions, and cities with location-based filtering.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <List className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Duplicate Removal</h3>
                  <p className="text-gray-300">
                    Our smart algorithms identify and remove duplicate job listings so you don't waste time on the same opportunity.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <BarChart2 className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Salary Insights</h3>
                  <p className="text-gray-300">
                    Compare compensation across companies and roles with our comprehensive salary data integration.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
        
        <section className="text-center">
          <h2 className="text-2xl font-semibold mb-6">Ready to Find Your Next Opportunity?</h2>
          <div className="flex flex-wrap justify-center gap-4">
            <Button asChild className="bg-gradient-to-r from-electric-500 to-electric-600 hover:from-electric-600 hover:to-electric-700 text-white">
              <Link to="/search">Start Searching Now</Link>
            </Button>
            <Button asChild variant="outline" className="border-white/10 hover:bg-white/10">
              <Link to="/auth/register">Create an Account</Link>
            </Button>
          </div>
        </section>
      </div>
    </FeatureLayout>
  );
};

export default JobAggregation;
