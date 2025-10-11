
import { Link } from 'react-router-dom';
import { Briefcase, ArrowRight } from 'lucide-react';
import JobCard from '../JobCard';

// Mock data for jobs
const featuredJobs = [
  {
    id: '1',
    title: 'Senior Frontend Developer',
    company: 'TechCorp Solutions',
    location: 'San Francisco, CA',
    salary: '$120k - $150k',
    logo_url: '',
    description: 'We are looking for an experienced Frontend Developer to join our team and help build amazing web applications using React, TypeScript, and more.',
    source: 'LinkedIn',
    source_url: '#',
    posted_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
    is_remote: false,
    experience_level: 'senior',
    matchPercentage: 95,
  },
  {
    id: '2',
    title: 'Full Stack Engineer',
    company: 'InnovateTech',
    location: 'New York, NY',
    salary: '$130k - $160k',
    logo_url: '',
    description: 'Join our dynamic team building next-generation software with Node.js, React, and AWS. Focus on scalable architecture and performance.',
    source: 'Indeed',
    source_url: '#',
    posted_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
    is_remote: false,
    experience_level: 'mid-senior',
    matchPercentage: 88,
  },
  {
    id: '3',
    title: 'Product Designer',
    company: 'DesignWorks Agency',
    location: 'Austin, TX',
    salary: '$90k - $120k',
    logo_url: '',
    description: 'Creative and detail-oriented Product Designer needed for our growing UX team. Work on exciting projects for major tech clients.',
    source: 'Glassdoor',
    source_url: '#',
    posted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), // 3 days ago
    is_remote: true,
    experience_level: 'mid-level',
    matchPercentage: 92,
  },
  {
    id: '4',
    title: 'DevOps Engineer',
    company: 'CloudScale Systems',
    location: 'Remote',
    salary: '$115k - $145k',
    logo_url: '',
    description: 'Help us build and maintain our cloud infrastructure using Docker, Kubernetes, and AWS. CI/CD pipeline expertise is a must.',
    source: 'Monster',
    source_url: '#',
    posted_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
    is_remote: true,
    experience_level: 'senior',
    matchPercentage: 84,
  },
];

const FeaturedJobsSection = () => {
  return (
    <section className="py-24 relative">
      <div className="absolute top-0 right-0 w-1/3 h-2/3 bg-gradient-radial from-electric-500/5 to-transparent opacity-50"></div>
      <div className="absolute bottom-0 left-0 w-1/3 h-1/2 bg-gradient-radial from-purple-500/5 to-transparent opacity-40"></div>
      
      <div className="container mx-auto px-4 md:px-6 relative">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center mb-4">
            <div className="h-px w-12 bg-gradient-to-r from-transparent to-electric-400/50"></div>
            <h2 className="mx-4 text-sm uppercase tracking-wider text-electric-400 font-semibold">Career Opportunities</h2>
            <div className="h-px w-12 bg-gradient-to-l from-transparent to-electric-400/50"></div>
          </div>
        </div>
        
        <div className="flex justify-between items-end mb-12">
          <div>
            <h2 className="text-3xl font-bold mb-3 flex items-center">
              <Briefcase className="mr-3 h-7 w-7 text-electric-400" />
              Featured Jobs
            </h2>
            <p className="text-gray-300">Curated opportunities matching popular searches</p>
          </div>
          <Link 
            to="/search" 
            className="text-electric-400 font-medium flex items-center hover:text-electric-300 transition-colors group"
          >
            View all jobs
            <ArrowRight className="h-5 w-5 ml-1 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {featuredJobs.map((job, index) => (
            <div key={job.id} className="animate-fade-in" style={{ animationDelay: `${index * 100 + 100}ms` }}>
              <JobCard job={job} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturedJobsSection;
