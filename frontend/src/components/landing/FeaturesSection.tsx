
import { Link } from 'react-router-dom';
import { Search, Brain, BarChart2, Bell, ChevronRight, LucideProps } from 'lucide-react';
import FeatureCard from '../FeatureCard';

// Custom icons for feature cards
const AggregationIcon = (props: LucideProps) => (
  <Search {...props} className="h-6 w-6 text-white" />
);

const AIIcon = (props: LucideProps) => (
  <Brain {...props} className="h-6 w-6 text-white" />
);

const TrackerIcon = (props: LucideProps) => (
  <BarChart2 {...props} className="h-6 w-6 text-white" />
);

const AlertsIcon = (props: LucideProps) => (
  <Bell {...props} className="h-6 w-6 text-white" />
);

const FeaturesSection = () => {
  return (
    <section id="features-section" className="py-24 relative">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-1/2 bg-gradient-radial from-electric-500/5 to-transparent opacity-30"></div>
        <div className="absolute bottom-0 left-0 w-1/2 h-1/2 bg-gradient-radial from-purple-500/5 to-transparent opacity-30"></div>
      </div>
      
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center justify-center mb-4">
            <div className="h-px w-12 bg-gradient-to-r from-transparent to-electric-400/50"></div>
            <h2 className="mx-4 text-sm uppercase tracking-wider text-electric-400 font-semibold">Features</h2>
            <div className="h-px w-12 bg-gradient-to-l from-transparent to-electric-400/50"></div>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Modern Tools for the Modern <span className="text-gradient bg-gradient-to-r from-electric-400 to-purple-500 bg-clip-text text-transparent">Job Seeker</span>
          </h2>
          <p className="text-gray-300 text-lg leading-relaxed">
            CareerSync combines cutting-edge technology with intuitive design to streamline 
            every aspect of your job search journey.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          <FeatureCard
            icon={<AggregationIcon />}
            title="Job Aggregation"
            description="Search thousands of jobs from hundreds of sources in one place, saving you time and effort."
            gradient="blue"
            delay={100}
            link="/features/job-aggregation"
          />
          <FeatureCard
            icon={<AIIcon />}
            title="AI Recommendations"
            description="Get personalized job recommendations based on your skills, experience, and preferences."
            gradient="purple"
            delay={200}
            link="/features/ai-recommendations"
          />
          <FeatureCard
            icon={<TrackerIcon />}
            title="Application Tracker"
            description="Keep track of all your applications in one place with our intuitive kanban board."
            gradient="mixed"
            delay={300}
            link="/features/application-tracker"
          />
          <FeatureCard
            icon={<AlertsIcon />}
            title="Real-time Alerts"
            description="Receive notifications about new job postings, application updates, and interview reminders."
            gradient="blue"
            delay={400}
            link="/features/real-time-alerts"
          />
        </div>
        
        <div className="text-center">
          <Link 
            to="/search" 
            className="inline-flex items-center text-electric-400 font-medium hover:text-electric-300 transition-colors hover:underline group"
          >
            Explore all features
            <ChevronRight className="h-5 w-5 ml-1 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
