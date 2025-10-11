
import { Link } from 'react-router-dom';
import { Zap, Activity, CheckCircle2 } from 'lucide-react';

const HowItWorksSection = () => {
  return (
    <section className="py-24 bg-navy-800/60 backdrop-blur-sm border-y border-white/5 relative overflow-hidden">
      {/* Enhanced background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-64 h-64 rounded-full bg-electric-500/5 blur-3xl"></div>
        <div className="absolute bottom-1/4 left-1/4 w-64 h-64 rounded-full bg-purple-500/5 blur-3xl"></div>
      </div>
      
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center justify-center mb-4">
            <div className="h-px w-12 bg-gradient-to-r from-transparent to-electric-400/50"></div>
            <h2 className="mx-4 text-sm uppercase tracking-wider text-electric-400 font-semibold">Process</h2>
            <div className="h-px w-12 bg-gradient-to-l from-transparent to-electric-400/50"></div>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            How <span className="text-gradient bg-gradient-to-r from-electric-400 to-purple-500 bg-clip-text text-transparent">CareerSync</span> Works
          </h2>
          <p className="text-gray-300 text-lg leading-relaxed">
            Our platform simplifies your job search with powerful automation and intelligent features.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 mb-16 relative">
          {/* Connection lines */}
          <div className="hidden md:block absolute top-20 left-1/3 right-1/3 h-1 bg-gradient-to-r from-electric-500 to-purple-500 z-0"></div>
          
          <div className="glass-card p-8 rounded-xl text-center relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow hover:-translate-y-2 transition-all duration-300 z-10">
            <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-electric-500 to-electric-600 w-12 h-12 rounded-full flex items-center justify-center text-xl font-semibold shadow-lg">
              1
            </div>
            <div className="mb-6 h-16 flex items-center justify-center">
              <Zap className="h-14 w-14 text-electric-400" />
            </div>
            <h3 className="text-xl font-semibold mb-4">Create Your Profile</h3>
            <p className="text-gray-300 leading-relaxed">
              Upload your resume or fill out your profile details. Our AI will analyze your skills and experience.
            </p>
          </div>

          <div className="glass-card p-8 rounded-xl text-center relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow hover:-translate-y-2 transition-all duration-300 z-10">
            <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-purple-500 to-purple-600 w-12 h-12 rounded-full flex items-center justify-center text-xl font-semibold shadow-lg">
              2
            </div>
            <div className="mb-6 h-16 flex items-center justify-center">
              <Activity className="h-14 w-14 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold mb-4">Discover Opportunities</h3>
            <p className="text-gray-300 leading-relaxed">
              Browse personalized job recommendations or search across multiple platforms with advanced filters.
            </p>
          </div>

          <div className="glass-card p-8 rounded-xl text-center relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow hover:-translate-y-2 transition-all duration-300 z-10">
            <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-electric-500 to-purple-500 w-12 h-12 rounded-full flex items-center justify-center text-xl font-semibold shadow-lg">
              3
            </div>
            <div className="mb-6 h-16 flex items-center justify-center">
              <div className="bg-gradient-to-r from-electric-400 to-purple-400 bg-clip-text">
                <CheckCircle2 className="h-14 w-14 text-transparent stroke-current" />
              </div>
            </div>
            <h3 className="text-xl font-semibold mb-4">Apply & Track</h3>
            <p className="text-gray-300 leading-relaxed">
              Apply directly or save jobs for later. Track your application status and receive timely updates.
            </p>
          </div>
        </div>

        <div className="text-center">
          <Link to="/auth">
            <button className="bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white px-8 py-4 rounded-lg shadow-glow transition-all duration-300 font-medium text-lg transform hover:-translate-y-1">
              Get Started Now
            </button>
          </Link>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;
