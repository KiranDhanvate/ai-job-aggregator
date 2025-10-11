
import { useState } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { 
  BarChart2, 
  PieChart, 
  Calendar, 
  Clock, 
  Plus, 
  ChevronRight, 
  Clock3, 
  TrendingUp,
  CheckCircle,
  X,
  AlertCircle,
  MessageCircle,
  Mail
} from 'lucide-react';

// Mock application data
const applications = [
  {
    id: '1',
    company: 'TechCorp Solutions',
    position: 'Senior Frontend Developer',
    status: 'interview',
    logoUrl: '',
    appliedDate: '2023-05-15',
    interviewDate: '2023-05-30',
  },
  {
    id: '2',
    company: 'InnovateTech',
    position: 'Full Stack Engineer',
    status: 'applied',
    logoUrl: '',
    appliedDate: '2023-05-18',
  },
  {
    id: '3',
    company: 'DesignWorks Agency',
    position: 'Product Designer',
    status: 'offered',
    logoUrl: '',
    appliedDate: '2023-05-05',
    offerDate: '2023-05-25',
  },
  {
    id: '4',
    company: 'CloudScale Systems',
    position: 'DevOps Engineer',
    status: 'rejected',
    logoUrl: '',
    appliedDate: '2023-04-28',
    rejectedDate: '2023-05-20',
  },
  {
    id: '5',
    company: 'AnalyticsPro',
    position: 'Data Scientist',
    status: 'applied',
    logoUrl: '',
    appliedDate: '2023-05-22',
  },
];

// Mock recommended jobs
const recommendedJobs = [
  {
    id: '1',
    title: 'Senior UI Engineer',
    company: 'DesignTech Solutions',
    location: 'San Francisco, CA',
    matchScore: 92,
    logoUrl: '',
    postedAt: '2023-05-25',
  },
  {
    id: '2',
    title: 'Frontend Architect',
    company: 'WebScape',
    location: 'Remote',
    matchScore: 88,
    logoUrl: '',
    postedAt: '2023-05-24',
  },
  {
    id: '3',
    title: 'Lead React Developer',
    company: 'AppWorks',
    location: 'Austin, TX',
    matchScore: 85,
    logoUrl: '',
    postedAt: '2023-05-26',
  },
];

// Mock upcoming events
const upcomingEvents = [
  {
    id: '1',
    title: 'Interview with TechCorp Solutions',
    time: '2023-05-30T14:00:00',
    type: 'interview',
  },
  {
    id: '2',
    title: 'Follow-up with DesignWorks Agency',
    time: '2023-05-29T11:00:00',
    type: 'follow-up',
  },
  {
    id: '3',
    title: 'Technical Assessment for WebScape',
    time: '2023-05-31T10:00:00',
    type: 'assessment',
  },
];

// Mock in-demand skills
const inDemandSkills = [
  { name: 'React', growth: 24, level: 'expert' },
  { name: 'TypeScript', growth: 32, level: 'intermediate' },
  { name: 'Node.js', growth: 18, level: 'intermediate' },
  { name: 'GraphQL', growth: 41, level: 'beginner' },
  { name: 'AWS', growth: 27, level: 'beginner' },
];

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');

  // Helper functions
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    }).format(date);
  };

  const formatTime = (dateTimeString: string) => {
    const date = new Date(dateTimeString);
    return new Intl.DateTimeFormat('en-US', { 
      hour: 'numeric', 
      minute: '2-digit', 
      hour12: true 
    }).format(date);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'applied':
        return 'bg-blue-500/20 text-blue-400';
      case 'interview':
        return 'bg-purple-500/20 text-purple-400';
      case 'offered':
        return 'bg-green-500/20 text-green-400';
      case 'rejected':
        return 'bg-red-500/20 text-red-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'applied':
        return <Mail className="h-4 w-4" />;
      case 'interview':
        return <MessageCircle className="h-4 w-4" />;
      case 'offered':
        return <CheckCircle className="h-4 w-4" />;
      case 'rejected':
        return <X className="h-4 w-4" />;
      default:
        return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'interview':
        return <MessageCircle className="h-5 w-5 text-purple-400" />;
      case 'follow-up':
        return <Mail className="h-5 w-5 text-blue-400" />;
      case 'assessment':
        return <AlertCircle className="h-5 w-5 text-orange-400" />;
      default:
        return <Calendar className="h-5 w-5 text-gray-400" />;
    }
  };

  const getSkillLevelColor = (level: string) => {
    switch (level) {
      case 'expert':
        return 'bg-green-500';
      case 'intermediate':
        return 'bg-blue-500';
      case 'beginner':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-navy-900 text-white">
      <Navbar />
      
      <main className="pt-28 pb-20">
        <div className="container mx-auto px-4 md:px-6">
          {/* Dashboard Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
            <p className="text-gray-300">
              Track your applications, upcoming interviews, and career growth
            </p>
          </div>
          
          {/* Dashboard Tabs */}
          <div className="flex border-b border-white/10 mb-8 overflow-x-auto no-scrollbar">
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeTab === 'overview'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeTab === 'applications'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('applications')}
            >
              Applications
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeTab === 'saved'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('saved')}
            >
              Saved Jobs
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeTab === 'recommendations'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('recommendations')}
            >
              Recommendations
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeTab === 'analytics'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveTab('analytics')}
            >
              Analytics
            </button>
          </div>
          
          {/* Dashboard Content */}
          {activeTab === 'overview' && (
            <div className="space-y-8">
              {/* Statistics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="glass-card rounded-xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-gray-400 mb-1 text-sm">Total Applications</p>
                      <h3 className="text-3xl font-bold">{applications.length}</h3>
                    </div>
                    <div className="h-10 w-10 rounded-lg bg-electric-500/20 flex items-center justify-center">
                      <BarChart2 className="h-5 w-5 text-electric-400" />
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <span className="text-green-400 text-sm flex items-center">
                      <TrendingUp className="h-4 w-4 mr-1" />
                      24% more than last month
                    </span>
                  </div>
                </div>
                
                <div className="glass-card rounded-xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-gray-400 mb-1 text-sm">Interviews Scheduled</p>
                      <h3 className="text-3xl font-bold">2</h3>
                    </div>
                    <div className="h-10 w-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                      <Calendar className="h-5 w-5 text-purple-400" />
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <span className="text-gray-400 text-sm flex items-center">
                      Next interview in 2 days
                    </span>
                  </div>
                </div>
                
                <div className="glass-card rounded-xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-gray-400 mb-1 text-sm">Offers Received</p>
                      <h3 className="text-3xl font-bold">1</h3>
                    </div>
                    <div className="h-10 w-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                      <CheckCircle className="h-5 w-5 text-green-400" />
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <span className="text-green-400 text-sm flex items-center">
                      <TrendingUp className="h-4 w-4 mr-1" />
                      1 new offer this month
                    </span>
                  </div>
                </div>
                
                <div className="glass-card rounded-xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-gray-400 mb-1 text-sm">Response Rate</p>
                      <h3 className="text-3xl font-bold">45%</h3>
                    </div>
                    <div className="h-10 w-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                      <PieChart className="h-5 w-5 text-blue-400" />
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <span className="text-green-400 text-sm flex items-center">
                      <TrendingUp className="h-4 w-4 mr-1" />
                      12% higher than average
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Recent Applications */}
              <div className="glass rounded-xl p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold">Recent Applications</h2>
                  <button className="text-electric-400 text-sm flex items-center hover:text-electric-300 transition-colors">
                    View All
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </button>
                </div>
                
                <div className="space-y-4">
                  {applications.slice(0, 3).map((app) => (
                    <div key={app.id} className="flex items-center p-3 rounded-lg hover:bg-white/5 transition-colors">
                      <div className="w-10 h-10 rounded-md bg-white/10 mr-4 flex items-center justify-center">
                        {app.logoUrl ? (
                          <img 
                            src={app.logoUrl} 
                            alt={`${app.company} logo`} 
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <span className="text-lg font-semibold text-gray-400">
                            {app.company.charAt(0)}
                          </span>
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <h3 className="text-base font-medium truncate">{app.position}</h3>
                        <p className="text-sm text-gray-400 truncate">{app.company}</p>
                      </div>
                      
                      <div className="ml-4">
                        <span 
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(app.status)}`}
                        >
                          {getStatusIcon(app.status)}
                          <span className="ml-1 capitalize">{app.status}</span>
                        </span>
                      </div>
                      
                      <div className="ml-6 text-right">
                        <p className="text-sm text-gray-400">Applied</p>
                        <p className="text-sm">{formatDate(app.appliedDate)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Two-column layout */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Upcoming Events */}
                <div className="glass rounded-xl p-6 lg:col-span-1">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Upcoming Events</h2>
                    <button className="text-purple-400 text-sm flex items-center hover:text-purple-300 transition-colors">
                      <Plus className="h-4 w-4 mr-1" />
                      Add
                    </button>
                  </div>
                  
                  <div className="space-y-5">
                    {upcomingEvents.map((event) => (
                      <div key={event.id} className="flex items-start">
                        <div className="flex-shrink-0 mt-1">
                          {getEventIcon(event.type)}
                        </div>
                        <div className="ml-4">
                          <h3 className="text-sm font-medium">{event.title}</h3>
                          <div className="flex items-center mt-1 text-xs text-gray-400">
                            <Calendar className="h-3.5 w-3.5 mr-1" />
                            {formatDate(event.time)}
                            <Clock3 className="h-3.5 w-3.5 ml-3 mr-1" />
                            {formatTime(event.time)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Recommended Jobs */}
                <div className="glass rounded-xl p-6 lg:col-span-2">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Recommended for You</h2>
                    <button className="text-electric-400 text-sm flex items-center hover:text-electric-300 transition-colors">
                      View All
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </button>
                  </div>
                  
                  <div className="space-y-4">
                    {recommendedJobs.map((job) => (
                      <div key={job.id} className="p-4 rounded-lg border border-white/10 hover:bg-white/5 transition-colors">
                        <div className="flex items-center">
                          <div className="w-10 h-10 rounded-md bg-white/10 mr-4 flex items-center justify-center">
                            {job.logoUrl ? (
                              <img 
                                src={job.logoUrl} 
                                alt={`${job.company} logo`} 
                                className="w-full h-full object-contain"
                              />
                            ) : (
                              <span className="text-lg font-semibold text-gray-400">
                                {job.company.charAt(0)}
                              </span>
                            )}
                          </div>
                          
                          <div className="flex-1">
                            <h3 className="text-base font-medium">{job.title}</h3>
                            <p className="text-sm text-gray-400">{job.company} • {job.location}</p>
                          </div>
                          
                          <div className="ml-4 text-right">
                            <div className="text-sm font-medium mb-1 text-electric-400">
                              {job.matchScore}% Match
                            </div>
                            <p className="text-xs text-gray-400">
                              Posted {formatDate(job.postedAt)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Skills in Demand */}
              <div className="glass rounded-xl p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold">Skills in Demand</h2>
                  <button className="text-electric-400 text-sm flex items-center hover:text-electric-300 transition-colors">
                    View All
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {inDemandSkills.map((skill) => (
                    <div key={skill.name} className="flex items-center p-3 rounded-lg border border-white/10">
                      <div className={`w-2 h-8 ${getSkillLevelColor(skill.level)} rounded-full mr-3`}></div>
                      <div className="flex-1">
                        <h3 className="text-base font-medium">{skill.name}</h3>
                        <div className="flex items-center mt-1">
                          <div className="flex-1 bg-white/10 rounded-full h-1.5">
                            <div 
                              className="bg-gradient-to-r from-electric-500 to-purple-500 h-1.5 rounded-full"
                              style={{ width: `${skill.level === 'expert' ? 90 : skill.level === 'intermediate' ? 60 : 30}%` }}
                            ></div>
                          </div>
                          <span className="ml-3 text-xs text-gray-400 whitespace-nowrap">{skill.level}</span>
                        </div>
                      </div>
                      <div className="ml-4 px-2 py-1 rounded bg-green-500/10 text-green-400 text-xs flex items-center">
                        <TrendingUp className="h-3 w-3 mr-1" />
                        {skill.growth}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          
          {/* Placeholder for other tabs */}
          {activeTab !== 'overview' && (
            <div className="glass rounded-xl p-12 text-center">
              <h3 className="text-2xl font-semibold mb-4">
                {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Dashboard
              </h3>
              <p className="text-gray-300 mb-6">
                This feature is coming soon. We're working hard to bring you the best job search experience.
              </p>
              <button 
                onClick={() => setActiveTab('overview')}
                className="bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white px-6 py-3 rounded-lg shadow-glow transition-all duration-300 font-medium"
              >
                Back to Overview
              </button>
            </div>
          )}
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Dashboard;
