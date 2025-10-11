
import { useState } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { 
  User, 
  Mail, 
  Phone, 
  MapPin, 
  Briefcase, 
  GraduationCap, 
  Award, 
  Globe, 
  Plus,
  Edit,
  Save,
  X,
  FileText,
  Upload,
  Check,
  ChevronDown,
  ChevronRight,
  AlertCircle
} from 'lucide-react';

const Profile = () => {
  const [editMode, setEditMode] = useState(false);
  const [activeSection, setActiveSection] = useState('profile');
  
  // User profile mock data
  const [profile, setProfile] = useState({
    name: 'Alex Johnson',
    title: 'Senior Frontend Developer',
    email: 'alex.johnson@example.com',
    phone: '+1 (555) 123-4567',
    location: 'San Francisco, CA',
    about: 'Experienced frontend developer with 7+ years of experience building responsive web applications using React, TypeScript, and modern frontend technologies. Passionate about creating intuitive user experiences and clean, maintainable code.',
    experience: [
      {
        id: '1',
        title: 'Senior Frontend Developer',
        company: 'TechCorp Solutions',
        location: 'San Francisco, CA',
        startDate: '2021-05',
        endDate: null,
        description: 'Leading frontend development for enterprise SaaS platform. Implemented component library, improved performance, and mentored junior developers.'
      },
      {
        id: '2',
        title: 'Frontend Developer',
        company: 'WebScape',
        location: 'Remote',
        startDate: '2018-03',
        endDate: '2021-04',
        description: 'Developed responsive web applications using React, Redux, and GraphQL. Collaborated with design team to implement UI/UX improvements.'
      }
    ],
    education: [
      {
        id: '1',
        degree: 'Bachelor of Science in Computer Science',
        institution: 'University of California, Berkeley',
        location: 'Berkeley, CA',
        startDate: '2014-09',
        endDate: '2018-05'
      }
    ],
    skills: [
      { name: 'React', level: 'expert' },
      { name: 'TypeScript', level: 'expert' },
      { name: 'JavaScript', level: 'expert' },
      { name: 'HTML/CSS', level: 'expert' },
      { name: 'GraphQL', level: 'intermediate' },
      { name: 'Node.js', level: 'intermediate' },
      { name: 'Redux', level: 'expert' },
      { name: 'Webpack', level: 'intermediate' },
      { name: 'Tailwind CSS', level: 'expert' },
      { name: 'Jest', level: 'intermediate' },
      { name: 'CI/CD', level: 'intermediate' },
      { name: 'AWS', level: 'beginner' }
    ],
    jobPreferences: {
      roles: ['Frontend Developer', 'UI Engineer', 'Full Stack Developer'],
      locations: ['San Francisco, CA', 'Remote'],
      salary: '$120,000 - $150,000',
      workType: 'Full-time'
    },
    resumeUrl: null
  });

  // Helper functions
  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Present';
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', 
      year: 'numeric' 
    }).format(date);
  };

  const handleSaveProfile = () => {
    // In a real app, this would save to the backend
    setEditMode(false);
  };

  const handleResumeUpload = () => {
    // In a real app, this would handle file upload
    setProfile({
      ...profile,
      resumeUrl: '/resume.pdf'
    });
  };

  const getSkillLevelColor = (level: string) => {
    switch (level) {
      case 'expert':
        return 'bg-electric-500';
      case 'intermediate':
        return 'bg-purple-500';
      case 'beginner':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getSkillLevelWidth = (level: string) => {
    switch (level) {
      case 'expert':
        return 'w-full';
      case 'intermediate':
        return 'w-2/3';
      case 'beginner':
        return 'w-1/3';
      default:
        return 'w-0';
    }
  };

  return (
    <div className="min-h-screen bg-navy-900 text-white">
      <Navbar />
      
      <main className="pt-28 pb-20">
        <div className="container mx-auto px-4 md:px-6">
          {/* Profile Header */}
          <div className="glass-card rounded-xl p-6 md:p-8 mb-8 relative">
            {/* Edit button */}
            <button
              onClick={() => setEditMode(!editMode)}
              className="absolute top-6 right-6 p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
            >
              {editMode ? (
                <Save className="h-5 w-5 text-electric-400" />
              ) : (
                <Edit className="h-5 w-5 text-gray-300" />
              )}
            </button>
            
            <div className="flex flex-col md:flex-row md:items-center gap-6">
              {/* Profile Image */}
              <div className="relative w-24 h-24 md:w-32 md:h-32 rounded-full bg-gradient-to-br from-electric-500 to-purple-500 flex items-center justify-center text-3xl font-bold">
                {profile.name.charAt(0)}
                {editMode && (
                  <button className="absolute -right-2 -bottom-2 p-1.5 bg-navy-800 rounded-full border border-white/10">
                    <Edit className="h-4 w-4 text-gray-300" />
                  </button>
                )}
              </div>
              
              {/* Profile Info */}
              <div className="flex-1">
                {editMode ? (
                  <input
                    type="text"
                    className="bg-navy-800 border border-white/10 rounded-md px-3 py-2 text-xl font-bold w-full mb-2"
                    value={profile.name}
                    onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                  />
                ) : (
                  <h1 className="text-2xl md:text-3xl font-bold mb-2">{profile.name}</h1>
                )}
                
                {editMode ? (
                  <input
                    type="text"
                    className="bg-navy-800 border border-white/10 rounded-md px-3 py-2 text-gray-300 w-full mb-4"
                    value={profile.title}
                    onChange={(e) => setProfile({ ...profile, title: e.target.value })}
                  />
                ) : (
                  <h2 className="text-xl text-gray-300 mb-4">{profile.title}</h2>
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-center text-gray-300">
                    <Mail className="h-5 w-5 mr-2 text-electric-400" />
                    {editMode ? (
                      <input
                        type="email"
                        className="bg-navy-800 border border-white/10 rounded-md px-3 py-1 w-full"
                        value={profile.email}
                        onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                      />
                    ) : (
                      <span>{profile.email}</span>
                    )}
                  </div>
                  
                  <div className="flex items-center text-gray-300">
                    <Phone className="h-5 w-5 mr-2 text-electric-400" />
                    {editMode ? (
                      <input
                        type="text"
                        className="bg-navy-800 border border-white/10 rounded-md px-3 py-1 w-full"
                        value={profile.phone}
                        onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                      />
                    ) : (
                      <span>{profile.phone}</span>
                    )}
                  </div>
                  
                  <div className="flex items-center text-gray-300">
                    <MapPin className="h-5 w-5 mr-2 text-electric-400" />
                    {editMode ? (
                      <input
                        type="text"
                        className="bg-navy-800 border border-white/10 rounded-md px-3 py-1 w-full"
                        value={profile.location}
                        onChange={(e) => setProfile({ ...profile, location: e.target.value })}
                      />
                    ) : (
                      <span>{profile.location}</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* About section */}
            <div className="mt-6 pt-6 border-t border-white/10">
              <h3 className="text-lg font-semibold mb-3">About</h3>
              {editMode ? (
                <textarea
                  className="bg-navy-800 border border-white/10 rounded-md px-3 py-2 w-full h-32 resize-none"
                  value={profile.about}
                  onChange={(e) => setProfile({ ...profile, about: e.target.value })}
                />
              ) : (
                <p className="text-gray-300">{profile.about}</p>
              )}
            </div>
            
            {/* Save button */}
            {editMode && (
              <div className="flex justify-end mt-6">
                <button
                  onClick={() => setEditMode(false)}
                  className="px-4 py-2 border border-white/10 rounded-lg text-gray-300 hover:bg-white/5 transition-colors mr-3"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveProfile}
                  className="px-4 py-2 bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white rounded-lg transition-all duration-300"
                >
                  Save Changes
                </button>
              </div>
            )}
          </div>
          
          {/* Profile Navigation */}
          <div className="flex border-b border-white/10 mb-8 overflow-x-auto no-scrollbar">
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeSection === 'profile'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveSection('profile')}
            >
              Profile
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeSection === 'resume'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveSection('resume')}
            >
              Resume
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeSection === 'preferences'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveSection('preferences')}
            >
              Job Preferences
            </button>
            <button
              className={`pb-3 px-4 font-medium text-sm whitespace-nowrap ${
                activeSection === 'settings'
                  ? 'text-electric-400 border-b-2 border-electric-400'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setActiveSection('settings')}
            >
              Settings
            </button>
          </div>
          
          {/* Profile Content */}
          {activeSection === 'profile' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Left column */}
              <div className="lg:col-span-2 space-y-8">
                {/* Experience section */}
                <div className="glass rounded-xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Experience</h2>
                    {editMode && (
                      <button className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">
                        <Plus className="h-5 w-5 text-electric-400" />
                      </button>
                    )}
                  </div>
                  
                  <div className="space-y-6">
                    {profile.experience.map((exp) => (
                      <div key={exp.id} className="relative">
                        {editMode && (
                          <div className="absolute -right-2 -top-2 flex space-x-2">
                            <button className="p-1.5 bg-navy-800 rounded-full border border-white/10">
                              <Edit className="h-3.5 w-3.5 text-gray-300" />
                            </button>
                            <button className="p-1.5 bg-navy-800 rounded-full border border-white/10">
                              <X className="h-3.5 w-3.5 text-red-400" />
                            </button>
                          </div>
                        )}
                        
                        <div className="flex items-start">
                          <div className="h-12 w-12 rounded-lg bg-white/10 flex items-center justify-center text-lg font-semibold mr-4 flex-shrink-0">
                            {exp.company.charAt(0)}
                          </div>
                          <div>
                            <h3 className="text-lg font-medium">{exp.title}</h3>
                            <p className="text-gray-300">{exp.company} • {exp.location}</p>
                            <p className="text-sm text-gray-400 mt-1">
                              {formatDate(exp.startDate)} - {formatDate(exp.endDate)}
                            </p>
                            <p className="mt-3 text-gray-300">{exp.description}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Education section */}
                <div className="glass rounded-xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Education</h2>
                    {editMode && (
                      <button className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">
                        <Plus className="h-5 w-5 text-electric-400" />
                      </button>
                    )}
                  </div>
                  
                  <div className="space-y-6">
                    {profile.education.map((edu) => (
                      <div key={edu.id} className="relative">
                        {editMode && (
                          <div className="absolute -right-2 -top-2 flex space-x-2">
                            <button className="p-1.5 bg-navy-800 rounded-full border border-white/10">
                              <Edit className="h-3.5 w-3.5 text-gray-300" />
                            </button>
                            <button className="p-1.5 bg-navy-800 rounded-full border border-white/10">
                              <X className="h-3.5 w-3.5 text-red-400" />
                            </button>
                          </div>
                        )}
                        
                        <div className="flex items-start">
                          <div className="h-12 w-12 rounded-lg bg-white/10 flex items-center justify-center mr-4 flex-shrink-0">
                            <GraduationCap className="h-6 w-6 text-gray-300" />
                          </div>
                          <div>
                            <h3 className="text-lg font-medium">{edu.degree}</h3>
                            <p className="text-gray-300">{edu.institution} • {edu.location}</p>
                            <p className="text-sm text-gray-400 mt-1">
                              {formatDate(edu.startDate)} - {formatDate(edu.endDate)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Right column */}
              <div className="space-y-8">
                {/* Skills section */}
                <div className="glass rounded-xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Skills</h2>
                    {editMode && (
                      <button className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">
                        <Plus className="h-5 w-5 text-electric-400" />
                      </button>
                    )}
                  </div>
                  
                  <div className="space-y-4">
                    {profile.skills.map((skill) => (
                      <div key={skill.name} className="relative">
                        {editMode && (
                          <button className="absolute -right-2 -top-2 p-1.5 bg-navy-800 rounded-full border border-white/10">
                            <X className="h-3.5 w-3.5 text-red-400" />
                          </button>
                        )}
                        
                        <div className="flex items-center justify-between mb-1">
                          <h3 className="text-sm font-medium">{skill.name}</h3>
                          <span className="text-xs text-gray-400 capitalize">{skill.level}</span>
                        </div>
                        
                        <div className="w-full bg-white/10 rounded-full h-1.5">
                          <div 
                            className={`${getSkillLevelColor(skill.level)} h-1.5 rounded-full ${getSkillLevelWidth(skill.level)}`}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Job Preferences Overview */}
                <div className="glass rounded-xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Job Preferences</h2>
                    <button 
                      onClick={() => setActiveSection('preferences')}
                      className="text-electric-400 text-sm flex items-center hover:text-electric-300 transition-colors"
                    >
                      Edit
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </button>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Desired Roles</h3>
                      <div className="flex flex-wrap gap-2">
                        {profile.jobPreferences.roles.map((role) => (
                          <span key={role} className="px-2.5 py-1 rounded-full text-xs bg-electric-500/20 text-electric-400">
                            {role}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Preferred Locations</h3>
                      <div className="flex flex-wrap gap-2">
                        {profile.jobPreferences.locations.map((location) => (
                          <span key={location} className="px-2.5 py-1 rounded-full text-xs bg-purple-500/20 text-purple-400">
                            {location}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Salary Expectation</h3>
                      <p className="text-white">{profile.jobPreferences.salary}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Work Type</h3>
                      <p className="text-white">{profile.jobPreferences.workType}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Resume Section */}
          {activeSection === 'resume' && (
            <div className="glass rounded-xl p-6 md:p-8">
              <h2 className="text-xl font-semibold mb-6">Resume Management</h2>
              
              {profile.resumeUrl ? (
                <div className="mb-8">
                  <div className="bg-navy-800/70 border border-white/10 rounded-lg p-6 flex flex-col md:flex-row items-center justify-between">
                    <div className="flex items-center mb-4 md:mb-0">
                      <div className="h-12 w-12 rounded-lg bg-electric-500/20 flex items-center justify-center mr-4 flex-shrink-0">
                        <FileText className="h-6 w-6 text-electric-400" />
                      </div>
                      <div>
                        <h3 className="text-lg font-medium">resume.pdf</h3>
                        <p className="text-sm text-gray-400">Uploaded on May 24, 2023</p>
                      </div>
                    </div>
                    
                    <div className="flex space-x-3">
                      <button className="px-4 py-2 bg-navy-700 hover:bg-navy-600 text-white rounded-lg transition-colors">
                        View
                      </button>
                      <button className="px-4 py-2 bg-navy-700 hover:bg-navy-600 text-white rounded-lg transition-colors">
                        Replace
                      </button>
                      <button className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg transition-colors">
                        Delete
                      </button>
                    </div>
                  </div>
                  
                  <div className="mt-8 bg-navy-800/70 border border-white/10 rounded-lg p-6">
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="text-lg font-medium">Resume Insights</h3>
                      <span className="text-sm px-3 py-1 bg-green-500/20 text-green-400 rounded-full">
                        Good Match
                      </span>
                    </div>
                    
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="text-sm font-medium">ATS Compatibility</h4>
                          <span className="text-xs text-green-400">92%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-1.5">
                          <div className="bg-green-500 h-1.5 rounded-full w-[92%]"></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="text-sm font-medium">Keyword Optimization</h4>
                          <span className="text-xs text-yellow-400">78%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-1.5">
                          <div className="bg-yellow-500 h-1.5 rounded-full w-[78%]"></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="text-sm font-medium">Completeness</h4>
                          <span className="text-xs text-green-400">95%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-1.5">
                          <div className="bg-green-500 h-1.5 rounded-full w-[95%]"></div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-6 pt-6 border-t border-white/10">
                      <h4 className="text-sm font-medium mb-3">Recommendations</h4>
                      <ul className="space-y-2">
                        <li className="flex items-start">
                          <Check className="h-5 w-5 text-green-400 mr-2 flex-shrink-0 mt-0.5" />
                          <span className="text-sm text-gray-300">Your contact information is clear and accessible.</span>
                        </li>
                        <li className="flex items-start">
                          <Check className="h-5 w-5 text-green-400 mr-2 flex-shrink-0 mt-0.5" />
                          <span className="text-sm text-gray-300">Your work experience highlights your achievements well.</span>
                        </li>
                        <li className="flex items-start">
                          <AlertCircle className="h-5 w-5 text-yellow-400 mr-2 flex-shrink-0 mt-0.5" />
                          <span className="text-sm text-gray-300">Consider adding more industry-specific keywords to improve your match rate.</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mb-8 bg-navy-800/70 border border-white/10 border-dashed rounded-lg p-12 text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-electric-500/20 text-electric-400 mb-4">
                    <Upload className="h-8 w-8" />
                  </div>
                  <h3 className="text-xl font-medium mb-2">Upload Your Resume</h3>
                  <p className="text-gray-300 mb-6 max-w-md mx-auto">
                    Upload your resume to improve your job matches and enable one-click applications.
                  </p>
                  <button
                    onClick={handleResumeUpload}
                    className="px-6 py-3 bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white rounded-lg transition-all duration-300"
                  >
                    Upload Resume
                  </button>
                  <p className="text-xs text-gray-400 mt-4">
                    Supported formats: PDF, DOCX, RTF (Max 5MB)
                  </p>
                </div>
              )}
              
              <div className="rounded-xl p-6 border border-white/10 bg-navy-800/50">
                <h3 className="text-lg font-medium mb-4">Resume Builder</h3>
                <p className="text-gray-300 mb-6">
                  Don't have a resume? Use our AI-powered resume builder to create a professional resume 
                  based on your profile information.
                </p>
                <button className="px-6 py-3 bg-white/10 hover:bg-white/15 text-white rounded-lg transition-colors">
                  Build My Resume
                </button>
              </div>
            </div>
          )}
          
          {/* Job Preferences Section */}
          {activeSection === 'preferences' && (
            <div className="glass rounded-xl p-6 md:p-8">
              <h2 className="text-xl font-semibold mb-6">Job Preferences</h2>
              
              <form className="space-y-8">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    What roles are you interested in?
                  </label>
                  <div className="bg-navy-800 border border-white/10 rounded-lg p-4">
                    <div className="flex flex-wrap gap-2 mb-3">
                      {profile.jobPreferences.roles.map((role) => (
                        <div key={role} className="flex items-center bg-electric-500/20 text-electric-400 rounded-full pl-3 pr-1 py-1 text-sm">
                          {role}
                          <button className="ml-1 p-1 hover:bg-electric-500/30 rounded-full">
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                    <div className="flex">
                      <input
                        type="text"
                        placeholder="Add role (e.g. Frontend Developer)"
                        className="bg-navy-700 border border-white/10 rounded-l-md px-3 py-2 w-full focus:outline-none focus:ring-1 focus:ring-electric-500"
                      />
                      <button className="bg-electric-500 hover:bg-electric-600 text-white px-4 py-2 rounded-r-md transition-colors">
                        Add
                      </button>
                    </div>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Where would you like to work?
                  </label>
                  <div className="bg-navy-800 border border-white/10 rounded-lg p-4">
                    <div className="flex flex-wrap gap-2 mb-3">
                      {profile.jobPreferences.locations.map((location) => (
                        <div key={location} className="flex items-center bg-purple-500/20 text-purple-400 rounded-full pl-3 pr-1 py-1 text-sm">
                          {location}
                          <button className="ml-1 p-1 hover:bg-purple-500/30 rounded-full">
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                    <div className="flex">
                      <input
                        type="text"
                        placeholder="Add location (e.g. San Francisco, CA)"
                        className="bg-navy-700 border border-white/10 rounded-l-md px-3 py-2 w-full focus:outline-none focus:ring-1 focus:ring-purple-500"
                      />
                      <button className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-r-md transition-colors">
                        Add
                      </button>
                    </div>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Salary Expectations
                  </label>
                  <div className="flex gap-4">
                    <div className="w-1/2">
                      <label className="block text-xs text-gray-400 mb-1">Minimum</label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">$</div>
                        <input
                          type="text"
                          className="bg-navy-800 border border-white/10 rounded-md pl-8 pr-3 py-2 w-full focus:outline-none focus:ring-1 focus:ring-electric-500"
                          placeholder="Minimum"
                          value="120,000"
                        />
                      </div>
                    </div>
                    <div className="w-1/2">
                      <label className="block text-xs text-gray-400 mb-1">Maximum</label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">$</div>
                        <input
                          type="text"
                          className="bg-navy-800 border border-white/10 rounded-md pl-8 pr-3 py-2 w-full focus:outline-none focus:ring-1 focus:ring-electric-500"
                          placeholder="Maximum"
                          value="150,000"
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Work Type
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-electric-500 rounded-lg cursor-pointer">
                      <input type="radio" name="workType" className="sr-only" defaultChecked />
                      <span className="text-electric-400">Full-time</span>
                    </label>
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-white/10 rounded-lg cursor-pointer hover:border-electric-500/50 transition-colors">
                      <input type="radio" name="workType" className="sr-only" />
                      <span className="text-gray-300">Part-time</span>
                    </label>
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-white/10 rounded-lg cursor-pointer hover:border-electric-500/50 transition-colors">
                      <input type="radio" name="workType" className="sr-only" />
                      <span className="text-gray-300">Contract</span>
                    </label>
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-white/10 rounded-lg cursor-pointer hover:border-electric-500/50 transition-colors">
                      <input type="radio" name="workType" className="sr-only" />
                      <span className="text-gray-300">Internship</span>
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Remote Preferences
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-white/10 rounded-lg cursor-pointer hover:border-electric-500/50 transition-colors">
                      <input type="radio" name="remote" className="sr-only" />
                      <span className="text-gray-300">On-site only</span>
                    </label>
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-electric-500 rounded-lg cursor-pointer">
                      <input type="radio" name="remote" className="sr-only" defaultChecked />
                      <span className="text-electric-400">Hybrid</span>
                    </label>
                    <label className="flex items-center justify-center p-3 bg-navy-800 border border-white/10 rounded-lg cursor-pointer hover:border-electric-500/50 transition-colors">
                      <input type="radio" name="remote" className="sr-only" />
                      <span className="text-gray-300">Remote only</span>
                    </label>
                  </div>
                </div>
                
                <div className="pt-4 flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={() => setActiveSection('profile')}
                    className="px-4 py-2 border border-white/10 rounded-lg text-gray-300 hover:bg-white/5 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveSection('profile')}
                    className="px-4 py-2 bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white rounded-lg transition-all duration-300"
                  >
                    Save Preferences
                  </button>
                </div>
              </form>
            </div>
          )}
          
          {/* Settings Section */}
          {activeSection === 'settings' && (
            <div className="glass rounded-xl p-6 md:p-8">
              <h2 className="text-xl font-semibold mb-6">Account Settings</h2>
              
              <div className="space-y-8">
                <div className="border-b border-white/10 pb-8">
                  <h3 className="text-lg font-medium mb-4">Notification Preferences</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-base font-medium">Email Notifications</h4>
                        <p className="text-sm text-gray-400">Receive updates about new job matches, application status, and more.</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" value="" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-electric-500"></div>
                      </label>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-base font-medium">Mobile Notifications</h4>
                        <p className="text-sm text-gray-400">Receive push notifications on your mobile device.</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" value="" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-electric-500"></div>
                      </label>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-base font-medium">Weekly Digest</h4>
                        <p className="text-sm text-gray-400">Receive a weekly summary of your job search activity.</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" value="" className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-electric-500"></div>
                      </label>
                    </div>
                  </div>
                </div>
                
                <div className="border-b border-white/10 pb-8">
                  <h3 className="text-lg font-medium mb-4">Privacy Settings</h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-base font-medium">Profile Visibility</h4>
                        <p className="text-sm text-gray-400">Allow recruiters to discover your profile.</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" value="" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-electric-500"></div>
                      </label>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-base font-medium">Data Sharing</h4>
                        <p className="text-sm text-gray-400">Share anonymized data to improve job recommendations.</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" value="" className="sr-only peer" defaultChecked />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-electric-500"></div>
                      </label>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-4">Account Actions</h3>
                  
                  <div className="space-y-4">
                    <button className="flex items-center text-gray-300 hover:text-white transition-colors">
                      <ChevronDown className="h-5 w-5 mr-2" />
                      Export My Data
                    </button>
                    
                    <button className="flex items-center text-red-400 hover:text-red-300 transition-colors">
                      <ChevronDown className="h-5 w-5 mr-2" />
                      Delete Account
                    </button>
                  </div>
                </div>
                
                <div className="pt-4 flex justify-end">
                  <button
                    type="button"
                    className="px-4 py-2 bg-gradient-to-r from-electric-500 to-purple-500 hover:from-electric-600 hover:to-purple-600 text-white rounded-lg transition-all duration-300"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Profile;
