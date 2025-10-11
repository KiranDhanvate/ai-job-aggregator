
import { useState, useEffect } from 'react';
import { BarChart2, Globe, Users, Award } from 'lucide-react';

interface Statistic {
  label: string;
  value: string;
  target: number;
  prefix?: string;
  suffix?: string;
  icon: React.ReactNode;
}

const statistics: Statistic[] = [
  { 
    label: 'Jobs Aggregated', 
    value: '0', 
    target: 1250000, 
    suffix: '+',
    icon: <BarChart2 className="h-6 w-6 text-electric-400" />
  },
  { 
    label: 'Partner Platforms', 
    value: '0', 
    target: 42,
    icon: <Globe className="h-6 w-6 text-purple-400" />
  },
  { 
    label: 'Successful Matches', 
    value: '0', 
    target: 85000, 
    suffix: '+',
    icon: <Users className="h-6 w-6 text-electric-400" />
  },
  { 
    label: 'User Success Rate', 
    value: '0', 
    target: 92, 
    suffix: '%',
    icon: <Award className="h-6 w-6 text-purple-400" />
  },
];

const StatisticsBar = () => {
  const [counts, setCounts] = useState<string[]>(statistics.map(() => '0'));
  const [animationStarted, setAnimationStarted] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !animationStarted) {
          setAnimationStarted(true);
          statistics.forEach((stat, index) => {
            animateCounter(index, stat.target);
          });
        }
      },
      { threshold: 0.1 }
    );

    const element = document.getElementById('statistics-section');
    if (element) {
      observer.observe(element);
    }

    return () => {
      if (element) {
        observer.unobserve(element);
      }
    };
  }, [animationStarted]);

  const animateCounter = (index: number, target: number) => {
    const duration = 2000; // 2 seconds
    const frameDuration = 1000 / 60; // 60 fps
    const totalFrames = Math.round(duration / frameDuration);
    const increment = target / totalFrames;
    
    let currentFrame = 0;
    let currentCount = 0;
    
    const counter = setInterval(() => {
      currentCount += increment;
      currentFrame++;
      
      setCounts(prev => {
        const newCounts = [...prev];
        
        if (target >= 1000) {
          // Format large numbers with commas
          newCounts[index] = Math.min(Math.floor(currentCount), target).toLocaleString();
        } else {
          // Format percentages or smaller numbers
          newCounts[index] = Math.min(Math.floor(currentCount), target).toString();
        }
        
        return newCounts;
      });
      
      if (currentFrame === totalFrames) {
        clearInterval(counter);
      }
    }, frameDuration);
  };

  return (
    <div id="statistics-section" className="relative py-16 backdrop-blur-md border-y border-white/10 overflow-hidden">
      {/* Enhanced background with subtle patterns */}
      <div className="absolute inset-0 bg-gradient-to-r from-navy-800/80 via-navy-800/90 to-navy-800/80"></div>
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-electric-500/30 to-transparent"></div>
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500/30 to-transparent"></div>
      
      {/* Subtle background patterns */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-1/4 left-1/4 w-32 h-32 rounded-full bg-electric-500/30 blur-xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-32 h-32 rounded-full bg-purple-500/30 blur-xl"></div>
      </div>
      
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
          {statistics.map((stat, index) => (
            <div key={stat.label} className="text-center transform transition-all duration-300 hover:scale-105">
              <div className="flex justify-center mb-4">
                <div className="p-3 rounded-full bg-white/5 border border-white/10">{stat.icon}</div>
              </div>
              <div className="text-3xl md:text-4xl font-bold mb-2 bg-gradient-to-r from-electric-400 to-purple-500 bg-clip-text text-transparent">
                {stat.prefix}{counts[index]}{stat.suffix}
              </div>
              <div className="text-sm md:text-base text-gray-300">{stat.label}</div>
              <div className="w-16 h-1 bg-gradient-to-r from-electric-500/50 to-purple-500/50 mx-auto mt-3 rounded-full"></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StatisticsBar;
