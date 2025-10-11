
import { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  gradient?: 'blue' | 'purple' | 'mixed' | 'default';
  delay?: number;
  link?: string;
}

const FeatureCard = ({ 
  icon, 
  title, 
  description, 
  gradient = 'default',
  delay = 0,
  link = '#'
}: FeatureCardProps) => {
  const getGradientClass = () => {
    switch (gradient) {
      case 'blue':
        return 'from-electric-500/30 to-electric-500/5';
      case 'purple':
        return 'from-purple-500/30 to-purple-500/5';
      case 'mixed':
        return 'from-electric-500/30 to-purple-500/5';
      default:
        return 'from-white/10 to-white/5';
    }
  };

  const getIconClass = () => {
    switch (gradient) {
      case 'blue':
        return 'bg-gradient-to-br from-electric-400 to-electric-600 shadow-glow';
      case 'purple':
        return 'bg-gradient-to-br from-purple-400 to-purple-600 shadow-glow-purple';
      case 'mixed':
        return 'bg-gradient-to-br from-electric-500 to-purple-500 shadow-glow';
      default:
        return 'bg-white/10';
    }
  };

  return (
    <div 
      className={`group bg-gradient-to-b ${getGradientClass()} backdrop-blur-xl rounded-xl border border-white/20 p-8 transition-all duration-300 hover:shadow-glow hover:-translate-y-2 h-full flex flex-col animate-fade-in relative overflow-hidden`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Enhanced background elements */}
      <div className="absolute -bottom-16 -right-16 w-32 h-32 rounded-full bg-white/5 blur-xl"></div>
      
      <div className={`${getIconClass()} rounded-xl p-4 w-14 h-14 flex items-center justify-center mb-6 transform transition-transform group-hover:scale-110`}>
        {icon}
      </div>
      
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      
      <p className="text-gray-300 mb-6 leading-relaxed flex-grow">{description}</p>
      
      <div className="mt-auto pt-4">
        <Link 
          to={link} 
          className="text-electric-400 font-medium flex items-center group-hover:text-electric-300 transition-colors"
        >
          Learn more
          <ChevronRight 
            className="h-5 w-5 ml-1 transform transition-transform group-hover:translate-x-1" 
          />
        </Link>
      </div>
    </div>
  );
};

export default FeatureCard;
