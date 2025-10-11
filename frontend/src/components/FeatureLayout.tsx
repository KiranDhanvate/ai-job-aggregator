
import { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Helmet } from 'react-helmet-async';
import Footer from './Footer';

interface FeatureLayoutProps {
  title: string;
  description: string;
  icon: ReactNode;
  gradient: 'blue' | 'purple' | 'mixed';
  children: ReactNode;
}

const FeatureLayout = ({ title, description, icon, gradient, children }: FeatureLayoutProps) => {
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
    <>
      <Helmet>
        <title>{title} - CareerSync</title>
        <meta name="description" content={description} />
      </Helmet>
    
      <div className="min-h-screen bg-navy-900 text-white">
        <div className={`bg-gradient-to-b ${getGradientClass()} pt-28 pb-20 border-b border-white/5`}>
          <div className="container mx-auto px-4 md:px-6">
            <Link to="/" className="inline-flex items-center text-gray-300 hover:text-white mb-6 transition-colors">
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back to home
            </Link>
            
            <div className="flex items-center mb-6">
              <div className={`${getIconClass()} rounded-lg p-3 w-14 h-14 flex items-center justify-center mr-4`}>
                {icon}
              </div>
              <h1 className="text-3xl md:text-4xl font-bold">{title}</h1>
            </div>
            
            <p className="text-xl text-gray-300 max-w-3xl">
              {description}
            </p>
          </div>
        </div>
        
        <div className="container mx-auto px-4 md:px-6 py-12">
          {children}
        </div>
        
        <Footer />
      </div>
    </>
  );
};

export default FeatureLayout;
