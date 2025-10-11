
import { useState, useEffect } from 'react';
import Hero from '../components/Hero';
import StatisticsBar from '../components/StatisticsBar';
import Footer from '../components/Footer';
import FeaturesSection from '../components/landing/FeaturesSection';
import FeaturedJobsSection from '../components/landing/FeaturedJobsSection';
import HowItWorksSection from '../components/landing/HowItWorksSection';
import TestimonialsSection from '../components/landing/TestimonialsSection';
import CtaSection from '../components/landing/CtaSection';

const Index = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div className={`min-h-screen bg-navy-900 text-white transition-opacity duration-500 ${isVisible ? 'opacity-100' : 'opacity-0'}`}>
      <Hero />
      <FeaturesSection />
      <StatisticsBar />
      <FeaturedJobsSection />
      <HowItWorksSection />
      <TestimonialsSection />
      <CtaSection />
      <Footer />
    </div>
  );
};

export default Index;
