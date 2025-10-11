
import { Users, Briefcase, Star } from 'lucide-react';

const TestimonialsSection = () => {
  return (
    <section className="py-24 relative">
      <div className="absolute top-20 right-20 w-32 h-32 rounded-full bg-electric-500/5 blur-2xl"></div>
      <div className="absolute bottom-20 left-20 w-32 h-32 rounded-full bg-purple-500/5 blur-2xl"></div>
        
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-flex items-center justify-center mb-4">
            <div className="h-px w-12 bg-gradient-to-r from-transparent to-electric-400/50"></div>
            <h2 className="mx-4 text-sm uppercase tracking-wider text-electric-400 font-semibold">Testimonials</h2>
            <div className="h-px w-12 bg-gradient-to-l from-transparent to-electric-400/50"></div>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-6 flex items-center justify-center">
            <Users className="mr-3 h-7 w-7 text-electric-400" />
            Trusted by <span className="text-gradient bg-gradient-to-r from-electric-400 to-purple-500 bg-clip-text text-transparent ml-2">Thousands</span> of Job Seekers
          </h2>
          <p className="text-gray-300 text-lg leading-relaxed">
            Don't just take our word for it. Here's what our users have to say about CareerSync.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="glass-card rounded-xl p-8 relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow transition-all duration-300 hover:-translate-y-1">
            <div className="absolute -top-4 -left-2 text-5xl text-electric-500 opacity-50">"</div>
            <div className="absolute top-4 right-6">
              <div className="flex">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} className="h-4 w-4 text-yellow-400 fill-yellow-400" />
                ))}
              </div>
            </div>
            <p className="text-gray-300 mb-8 leading-relaxed pt-6">
              CareerSync helped me land my dream job in just three weeks. The AI recommendations were spot-on and perfectly matched my skills!
            </p>
            <div className="flex items-center">
              <div className="bg-gradient-to-br from-electric-500 to-electric-700 h-12 w-12 rounded-full mr-4 flex items-center justify-center text-lg font-bold text-white">
                SJ
              </div>
              <div>
                <p className="font-medium">Sarah Johnson</p>
                <p className="text-sm text-gray-400 flex items-center">
                  <Briefcase className="h-3 w-3 mr-1" /> UX Designer at TechCorp
                </p>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-8 relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow transition-all duration-300 md:translate-y-4 hover:-translate-y-1">
            <div className="absolute -top-4 -left-2 text-5xl text-electric-500 opacity-50">"</div>
            <div className="absolute top-4 right-6">
              <div className="flex">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} className="h-4 w-4 text-yellow-400 fill-yellow-400" />
                ))}
              </div>
            </div>
            <p className="text-gray-300 mb-8 leading-relaxed pt-6">
              The application tracking feature is a game-changer. I can finally keep track of everything in one place and never miss an important follow-up.
            </p>
            <div className="flex items-center">
              <div className="bg-gradient-to-br from-purple-500 to-purple-700 h-12 w-12 rounded-full mr-4 flex items-center justify-center text-lg font-bold text-white">
                MC
              </div>
              <div>
                <p className="font-medium">Michael Chen</p>
                <p className="text-sm text-gray-400 flex items-center">
                  <Briefcase className="h-3 w-3 mr-1" /> Software Engineer
                </p>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-8 relative backdrop-blur-md bg-white/5 border border-white/10 hover:shadow-glow transition-all duration-300 hover:-translate-y-1">
            <div className="absolute -top-4 -left-2 text-5xl text-electric-500 opacity-50">"</div>
            <div className="absolute top-4 right-6">
              <div className="flex">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} className="h-4 w-4 text-yellow-400 fill-yellow-400" />
                ))}
              </div>
            </div>
            <p className="text-gray-300 mb-8 leading-relaxed pt-6">
              As a career changer, CareerSync's skill recommendations helped me identify transferable skills for my resume and highlight my unique strengths.
            </p>
            <div className="flex items-center">
              <div className="bg-gradient-to-br from-electric-500 to-purple-500 h-12 w-12 rounded-full mr-4 flex items-center justify-center text-lg font-bold text-white">
                EP
              </div>
              <div>
                <p className="font-medium">Emily Patel</p>
                <p className="text-sm text-gray-400 flex items-center">
                  <Briefcase className="h-3 w-3 mr-1" /> Marketing Specialist
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TestimonialsSection;
