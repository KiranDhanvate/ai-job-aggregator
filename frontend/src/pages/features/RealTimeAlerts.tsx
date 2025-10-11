
import { Bell, MessageSquare, Calendar, Smartphone, Mail, Clock, BellRing } from 'lucide-react';
import FeatureLayout from '@/components/FeatureLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const RealTimeAlerts = () => {
  return (
    <FeatureLayout
      title="Real-time Alerts"
      description="Receive notifications about new job postings, application updates, and interview reminders."
      icon={<Bell className="h-8 w-8 text-white" />}
      gradient="blue"
    >
      <div className="max-w-4xl mx-auto">
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-6">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <Smartphone className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Multi-Channel Delivery</CardTitle>
                <CardDescription className="text-gray-300">
                  Get notifications via email, browser notifications, and mobile push alerts based on your preferences.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <Clock className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Real-Time Processing</CardTitle>
                <CardDescription className="text-gray-300">
                  Our system processes events and delivers notifications within seconds of them occurring.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="bg-navy-800 border-white/10 text-white">
              <CardHeader>
                <BellRing className="h-10 w-10 text-electric-400 mb-3" />
                <CardTitle>Custom Alert Preferences</CardTitle>
                <CardDescription className="text-gray-300">
                  Customize which notifications you receive and how they're delivered to match your workflow.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>
        
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-6">Key Features</h2>
          <div className="space-y-6">
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <Mail className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Job Match Alerts</h3>
                  <p className="text-gray-300">
                    Receive alerts when new job listings match your skills, experience, and preferences - even when you're not actively searching.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <MessageSquare className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Application Status Updates</h3>
                  <p className="text-gray-300">
                    Get notified when recruiters view your application, when your status changes, or when follow-up is recommended.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-start mb-4">
                <div className="bg-electric-500/20 rounded-lg p-2 mr-4">
                  <Calendar className="h-6 w-6 text-electric-400" />
                </div>
                <div>
                  <h3 className="text-xl font-medium mb-2">Calendar Integration</h3>
                  <p className="text-gray-300">
                    Sync with your calendar to receive timely reminders for interviews, follow-ups, and application deadlines.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
        
        <section className="text-center">
          <h2 className="text-2xl font-semibold mb-6">Never Miss an Opportunity Again</h2>
          <div className="flex flex-wrap justify-center gap-4">
            <Button asChild className="bg-gradient-to-r from-electric-500 to-electric-600 hover:from-electric-600 hover:to-electric-700 text-white">
              <Link to="/profile">Set Up Alerts</Link>
            </Button>
            <Button asChild variant="outline" className="border-white/10 hover:bg-white/10">
              <Link to="/auth/register">Create an Account</Link>
            </Button>
          </div>
        </section>
      </div>
    </FeatureLayout>
  );
};

export default RealTimeAlerts;
