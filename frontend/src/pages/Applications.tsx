
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Helmet } from 'react-helmet-async';
import { PlusCircle, ClipboardList, CheckCircle2, XCircle, Clock, AlertCircle, Filter } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getApplications } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  logo_url?: string;
}

interface Application {
  id: string;
  status: string;
  applied_at: string;
  last_updated_at: string;
  jobs: Job;
}

const statusColors = {
  applied: "bg-blue-500/20 text-blue-500 border-blue-500/30",
  interview: "bg-purple-500/20 text-purple-500 border-purple-500/30",
  offer: "bg-green-500/20 text-green-500 border-green-500/30",
  rejected: "bg-red-500/20 text-red-500 border-red-500/30",
  withdrawn: "bg-amber-500/20 text-amber-500 border-amber-500/30",
};

const statusIcons = {
  applied: <ClipboardList className="h-4 w-4" />,
  interview: <Clock className="h-4 w-4" />,
  offer: <CheckCircle2 className="h-4 w-4" />,
  rejected: <XCircle className="h-4 w-4" />,
  withdrawn: <AlertCircle className="h-4 w-4" />,
};

const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
};

const ApplicationsPage = () => {
  const { user, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("all");
  
  // Redirect if not logged in
  useEffect(() => {
    if (!authLoading && !user) {
      navigate('/auth');
    }
  }, [user, authLoading, navigate]);

  const { 
    data: applications,
    isLoading, 
    error 
  } = useQuery({
    queryKey: ['applications'],
    queryFn: getApplications,
    enabled: !!user,
  });

  const getFilteredApplications = () => {
    if (!applications) return [];
    
    if (activeTab === "all") return applications;
    return applications.filter((app: Application) => app.status === activeTab);
  };

  const getStatusCounts = () => {
    if (!applications) return {};
    
    return applications.reduce((acc, app: Application) => {
      acc[app.status] = (acc[app.status] || 0) + 1;
      return acc;
    }, { all: applications.length } as Record<string, number>);
  };

  const statusCounts = getStatusCounts();
  const filteredApplications = getFilteredApplications();

  if (authLoading) {
    return (
      <div className="container mx-auto py-16 px-4">
        <div className="flex flex-col gap-8">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-12 w-full" />
          <div className="grid gap-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        </div>
      </div>
    );
  }

  if (!user) {
    return null; // Will redirect in the useEffect
  }

  return (
    <>
      <Helmet>
        <title>My Applications | CareerSync</title>
      </Helmet>
      
      <div className="container mx-auto py-12 px-4">
        <div className="flex flex-col gap-8">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white">My Applications</h1>
              <p className="text-silver-400 mt-1">Track and manage your job applications</p>
            </div>
            
            <Button className="premium-button">
              <PlusCircle className="h-4 w-4 mr-2" />
              Add Application
            </Button>
          </div>
          
          <Tabs defaultValue="all" className="w-full" value={activeTab} onValueChange={setActiveTab}>
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
              <TabsList className="premium-glass">
                <TabsTrigger value="all">
                  All
                  {statusCounts.all && <Badge className="ml-2 bg-white/10">{statusCounts.all}</Badge>}
                </TabsTrigger>
                <TabsTrigger value="applied">
                  Applied
                  {statusCounts.applied && <Badge className="ml-2 bg-blue-500/20 text-blue-400">{statusCounts.applied}</Badge>}
                </TabsTrigger>
                <TabsTrigger value="interview">
                  Interview
                  {statusCounts.interview && <Badge className="ml-2 bg-purple-500/20 text-purple-400">{statusCounts.interview}</Badge>}
                </TabsTrigger>
                <TabsTrigger value="offer">
                  Offer
                  {statusCounts.offer && <Badge className="ml-2 bg-green-500/20 text-green-400">{statusCounts.offer}</Badge>}
                </TabsTrigger>
                <TabsTrigger value="rejected">
                  Rejected
                  {statusCounts.rejected && <Badge className="ml-2 bg-red-500/20 text-red-400">{statusCounts.rejected}</Badge>}
                </TabsTrigger>
              </TabsList>
              
              <Button variant="outline" size="sm" className="premium-button-outline">
                <Filter className="h-4 w-4 mr-2" />
                Filter
              </Button>
            </div>
            
            <TabsContent value={activeTab} className="mt-0">
              {isLoading ? (
                <div className="grid gap-4">
                  <Skeleton className="h-32 w-full" />
                  <Skeleton className="h-32 w-full" />
                  <Skeleton className="h-32 w-full" />
                </div>
              ) : error ? (
                <Card className="premium-card">
                  <CardContent className="pt-6">
                    <div className="text-center p-4">
                      <AlertCircle className="mx-auto h-10 w-10 text-red-500 mb-2" />
                      <h3 className="text-lg font-medium">Error loading applications</h3>
                      <p className="text-silver-400 mt-1">Please try again later</p>
                    </div>
                  </CardContent>
                </Card>
              ) : filteredApplications.length === 0 ? (
                <Card className="premium-card">
                  <CardContent className="pt-6">
                    <div className="text-center p-4">
                      <ClipboardList className="mx-auto h-10 w-10 text-silver-500 mb-2" />
                      <h3 className="text-lg font-medium">No applications found</h3>
                      <p className="text-silver-400 mt-1">
                        {activeTab === "all" 
                          ? "You haven't applied to any jobs yet" 
                          : `No applications with status "${activeTab}"`}
                      </p>
                      <Button className="premium-button mt-4">
                        <PlusCircle className="h-4 w-4 mr-2" />
                        Add Application
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <AnimatePresence>
                  <div className="grid gap-4">
                    {filteredApplications.map((application: Application) => (
                      <motion.div
                        key={application.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                        transition={{ duration: 0.3 }}
                      >
                        <Card className="premium-card-interactive cursor-pointer" onClick={() => navigate(`/applications/${application.id}`)}>
                          <CardContent className="p-6">
                            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                              <div className="h-12 w-12 rounded-md bg-white/10 flex items-center justify-center overflow-hidden flex-shrink-0 border border-white/5">
                                {application.jobs.logo_url ? (
                                  <img 
                                    src={application.jobs.logo_url} 
                                    alt={`${application.jobs.company} logo`} 
                                    className="w-full h-full object-contain p-1" 
                                  />
                                ) : (
                                  <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-navy-500 to-teal-500">
                                    <span className="font-bold text-white text-lg">
                                      {application.jobs.company.charAt(0)}
                                    </span>
                                  </div>
                                )}
                              </div>
                              
                              <div className="flex-1">
                                <h3 className="text-lg font-medium text-white">{application.jobs.title}</h3>
                                <p className="text-silver-300">{application.jobs.company}</p>
                                <p className="text-silver-500 text-sm">{application.jobs.location}</p>
                              </div>
                              
                              <div className="flex flex-col items-end gap-2">
                                <Badge className={`${statusColors[application.status as keyof typeof statusColors]} flex items-center gap-1`}>
                                  {statusIcons[application.status as keyof typeof statusIcons]}
                                  <span className="capitalize">{application.status}</span>
                                </Badge>
                                <div className="flex flex-col items-end text-sm">
                                  <span className="text-silver-400">Applied: {formatDate(application.applied_at)}</span>
                                  <span className="text-silver-500 text-xs">Last updated: {formatDate(application.last_updated_at)}</span>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </AnimatePresence>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </>
  );
};

export default ApplicationsPage;
