
import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bell, CheckCheck, Trash2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getNotifications, markNotificationAsRead, markAllNotificationsAsRead } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';

const formatTimeAgo = (dateString: string) => {
  const now = new Date();
  const date = new Date(dateString);
  const diffInMs = now.getTime() - date.getTime();
  const diffInSecs = Math.floor(diffInMs / 1000);
  
  if (diffInSecs < 60) return 'Just now';
  if (diffInSecs < 3600) return `${Math.floor(diffInSecs / 60)}m ago`;
  if (diffInSecs < 86400) return `${Math.floor(diffInSecs / 3600)}h ago`;
  if (diffInSecs < 604800) return `${Math.floor(diffInSecs / 86400)}d ago`;
  
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric'
  });
};

const getNotificationIcon = (type: string) => {
  switch (type) {
    case 'application':
      return <span className="h-8 w-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-500">📝</span>;
    case 'interview':
      return <span className="h-8 w-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-500">🗓️</span>;
    case 'offer':
      return <span className="h-8 w-8 rounded-full bg-green-500/20 flex items-center justify-center text-green-500">🎉</span>;
    case 'rejection':
      return <span className="h-8 w-8 rounded-full bg-red-500/20 flex items-center justify-center text-red-500">❌</span>;
    case 'system':
      return <span className="h-8 w-8 rounded-full bg-silver-500/20 flex items-center justify-center text-silver-500">⚙️</span>;
    default:
      return <span className="h-8 w-8 rounded-full bg-navy-500/20 flex items-center justify-center text-navy-500">📢</span>;
  }
};

const NotificationsMenu = () => {
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const queryClient = useQueryClient();
  
  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (isOpen && !target.closest('.notifications-menu')) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);
  
  const { 
    data,
    isLoading,
    error
  } = useQuery({
    queryKey: ['notifications'],
    queryFn: () => getNotifications({ limit: 10 }),
    enabled: !!user,
    refetchInterval: 30000 // Refetch every 30 seconds
  });
  
  const markAsReadMutation = useMutation({
    mutationFn: markNotificationAsRead,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    }
  });
  
  const markAllAsReadMutation = useMutation({
    mutationFn: markAllNotificationsAsRead,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    }
  });
  
  const handleMarkAsRead = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    markAsReadMutation.mutate(id);
  };
  
  const handleMarkAllAsRead = (e: React.MouseEvent) => {
    e.stopPropagation();
    markAllAsReadMutation.mutate();
  };
  
  if (!user) return null;
  
  const unreadCount = data?.unreadCount || 0;
  
  return (
    <div className="notifications-menu relative">
      <button
        className="premium-button-outline rounded-full p-2 relative"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Notifications"
      >
        <Bell className="h-5 w-5" />
        {unreadCount > 0 && (
          <Badge className="absolute -top-1 -right-1 h-5 min-w-5 rounded-full bg-red-500 flex items-center justify-center text-xs p-0">
            {unreadCount > 9 ? '9+' : unreadCount}
          </Badge>
        )}
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute right-0 mt-2 w-80 sm:w-96 premium-glass rounded-lg shadow-premium overflow-hidden z-50"
          >
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
              <h3 className="font-medium">Notifications</h3>
              {unreadCount > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-silver-400 hover:text-white flex items-center gap-1"
                  onClick={handleMarkAllAsRead}
                >
                  <CheckCheck className="h-3.5 w-3.5" />
                  Mark all as read
                </Button>
              )}
            </div>
            
            <div className="max-h-[60vh] overflow-y-auto">
              {isLoading ? (
                <div className="p-4 space-y-4">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : error ? (
                <div className="p-8 text-center">
                  <p className="text-silver-400">Failed to load notifications</p>
                </div>
              ) : data?.notifications?.length === 0 ? (
                <div className="p-8 text-center">
                  <p className="text-silver-400">No notifications yet</p>
                </div>
              ) : (
                <div>
                  {data?.notifications.map((notification: any) => (
                    <div 
                      key={notification.id}
                      className={`p-4 border-b border-white/5 flex gap-3 hover:bg-white/5 transition-colors ${
                        !notification.is_read ? 'bg-navy-900/50' : ''
                      }`}
                    >
                      {getNotificationIcon(notification.type)}
                      
                      <div className="flex-1">
                        <div className="flex justify-between items-start">
                          <h4 className="font-medium text-sm">{notification.title}</h4>
                          <span className="text-xs text-silver-500">{formatTimeAgo(notification.created_at)}</span>
                        </div>
                        <p className="text-sm text-silver-400 mt-1">{notification.message}</p>
                      </div>
                      
                      {!notification.is_read && (
                        <button
                          className="text-silver-500 hover:text-white self-center"
                          onClick={(e) => handleMarkAsRead(notification.id, e)}
                          aria-label="Mark as read"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-3 border-t border-white/10 flex justify-center">
              <Button 
                variant="link" 
                size="sm"
                className="text-sm text-silver-400 hover:text-white"
              >
                View all notifications
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default NotificationsMenu;
