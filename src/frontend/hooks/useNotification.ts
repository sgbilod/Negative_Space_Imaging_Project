import { useContext, useCallback } from 'react';
import { NotificationContext } from '@/context/NotificationContext';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;
}

export interface UseNotificationReturn {
  notifications: Notification[];
  showSuccess: (message: string, duration?: number) => void;
  showError: (message: string, duration?: number) => void;
  showWarning: (message: string, duration?: number) => void;
  showInfo: (message: string, duration?: number) => void;
  dismissNotification: (id: string) => void;
  clearAll: () => void;
}

export const useNotification = (): UseNotificationReturn => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within NotificationProvider');
  }

  const { notifications, addNotification, removeNotification, clearNotifications } = context;

  const showNotification = useCallback(
    (type: NotificationType, message: string, duration?: number) => {
      addNotification({ type, message, duration });
    },
    [addNotification]
  );

  const showSuccess = useCallback(
    (message: string, duration?: number) => {
      showNotification('success', message, duration);
    },
    [showNotification]
  );

  const showError = useCallback(
    (message: string, duration?: number) => {
      showNotification('error', message, duration);
    },
    [showNotification]
  );

  const showWarning = useCallback(
    (message: string, duration?: number) => {
      showNotification('warning', message, duration);
    },
    [showNotification]
  );

  const showInfo = useCallback(
    (message: string, duration?: number) => {
      showNotification('info', message, duration);
    },
    [showNotification]
  );

  const dismissNotification = useCallback(
    (id: string) => {
      removeNotification(id);
    },
    [removeNotification]
  );

  const clearAll = useCallback(() => {
    clearNotifications();
  }, [clearNotifications]);

  return {
    notifications,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    dismissNotification,
    clearAll,
  };
};
