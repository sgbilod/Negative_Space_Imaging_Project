/**
 * Notification Hook
 * Provides notification/toast functionality
 */

import { useCallback, useState, useEffect } from 'react';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  message: string;
  type: NotificationType;
  duration?: number;
  dismissible?: boolean;
  createdAt: number;
}

// Create a simple notification store that can be shared
let notificationListeners: ((notifications: Notification[]) => void)[] = [];
let notifications: Notification[] = [];

const notifyListeners = (): void => {
  notificationListeners.forEach((listener) => listener([...notifications]));
};

const dismissNotification = (id: string): void => {
  notifications = notifications.filter((n) => n.id !== id);
  notifyListeners();
};

export const useNotification = () => {
  const [, setUpdate] = useState(0);

  // Subscribe to notifications on mount
  useEffect(() => {
    const listener = (): void => setUpdate((prev) => prev + 1);
    notificationListeners.push(listener);
    return () => {
      notificationListeners = notificationListeners.filter((l) => l !== listener);
    };
  }, []);

  const notify = useCallback(
    (message: string, type: NotificationType = 'info', duration = 5000): string => {
      const id = `notification-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;

      const notification: Notification = {
        id,
        message,
        type,
        duration,
        dismissible: true,
        createdAt: Date.now(),
      };

      notifications = [...notifications, notification];
      notifyListeners();

      // Auto-dismiss after duration
      if (duration > 0) {
        setTimeout(() => {
          dismissNotification(id);
        }, duration);
      }

      return id;
    },
    [],
  );

  const dismiss = useCallback((id: string) => {
    dismissNotification(id);
  }, []);

  const dismissAll = useCallback(() => {
    notifications = [];
    notifyListeners();
  }, []);

  const getNotifications = useCallback((): Notification[] => {
    return [...notifications];
  }, []);

  return {
    notify,
    showNotification: notify, // Alias for backward compatibility
    dismiss,
    dismissAll,
    getNotifications,
    notifications,
    success: (message: string, duration?: number): string => notify(message, 'success', duration),
    error: (message: string, duration?: number): string => notify(message, 'error', duration ?? 8000),
    warning: (message: string, duration?: number): string => notify(message, 'warning', duration),
    info: (message: string, duration?: number): string => notify(message, 'info', duration),
  };
};
