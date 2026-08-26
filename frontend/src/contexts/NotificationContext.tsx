/**
 * Notification Context
 * Provides global toast/notification system to all components
 * Manages notification queue and display
 */

import React, { createContext, useState, useCallback, useMemo } from 'react';

/**
 * Notification severity levels
 */
export type NotificationSeverity = 'success' | 'error' | 'warning' | 'info';

/**
 * Notification object structure
 */
export interface Notification {
  id: string;
  message: string;
  severity: NotificationSeverity;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

/**
 * Notification context state shape
 */
interface NotificationContextType {
  notifications: Notification[];
  addNotification: (
    message: string,
    severity?: NotificationSeverity,
    duration?: number,
    action?: Notification['action'],
  ) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  success: (message: string, duration?: number) => string;
  error: (message: string, duration?: number) => string;
  warning: (message: string, duration?: number) => string;
  info: (message: string, duration?: number) => string;
}

/**
 * Create notification context
 */
export const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

/**
 * Notification Context Provider Props
 */
interface NotificationProviderProps {
  children: React.ReactNode;
  maxNotifications?: number;
}

/**
 * Generate unique ID for notifications
 */
const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Notification Provider Component
 * Wraps the app with notification state management
 *
 * @param props - Provider props
 * @returns Notification provider component
 *
 * @example
 * <NotificationProvider maxNotifications={5}>
 *   <App />
 * </NotificationProvider>
 */
export const NotificationProvider: React.FC<NotificationProviderProps> = ({
  children,
  maxNotifications = 5,
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  /**
   * Add a new notification
   */
  const addNotification = useCallback(
    (
      message: string,
      severity: NotificationSeverity = 'info',
      duration = 5000,
      action?: Notification['action'],
    ): string => {
      const id = generateId();
      const notification: Notification = {
        id,
        message,
        severity,
        duration,
        action,
      };

      setNotifications((prev) => {
        const updated = [...prev, notification];
        // Keep only the most recent notifications
        return updated.slice(Math.max(0, updated.length - maxNotifications));
      });

      // Auto-remove notification after duration
      if (duration && duration > 0) {
        setTimeout(() => {
          removeNotification(id);
        }, duration);
      }

      return id;
    },
    [maxNotifications],
  );

  /**
   * Remove a notification by ID
   */
  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  /**
   * Clear all notifications
   */
  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  /**
   * Add success notification
   */
  const success = useCallback(
    (message: string, duration = 5000) => addNotification(message, 'success', duration),
    [addNotification],
  );

  /**
   * Add error notification
   */
  const error = useCallback(
    (message: string, duration = 5000) => addNotification(message, 'error', duration),
    [addNotification],
  );

  /**
   * Add warning notification
   */
  const warning = useCallback(
    (message: string, duration = 5000) => addNotification(message, 'warning', duration),
    [addNotification],
  );

  /**
   * Add info notification
   */
  const info = useCallback(
    (message: string, duration = 5000) => addNotification(message, 'info', duration),
    [addNotification],
  );

  /**
   * Memoized context value
   */
  const value = useMemo<NotificationContextType>(
    () => ({
      notifications,
      addNotification,
      removeNotification,
      clearNotifications,
      success,
      error,
      warning,
      info,
    }),
    [
      notifications,
      addNotification,
      removeNotification,
      clearNotifications,
      success,
      error,
      warning,
      info,
    ],
  );

  return <NotificationContext.Provider value={value}>{children}</NotificationContext.Provider>;
};

/**
 * useNotificationContext - Custom hook to use notification context
 * Must be used within NotificationProvider
 *
 * @returns {NotificationContextType} Notification context value
 * @throws {Error} If used outside of NotificationProvider
 *
 * @example
 * const { success, error } = useNotificationContext();
 * success('Operation completed');
 */
export const useNotificationContext = (): NotificationContextType => {
  const context = React.useContext(NotificationContext);

  if (context === undefined) {
    throw new Error('useNotificationContext must be used within a NotificationProvider');
  }

  return context;
};

export default NotificationProvider;
