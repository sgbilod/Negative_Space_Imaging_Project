/**
 * Theme Context
 * Provides global theme state (light/dark mode) to all components
 * Manages theme preferences and persistence
 */

import React, { createContext, useState, useCallback, useMemo, useEffect } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';

/**
 * Theme mode type
 */
export type ThemeMode = 'light' | 'dark' | 'auto';

/**
 * Theme context state shape
 */
interface ThemeContextType {
  mode: ThemeMode;
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (theme: ThemeMode) => void;
}

/**
 * Create theme context
 */
export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

/**
 * Theme Context Provider Props
 */
interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: ThemeMode;
}

/**
 * Detect system theme preference
 */
const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') {
    return 'light';
  }

  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }

  return 'light';
};

/**
 * Theme Provider Component
 * Wraps the app with theme state management
 *
 * @param props - Provider props
 * @returns Theme provider component
 *
 * @example
 * <ThemeProvider defaultTheme="dark">
 *   <App />
 * </ThemeProvider>
 */
export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme = 'auto',
}) => {
  const [savedTheme, setSavedTheme] = useLocalStorage<ThemeMode>('theme', defaultTheme);
  const [mode, setMode] = useState<ThemeMode>(savedTheme);
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(() => getSystemTheme());

  /**
   * Determine if current theme is dark
   */
  const isDark = useMemo(() => {
    if (mode === 'auto') {
      return systemTheme === 'dark';
    }
    return mode === 'dark';
  }, [mode, systemTheme]);

  /**
   * Listen for system theme changes
   */
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  /**
   * Apply theme to DOM
   */
  useEffect(() => {
    const htmlElement = document.documentElement;

    if (isDark) {
      htmlElement.classList.add('dark');
      htmlElement.style.colorScheme = 'dark';
    } else {
      htmlElement.classList.remove('dark');
      htmlElement.style.colorScheme = 'light';
    }
  }, [isDark]);

  /**
   * Toggle between light and dark theme
   */
  const toggleTheme = useCallback(() => {
    setMode((prevMode) => {
      const newMode: ThemeMode = prevMode === 'light' ? 'dark' : 'light';
      setSavedTheme(newMode);
      return newMode;
    });
  }, [setSavedTheme]);

  /**
   * Set specific theme
   */
  const setTheme = useCallback(
    (theme: ThemeMode) => {
      setMode(theme);
      setSavedTheme(theme);
    },
    [setSavedTheme],
  );

  /**
   * Memoized context value
   */
  const value = useMemo<ThemeContextType>(
    () => ({
      mode,
      isDark,
      toggleTheme,
      setTheme,
    }),
    [mode, isDark, toggleTheme, setTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

/**
 * useThemeContext - Custom hook to use theme context
 * Must be used within ThemeProvider
 *
 * @returns {ThemeContextType} Theme context value
 * @throws {Error} If used outside of ThemeProvider
 *
 * @example
 * const { isDark, toggleTheme } = useThemeContext();
 */
export const useThemeContext = (): ThemeContextType => {
  const context = React.useContext(ThemeContext);

  if (context === undefined) {
    throw new Error('useThemeContext must be used within a ThemeProvider');
  }

  return context;
};

export default ThemeProvider;
