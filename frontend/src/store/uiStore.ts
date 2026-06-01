/**
 * UI State Store
 * Manages global UI state (theme, sidebar, modals, etc.)
 */

import { create } from 'zustand';

interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  theme: 'light',
  setSidebarOpen: (open: boolean) => set({ sidebarOpen: open }),
  setTheme: (theme: 'light' | 'dark') => set({ theme }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
