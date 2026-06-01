/**
 * App Store
 * Central application state store
 */

import { create } from 'zustand';

type AppState = Record<string, unknown>;

export const useAppStore = create<AppState>(() => ({}));
