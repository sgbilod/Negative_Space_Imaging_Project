/**
 * Analysis Store
 * Manages analysis-related state
 */

import { create } from 'zustand';

type AnalysisState = Record<string, unknown>;

export const useAnalysisStore = create<AnalysisState>(() => ({}));
