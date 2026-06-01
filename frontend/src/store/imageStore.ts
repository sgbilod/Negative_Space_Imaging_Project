/**
 * Image Store
 * Manages image-related state
 */

import { create } from 'zustand';

type ImageState = Record<string, unknown>;

export const useImageStore = create<ImageState>(() => ({}));
