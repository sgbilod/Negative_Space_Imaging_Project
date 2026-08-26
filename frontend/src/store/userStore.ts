/**
 * User Store
 * Manages user-related state
 */

import { create } from 'zustand';

type UserState = Record<string, unknown>;

export const useUserStore = create<UserState>(() => ({}));
