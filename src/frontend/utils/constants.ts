export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    ME: '/auth/me',
  },
  IMAGES: {
    LIST: '/images',
    GET: (id: string) => `/images/${id}`,
    UPLOAD: '/images/upload',
    DELETE: (id: string) => `/images/${id}`,
    SEARCH: '/images/search',
  },
  ANALYSIS: {
    START: '/analysis/start',
    GET: (id: string) => `/analysis/${id}`,
    LIST: '/analysis',
    IMAGE: (imageId: string) => `/analysis/image/${imageId}`,
  },
  USERS: {
    PROFILE: '/users/profile',
    UPDATE: '/users/profile',
    CHANGE_PASSWORD: '/users/change-password',
    DELETE: '/users/account',
  },
};

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  UNAUTHORIZED: 'You are not authorized. Please log in again.',
  FORBIDDEN: 'You do not have permission to perform this action.',
  NOT_FOUND: 'The requested resource was not found.',
  SERVER_ERROR: 'Server error. Please try again later.',
  INVALID_CREDENTIALS: 'Invalid email or password.',
  EMAIL_EXISTS: 'This email is already registered.',
  VALIDATION_ERROR: 'Please check your input and try again.',
};

export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Successfully logged in.',
  REGISTER_SUCCESS: 'Account created successfully.',
  LOGOUT_SUCCESS: 'Successfully logged out.',
  IMAGE_UPLOADED: 'Image uploaded successfully.',
  IMAGE_DELETED: 'Image deleted successfully.',
  ANALYSIS_STARTED: 'Analysis started.',
  PROFILE_UPDATED: 'Profile updated successfully.',
  PASSWORD_CHANGED: 'Password changed successfully.',
};

export const UI_CONSTANTS = {
  PAGINATION_SIZE: 10,
  IMAGE_UPLOAD_SIZE_LIMIT: 50 * 1024 * 1024, // 50MB
  ANALYSIS_POLL_INTERVAL: 2000, // 2 seconds
  ANALYSIS_TIMEOUT: 300000, // 5 minutes
  NOTIFICATION_DURATION: 5000, // 5 seconds
};
