/**
 * Security-related TypeScript types for Negative Space Imaging System
 */

/**
 * Configuration for rate limiting
 */
export interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  message: string;
}

/**
 * CORS configuration options
 */
export interface CorsConfig {
  origin: string | string[] | boolean;
  methods: string[];
  allowedHeaders: string[];
  exposedHeaders: string[];
  credentials: boolean;
  maxAge: number;
}

/**
 * Content Security Policy directives
 */
export interface CSPDirectives {
  defaultSrc?: string[];
  scriptSrc?: string[];
  styleSrc?: string[];
  imgSrc?: string[];
  fontSrc?: string[];
  connectSrc?: string[];
  frameSrc?: string[];
  objectSrc?: string[];
  mediaSrc?: string[];
  workerSrc?: string[];
  frameAncestors?: string[];
  formAction?: string[];
  baseUri?: string[];
  upgradeInsecureRequests?: string[];
  [key: string]: string[] | undefined;
}

/**
 * Content Security Policy configuration
 */
export interface CSPConfig {
  directives: CSPDirectives;
  reportOnly?: boolean;
}

/**
 * Cross-Origin Resource Policy configuration
 */
export interface CORPConfig {
  policy: 'same-origin' | 'same-site' | 'cross-origin';
}

/**
 * DNS Prefetch Control configuration
 */
export interface DNSPrefetchControlConfig {
  allow: boolean;
}

/**
 * Expect-CT configuration
 */
export interface ExpectCTConfig {
  maxAge: number;
  enforce: boolean;
  reportUri?: string;
}

/**
 * Frameguard configuration to prevent clickjacking
 */
export interface FrameguardConfig {
  action: 'deny' | 'sameorigin';
}

/**
 * HTTP Strict Transport Security configuration
 */
export interface HSTSConfig {
  maxAge: number;
  includeSubDomains: boolean;
  preload: boolean;
}

/**
 * Permitted Cross-Domain Policies configuration
 */
export interface PermittedCrossDomainPoliciesConfig {
  permittedPolicies: 'none' | 'master-only' | 'by-content-type' | 'all';
}

/**
 * Referrer Policy configuration
 */
export interface ReferrerPolicyConfig {
  policy: 
    | 'no-referrer'
    | 'no-referrer-when-downgrade'
    | 'origin'
    | 'origin-when-cross-origin'
    | 'same-origin'
    | 'strict-origin'
    | 'strict-origin-when-cross-origin'
    | 'unsafe-url';
}

/**
 * Encryption configuration
 */
export interface EncryptionConfig {
  defaultKey: string;
  defaultIv: string;
}

/**
 * JWT configuration
 */
export interface JWTConfig {
  defaultSecret: string;
  expiresIn: {
    access: string;
    refresh: string;
  };
}

/**
 * Password policy configuration
 */
export interface PasswordPolicyConfig {
  minLength: number;
  requireUppercase: boolean;
  requireLowercase: boolean;
  requireNumbers: boolean;
  requireSpecialChars: boolean;
  passwordHistoryLimit: number;
  maxPasswordAge: number;
}

/**
 * Session configuration
 */
export interface SessionConfig {
  name: string;
  secure: boolean;
  httpOnly: boolean;
  maxAge: number;
  sameSite: 'strict' | 'lax' | 'none';
}

/**
 * Audit logging configuration
 */
export interface AuditConfig {
  enabled: boolean;
  logLevel: string;
  sensitiveFields: string[];
}

/**
 * HIPAA compliance configuration
 */
export interface HIPAAConfig {
  enabledFor: {
    audit: boolean;
    encryption: boolean;
    authentication: boolean;
    authorization: boolean;
  };
  dataRetentionPeriod: number;
  requiredRoles: string[];
}

/**
 * Main security configuration interface
 */
export interface SecurityConfig {
  rateLimit: RateLimitConfig;
  cors: CorsConfig;
  contentSecurityPolicy: CSPConfig;
  crossOriginEmbedderPolicy: boolean;
  crossOriginOpenerPolicy: boolean;
  crossOriginResourcePolicy: CORPConfig;
  dnsPrefetchControl: DNSPrefetchControlConfig;
  expectCt: ExpectCTConfig;
  frameguard: FrameguardConfig;
  hidePoweredBy: boolean;
  hsts: HSTSConfig;
  ieNoOpen: boolean;
  noSniff: boolean;
  originAgentCluster: boolean;
  permittedCrossDomainPolicies: PermittedCrossDomainPoliciesConfig;
  referrerPolicy: ReferrerPolicyConfig;
  xssFilter: boolean;
  signedRoutes: string[];
  signatureTimeout: number;
  encryption: EncryptionConfig;
  jwt: JWTConfig;
  passwordPolicy: PasswordPolicyConfig;
  session: SessionConfig;
  audit: AuditConfig;
  hipaa: HIPAAConfig;
}

/**
 * Interface for token payload in authentication
 */
export interface TokenPayload {
  id: string;
  email: string;
  role: string;
  permissions: string[];
  iat?: number;
  exp?: number;
}

/**
 * Interface for encrypted data
 */
export interface EncryptedData {
  data: string;
  iv: string;
  tag?: string;
}

/**
 * Interface for audit log entry
 */
export interface AuditLogEntry {
  id: string;
  timestamp: Date;
  userId: string;
  action: string;
  resource: string;
  resourceId?: string;
  ipAddress: string;
  userAgent?: string;
  requestId?: string;
  details?: Record<string, any>;
  status: 'success' | 'failure' | 'warning';
}

/**
 * Interface for security-related user attributes
 */
export interface SecurityUserAttributes {
  passwordLastChanged: Date;
  passwordHistory: string[];
  failedLoginAttempts: number;
  lastFailedLogin?: Date;
  lockedUntil?: Date;
  mfaEnabled: boolean;
  mfaMethod?: 'app' | 'sms' | 'email';
  mfaSecret?: string;
}

/**
 * User role with associated permissions
 */
export interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  isHIPAARole: boolean;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Permission definition
 */
export interface Permission {
  id: string;
  name: string;
  description: string;
  resource: string;
  action: 'create' | 'read' | 'update' | 'delete' | 'manage';
  conditions?: Record<string, any>;
}

/**
 * Security event types for audit logging
 */
export enum SecurityEventType {
  LOGIN_SUCCESS = 'login.success',
  LOGIN_FAILURE = 'login.failure',
  LOGOUT = 'logout',
  PASSWORD_CHANGE = 'password.change',
  PASSWORD_RESET_REQUEST = 'password.reset.request',
  PASSWORD_RESET_COMPLETE = 'password.reset.complete',
  MFA_ENABLED = 'mfa.enabled',
  MFA_DISABLED = 'mfa.disabled',
  MFA_CHALLENGE_SUCCESS = 'mfa.challenge.success',
  MFA_CHALLENGE_FAILURE = 'mfa.challenge.failure',
  ACCOUNT_LOCKED = 'account.locked',
  ACCOUNT_UNLOCKED = 'account.unlocked',
  PERMISSION_CHANGE = 'permission.change',
  ROLE_CHANGE = 'role.change',
  USER_CREATED = 'user.created',
  USER_UPDATED = 'user.updated',
  USER_DELETED = 'user.deleted',
  IMAGE_UPLOAD = 'image.upload',
  IMAGE_DOWNLOAD = 'image.download',
  IMAGE_DELETED = 'image.deleted',
  IMAGE_ACCESSED = 'image.accessed',
  IMAGE_PROCESSED = 'image.processed',
  ANALYSIS_CREATED = 'analysis.created',
  ANALYSIS_ACCESSED = 'analysis.accessed',
  SETTINGS_CHANGED = 'settings.changed',
  API_KEY_CREATED = 'apikey.created',
  API_KEY_REVOKED = 'apikey.revoked',
  UNAUTHORIZED_ACCESS = 'unauthorized.access',
  SUSPICIOUS_ACTIVITY = 'suspicious.activity',
}
