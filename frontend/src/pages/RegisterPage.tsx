/**
 * Register Page Component
 * Handles user registration with email validation, password strength checking, and terms acceptance
 * @author Negative Space Imaging Project
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  FormControlLabel,
  Checkbox,
  Link,
  Paper,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';

/**
 * Validation errors interface
 */
interface ValidationErrors {
  email?: string;
  password?: string;
  confirmPassword?: string;
  firstName?: string;
  lastName?: string;
  terms?: string;
}

/**
 * Password strength interface
 */
interface PasswordStrength {
  score: number; // 0-4
  feedback: string;
  hasMinLength: boolean;
  hasUppercase: boolean;
  hasLowercase: boolean;
  hasNumbers: boolean;
  hasSpecialChars: boolean;
}

/**
 * Calculate password strength
 */
const calculatePasswordStrength = (password: string): PasswordStrength => {
  let score = 0;
  const checks = {
    hasMinLength: password.length >= 8,
    hasUppercase: /[A-Z]/.test(password),
    hasLowercase: /[a-z]/.test(password),
    hasNumbers: /[0-9]/.test(password),
    hasSpecialChars: /[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(password),
  };

  Object.values(checks).forEach((check) => {
    if (check) score++;
  });

  let feedback = '';
  if (score === 0) feedback = 'Very Weak';
  else if (score === 1) feedback = 'Weak';
  else if (score === 2) feedback = 'Fair';
  else if (score === 3) feedback = 'Good';
  else if (score === 4) feedback = 'Strong';
  else feedback = 'Very Strong';

  return {
    score,
    feedback,
    ...checks,
  };
};

/**
 * Register Page Component
 * Features:
 * - Email and password validation
 * - Password strength indicator
 * - Password confirmation
 * - Terms of service checkbox
 * - Form validation
 * - Success message and redirect
 * - Responsive design
 */
const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const { register, loading, isAuthenticated } = useAuth();
  const { showNotification } = useNotification();

  // Form state
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
  });

  // UI state
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [serverError, setServerError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState<PasswordStrength | null>(null);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  /**
   * Update password strength when password changes
   */
  useEffect(() => {
    if (formData.password) {
      setPasswordStrength(calculatePasswordStrength(formData.password));
    } else {
      setPasswordStrength(null);
    }
  }, [formData.password]);

  /**
   * Validate form fields
   */
  const validateForm = (): boolean => {
    const newErrors: ValidationErrors = {};

    // First name validation
    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required';
    } else if (formData.firstName.length < 2) {
      newErrors.firstName = 'First name must be at least 2 characters';
    }

    // Last name validation
    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required';
    } else if (formData.lastName.length < 2) {
      newErrors.lastName = 'Last name must be at least 2 characters';
    }

    // Email validation
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    } else if (!passwordStrength || passwordStrength.score < 2) {
      newErrors.password =
        'Password is too weak. Use uppercase, lowercase, numbers, and special characters';
    }

    // Confirm password validation
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    // Terms validation
    if (!formData.acceptTerms) {
      newErrors.terms = 'You must accept the terms of service';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handle input changes
   */
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'acceptTerms' ? checked : value,
    }));
    // Clear field error when user starts typing
    if (errors[name as keyof ValidationErrors]) {
      setErrors((prev) => ({
        ...prev,
        [name]: undefined,
      }));
    }
  };

  /**
   * Handle form submission
   */
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setServerError('');

    // Validate before submitting
    if (!validateForm()) {
      showNotification('Please fix the errors below', 'error');
      return;
    }

    try {
      await register(formData.email, formData.password, formData.firstName, formData.lastName);

      showNotification('Registration successful! Redirecting to dashboard...', 'success');
      navigate('/dashboard');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Registration failed';
      setServerError(message);
      showNotification(message, 'error');
    }
  };

  const strengthColor = passwordStrength
    ? passwordStrength.score <= 1
      ? '#f44336'
      : passwordStrength.score === 2
        ? '#ff9800'
        : passwordStrength.score === 3
          ? '#ffc107'
          : '#4caf50'
    : '#e0e0e0';

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: 2,
      }}
    >
      <Container component="main" maxWidth="sm">
        <Paper
          elevation={24}
          sx={{
            borderRadius: 2,
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            maxHeight: '90vh',
            overflowY: 'auto',
          }}
        >
          {/* Header */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 56,
              height: 56,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              marginBottom: 2,
            }}
          >
            <LockOutlinedIcon sx={{ fontSize: 32, color: 'white' }} />
          </Box>

          <Typography
            component="h1"
            variant="h4"
            sx={{
              fontWeight: 700,
              marginBottom: 0.5,
              color: '#333',
            }}
          >
            Create Account
          </Typography>

          <Typography variant="body2" color="textSecondary" sx={{ marginBottom: 3 }}>
            Join Negative Space Imaging Project
          </Typography>

          {/* Error Alert */}
          {serverError && (
            <Alert
              severity="error"
              sx={{
                width: '100%',
                marginBottom: 2,
              }}
              onClose={() => setServerError('')}
            >
              {serverError}
            </Alert>
          )}

          {/* Form */}
          <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
            {/* First Name */}
            <TextField
              margin="normal"
              required
              fullWidth
              id="firstName"
              label="First Name"
              name="firstName"
              autoComplete="given-name"
              value={formData.firstName}
              onChange={handleChange}
              error={!!errors.firstName}
              helperText={errors.firstName}
              disabled={loading}
              sx={{ mb: 2 }}
            />

            {/* Last Name */}
            <TextField
              margin="normal"
              required
              fullWidth
              id="lastName"
              label="Last Name"
              name="lastName"
              autoComplete="family-name"
              value={formData.lastName}
              onChange={handleChange}
              error={!!errors.lastName}
              helperText={errors.lastName}
              disabled={loading}
              sx={{ mb: 2 }}
            />

            {/* Email */}
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              value={formData.email}
              onChange={handleChange}
              error={!!errors.email}
              helperText={errors.email}
              disabled={loading}
              sx={{ mb: 2 }}
            />

            {/* Password */}
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              id="password"
              value={formData.password}
              onChange={handleChange}
              error={!!errors.password}
              helperText={errors.password}
              disabled={loading}
              sx={{ mb: 1 }}
            />

            {/* Password Strength Indicator */}
            {passwordStrength && (
              <Box sx={{ mb: 2 }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    mb: 1,
                  }}
                >
                  <Typography variant="caption" color="textSecondary">
                    Password Strength:
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 600,
                      color: strengthColor,
                    }}
                  >
                    {passwordStrength.feedback}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(passwordStrength.score / 5) * 100}
                  sx={{
                    backgroundColor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: strengthColor,
                    },
                    mb: 1,
                  }}
                />
                <List sx={{ py: 0 }}>
                  {[
                    { check: passwordStrength.hasMinLength, label: 'At least 8 characters' },
                    { check: passwordStrength.hasUppercase, label: 'Contains uppercase letter' },
                    { check: passwordStrength.hasLowercase, label: 'Contains lowercase letter' },
                    { check: passwordStrength.hasNumbers, label: 'Contains numbers' },
                    {
                      check: passwordStrength.hasSpecialChars,
                      label: 'Contains special characters',
                    },
                  ].map((item, index) => (
                    <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        {item.check ? (
                          <CheckCircleIcon sx={{ fontSize: 18, color: '#4caf50' }} />
                        ) : (
                          <CancelIcon sx={{ fontSize: 18, color: '#e0e0e0' }} />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={item.label}
                        sx={{
                          '& .MuiTypography-root': {
                            fontSize: '0.75rem',
                            color: item.check ? '#4caf50' : '#999',
                          },
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}

            {/* Confirm Password */}
            <TextField
              margin="normal"
              required
              fullWidth
              name="confirmPassword"
              label="Confirm Password"
              type={showConfirmPassword ? 'text' : 'password'}
              id="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              error={!!errors.confirmPassword}
              helperText={errors.confirmPassword}
              disabled={loading}
              sx={{ mb: 2 }}
            />

            {/* Terms Checkbox */}
            <FormControlLabel
              control={
                <Checkbox
                  name="acceptTerms"
                  color="primary"
                  checked={formData.acceptTerms}
                  onChange={handleChange}
                  disabled={loading}
                />
              }
              label={
                <Typography variant="body2">
                  I agree to the{' '}
                  <Link
                    href="#"
                    sx={{
                      color: '#667eea',
                      textDecoration: 'none',
                      fontWeight: 600,
                      '&:hover': { textDecoration: 'underline' },
                    }}
                  >
                    Terms of Service
                  </Link>{' '}
                  and{' '}
                  <Link
                    href="#"
                    sx={{
                      color: '#667eea',
                      textDecoration: 'none',
                      fontWeight: 600,
                      '&:hover': { textDecoration: 'underline' },
                    }}
                  >
                    Privacy Policy
                  </Link>
                </Typography>
              }
              sx={{ mb: errors.terms ? 0 : 2 }}
            />
            {errors.terms && (
              <Typography variant="caption" color="error" sx={{ display: 'block', mb: 2 }}>
                {errors.terms}
              </Typography>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{
                py: 1.5,
                mb: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: 1,
                fontWeight: 600,
                fontSize: '1rem',
                textTransform: 'none',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5568d3 0%, #673a91 100%)',
                },
                '&:disabled': {
                  background: '#ccc',
                },
              }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Create Account'}
            </Button>
          </Box>

          {/* Login Link */}
          <Typography variant="body2" color="textSecondary">
            Already have an account?{' '}
            <Link
              component={RouterLink}
              to="/login"
              sx={{
                color: '#667eea',
                fontWeight: 600,
                textDecoration: 'none',
                '&:hover': {
                  textDecoration: 'underline',
                },
              }}
            >
              Sign In
            </Link>
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
};

export default RegisterPage;
