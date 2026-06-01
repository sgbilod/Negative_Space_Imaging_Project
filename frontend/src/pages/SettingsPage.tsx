/**
 * Settings Page Component
 * User profile settings, password change, preferences, and privacy settings
 * @author Negative Space Imaging Project
 * @version 1.0.0
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  Card,
  CardContent,
  CardHeader,
  Divider,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  FormControlLabel,
  Switch,
  Grid,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Save as SaveIcon,
  Lock as LockIcon,
  Notifications as NotificationsIcon,
  VpnLock as PrivacyIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';

/**
 * Settings state interface
 */
interface SettingsState {
  profile: {
    firstName: string;
    lastName: string;
    email: string;
  };
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    emailNotifications: boolean;
    analysisAlerts: boolean;
    weeklyReport: boolean;
  };
  privacy: {
    profilePublic: boolean;
    allowSharing: boolean;
    dataRetention: '30' | '90' | '365' | 'unlimited';
  };
  password: {
    current: string;
    new: string;
    confirm: string;
  };
}

/**
 * Settings Page Component
 * Features:
 * - Profile information editing
 * - Password change
 * - Account preferences
 * - Privacy settings
 * - Notification preferences
 * - Data deletion
 * - Save and cancel buttons
 */
const SettingsPage: React.FC = () => {
  const { user } = useAuth();
  const { showNotification } = useNotification();

  // State
  const [settings, setSettings] = useState<SettingsState>({
    profile: {
      firstName: user?.firstName || '',
      lastName: user?.lastName || '',
      email: user?.email || '',
    },
    preferences: {
      theme: 'auto',
      emailNotifications: true,
      analysisAlerts: true,
      weeklyReport: false,
    },
    privacy: {
      profilePublic: false,
      allowSharing: true,
      dataRetention: '90',
    },
    password: {
      current: '',
      new: '',
      confirm: '',
    },
  });

  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState('profile');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState('');

  /**
   * Handle profile field change
   */
  const handleProfileChange = (field: string, value: string) => {
    setSettings((prev) => ({
      ...prev,
      profile: {
        ...prev.profile,
        [field]: value,
      },
    }));
    if (errors[field]) {
      setErrors((prev) => ({
        ...prev,
        [field]: '',
      }));
    }
  };

  /**
   * Handle preferences change
   */
  const handlePreferencesChange = (field: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        [field]: value,
      },
    }));
  };

  /**
   * Handle privacy change
   */
  const handlePrivacyChange = (field: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      privacy: {
        ...prev.privacy,
        [field]: value,
      },
    }));
  };

  /**
   * Handle password field change
   */
  const handlePasswordChange = (field: string, value: string) => {
    setSettings((prev) => ({
      ...prev,
      password: {
        ...prev.password,
        [field]: value,
      },
    }));
    if (errors[field]) {
      setErrors((prev) => ({
        ...prev,
        [field]: '',
      }));
    }
  };

  /**
   * Validate password change
   */
  const validatePasswordChange = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!settings.password.current) {
      newErrors.current = 'Current password is required';
    }

    if (!settings.password.new) {
      newErrors.new = 'New password is required';
    } else if (settings.password.new.length < 8) {
      newErrors.new = 'Password must be at least 8 characters';
    }

    if (!settings.password.confirm) {
      newErrors.confirm = 'Please confirm your password';
    } else if (settings.password.new !== settings.password.confirm) {
      newErrors.confirm = 'Passwords do not match';
    }

    if (settings.password.new === settings.password.current) {
      newErrors.new = 'New password must be different from current password';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handle save profile
   */
  const handleSaveProfile = async () => {
    try {
      setLoading(true);
      // API call would go here
      // await updateProfile(settings.profile);
      showNotification('Profile updated successfully', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update profile';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle change password
   */
  const handleChangePassword = async () => {
    if (!validatePasswordChange()) {
      showNotification('Please fix the errors below', 'error');
      return;
    }

    try {
      setLoading(true);
      // API call would go here
      // await changePassword(settings.password.current, settings.password.new);

      setSettings((prev) => ({
        ...prev,
        password: { current: '', new: '', confirm: '' },
      }));

      showNotification('Password changed successfully', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to change password';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle save preferences
   */
  const handleSavePreferences = async () => {
    try {
      setLoading(true);
      // API call would go here
      // await updatePreferences(settings.preferences);
      showNotification('Preferences updated successfully', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update preferences';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle save privacy
   */
  const handleSavePrivacy = async () => {
    try {
      setLoading(true);
      // API call would go here
      // await updatePrivacy(settings.privacy);
      showNotification('Privacy settings updated successfully', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update privacy settings';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle delete account
   */
  const handleDeleteAccount = async () => {
    if (deleteConfirm !== 'DELETE') {
      showNotification('Please type "DELETE" to confirm', 'error');
      return;
    }

    try {
      setLoading(true);
      // API call would go here
      // await deleteAccount();
      showNotification('Account deleted', 'success');
      // Redirect to login
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete account';
      showNotification(message, 'error');
    } finally {
      setLoading(false);
      setDeleteDialogOpen(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      {/* Header */}
      <Typography
        variant="h3"
        component="h1"
        sx={{
          fontWeight: 700,
          marginBottom: 4,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Settings
      </Typography>

      {/* Tabs */}
      <Box sx={{ display: 'flex', gap: 1, mb: 3, overflowX: 'auto' }}>
        {[
          { id: 'profile', label: 'Profile', icon: '👤' },
          { id: 'preferences', label: 'Preferences', icon: '⚙️' },
          { id: 'privacy', label: 'Privacy', icon: '🔒' },
          { id: 'password', label: 'Password', icon: '🔑' },
        ].map((tab) => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? 'contained' : 'outlined'}
            onClick={() => setActiveTab(tab.id)}
            sx={{
              background:
                activeTab === tab.id
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : 'transparent',
              borderColor: '#667eea',
              color: activeTab === tab.id ? 'white' : '#667eea',
              textTransform: 'none',
              fontWeight: 600,
              whiteSpace: 'nowrap',
            }}
          >
            {tab.icon} {tab.label}
          </Button>
        ))}
      </Box>

      {/* Profile Settings */}
      {activeTab === 'profile' && (
        <Card>
          <CardHeader title="Profile Information" />
          <Divider />
          <CardContent sx={{ pt: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="First Name"
                  value={settings.profile.firstName}
                  onChange={(e) => handleProfileChange('firstName', e.target.value)}
                  error={!!errors.firstName}
                  helperText={errors.firstName}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Last Name"
                  value={settings.profile.lastName}
                  onChange={(e) => handleProfileChange('lastName', e.target.value)}
                  error={!!errors.lastName}
                  helperText={errors.lastName}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Email Address"
                  type="email"
                  value={settings.profile.email}
                  disabled
                  helperText="Email cannot be changed"
                />
              </Grid>
            </Grid>

            <Box sx={{ display: 'flex', gap: 2, mt: 4 }}>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveProfile}
                disabled={loading}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  textTransform: 'none',
                }}
              >
                {loading ? <CircularProgress size={24} /> : 'Save Changes'}
              </Button>
              <Button variant="outlined" sx={{ borderColor: '#667eea', color: '#667eea' }}>
                Cancel
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Preferences Settings */}
      {activeTab === 'preferences' && (
        <Card>
          <CardHeader title="Preferences" />
          <Divider />
          <CardContent sx={{ pt: 3 }}>
            <List>
              <ListItem>
                <ListItemText
                  primary="Email Notifications"
                  secondary="Receive email updates about your analyses"
                />
                <Switch
                  checked={settings.preferences.emailNotifications}
                  onChange={(e) => handlePreferencesChange('emailNotifications', e.target.checked)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Analysis Alerts"
                  secondary="Get notified when analysis completes"
                />
                <Switch
                  checked={settings.preferences.analysisAlerts}
                  onChange={(e) => handlePreferencesChange('analysisAlerts', e.target.checked)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Weekly Report"
                  secondary="Receive a weekly summary of your activity"
                />
                <Switch
                  checked={settings.preferences.weeklyReport}
                  onChange={(e) => handlePreferencesChange('weeklyReport', e.target.checked)}
                />
              </ListItem>
            </List>

            <Box sx={{ display: 'flex', gap: 2, mt: 4 }}>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSavePreferences}
                disabled={loading}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  textTransform: 'none',
                }}
              >
                {loading ? <CircularProgress size={24} /> : 'Save Changes'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Privacy Settings */}
      {activeTab === 'privacy' && (
        <Card>
          <CardHeader title="Privacy Settings" />
          <Divider />
          <CardContent sx={{ pt: 3 }}>
            <List>
              <ListItem>
                <ListItemText
                  primary="Public Profile"
                  secondary="Allow others to see your profile"
                />
                <Switch
                  checked={settings.privacy.profilePublic}
                  onChange={(e) => handlePrivacyChange('profilePublic', e.target.checked)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Allow Sharing"
                  secondary="Allow your analysis results to be shared"
                />
                <Switch
                  checked={settings.privacy.allowSharing}
                  onChange={(e) => handlePrivacyChange('allowSharing', e.target.checked)}
                />
              </ListItem>
            </List>

            <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 3, mb: 1 }}>
              Data Retention
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              How long to keep your analysis data
            </Typography>

            {(
              [
                { value: '30', label: '30 days' },
                { value: '90', label: '90 days' },
                { value: '365', label: '1 year' },
                { value: 'unlimited', label: 'Unlimited' },
              ] as const
            ).map((option) => (
              <FormControlLabel
                key={option.value}
                control={
                  <Switch
                    checked={settings.privacy.dataRetention === option.value}
                    onChange={() => handlePrivacyChange('dataRetention', option.value)}
                  />
                }
                label={option.label}
                sx={{ display: 'block', mb: 1 }}
              />
            ))}

            <Box sx={{ display: 'flex', gap: 2, mt: 4 }}>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSavePrivacy}
                disabled={loading}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  textTransform: 'none',
                }}
              >
                {loading ? <CircularProgress size={24} /> : 'Save Changes'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Password Change */}
      {activeTab === 'password' && (
        <Card>
          <CardHeader title="Change Password" />
          <Divider />
          <CardContent sx={{ pt: 3 }}>
            <Alert severity="info" sx={{ mb: 3 }}>
              Use a strong password with at least 8 characters, including uppercase, lowercase,
              numbers, and special characters.
            </Alert>

            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="Current Password"
                  value={settings.password.current}
                  onChange={(e) => handlePasswordChange('current', e.target.value)}
                  error={!!errors.current}
                  helperText={errors.current}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="New Password"
                  value={settings.password.new}
                  onChange={(e) => handlePasswordChange('new', e.target.value)}
                  error={!!errors.new}
                  helperText={errors.new}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="Confirm New Password"
                  value={settings.password.confirm}
                  onChange={(e) => handlePasswordChange('confirm', e.target.value)}
                  error={!!errors.confirm}
                  helperText={errors.confirm}
                />
              </Grid>
            </Grid>

            <Box sx={{ display: 'flex', gap: 2, mt: 4 }}>
              <Button
                variant="contained"
                startIcon={<LockIcon />}
                onClick={handleChangePassword}
                disabled={loading}
                sx={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  textTransform: 'none',
                }}
              >
                {loading ? <CircularProgress size={24} /> : 'Change Password'}
              </Button>
            </Box>

            {/* Danger Zone */}
            <Divider sx={{ my: 4 }} />

            <Box sx={{ backgroundColor: '#fff3e0', p: 2, borderRadius: 1, mb: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#e65100', mb: 1 }}>
                ⚠️ Danger Zone
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Permanently delete your account and all associated data.
              </Typography>
            </Box>

            <Button
              variant="outlined"
              color="error"
              startIcon={<DeleteIcon />}
              onClick={() => setDeleteDialogOpen(true)}
              sx={{ borderColor: '#f44336', color: '#f44336' }}
            >
              Delete Account
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Delete Account Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete Account</DialogTitle>
        <DialogContent>
          <Alert severity="error" sx={{ mb: 2 }}>
            This action cannot be undone. All your data will be permanently deleted.
          </Alert>
          <TextField
            fullWidth
            placeholder='Type "DELETE" to confirm'
            value={deleteConfirm}
            onChange={(e) => setDeleteConfirm(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleDeleteAccount}
            disabled={deleteConfirm !== 'DELETE' || loading}
          >
            Delete Account
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default SettingsPage;
