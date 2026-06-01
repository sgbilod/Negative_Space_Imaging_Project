import React from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Box,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  AppBarProps,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';
import { useAuth } from '../../hooks';
import { useNavigate } from 'react-router-dom';

interface NavigationBarProps extends AppBarProps {
  onMenuToggle?: () => void;
}

/**
 * NavigationBar Component
 *
 * Top navigation bar with:
 * - Menu toggle button
 * - Application title
 * - User profile menu
 * - Responsive design
 */
const NavigationBar: React.FC<NavigationBarProps> = ({ onMenuToggle, ...appBarProps }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const menuOpen = Boolean(anchorEl);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>): void => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = (): void => {
    setAnchorEl(null);
  };

  const handleLogout = (): void => {
    handleMenuClose();
    logout();
    navigate('/login');
  };

  const handleSettings = (): void => {
    handleMenuClose();
    navigate('/settings');
  };

  const userInitials = user?.email
    ? user.email
        .split('@')[0]
        .split('.')
        .map((n: string) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : 'U';

  return (
    <AppBar position="fixed" {...appBarProps}>
      <Toolbar>
        {/* Menu Toggle Button */}
        <IconButton color="inherit" aria-label="toggle menu" onClick={onMenuToggle} sx={{ mr: 2 }}>
          <MenuIcon />
        </IconButton>

        {/* Application Title */}
        <Typography
          variant="h6"
          sx={{
            flexGrow: 1,
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          Negative Space Imaging
        </Typography>

        {/* User Profile Menu */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" sx={{ display: { xs: 'none', sm: 'block' } }}>
            {user?.email?.split('@')[0] || 'User'}
          </Typography>
          <IconButton
            onClick={handleMenuOpen}
            size="small"
            sx={{ ml: 2 }}
            aria-controls={menuOpen ? 'profile-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={menuOpen ? 'true' : undefined}
          >
            <Avatar
              sx={{
                width: 36,
                height: 36,
                backgroundColor: 'secondary.main',
                fontSize: '0.875rem',
                fontWeight: 600,
              }}
            >
              {userInitials}
            </Avatar>
          </IconButton>
        </Box>

        {/* Profile Menu Dropdown */}
        <Menu
          id="profile-menu"
          anchorEl={anchorEl}
          open={menuOpen}
          onClose={handleMenuClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          <MenuItem disabled>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              {user?.email || 'No email'}
            </Typography>
          </MenuItem>
          <Divider />
          <MenuItem onClick={handleSettings}>
            <SettingsIcon sx={{ mr: 1 }} fontSize="small" />
            Settings
          </MenuItem>
          <Divider />
          <MenuItem onClick={handleLogout}>
            <LogoutIcon sx={{ mr: 1 }} fontSize="small" />
            Logout
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default NavigationBar;
