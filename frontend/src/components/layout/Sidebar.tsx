import React from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  useTheme,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SecurityIcon from '@mui/icons-material/Security';
import SettingsIcon from '@mui/icons-material/Settings';
import PeopleIcon from '@mui/icons-material/People';
import LogoutIcon from '@mui/icons-material/Logout';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  width: number;
  mobile?: boolean;
}

interface NavItem {
  label: string;
  path: string;
  icon: React.ComponentType<any>;
  requiredRoles?: string[];
}

/**
 * Sidebar Component
 *
 * Navigation sidebar with:
 * - Main navigation items
 * - Role-based access control
 * - Active route highlighting
 * - Collapsible drawer option
 */
const Sidebar: React.FC<SidebarProps> = ({ open, onClose, width, mobile = false }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const { user, logout } = useAuth();

  const navigationItems: NavItem[] = [
    {
      label: 'Dashboard',
      path: '/dashboard',
      icon: DashboardIcon,
    },
    {
      label: 'Upload',
      path: '/upload',
      icon: CloudUploadIcon,
    },
    {
      label: 'Analysis',
      path: '/analysis',
      icon: AnalyticsIcon,
    },
    {
      label: 'Security Monitor',
      path: '/security',
      icon: SecurityIcon,
      requiredRoles: ['admin', 'operator'],
    },
    {
      label: 'User Management',
      path: '/users',
      icon: PeopleIcon,
      requiredRoles: ['admin'],
    },
  ];

  const canAccessItem = (item: NavItem): boolean => {
    if (!item.requiredRoles) return true;
    return item.requiredRoles.some((role) => user?.roles?.includes(role));
  };

  const handleNavigate = (path: string): void => {
    navigate(path);
    if (mobile) onClose();
  };

  const handleLogout = (): void => {
    logout();
    navigate('/login');
    if (mobile) onClose();
  };

  const sidebarContent = (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Sidebar Header */}
      <Box sx={{ p: 2 }}>
        <Typography
          variant="subtitle1"
          sx={{
            fontWeight: 600,
            color: 'primary.main',
            mb: 1,
          }}
        >
          Navigation
        </Typography>
        <Divider />
      </Box>

      {/* Navigation Items */}
      <List sx={{ flex: 1, px: 0 }}>
        {navigationItems.map((item) => {
          if (!canAccessItem(item)) return null;

          const isActive = location.pathname.startsWith(item.path);
          const Icon = item.icon;

          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                onClick={() => handleNavigate(item.path)}
                sx={{
                  backgroundColor: isActive ? theme.palette.action.selected : 'transparent',
                  borderLeft: isActive ? `3px solid ${theme.palette.primary.main}` : 'none',
                  pl: isActive ? 1 : 2,
                  '&:hover': {
                    backgroundColor: theme.palette.action.hover,
                  },
                  transition: theme.transitions.create(['background-color', 'padding'], {
                    duration: theme.transitions.duration.shorter,
                  }),
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 40,
                    color: isActive ? 'primary.main' : 'inherit',
                  }}
                >
                  <Icon />
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    variant: 'body2',
                    fontWeight: isActive ? 600 : 500,
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {/* Sidebar Footer */}
      <Box sx={{ borderTop: `1px solid ${theme.palette.divider}` }}>
        <List sx={{ px: 0 }}>
          <ListItem disablePadding>
            <ListItemButton
              onClick={() => handleNavigate('/settings')}
              sx={{
                backgroundColor:
                  location.pathname === '/settings' ? theme.palette.action.selected : 'transparent',
                borderLeft:
                  location.pathname === '/settings'
                    ? `3px solid ${theme.palette.primary.main}`
                    : 'none',
                pl: location.pathname === '/settings' ? 1 : 2,
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                <SettingsIcon />
              </ListItemIcon>
              <ListItemText primary="Settings" primaryTypographyProps={{ variant: 'body2' }} />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton
              onClick={handleLogout}
              sx={{
                '&:hover': {
                  backgroundColor: theme.palette.error.light || 'rgba(244, 67, 54, 0.1)',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40, color: 'error.main' }}>
                <LogoutIcon />
              </ListItemIcon>
              <ListItemText
                primary="Logout"
                primaryTypographyProps={{ variant: 'body2', color: 'error' }}
              />
            </ListItemButton>
          </ListItem>
        </List>
      </Box>
    </Box>
  );

  if (mobile) {
    return <>{sidebarContent}</>;
  }

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: width,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: width,
          boxSizing: 'border-box',
          mt: '64px',
          height: 'calc(100vh - 64px)',
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
        },
      }}
    >
      {sidebarContent}
    </Drawer>
  );
};

export default Sidebar;
