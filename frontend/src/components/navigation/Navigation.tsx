import { useState, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Breadcrumbs,
  Link as MuiLink,
  useMediaQuery,
  useTheme,
  Avatar,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  Image as ImageIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  AdminPanelSettings as AdminIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useAuth } from '../../hooks';
import { useUIStore } from '../../store/uiStore';
import { getBreadcrumbs, isAdminRoute } from '../../router/routes';

const routeIcons: Record<string, React.ReactNode> = {
  '/dashboard': <DashboardIcon />,
  '/upload': <UploadIcon />,
  '/images': <ImageIcon />,
  '/analysis': <AnalyticsIcon />,
  '/settings': <SettingsIcon />,
  '/admin': <AdminIcon />,
};

export default function Navigation() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const navigate = useNavigate();
  const { logout, user } = useAuth();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const breadcrumbs = useMemo(() => getBreadcrumbs(location.pathname), [location.pathname]);

  const navigationItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/upload', label: 'Upload', icon: <UploadIcon /> },
    { path: '/analysis', label: 'Analysis', icon: <AnalyticsIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
    ...(user?.roles?.includes('admin')
      ? [{ path: '/admin', label: 'Admin', icon: <AdminIcon /> }]
      : []),
  ];

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleMenuClose();
    logout();
    navigate('/login');
  };

  const handleNavClick = (path: string) => {
    navigate(path);
    if (isMobile) {
      toggleSidebar();
    }
  };

  const sidebarContent = (
    <Box sx={{ width: 250, overflow: 'auto' }}>
      <Box sx={{ p: 2, textAlign: 'center', borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          NSIP
        </Typography>
      </Box>

      <List>
        {navigationItems.map((item) => (
          <ListItem
            button
            key={item.path}
            selected={location.pathname === item.path || location.pathname.startsWith(item.path)}
            onClick={() => handleNavClick(item.path)}
            sx={{
              '&.Mui-selected': {
                backgroundColor: 'action.selected',
                '&:hover': {
                  backgroundColor: 'action.selected',
                },
              },
            }}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      {/* App Bar */}
      <AppBar position="sticky" elevation={1}>
        <Toolbar>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleSidebar}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
            Negative Space Imaging
          </Typography>

          {/* User Menu */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Avatar sx={{ cursor: 'pointer', bgcolor: 'secondary.main' }} onClick={handleMenuOpen}>
              {user?.email?.charAt(0).toUpperCase() || 'U'}
            </Avatar>
            <Typography variant="body2" sx={{ display: { xs: 'none', sm: 'block' } }}>
              {user?.email || 'User'}
            </Typography>
          </Box>

          {/* User Menu Dropdown */}
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem onClick={() => handleNavClick('/settings')}>
              <SettingsIcon sx={{ mr: 1 }} />
              Settings
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <LogoutIcon sx={{ mr: 1 }} />
              Logout
            </MenuItem>
          </Menu>
        </Toolbar>

        {/* Breadcrumbs */}
        {breadcrumbs.length > 1 && (
          <Box sx={{ px: 2, pb: 1 }}>
            <Breadcrumbs aria-label="breadcrumb">
              {breadcrumbs.map((crumb, index) => (
                <MuiLink
                  key={crumb.path}
                  component="button"
                  color="inherit"
                  onClick={() => navigate(crumb.path)}
                  underline="hover"
                  sx={{
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    '&:hover': { textDecoration: 'underline' },
                  }}
                >
                  {crumb.title}
                </MuiLink>
              ))}
            </Breadcrumbs>
          </Box>
        )}
      </AppBar>

      {/* Sidebar Drawer */}
      {isMobile ? (
        <Drawer
          anchor="left"
          open={sidebarOpen}
          onClose={toggleSidebar}
          sx={{ display: { xs: 'block', md: 'none' } }}
        >
          {sidebarContent}
        </Drawer>
      ) : (
        <Box
          sx={{
            width: 250,
            position: 'fixed',
            left: 0,
            top: 112,
            height: 'calc(100vh - 112px)',
            borderRight: `1px solid ${theme.palette.divider}`,
            display: { xs: 'none', md: 'block' },
            overflow: 'auto',
            backgroundColor: 'background.paper',
          }}
        >
          {sidebarContent}
        </Box>
      )}
    </>
  );
}
