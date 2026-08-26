import React, { ReactNode, useState } from 'react';
import { Box, Container, Drawer, useMediaQuery, useTheme } from '@mui/material';
import NavigationBar from './NavigationBar';
import Sidebar from './Sidebar';
import Footer from './Footer';

interface MainLayoutProps {
  children: ReactNode;
}

/**
 * MainLayout Component
 *
 * Wrapper component that provides consistent layout structure across the application.
 * Includes responsive navigation bar, sidebar, and footer.
 *
 * Features:
 * - Responsive design (sidebar toggles on mobile)
 * - Collapsible sidebar on desktop
 * - Fixed navigation bar
 * - Sticky footer
 * - Content container with proper spacing
 */
const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleSidebarToggle = (): void => {
    if (isMobile) {
      setMobileMenuOpen(!mobileMenuOpen);
    } else {
      setSidebarOpen(!sidebarOpen);
    }
  };

  const handleMobileMenuClose = (): void => {
    setMobileMenuOpen(false);
  };

  const sidebarWidth = 280;
  const navBarHeight = 64;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Navigation Bar */}
      <NavigationBar
        onMenuToggle={handleSidebarToggle}
        elevation={1}
        sx={{ zIndex: theme.zIndex.drawer + 1 }}
      />

      {/* Main Content Area */}
      <Box sx={{ display: 'flex', flex: 1, mt: `${navBarHeight}px` }}>
        {/* Sidebar - Desktop */}
        {!isMobile && (
          <Sidebar open={sidebarOpen} onClose={handleSidebarToggle} width={sidebarWidth} />
        )}

        {/* Sidebar - Mobile Drawer */}
        {isMobile && (
          <Drawer
            anchor="left"
            open={mobileMenuOpen}
            onClose={handleMobileMenuClose}
            sx={{
              display: { xs: 'block', md: 'none' },
              '& .MuiDrawer-paper': {
                width: sidebarWidth,
                mt: `${navBarHeight}px`,
              },
            }}
          >
            <Sidebar open={true} onClose={handleMobileMenuClose} width={sidebarWidth} mobile />
          </Drawer>
        )}

        {/* Content */}
        <Box
          sx={{
            flex: 1,
            overflow: 'auto',
            transition: theme.transitions.create('margin', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
            marginLeft: !isMobile && sidebarOpen ? `${sidebarWidth}px` : 0,
          }}
        >
          <Container maxWidth="lg" sx={{ py: 4 }}>
            {children}
          </Container>
        </Box>
      </Box>

      {/* Footer */}
      <Footer />
    </Box>
  );
};

export default MainLayout;
