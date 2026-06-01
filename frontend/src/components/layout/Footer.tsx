import React from 'react';
import { Box, Container, Typography, Link, Grid, useTheme, useMediaQuery } from '@mui/material';
import CopyrightIcon from '@mui/icons-material/Copyright';

interface FooterProps {
  companyName?: string;
  year?: number;
}

/**
 * Footer Component
 *
 * Application footer with:
 * - Copyright information
 * - Navigation links
 * - Contact information
 * - Responsive layout
 */
const Footer: React.FC<FooterProps> = ({
  companyName = 'Negative Space Imaging',
  year = new Date().getFullYear(),
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const footerLinks = [
    { label: 'Privacy Policy', href: '/privacy' },
    { label: 'Terms of Service', href: '/terms' },
    { label: 'Contact Us', href: '/contact' },
    { label: 'Documentation', href: '/docs' },
    { label: 'API Reference', href: '/api' },
  ];

  const socialLinks = [
    { label: 'GitHub', href: 'https://github.com' },
    { label: 'Twitter', href: 'https://twitter.com' },
    { label: 'LinkedIn', href: 'https://linkedin.com' },
  ];

  return (
    <Box
      component="footer"
      sx={{
        bgcolor: theme.palette.background.paper,
        borderTop: `1px solid ${theme.palette.divider}`,
        py: { xs: 3, sm: 4, md: 6 },
        mt: 'auto',
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={isMobile ? 3 : 4}>
          {/* About Section */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              {companyName}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
              Transforming visual data through advanced imaging technology and AI-driven analysis.
            </Typography>
          </Grid>

          {/* Quick Links */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Quick Links
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {footerLinks.map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  underline="hover"
                  sx={{
                    color: 'text.secondary',
                    fontSize: 'body2.fontSize',
                    transition: theme.transitions.create('color'),
                    '&:hover': {
                      color: 'primary.main',
                    },
                  }}
                >
                  {link.label}
                </Link>
              ))}
            </Box>
          </Grid>

          {/* Resources */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Resources
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Link
                href="/documentation"
                underline="hover"
                sx={{
                  color: 'text.secondary',
                  fontSize: 'body2.fontSize',
                  transition: theme.transitions.create('color'),
                  '&:hover': {
                    color: 'primary.main',
                  },
                }}
              >
                Documentation
              </Link>
              <Link
                href="/support"
                underline="hover"
                sx={{
                  color: 'text.secondary',
                  fontSize: 'body2.fontSize',
                  transition: theme.transitions.create('color'),
                  '&:hover': {
                    color: 'primary.main',
                  },
                }}
              >
                Support
              </Link>
              <Link
                href="/roadmap"
                underline="hover"
                sx={{
                  color: 'text.secondary',
                  fontSize: 'body2.fontSize',
                  transition: theme.transitions.create('color'),
                  '&:hover': {
                    color: 'primary.main',
                  },
                }}
              >
                Roadmap
              </Link>
            </Box>
          </Grid>

          {/* Connect */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Connect
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {socialLinks.map((link) => (
                <Link
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  underline="hover"
                  sx={{
                    color: 'text.secondary',
                    fontSize: 'body2.fontSize',
                    transition: theme.transitions.create('color'),
                    '&:hover': {
                      color: 'primary.main',
                    },
                  }}
                >
                  {link.label}
                </Link>
              ))}
            </Box>
          </Grid>
        </Grid>

        {/* Divider */}
        <Box sx={{ my: 4, borderTop: `1px solid ${theme.palette.divider}` }} />

        {/* Copyright Section */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: isMobile ? 'column' : 'row',
            justifyContent: 'space-between',
            alignItems: isMobile ? 'flex-start' : 'center',
            gap: 2,
          }}
        >
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
          >
            <CopyrightIcon sx={{ fontSize: 16 }} />
            {year} {companyName}. All rights reserved.
          </Typography>

          <Typography variant="caption" color="text.secondary">
            v1.0.0 • Built with React & Material-UI
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
