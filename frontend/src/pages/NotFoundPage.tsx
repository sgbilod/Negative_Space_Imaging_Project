import { useNavigate } from 'react-router-dom';
import { Box, Container, Typography, Button, Stack } from '@mui/material';
import { ErrorOutline as NotFoundIcon } from '@mui/icons-material';

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <Container maxWidth="sm">
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        gap={3}
      >
        <Typography
          variant="h1"
          component="div"
          sx={{
            fontSize: { xs: 60, sm: 100, md: 120 },
            fontWeight: 'bold',
            color: 'primary.main',
          }}
        >
          404
        </Typography>

        <Typography variant="h3" component="h1" gutterBottom align="center">
          Page Not Found
        </Typography>

        <Typography variant="body1" align="center" color="textSecondary">
          The page you are looking for doesn't exist or has been moved. Please check the URL and try
          again.
        </Typography>

        <Stack direction="row" spacing={2}>
          <Button variant="contained" onClick={() => navigate(-1)}>
            Go Back
          </Button>
          <Button variant="outlined" onClick={() => navigate('/dashboard')}>
            Go to Dashboard
          </Button>
        </Stack>
      </Box>
    </Container>
  );
}
