import { useNavigate } from 'react-router-dom';
import { Box, Container, Typography, Button, Stack } from '@mui/material';
import { ErrorOutline } from '@mui/icons-material';

export default function ErrorPage() {
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
        <ErrorOutline sx={{ fontSize: 80, color: 'error.main' }} />

        <Typography variant="h3" component="h1" gutterBottom align="center">
          Something Went Wrong
        </Typography>

        <Typography variant="body1" align="center" color="textSecondary">
          An unexpected error occurred. Please try again or contact support if the problem persists.
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
