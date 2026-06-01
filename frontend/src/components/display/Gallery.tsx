import React, { useState } from 'react';
import {
  Grid,
  Box,
  ImageListItem,
  Modal,
  IconButton,
  useMediaQuery,
  useTheme,
  Paper,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';

export interface GalleryImage {
  id: string | number;
  src: string;
  alt?: string;
  title?: string;
  description?: string;
  thumbnail?: string;
}

interface GalleryProps {
  images: GalleryImage[];
  columns?: number;
  columnsMobile?: number;
  columnsMedium?: number;
  onClick?: (image: GalleryImage, index: number) => void;
  loading?: boolean;
  empty?: React.ReactNode;
}

/**
 * Gallery Component
 *
 * Responsive image gallery with:
 * - Responsive grid layout
 * - Lightbox modal preview
 * - Keyboard navigation
 * - Touch-friendly controls
 * - Customizable columns
 */
const Gallery: React.FC<GalleryProps> = ({
  images,
  columns = 4,
  columnsMobile = 1,
  columnsMedium = 2,
  onClick,
  loading = false,
  empty,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isMedium = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const currentColumns = isMobile ? columnsMobile : isMedium ? columnsMedium : columns;

  const handleImageClick = (image: GalleryImage, index: number) => {
    setSelectedIndex(index);
    if (onClick) {
      onClick(image, index);
    }
  };

  const handleNextImage = () => {
    if (selectedIndex !== null && selectedIndex < images.length - 1) {
      setSelectedIndex(selectedIndex + 1);
    }
  };

  const handlePreviousImage = () => {
    if (selectedIndex !== null && selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowRight') handleNextImage();
    if (event.key === 'ArrowLeft') handlePreviousImage();
    if (event.key === 'Escape') setSelectedIndex(null);
  };

  const selectedImage = selectedIndex !== null ? images[selectedIndex] : null;

  if (images.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        {empty || <Typography color="text.secondary">No images available</Typography>}
      </Paper>
    );
  }

  return (
    <>
      {/* Gallery Grid */}
      <Grid container spacing={2}>
        {images.map((image, index) => (
          <Grid
            item
            xs={12 / currentColumns}
            key={image.id}
            onClick={() => handleImageClick(image, index)}
            sx={{
              cursor: 'pointer',
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 1,
            }}
          >
            <Box
              component="img"
              src={image.thumbnail || image.src}
              alt={image.alt || image.title || `Gallery item ${index}`}
              sx={{
                width: '100%',
                height: 200,
                objectFit: 'cover',
                transition: theme.transitions.create(['transform', 'filter'], {
                  duration: theme.transitions.duration.shorter,
                }),
                '&:hover': {
                  transform: 'scale(1.08)',
                  filter: 'brightness(0.85)',
                },
              }}
            />

            {/* Image Overlay */}
            {(image.title || image.description) && (
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: 'linear-gradient(to top, rgba(0,0,0,0.8), transparent)',
                  color: '#fff',
                  p: 1,
                  minHeight: 60,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'flex-end',
                }}
              >
                {image.title && (
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {image.title}
                  </Typography>
                )}
                {image.description && (
                  <Typography variant="caption">{image.description}</Typography>
                )}
              </Box>
            )}
          </Grid>
        ))}
      </Grid>

      {/* Lightbox Modal */}
      <Modal
        open={selectedIndex !== null}
        onClose={() => setSelectedIndex(null)}
        onKeyDown={handleKeyDown}
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
        }}
      >
        <Box
          sx={{
            position: 'relative',
            width: '100%',
            height: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            flexDirection: 'column',
          }}
          onKeyDown={handleKeyDown}
          tabIndex={0}
        >
          {selectedImage && (
            <>
              {/* Main Image */}
              <Box
                component="img"
                src={selectedImage.src}
                alt={selectedImage.alt || selectedImage.title}
                sx={{
                  maxWidth: '90%',
                  maxHeight: '80%',
                  objectFit: 'contain',
                }}
              />

              {/* Image Info */}
              {(selectedImage.title || selectedImage.description) && (
                <Box sx={{ mt: 2, textAlign: 'center', color: '#fff', px: 2 }}>
                  {selectedImage.title && (
                    <Typography variant="h6">{selectedImage.title}</Typography>
                  )}
                  {selectedImage.description && (
                    <Typography variant="body2">{selectedImage.description}</Typography>
                  )}
                  <Typography variant="caption">
                    {selectedIndex! + 1} / {images.length}
                  </Typography>
                </Box>
              )}

              {/* Navigation Controls */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 20,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  display: 'flex',
                  gap: 2,
                }}
              >
                <IconButton
                  onClick={handlePreviousImage}
                  disabled={selectedIndex === 0}
                  sx={{ color: '#fff' }}
                >
                  <NavigateBeforeIcon />
                </IconButton>
                <IconButton onClick={() => setSelectedIndex(null)} sx={{ color: '#fff' }}>
                  <CloseIcon />
                </IconButton>
                <IconButton
                  onClick={handleNextImage}
                  disabled={selectedIndex === images.length - 1}
                  sx={{ color: '#fff' }}
                >
                  <NavigateNextIcon />
                </IconButton>
              </Box>

              {/* Close Button */}
              <IconButton
                onClick={() => setSelectedIndex(null)}
                sx={{
                  position: 'absolute',
                  top: 20,
                  right: 20,
                  color: '#fff',
                }}
              >
                <CloseIcon />
              </IconButton>
            </>
          )}
        </Box>
      </Modal>
    </>
  );
};

export default Gallery;
