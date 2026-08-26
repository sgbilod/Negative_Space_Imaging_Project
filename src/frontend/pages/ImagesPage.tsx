/**
 * ImagesPage Component
 *
 * Gallery page displaying all uploaded images with pagination and search.
 *
 * @component
 */

import React, { useEffect, useState } from 'react';
import { Spinner } from '../components/Spinner';
import { ImageCard } from '../components/ImageCard';
import { Input } from '../components/Input';

/**
 * ImagesPage component
 */
export const ImagesPage: React.FC = () => {
  const [images, setImages] = useState<any[]>([]);
  const [filteredImages, setFilteredImages] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState('');
  const itemsPerPage = 12;

  useEffect(() => {
    const fetchImages = async () => {
      try {
        setLoading(true);

        // TODO: Replace with actual API call
        const mockImages = Array.from({ length: 36 }, (_, i) => ({
          id: (i + 1).toString(),
          name: `Image ${i + 1}`,
          imageUrl: `https://via.placeholder.com/300x200?text=Image${i + 1}`,
          uploadedAt: new Date(Date.now() - i * 86400000).toISOString(),
          fileSize: Math.random() * 5000000,
        }));

        setImages(mockImages);
        setFilteredImages(mockImages);
      } catch (err) {
        setError('Failed to load images');
      } finally {
        setLoading(false);
      }
    };

    fetchImages();
  }, []);

  useEffect(() => {
    if (searchQuery.trim()) {
      const filtered = images.filter((image) =>
        image.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredImages(filtered);
    } else {
      setFilteredImages(images);
    }
    setCurrentPage(1);
  }, [searchQuery, images]);

  const handleDelete = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this image?')) {
      try {
        setImages((prev) => prev.filter((img) => img.id !== id));
      } catch (err) {
        setError('Failed to delete image');
      }
    }
  };

  const totalPages = Math.ceil(filteredImages.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedImages = filteredImages.slice(startIndex, startIndex + itemsPerPage);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900">Image Gallery</h1>
        <p className="text-gray-600 mt-2">Browse and manage all your uploaded images.</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Search Bar */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <Input
          label="Search Images"
          placeholder="Search by image name..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <p className="text-sm text-gray-600 mt-2">
          Found {filteredImages.length} image{filteredImages.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Image Grid */}
      {paginatedImages.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {paginatedImages.map((image) => (
              <ImageCard
                key={image.id}
                id={image.id}
                name={image.name}
                imageUrl={image.imageUrl}
                uploadedAt={image.uploadedAt}
                fileSize={image.fileSize}
                onDelete={handleDelete}
              />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 mt-8">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>

              <div className="flex gap-2">
                {Array.from({ length: totalPages }).map((_, i) => (
                  <button
                    key={i + 1}
                    onClick={() => setCurrentPage(i + 1)}
                    className={`px-3 py-2 rounded-lg ${
                      currentPage === i + 1
                        ? 'bg-blue-600 text-white'
                        : 'border border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>

              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-12 bg-white rounded-lg">
          <p className="text-gray-600 text-lg">No images found</p>
          <p className="text-gray-500 mt-2">Upload an image to get started</p>
        </div>
      )}
    </div>
  );
};

export default ImagesPage;
