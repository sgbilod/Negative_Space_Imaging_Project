/**
 * ImageDetailPage Component
 *
 * Detailed view of a single image with metadata and analysis results.
 *
 * @component
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Spinner } from '../components/Spinner';
import { Button } from '../components/Button';

/**
 * ImageDetailPage component
 */
export const ImageDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [image, setImage] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const fetchImageDetail = async () => {
      try {
        setLoading(true);
        // TODO: Replace with actual API call
        const mockImage = {
          id,
          name: 'Sample Image',
          imageUrl: 'https://via.placeholder.com/600x400',
          uploadedAt: new Date().toISOString(),
          fileSize: 2048576,
          dimensions: { width: 1920, height: 1440 },
          format: 'JPG',
          description: 'A detailed landscape image',
          analysis: {
            id: 'analysis-1',
            negativeSpacePercentage: 65,
            regionsCount: 12,
            processingTime: 2.34,
            analyzedAt: new Date().toISOString(),
          },
        };
        setImage(mockImage);
      } catch (err) {
        setError('Failed to load image details');
      } finally {
        setLoading(false);
      }
    };

    fetchImageDetail();
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this image?')) {
      return;
    }

    try {
      setDeleting(true);
      // TODO: Replace with actual API call
      navigate('/images');
    } catch (err) {
      setError('Failed to delete image');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!image) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 text-lg">Image not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-gray-900">{image.name}</h1>
          <p className="text-gray-600 mt-2">Uploaded on {new Date(image.uploadedAt).toLocaleDateString()}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => navigate('/images')}>
            Back
          </Button>
          <Button variant="danger" loading={deleting} onClick={handleDelete}>
            Delete
          </Button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Image Preview */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <img src={image.imageUrl} alt={image.name} className="w-full h-auto" />
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Image Info Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Image Information</h3>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-gray-600">Format</p>
                <p className="text-gray-900 font-semibold">{image.format}</p>
              </div>
              <div>
                <p className="text-gray-600">Dimensions</p>
                <p className="text-gray-900 font-semibold">
                  {image.dimensions.width} x {image.dimensions.height} px
                </p>
              </div>
              <div>
                <p className="text-gray-600">File Size</p>
                <p className="text-gray-900 font-semibold">
                  {(image.fileSize / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          </div>

          {/* Analysis Card */}
          {image.analysis && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="font-semibold text-gray-900 mb-4">Analysis Results</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-gray-600">Negative Space</p>
                  <p className="text-2xl font-bold text-green-600">{image.analysis.negativeSpacePercentage}%</p>
                </div>
                <div>
                  <p className="text-gray-600">Regions Found</p>
                  <p className="text-2xl font-bold text-purple-600">{image.analysis.regionsCount}</p>
                </div>
                <div>
                  <p className="text-gray-600">Processing Time</p>
                  <p className="text-lg font-semibold">{image.analysis.processingTime.toFixed(2)}s</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Description */}
      {image.description && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="font-semibold text-gray-900 mb-3">Description</h3>
          <p className="text-gray-700">{image.description}</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="bg-white rounded-lg shadow-md p-6 flex gap-4">
        <Button variant="primary" className="flex-1">
          Run Analysis
        </Button>
        <Button variant="secondary" className="flex-1">
          Download
        </Button>
      </div>
    </div>
  );
};

export default ImageDetailPage;
