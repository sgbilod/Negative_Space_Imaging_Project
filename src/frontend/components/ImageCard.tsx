/**
 * ImageCard Component
 *
 * Card component for displaying image thumbnails with metadata.
 *
 * @component
 */

import React from 'react';
import { Link } from 'react-router-dom';

interface ImageCardProps {
  id: string;
  name: string;
  imageUrl: string;
  uploadedAt: string;
  fileSize: number;
  onDelete?: (id: string) => void;
}

/**
 * ImageCard component
 */
export const ImageCard: React.FC<ImageCardProps> = ({
  id,
  name,
  imageUrl,
  uploadedAt,
  fileSize,
  onDelete,
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition">
      {/* Image Container */}
      <Link
        to={`/images/${id}`}
        className="relative w-full h-48 bg-gray-200 overflow-hidden block"
      >
        <img
          src={imageUrl}
          alt={name}
          className="w-full h-full object-cover hover:scale-105 transition"
        />
      </Link>

      {/* Card Content */}
      <div className="p-4">
        {/* Name */}
        <Link
          to={`/images/${id}`}
          className="font-semibold text-gray-900 hover:text-blue-600 truncate block mb-2"
        >
          {name}
        </Link>

        {/* Metadata */}
        <div className="text-sm text-gray-600 space-y-1 mb-4">
          <p>Uploaded: {formatDate(uploadedAt)}</p>
          <p>Size: {formatFileSize(fileSize)}</p>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Link
            to={`/images/${id}`}
            className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-center text-sm"
          >
            View
          </Link>
          {onDelete && (
            <button
              onClick={() => onDelete(id)}
              className="px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition text-sm"
              aria-label="Delete image"
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageCard;
