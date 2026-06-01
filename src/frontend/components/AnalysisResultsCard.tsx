/**
 * AnalysisResultsCard Component
 *
 * Card component for displaying analysis results with metrics.
 *
 * @component
 */

import React from 'react';
import { Link } from 'react-router-dom';

interface AnalysisResultsCardProps {
  id: string;
  imageId: string;
  imageName: string;
  negativeSpacePercentage: number;
  regionsCount: number;
  processingTime: number;
  analyzedAt: string;
  thumbnailUrl: string;
}

/**
 * AnalysisResultsCard component
 */
export const AnalysisResultsCard: React.FC<AnalysisResultsCardProps> = ({
  id,
  imageId,
  imageName,
  negativeSpacePercentage,
  regionsCount,
  processingTime,
  analyzedAt,
  thumbnailUrl,
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getPercentageColor = (percentage: number) => {
    if (percentage > 70) return 'text-green-600';
    if (percentage > 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition">
      {/* Header with Image */}
      <div className="relative w-full h-40 bg-gray-200 overflow-hidden">
        <img
          src={thumbnailUrl}
          alt={imageName}
          className="w-full h-full object-cover"
        />
        <div className="absolute top-2 right-2 bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
          Analysis
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Title */}
        <h3 className="font-semibold text-gray-900 mb-3 truncate">
          {imageName}
        </h3>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Negative Space Percentage */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <p className="text-xs text-gray-600 mb-1">Negative Space</p>
            <p className={`text-lg font-bold ${getPercentageColor(negativeSpacePercentage)}`}>
              {negativeSpacePercentage}%
            </p>
          </div>

          {/* Regions Count */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <p className="text-xs text-gray-600 mb-1">Regions Found</p>
            <p className="text-lg font-bold text-purple-600">
              {regionsCount}
            </p>
          </div>

          {/* Processing Time */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <p className="text-xs text-gray-600 mb-1">Processing Time</p>
            <p className="text-lg font-bold text-orange-600">
              {processingTime.toFixed(2)}s
            </p>
          </div>

          {/* Date */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <p className="text-xs text-gray-600 mb-1">Analyzed</p>
            <p className="text-xs font-semibold text-gray-700">
              {formatDate(analyzedAt).split(' ')[0]}
            </p>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-600 mb-4">
          {negativeSpacePercentage > 70
            ? '✓ High negative space detected'
            : negativeSpacePercentage > 40
              ? '◐ Moderate negative space'
              : '✕ Low negative space detected'}
        </p>

        {/* Actions */}
        <Link
          to={`/analysis/${id}`}
          className="w-full block px-4 py-2 bg-blue-600 text-white text-center rounded-lg hover:bg-blue-700 transition"
        >
          View Full Results
        </Link>
      </div>
    </div>
  );
};

export default AnalysisResultsCard;
