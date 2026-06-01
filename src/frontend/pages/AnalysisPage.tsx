/**
 * AnalysisPage Component
 *
 * Detailed analysis results page with visualizations.
 *
 * @component
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Spinner } from '../components/Spinner';
import { Button } from '../components/Button';

/**
 * AnalysisPage component
 */
export const AnalysisPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        // TODO: Replace with actual API call
        const mockAnalysis = {
          id,
          imageName: 'Mountain Landscape',
          originalImageUrl: 'https://via.placeholder.com/400x300',
          resultImageUrl: 'https://via.placeholder.com/400x300',
          negativeSpacePercentage: 72,
          positiveSpacePercentage: 28,
          regionsCount: 15,
          largestRegionSize: 8500,
          smallestRegionSize: 250,
          averageRegionSize: 1200,
          processingTime: 2.45,
          analyzedAt: new Date().toISOString(),
          confidenceScore: 0.94,
        };
        setAnalysis(mockAnalysis);
      } catch (err) {
        setError('Failed to load analysis');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600 text-lg">Analysis not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-gray-900">Analysis Results</h1>
          <p className="text-gray-600 mt-2">{analysis.imageName}</p>
        </div>
        <Button variant="secondary" onClick={() => navigate(-1)}>
          Back
        </Button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Images Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="text-center p-4 bg-gray-50 border-b">
            <h3 className="font-semibold text-gray-900">Original Image</h3>
          </div>
          <img src={analysis.originalImageUrl} alt="Original" className="w-full h-auto" />
        </div>
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="text-center p-4 bg-gray-50 border-b">
            <h3 className="font-semibold text-gray-900">Analysis Result</h3>
          </div>
          <img src={analysis.resultImageUrl} alt="Result" className="w-full h-auto" />
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <p className="text-gray-600 text-sm mb-2">Negative Space</p>
          <p className="text-3xl font-bold text-green-600">{analysis.negativeSpacePercentage}%</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <p className="text-gray-600 text-sm mb-2">Positive Space</p>
          <p className="text-3xl font-bold text-orange-600">{analysis.positiveSpacePercentage}%</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <p className="text-gray-600 text-sm mb-2">Regions Found</p>
          <p className="text-3xl font-bold text-purple-600">{analysis.regionsCount}</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <p className="text-gray-600 text-sm mb-2">Confidence</p>
          <p className="text-3xl font-bold text-blue-600">{(analysis.confidenceScore * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Detailed Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Region Statistics</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Regions</span>
              <span className="font-semibold text-gray-900">{analysis.regionsCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Largest Region</span>
              <span className="font-semibold text-gray-900">{analysis.largestRegionSize} px²</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Smallest Region</span>
              <span className="font-semibold text-gray-900">{analysis.smallestRegionSize} px²</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Average Region Size</span>
              <span className="font-semibold text-gray-900">{analysis.averageRegionSize} px²</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Processing Information</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Processing Time</span>
              <span className="font-semibold text-gray-900">{analysis.processingTime.toFixed(2)}s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Analyzed At</span>
              <span className="font-semibold text-gray-900">
                {new Date(analysis.analyzedAt).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Confidence Score</span>
              <span className="font-semibold text-gray-900">
                {(analysis.confidenceScore * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4 justify-center">
        <Button variant="primary">📥 Download Report</Button>
        <Button variant="secondary">🖨️ Print Results</Button>
      </div>
    </div>
  );
};

export default AnalysisPage;
