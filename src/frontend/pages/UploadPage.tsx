/**
 * UploadPage Component
 *
 * Image upload page with drag-and-drop support.
 *
 * @component
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';
import { Alert } from '../components/Alert';

/**
 * UploadPage component
 */
export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setError('');
    setFile(file);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      processFile(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setProgress(0);

    try {
      // TODO: Replace with actual API call
      const formData = new FormData();
      formData.append('file', file);

      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return prev;
          }
          return prev + Math.random() * 30;
        });
      }, 500);

      // Simulate upload delay
      await new Promise((resolve) => setTimeout(resolve, 2000));
      clearInterval(interval);
      setProgress(100);

      navigate('/dashboard', { state: { message: 'Image uploaded successfully' } });
    } catch (err) {
      setError('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900">Upload Image</h1>
        <p className="text-gray-600 mt-2">Upload an image to analyze its negative space.</p>
      </div>

      {error && (
        <Alert type="error" message={error} dismissible onClose={() => setError('')} />
      )}

      {/* Upload Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-12 text-center transition ${
          dragActive ? 'border-blue-600 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        {!preview ? (
          <>
            <div className="text-4xl mb-4">📤</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Drag and drop your image</h3>
            <p className="text-gray-600 mb-4">or</p>
            <label className="inline-block">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
              <span className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer">
                Select File
              </span>
            </label>
            <p className="text-sm text-gray-500 mt-4">JPG, PNG, GIF up to 10MB</p>
          </>
        ) : (
          <div className="space-y-4">
            <img src={preview} alt="Preview" className="max-h-80 mx-auto rounded-lg" />
            <p className="text-sm text-gray-600">{file?.name}</p>
            <button
              onClick={() => {
                setFile(null);
                setPreview('');
                setProgress(0);
              }}
              className="px-4 py-2 text-blue-600 hover:text-blue-700"
            >
              Choose Different File
            </button>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {uploading && progress > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold">Uploading...</p>
            <p className="text-sm text-gray-600">{Math.round(progress)}%</p>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <Button
          variant="primary"
          size="md"
          onClick={handleUpload}
          loading={uploading}
          disabled={!file || uploading}
          className="flex-1"
        >
          Upload Image
        </Button>
        <Button variant="secondary" size="md" className="flex-1">
          Cancel
        </Button>
      </div>
    </div>
  );
};

export default UploadPage;
