import React, { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle2, X, Microscope } from 'lucide-react';
import { springs, useReducedMotion } from '../utils/motion';

/**
 * ImageUpload Component
 * Clinical-styled image upload with drag & drop support
 */
export default function ImageUpload({
  onImageUpload,
  onImageRemove,
  uploadedImage,
  uploadedFileName,
  accept = 'image/jpeg,image/png,image/jpg',
  maxSizeMB = 10,
  disabled = false,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const prefersReducedMotion = useReducedMotion();

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate file type
    const validTypes = accept.split(',').map((t) => t.trim());
    if (!validTypes.includes(file.type)) {
      setError(`Please upload a valid image file (${accept})`);
      return;
    }

    // Validate file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setError(`File size must be less than ${maxSizeMB}MB`);
      return;
    }

    setError(null);

    // Read file as base64
    const reader = new FileReader();
    reader.onload = (event) => {
      onImageUpload?.(event.target.result, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    handleFileSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onImageRemove?.();
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const hasImage = Boolean(uploadedImage);
  const displayName = uploadedFileName || 'Biopsy histopathology specimen';

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileInputChange}
        className="hidden"
        disabled={disabled}
      />

      <motion.div
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className="relative rounded-xl cursor-pointer overflow-hidden"
        style={{
          border: `2px dashed ${
            error
              ? 'var(--risk-high)'
              : isDragging
              ? 'var(--clinical-blue-500)'
              : hasImage
              ? 'var(--diagnostic-teal-500)'
              : 'var(--slate-300)'
          }`,
          backgroundColor: isDragging
            ? 'var(--clinical-blue-100)'
            : hasImage
            ? 'var(--diagnostic-teal-100)'
            : 'var(--slate-50)',
          transition: `border-color var(--duration-quick) var(--ease-out), background-color var(--duration-quick) var(--ease-out)`,
          opacity: disabled ? 0.5 : 1,
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
        whileHover={!disabled && !hasImage ? { scale: 1.01 } : {}}
        transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
      >
        <AnimatePresence mode="wait">
          {hasImage ? (
            <motion.div
              key="uploaded"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={prefersReducedMotion ? { duration: 0.01 } : springs.default}
              className="p-4 flex items-center gap-4"
            >
              {/* Thumbnail */}
              <div
                className="shrink-0 rounded-lg overflow-hidden"
                style={{
                  width: '64px',
                  height: '64px',
                  border: '1px solid var(--slate-200)',
                }}
              >
                <img
                  src={uploadedImage}
                  alt="Uploaded biopsy"
                  className="w-full h-full object-cover"
                />
              </div>

              {/* File info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2
                    className="shrink-0"
                    style={{
                      width: '16px',
                      height: '16px',
                      color: 'var(--diagnostic-teal-700)',
                    }}
                  />
                  <span
                    className="font-semibold truncate"
                    style={{
                      fontSize: 'var(--text-xs)',
                      color: 'var(--diagnostic-teal-700)',
                    }}
                  >
                    {displayName}
                  </span>
                </div>
                <p
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--slate-600)',
                  }}
                >
                  Click or drop to replace
                </p>
              </div>

              {/* Remove button */}
              <button
                onClick={handleRemove}
                className="shrink-0 p-1.5 rounded-lg hover:bg-white/50 transition-colors"
                style={{
                  color: 'var(--slate-600)',
                }}
                aria-label="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={prefersReducedMotion ? { duration: 0.01 } : springs.default}
              className="p-6 text-center"
            >
              <motion.div
                animate={isDragging ? { scale: 1.1 } : { scale: 1 }}
                transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
                className="inline-flex items-center justify-center w-12 h-12 rounded-xl mb-3"
                style={{
                  backgroundColor: isDragging
                    ? 'var(--clinical-blue-500)'
                    : 'var(--slate-200)',
                  color: isDragging ? 'white' : 'var(--slate-600)',
                }}
              >
                {isDragging ? (
                  <Microscope className="w-6 h-6" />
                ) : (
                  <UploadCloud className="w-6 h-6" />
                )}
              </motion.div>

              <p
                className="font-semibold mb-1"
                style={{
                  fontSize: 'var(--text-sm)',
                  color: 'var(--slate-900)',
                }}
              >
                {isDragging ? 'Drop image here' : 'Upload biopsy image'}
              </p>
              <p
                style={{
                  fontSize: 'var(--text-xs)',
                  color: 'var(--slate-600)',
                }}
              >
                Drag & drop or click to browse
              </p>
              <p
                className="mt-2"
                style={{
                  fontSize: 'var(--text-xs)',
                  color: 'var(--slate-500)',
                }}
              >
                JPG or PNG · Max {maxSizeMB}MB
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Error message */}
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
          className="mt-2"
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--risk-high)',
          }}
          role="alert"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
