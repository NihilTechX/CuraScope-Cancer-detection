import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, Check } from 'lucide-react';
import { generateId } from '../utils/accessibility';
import { springs, useReducedMotion } from '../utils/motion';

/**
 * Select Component
 * Custom dropdown with clinical styling
 * Note: Using native select for simplicity - can upgrade to Radix UI later
 */
export default function Select({
  label,
  value,
  onChange,
  options = [],
  error,
  helperText,
  required = false,
  disabled = false,
  fullWidth = false,
  placeholder = 'Select an option',
  id,
  className = '',
  ...props
}) {
  const selectId = id || generateId('select');
  const errorId = error ? `${selectId}-error` : undefined;
  const helperId = helperText ? `${selectId}-helper` : undefined;
  const prefersReducedMotion = useReducedMotion();

  const hasValue = value !== '' && value !== null && value !== undefined;

  return (
    <div className={fullWidth ? 'w-full' : ''}>
      {label && (
        <label
          htmlFor={selectId}
          className="block font-medium mb-1.5"
          style={{
            fontSize: 'var(--text-xs)',
            color: error ? 'var(--risk-high)' : 'var(--slate-700)',
            fontWeight: 'var(--font-semibold)',
          }}
        >
          {label}
          {required && (
            <span
              className="ml-1"
              style={{ color: 'var(--risk-high)' }}
              aria-label="required"
            >
              *
            </span>
          )}
        </label>
      )}

      <div className="relative">
        <select
          id={selectId}
          value={value}
          onChange={onChange}
          disabled={disabled}
          required={required}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? errorId : helperId}
          className={`
            w-full px-3 py-2 pr-10 rounded-lg
            appearance-none
            bg-white
            ${className}
          `}
          style={{
            fontSize: 'var(--text-sm)',
            color: hasValue ? 'var(--slate-900)' : 'var(--slate-500)',
            border: `1px solid ${
              error
                ? 'var(--risk-high)'
                : hasValue
                ? 'var(--slate-400)'
                : 'var(--slate-300)'
            }`,
            outline: 'none',
            transition: `border-color var(--duration-quick) var(--ease-out)`,
            opacity: disabled ? 0.5 : 1,
            cursor: disabled ? 'not-allowed' : 'pointer',
            backgroundColor: disabled ? 'var(--slate-50)' : 'white',
          }}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((option) => {
            const optionValue = typeof option === 'string' ? option : option.value;
            const optionLabel = typeof option === 'string' ? option : option.label;
            return (
              <option key={optionValue} value={optionValue}>
                {optionLabel}
              </option>
            );
          })}
        </select>

        {/* Chevron icon */}
        <div
          className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
          style={{
            color: disabled ? 'var(--slate-400)' : 'var(--slate-500)',
          }}
        >
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>

      {/* Error message */}
      {error && (
        <motion.p
          id={errorId}
          role="alert"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
          className="mt-1.5"
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--risk-high)',
          }}
        >
          {error}
        </motion.p>
      )}

      {/* Helper text */}
      {helperText && !error && (
        <p
          id={helperId}
          className="mt-1.5"
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--slate-500)',
          }}
        >
          {helperText}
        </p>
      )}
    </div>
  );
}
