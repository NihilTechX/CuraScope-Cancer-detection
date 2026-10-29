import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { springs, useReducedMotion } from '../utils/motion';

/**
 * Button Component
 * Clinical design with spring-based press feedback and design system tokens
 */
export default function Button({
  children,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  fullWidth = false,
  icon = null,
  iconPosition = 'left',
  onClick,
  type = 'button',
  className = '',
  ...props
}) {
  const prefersReducedMotion = useReducedMotion();

  // Variant styles using design tokens
  const variantStyles = {
    primary: {
      backgroundColor: 'var(--clinical-blue-700)',
      color: 'white',
      border: 'none',
      boxShadow: 'var(--shadow-sm)',
      hoverBg: 'var(--clinical-blue-900)',
      focusOutline: 'var(--clinical-blue-500)',
    },
    secondary: {
      backgroundColor: 'var(--slate-100)',
      color: 'var(--slate-900)',
      border: '1px solid var(--slate-200)',
      boxShadow: 'none',
      hoverBg: 'var(--slate-200)',
      focusOutline: 'var(--clinical-blue-500)',
    },
    ghost: {
      backgroundColor: 'transparent',
      color: 'var(--slate-700)',
      border: 'none',
      boxShadow: 'none',
      hoverBg: 'var(--slate-100)',
      focusOutline: 'var(--clinical-blue-500)',
    },
    danger: {
      backgroundColor: 'var(--risk-high)',
      color: 'white',
      border: 'none',
      boxShadow: 'var(--shadow-sm)',
      hoverBg: '#B71C1C',
      focusOutline: 'var(--risk-high)',
    },
    outline: {
      backgroundColor: 'transparent',
      color: 'var(--clinical-blue-700)',
      border: '1px solid var(--clinical-blue-700)',
      boxShadow: 'none',
      hoverBg: 'var(--clinical-blue-100)',
      focusOutline: 'var(--clinical-blue-500)',
    },
  };

  // Size styles
  const sizeStyles = {
    small: {
      padding: '0.375rem 0.75rem',
      fontSize: 'var(--text-xs)',
      gap: '0.375rem',
    },
    medium: {
      padding: '0.5rem 1rem',
      fontSize: 'var(--text-sm)',
      gap: '0.5rem',
    },
    large: {
      padding: '0.75rem 1.5rem',
      fontSize: 'var(--text-base)',
      gap: '0.625rem',
    },
  };

  const currentVariant = variantStyles[variant];
  const currentSize = sizeStyles[size];

  const [isHovered, setIsHovered] = React.useState(false);

  const motionProps = prefersReducedMotion
    ? {}
    : {
        whileTap: disabled || loading ? {} : { scale: 0.98 },
        transition: springs.quick,
      };

  return (
    <motion.button
      type={type}
      disabled={disabled || loading}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`
        inline-flex items-center justify-center
        font-medium rounded-lg
        focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2
        ${fullWidth ? 'w-full' : ''}
        ${className}
      `}
      style={{
        backgroundColor: isHovered && !disabled && !loading
          ? currentVariant.hoverBg
          : currentVariant.backgroundColor,
        color: currentVariant.color,
        border: currentVariant.border,
        boxShadow: currentVariant.boxShadow,
        padding: currentSize.padding,
        fontSize: currentSize.fontSize,
        gap: currentSize.gap,
        fontWeight: 'var(--font-medium)',
        opacity: disabled && !loading ? 0.5 : 1,
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
        transition: `background-color var(--duration-quick) var(--ease-out), opacity var(--duration-quick) var(--ease-out)`,
        outlineColor: currentVariant.focusOutline,
      }}
      {...motionProps}
      {...props}
    >
      {loading && (
        <Loader2
          className="animate-spin"
          style={{
            width: size === 'small' ? '14px' : size === 'large' ? '18px' : '16px',
            height: size === 'small' ? '14px' : size === 'large' ? '18px' : '16px',
          }}
        />
      )}
      {icon && !loading && iconPosition === 'left' && icon}
      {children}
      {icon && !loading && iconPosition === 'right' && icon}
    </motion.button>
  );
}
