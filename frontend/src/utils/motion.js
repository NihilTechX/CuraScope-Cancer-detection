/**
 * CuraScope Motion System
 * Apple-inspired spring animations with reduced motion support
 */

import { useEffect, useState } from 'react';

/**
 * Spring animation presets
 * All use critically damped springs (no bounce) by default
 */
export const springs = {
  // Default: critically damped, no overshoot
  default: {
    type: 'spring',
    bounce: 0,
    duration: 0.4,
  },

  // Quick: snappy interactions (buttons, toggles)
  quick: {
    type: 'spring',
    bounce: 0,
    duration: 0.25,
  },

  // Momentum: for drag/flick gestures (has slight bounce)
  momentum: {
    type: 'spring',
    bounce: 0.15,
    duration: 0.35,
  },

  // Gentle: large panels, sheets, modals
  gentle: {
    type: 'spring',
    bounce: 0,
    duration: 0.5,
  },

  // Stiff: immediate feedback (hover states)
  stiff: {
    type: 'spring',
    bounce: 0,
    duration: 0.15,
  },
};

/**
 * Reduced motion fallback
 * Instant transitions for users who prefer reduced motion
 */
export const reducedMotion = {
  transition: {
    duration: 0.01,
    ease: 'linear',
  },
};

/**
 * Hook to detect user's motion preference
 * @returns {boolean} true if user prefers reduced motion
 */
export function useReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (event) => {
      setPrefersReducedMotion(event.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return prefersReducedMotion;
}

/**
 * Get appropriate transition based on motion preference
 * @param {object} springConfig - Spring configuration object
 * @returns {object} Transition config (spring or reduced)
 */
export function getTransition(springConfig = springs.default) {
  const prefersReducedMotion = 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  
  return prefersReducedMotion ? reducedMotion.transition : springConfig;
}

/**
 * Common animation variants for Framer Motion
 */
export const variants = {
  // Fade in/out
  fade: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
  },

  // Slide up (for panels, modals)
  slideUp: {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 16 },
  },

  // Slide down (for dropdowns, tooltips)
  slideDown: {
    initial: { opacity: 0, y: -16 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -16 },
  },

  // Scale in (for popovers, dialogs)
  scaleIn: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
  },

  // Press feedback (for buttons)
  press: {
    initial: { scale: 1 },
    whileTap: { scale: 0.98 },
  },

  // Hover lift (for cards)
  hoverLift: {
    initial: { y: 0 },
    whileHover: { y: -2 },
  },
};

/**
 * Stagger children animation
 * @param {number} staggerDelay - Delay between each child (seconds)
 */
export function staggerChildren(staggerDelay = 0.05) {
  return {
    animate: {
      transition: {
        staggerChildren: staggerDelay,
      },
    },
  };
}

/**
 * Layout transition for shared element animations
 */
export const layoutTransition = {
  layout: true,
  transition: springs.quick,
};

/**
 * Drag constraints helper
 * @param {object} bounds - { top, right, bottom, left }
 */
export function dragConstraints(bounds) {
  return {
    drag: true,
    dragConstraints: bounds,
    dragElastic: 0.1,
    dragTransition: springs.momentum,
  };
}

/**
 * Interruptible animation helper
 * Allows animations to be interrupted mid-flight
 */
export const interruptible = {
  layout: true,
  transition: {
    ...springs.default,
    // Allow interruption
    restDelta: 0.01,
    restSpeed: 0.01,
  },
};

/**
 * Number count-up animation hook
 * @param {number} end - Target number
 * @param {number} duration - Animation duration in ms
 * @returns {number} Current animated value
 */
export function useCountUp(end, duration = 1000) {
  const [count, setCount] = useState(0);
  const prefersReducedMotion = useReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion) {
      setCount(end);
      return;
    }

    let startTime;
    let animationFrame;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      
      // Ease out cubic
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setCount(Math.floor(end * easeOut));

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration, prefersReducedMotion]);

  return count;
}

/**
 * Scroll-triggered animation hook
 * @param {number} threshold - Intersection threshold (0-1)
 * @returns {[ref, isVisible]} Ref to attach and visibility state
 */
export function useScrollReveal(threshold = 0.1) {
  const [isVisible, setIsVisible] = useState(false);
  const [ref, setRef] = useState(null);

  useEffect(() => {
    if (!ref) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold }
    );

    observer.observe(ref);
    return () => observer.disconnect();
  }, [ref, threshold]);

  return [setRef, isVisible];
}
