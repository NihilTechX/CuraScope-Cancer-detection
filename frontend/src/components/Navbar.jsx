import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';
import { springs, useReducedMotion } from '../utils/motion';

const TABS = [
  { id: 'prognosis', label: 'Assessment' },
  { id: 'insight', label: 'Insights' },
];

export default function Navbar({ activeTab, onTabChange }) {
  const [scrolled, setScrolled] = useState(false);
  const prefersReducedMotion = useReducedMotion();

  // Detect scroll for shadow effect
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header 
      className={`sticky top-0 z-30 transition-shadow ${
        scrolled ? 'shadow-lg' : 'shadow-sm'
      }`}
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.88)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderBottom: scrolled ? '1px solid rgba(226, 232, 240, 0.8)' : '1px solid transparent',
        transitionProperty: 'box-shadow, border-color',
        transitionDuration: 'var(--duration-default)',
        transitionTimingFunction: 'var(--ease-out)',
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">

          {/* Brand */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div 
              className="w-9 h-9 rounded-lg flex items-center justify-center text-white"
              style={{
                background: 'linear-gradient(135deg, var(--clinical-blue-700) 0%, var(--clinical-blue-900) 100%)',
                boxShadow: 'var(--shadow-sm)',
              }}
            >
              <Activity className="w-5 h-5" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <h1 
                className="font-semibold tracking-tight"
                style={{
                  fontSize: 'var(--text-base)',
                  color: 'var(--clinical-blue-900)',
                }}
              >
                CuraScope
              </h1>
              <p 
                className="hidden sm:block"
                style={{
                  fontSize: 'var(--text-xs)',
                  color: 'var(--slate-600)',
                }}
              >
                Survival Analysis
              </p>
            </div>
          </div>

          {/* Navigation with sliding indicator */}
          <nav className="flex items-center gap-1 relative" role="tablist">
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  role="tab"
                  aria-selected={isActive}
                  aria-controls={`${tab.id}-panel`}
                  className="relative px-4 py-2 font-medium"
                  style={{
                    fontSize: 'var(--text-sm)',
                    color: isActive ? 'var(--clinical-blue-700)' : 'var(--slate-600)',
                    transition: 'color var(--duration-quick) var(--ease-out)',
                    WebkitTapHighlightColor: 'transparent',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.color = 'var(--slate-900)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.color = 'var(--slate-600)';
                    }
                  }}
                >
                  {tab.label}
                  
                  {/* Sliding underline indicator */}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0"
                      transition={prefersReducedMotion ? { duration: 0.01 } : springs.quick}
                      style={{
                        height: '2px',
                        backgroundColor: 'var(--clinical-blue-700)',
                        borderRadius: '2px 2px 0 0',
                      }}
                    />
                  )}
                </button>
              );
            })}
          </nav>

        </div>
      </div>
    </header>
  );
}
