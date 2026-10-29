import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, ShieldHalf } from 'lucide-react';
import { springs, useReducedMotion, useCountUp } from '../utils/motion';

/**
 * RiskAssessment Component
 * Displays risk tier with clinical precision
 */

const RISK_CONFIG = {
  LOW: {
    label: 'Favorable Prognosis',
    icon: ShieldCheck,
    color: 'var(--risk-low)',
    bgColor: 'rgba(46, 125, 50, 0.08)',
    borderColor: 'rgba(46, 125, 50, 0.2)',
  },
  MEDIUM: {
    label: 'Moderate Risk',
    icon: ShieldHalf,
    color: 'var(--risk-medium)',
    bgColor: 'rgba(245, 124, 0, 0.08)',
    borderColor: 'rgba(245, 124, 0, 0.2)',
  },
  HIGH: {
    label: 'Elevated Risk',
    icon: ShieldAlert,
    color: 'var(--risk-high)',
    bgColor: 'rgba(198, 40, 40, 0.08)',
    borderColor: 'rgba(198, 40, 40, 0.2)',
  },
};

export default function RiskAssessment({
  riskTier = 'LOW',
  survival5yr,
  survival1yr,
  survival3yr,
  survival10yr,
  summary,
  className = '',
}) {
  const prefersReducedMotion = useReducedMotion();
  const config = RISK_CONFIG[riskTier] || RISK_CONFIG.LOW;
  const Icon = config.icon;

  // Animated survival percentage
  const survivalPct = survival5yr != null ? Math.round(survival5yr * 100) : null;
  const animatedPct = useCountUp(survivalPct || 0, 1200);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={prefersReducedMotion ? { duration: 0.01 } : springs.default}
      className={className}
    >
      {/* Risk Banner */}
      <div
        className="rounded-xl p-5 mb-4"
        style={{
          backgroundColor: config.bgColor,
          border: `1px solid ${config.borderColor}`,
        }}
      >
        <div className="flex items-center gap-3">
          <Icon
            className="shrink-0"
            style={{
              width: '32px',
              height: '32px',
              color: config.color,
            }}
          />
          <div>
            <p
              className="uppercase tracking-wider font-semibold mb-0.5"
              style={{
                fontSize: 'var(--text-xs)',
                color: config.color,
                opacity: 0.8,
              }}
            >
              Risk Assessment
            </p>
            <h3
              className="font-bold tracking-tight"
              style={{
                fontSize: 'var(--text-xl)',
                color: config.color,
              }}
            >
              {config.label}
            </h3>
          </div>
        </div>
      </div>

      {/* 5-Year Survival - Hero Metric */}
      {survivalPct !== null && (
        <div
          className="rounded-xl p-5 mb-4"
          style={{
            backgroundColor: 'var(--slate-50)',
            border: '1px solid var(--slate-200)',
          }}
        >
          <div className="flex items-baseline justify-between mb-3">
            <span
              className="uppercase tracking-wider font-semibold"
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--slate-700)',
              }}
            >
              5-Year Survival Probability
            </span>
            <span
              className="font-black tracking-tight tabular-nums"
              style={{
                fontSize: 'var(--text-2xl)',
                color: config.color,
              }}
            >
              {prefersReducedMotion ? survivalPct : animatedPct}%
            </span>
          </div>

          {/* Progress bar */}
          <div
            className="w-full rounded-full overflow-hidden"
            style={{
              height: '12px',
              backgroundColor: 'var(--slate-200)',
            }}
          >
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${survivalPct}%` }}
              transition={
                prefersReducedMotion
                  ? { duration: 0.01 }
                  : { ...springs.gentle, duration: 1.2 }
              }
              className="h-full rounded-full"
              style={{
                backgroundColor: config.color,
              }}
            />
          </div>

          <p
            className="mt-3 leading-relaxed"
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--slate-600)',
            }}
          >
            Based on patients with similar clinical profiles, this represents the estimated
            proportion surviving 5 years after diagnosis.
          </p>
        </div>
      )}

      {/* Additional Time Horizons */}
      {(survival1yr != null || survival3yr != null || survival10yr != null) && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          {[
            { label: '1 Year', value: survival1yr },
            { label: '3 Years', value: survival3yr },
            { label: '10 Years', value: survival10yr },
          ].map(
            ({ label, value }) =>
              value != null && (
                <motion.div
                  key={label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={
                    prefersReducedMotion
                      ? { duration: 0.01 }
                      : { ...springs.default, delay: 0.2 }
                  }
                  className="rounded-lg p-3 text-center"
                  style={{
                    backgroundColor: 'white',
                    border: '1px solid var(--slate-200)',
                  }}
                >
                  <p
                    className="font-semibold mb-1"
                    style={{
                      fontSize: 'var(--text-xs)',
                      color: 'var(--slate-600)',
                    }}
                  >
                    {label}
                  </p>
                  <p
                    className="font-black tabular-nums"
                    style={{
                      fontSize: 'var(--text-lg)',
                      color: 'var(--slate-900)',
                    }}
                  >
                    {Math.round(value * 100)}%
                  </p>
                </motion.div>
              )
          )}
        </div>
      )}

      {/* Clinical Summary */}
      {summary && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={
            prefersReducedMotion
              ? { duration: 0.01 }
              : { ...springs.default, delay: 0.4 }
          }
          className="rounded-xl p-4"
          style={{
            backgroundColor: 'var(--clinical-blue-100)',
            border: '1px solid var(--clinical-blue-500)',
            borderOpacity: 0.2,
          }}
        >
          <h4
            className="font-semibold mb-2"
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--clinical-blue-900)',
            }}
          >
            Clinical Interpretation
          </h4>
          <p
            className="leading-relaxed"
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--clinical-blue-900)',
              opacity: 0.9,
            }}
          >
            {summary}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}
