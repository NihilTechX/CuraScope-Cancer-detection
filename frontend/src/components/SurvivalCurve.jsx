import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { springs, useReducedMotion } from '../utils/motion';

/**
 * SurvivalCurve Component
 * Kaplan-Meier style survival curve visualization
 */
export default function SurvivalCurve({
  survivalData = [],
  patientSurvival,
  width = 600,
  height = 300,
  className = '',
}) {
  const canvasRef = useRef(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const prefersReducedMotion = useReducedMotion();

  // Default survival curve if no data provided (years 0 to 10)
  const defaultCurve = [
    { time: 0, survival: 1.0 },
    { time: 1, survival: 0.95 },
    { time: 2, survival: 0.88 },
    { time: 3, survival: 0.82 },
    { time: 4, survival: 0.78 },
    { time: 5, survival: 0.75 },
    { time: 6, survival: 0.72 },
    { time: 7, survival: 0.69 },
    { time: 8, survival: 0.67 },
    { time: 9, survival: 0.65 },
    { time: 10, survival: 0.63 },
  ];

  const rawData = survivalData && survivalData.length > 0 ? survivalData : defaultCurve;
  const curveData = rawData.map((d) => ({
    time: d.time != null ? Number(d.time) : (d.month != null ? Number(d.month) / 12 : 0),
    survival: typeof d.survival === 'number' ? d.survival : 1.0,
  }));

  const patientPt = patientSurvival ? {
    time: patientSurvival.time != null ? Number(patientSurvival.time) : (patientSurvival.month != null ? Number(patientSurvival.month) / 12 : 5),
    survival: typeof patientSurvival.survival === 'number' ? patientSurvival.survival : 0.75,
  } : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Set canvas dimensions
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Margins
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Scales
    const maxTime = Math.max(10, ...curveData.map((d) => d.time));
    const xScale = (time) => margin.left + (time / maxTime) * chartWidth;
    const yScale = (survival) => margin.top + (1 - survival) * chartHeight;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.5)';
    ctx.lineWidth = 1;

    // Horizontal grid lines (survival %)
    for (let i = 0; i <= 10; i++) {
      const y = yScale(i / 10);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();
    }

    // Vertical grid lines (years)
    for (let i = 0; i <= maxTime; i += 2) {
      const x = xScale(i);
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, height - margin.bottom);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.8)';
    ctx.lineWidth = 2;

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.stroke();

    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();

    // Draw survival curve (step function)
    ctx.strokeStyle = 'rgba(30, 73, 118, 1)';
    ctx.lineWidth = 3;
    ctx.beginPath();

    curveData.forEach((point, i) => {
      const x = xScale(point.time);
      const y = yScale(point.survival);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        // Step function: horizontal then vertical
        const prevX = xScale(curveData[i - 1].time);
        const prevY = yScale(curveData[i - 1].survival);
        ctx.lineTo(x, prevY); // Horizontal
        ctx.lineTo(x, y); // Vertical
      }
    });

    // Extend to end
    const lastPoint = curveData[curveData.length - 1];
    ctx.lineTo(width - margin.right, yScale(lastPoint.survival));
    ctx.stroke();

    // Fill area under curve
    ctx.fillStyle = 'rgba(30, 73, 118, 0.1)';
    ctx.beginPath();
    ctx.moveTo(xScale(0), height - margin.bottom);
    curveData.forEach((point, i) => {
      const x = xScale(point.time);
      const y = yScale(point.survival);
      if (i === 0) {
        ctx.lineTo(x, y);
      } else {
        const prevX = xScale(curveData[i - 1].time);
        const prevY = yScale(curveData[i - 1].survival);
        ctx.lineTo(x, prevY);
        ctx.lineTo(x, y);
      }
    });
    ctx.lineTo(width - margin.right, yScale(lastPoint.survival));
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.closePath();
    ctx.fill();

    // Draw patient marker if provided
    if (patientPt) {
      const px = xScale(patientPt.time || 5);
      const py = yScale(patientPt.survival);

      // Marker circle
      ctx.fillStyle = 'rgba(0, 172, 193, 1)';
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();

      // Marker border
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw axis labels
    ctx.fillStyle = 'rgba(71, 85, 105, 1)';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels (years)
    for (let i = 0; i <= maxTime; i += 2) {
      const x = xScale(i);
      ctx.fillText(`${i}`, x, height - margin.bottom + 20);
    }

    // X-axis title
    ctx.fillText('Years', width / 2, height - 5);

    // Y-axis labels (survival %)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 10; i += 2) {
      const y = yScale(i / 10);
      ctx.fillText(`${i * 10}%`, margin.left - 10, y);
    }

    // Y-axis title
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Survival Probability', 0, 0);
    ctx.restore();
  }, [curveData, patientPt, width, height]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={prefersReducedMotion ? { duration: 0.01 } : springs.default}
      className={`rounded-xl p-5 ${className}`}
      style={{
        backgroundColor: 'white',
        border: '1px solid var(--slate-200)',
      }}
    >
      <div className="mb-4">
        <h3
          className="font-semibold mb-1"
          style={{
            fontSize: 'var(--text-base)',
            color: 'var(--slate-900)',
          }}
        >
          Survival Curve
        </h3>
        <p
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--slate-600)',
          }}
        >
          Kaplan-Meier estimate for similar patient cohort
        </p>
      </div>

      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full"
          style={{
            maxWidth: '100%',
            height: 'auto',
          }}
        />

        {patientPt && (
          <div
            className="absolute top-4 right-4 px-3 py-2 rounded-lg"
            style={{
              backgroundColor: 'rgba(0, 172, 193, 0.1)',
              border: '1px solid rgba(0, 172, 193, 0.3)',
            }}
          >
            <p
              className="font-semibold"
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--diagnostic-teal-700)',
              }}
            >
              Patient: {Math.round(patientPt.survival * 100)}%
            </p>
          </div>
        )}
      </div>

      <div
        className="mt-4 pt-4"
        style={{
          borderTop: '1px solid var(--slate-200)',
        }}
      >
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div
              className="w-6 h-0.5"
              style={{ backgroundColor: 'var(--clinical-blue-700)' }}
            />
            <span style={{ color: 'var(--slate-600)' }}>Cohort survival</span>
          </div>
          {patientPt && (
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: 'var(--diagnostic-teal-500)' }}
              />
              <span style={{ color: 'var(--slate-600)' }}>Patient estimate</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
