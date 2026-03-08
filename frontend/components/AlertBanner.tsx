"use client";

import { useState } from "react";

interface AlertItem {
  code: string;
  message: string;
  layman: string;
}

interface AlertBannerProps {
  warnings: AlertItem[];
  errors: AlertItem[];
}

const laymanTitles: Record<string, string> = {
  LOW_FRAME_COUNT: "Video Too Short",
  CHAOTIC_MOTION: "Unstable Crowd Pattern",
  CHAOTIC_MOTION_WARN: "Crowd Movement Warning",
  PIPELINE_ERROR: "Processing Error",
};

export default function AlertBanner({ warnings, errors }: AlertBannerProps) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (warnings.length === 0 && errors.length === 0) return null;

  return (
    <div className="space-y-3">
      {errors.map((err, i) => (
        <div
          key={`err-${i}`}
          className="glass-card border-red-500/30 bg-red-500/5 p-4"
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-red-500/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-red-400 text-sm font-bold">!</span>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-red-400">
                  {laymanTitles[err.code] || "Error"}
                </h4>
                <p className="text-sm text-gray-300 mt-1">{err.layman}</p>
              </div>
            </div>
            <button
              onClick={() => setExpanded(expanded === `err-${i}` ? null : `err-${i}`)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex-shrink-0 ml-4"
            >
              {expanded === `err-${i}` ? "Hide" : "Details"}
            </button>
          </div>
          {expanded === `err-${i}` && (
            <div className="mt-3 pl-11">
              <p className="text-xs text-gray-500 font-mono bg-gray-900/50 p-2 rounded">
                {err.code}: {err.message}
              </p>
            </div>
          )}
        </div>
      ))}

      {warnings.map((warn, i) => (
        <div
          key={`warn-${i}`}
          className="glass-card border-amber-500/30 bg-amber-500/5 p-4"
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-amber-500/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-amber-400 text-sm font-bold">⚠</span>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-amber-400">
                  {laymanTitles[warn.code] || "Warning"}
                </h4>
                <p className="text-sm text-gray-300 mt-1">{warn.layman}</p>
              </div>
            </div>
            <button
              onClick={() => setExpanded(expanded === `warn-${i}` ? null : `warn-${i}`)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex-shrink-0 ml-4"
            >
              {expanded === `warn-${i}` ? "Hide" : "Details"}
            </button>
          </div>
          {expanded === `warn-${i}` && (
            <div className="mt-3 pl-11">
              <p className="text-xs text-gray-500 font-mono bg-gray-900/50 p-2 rounded">
                {warn.code}: {warn.message}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
