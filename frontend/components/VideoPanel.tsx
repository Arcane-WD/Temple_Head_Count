"use client";

import { useRef, useMemo } from "react";

interface TimelinePoint {
  frame: number;
  in_count: number;
}

interface VideoPanelProps {
  videoUrl: string | null;
  timeline: TimelinePoint[];
  fps: number;
  totalFrames: number;
}

export default function VideoPanel({ videoUrl, timeline, fps, totalFrames }: VideoPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Compute high-traffic regions (frames where IN count jumps significantly)
  const highTrafficMarkers = useMemo(() => {
    if (timeline.length < 2 || totalFrames === 0) return [];

    const markers: { position: number; label: string }[] = [];
    let maxDelta = 0;

    // Calculate deltas between consecutive points
    const deltas = timeline.slice(1).map((point, i) => ({
      frame: point.frame,
      delta: point.in_count - timeline[i].in_count,
    }));

    // Find max delta to set threshold
    deltas.forEach((d) => {
      if (d.delta > maxDelta) maxDelta = d.delta;
    });

    const threshold = Math.max(1, maxDelta * 0.6);

    deltas.forEach((d) => {
      if (d.delta >= threshold && d.delta > 0) {
        const position = (d.frame / totalFrames) * 100;
        const time = fps > 0 ? `${Math.floor(d.frame / fps / 60)}:${(Math.floor(d.frame / fps) % 60).toString().padStart(2, "0")}` : `F${d.frame}`;
        markers.push({ position, label: `High traffic at ${time}` });
      }
    });

    return markers;
  }, [timeline, totalFrames, fps]);

  if (!videoUrl) {
    return (
      <div className="glass-card p-6 flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mx-auto mb-3">
            <svg className="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-sm text-gray-500">Processed video will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="p-4 border-b border-gray-800">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
          Processed Output
        </h3>
      </div>

      {/* Video player */}
      <div className="bg-black aspect-video">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="w-full h-full"
          playsInline
        />
      </div>

      {/* Custom timeline with high-traffic markers */}
      {highTrafficMarkers.length > 0 && (
        <div className="p-4 border-t border-gray-800">
          <p className="text-xs text-gray-500 mb-2">
            High Traffic Zones ({highTrafficMarkers.length} detected)
          </p>
          <div className="relative h-6 bg-gray-800 rounded-full overflow-hidden">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 via-transparent to-emerald-500/10 rounded-full" />

            {/* Markers */}
            {highTrafficMarkers.map((marker, i) => (
              <div
                key={i}
                className="absolute top-0 h-full group cursor-pointer"
                style={{ left: `${marker.position}%` }}
                title={marker.label}
                onClick={() => {
                  if (videoRef.current && fps > 0 && totalFrames > 0) {
                    const duration = videoRef.current.duration;
                    if (isFinite(duration) && duration > 0) {
                      videoRef.current.currentTime = (marker.position / 100) * duration;
                    }
                  }
                }}
              >
                <div className="w-1.5 h-full bg-red-500 rounded-full opacity-80 group-hover:opacity-100 group-hover:w-2 transition-all" />
                <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 hidden group-hover:block whitespace-nowrap">
                  <div className="bg-gray-900 text-xs text-gray-300 px-2 py-1 rounded border border-gray-700">
                    {marker.label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
