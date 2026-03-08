"use client";

import { useState, useEffect, useCallback } from "react";
import StatCard from "@/components/StatCard";
import FlowChart from "@/components/FlowChart";
import ModelSidebar from "@/components/ModelSidebar";
import VideoPanel from "@/components/VideoPanel";
import AlertBanner from "@/components/AlertBanner";
import {
  fetchVideos,
  startProcessing,
  subscribeToStatus,
  fetchResults,
  getVideoUrl,
  uploadVideo,
  type StatusUpdate,
  type JobResults,
} from "@/lib/api";

type AppState = "idle" | "loading_videos" | "ready" | "processing" | "complete" | "error";

export default function DashboardPage() {
  const [appState, setAppState] = useState<AppState | "uploading">("idle");
  const [videos, setVideos] = useState<string[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string>("");
  const [jobId, setJobId] = useState<string | null>(null);

  // Live stats
  const [stats, setStats] = useState({
    in_count: 0,
    out_count: 0,
    male: 0,
    female: 0,
    unknown: 0,
    frame: 0,
    total_frames: 0,
    fps: 0,
  });

  const [timeline, setTimeline] = useState<JobResults["timeline"]>([]);
  const [warnings, setWarnings] = useState<StatusUpdate["warnings"]>([]);
  const [errors, setErrors] = useState<StatusUpdate["errors"]>([]);
  const [outputFile, setOutputFile] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Load video list on mount
  useEffect(() => {
    setAppState("loading_videos");
    fetchVideos()
      .then((vids) => {
        setVideos(vids);
        if (vids.length > 0) setSelectedVideo(vids[0]);
        setAppState("ready");
        setConnectionError(null);
      })
      .catch((err) => {
        setConnectionError(
          "Could not connect to the backend server. Make sure the backend is running on http://localhost:8000"
        );
        setAppState("error");
      });
  }, []);

  const handleProcess = useCallback(async () => {
    if (!selectedVideo) return;

    try {
      setAppState("processing");
      setWarnings([]);
      setErrors([]);
      setTimeline([]);
      setOutputFile(null);
      setConnectionError(null);

      const id = await startProcessing(selectedVideo);
      setJobId(id);

      subscribeToStatus(
        id,
        (update) => {
          setStats({
            in_count: update.in_count,
            out_count: update.out_count,
            male: update.male,
            female: update.female,
            unknown: update.unknown,
            frame: update.frame,
            total_frames: update.total_frames,
            fps: update.fps,
          });
          setWarnings(update.warnings);
          setErrors(update.errors);
        },
        async () => {
          // SSE done — fetch final results
          try {
            const results = await fetchResults(id);
            setTimeline(results.timeline);
            setWarnings(results.warnings);
            setErrors(results.errors);
            if (results.output_file) {
              setOutputFile(results.output_file);
            }
            setAppState(results.errors.length > 0 ? "error" : "complete");
          } catch {
            setAppState("complete");
          }
        }
      );
    } catch (err) {
      setConnectionError("Failed to start processing. Is the backend running?");
      setAppState("error");
    }
  }, [selectedVideo]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Quick validation
    if (!file.name.endsWith('.mp4')) {
      setConnectionError("Only .mp4 video files are supported.");
      return;
    }

    try {
      setAppState("uploading");
      setConnectionError(null);
      const filename = await uploadVideo(file);
      
      // Refresh video list and select new video
      const vids = await fetchVideos();
      setVideos(vids);
      setSelectedVideo(filename);
      setAppState("ready");
    } catch (err: any) {
      setConnectionError(`Failed to upload video: ${err.message}`);
      setAppState("error");
    } finally {
      // Clear input so same file can be uploaded again if needed
      e.target.value = '';
    }
  };

  const progress =
    stats.total_frames > 0 ? Math.round((stats.frame / stats.total_frames) * 100) : 0;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">Temple Surveillance Analytics</h1>
              <p className="text-xs text-gray-500">Real-time entry monitoring &amp; demographic classification</p>
            </div>
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-2">
            {appState === "processing" && (
              <div className="flex items-center gap-2 text-sm text-blue-400">
                <div className="w-2 h-2 rounded-full bg-blue-400 pulse-live" />
                Processing
              </div>
            )}
            {appState === "complete" && errors.length === 0 && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <div className="w-2 h-2 rounded-full bg-emerald-400" />
                Complete
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
        {connectionError && (
          <div className="mb-6 glass-card border-red-500/30 bg-red-500/5 p-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-red-500/15 flex items-center justify-center">
                <span className="text-red-400 text-sm font-bold">!</span>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-red-400">Connection Error</h4>
                <p className="text-sm text-gray-300 mt-0.5">{connectionError}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-6">
          {/* Left column */}
          <div className="space-y-6">
            {/* Video selector + process button */}
            <div className="glass-card p-5">
              <div className="flex flex-col sm:flex-row items-start sm:end gap-4">
                <div className="flex-1 w-full flex items-end gap-3">
                  <div className="flex-1">
                    <label className="text-xs text-gray-500 uppercase tracking-wide block mb-1.5">
                      Select Video
                    </label>
                    <div className="flex gap-2">
                      <select
                        id="video-selector"
                        value={selectedVideo}
                        onChange={(e) => setSelectedVideo(e.target.value)}
                        disabled={appState === "processing" || appState === "loading_videos" || appState === "uploading"}
                        className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
                      >
                        {videos.length === 0 && (
                          <option value="">No videos available</option>
                        )}
                        {videos.map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                      
                      {/* Hidden file input */}
                      <input 
                        type="file" 
                        id="video-upload" 
                        accept="video/mp4" 
                        className="hidden" 
                        onChange={handleFileUpload}
                      />
                      <label 
                        htmlFor="video-upload"
                        className={`px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 rounded-lg text-sm font-medium transition-colors cursor-pointer flex items-center shrink-0 ${
                          (appState === "processing" || appState === "uploading") ? "opacity-50 pointer-events-none" : ""
                        }`}
                        title="Upload new video"
                      >
                        {appState === "uploading" ? "Uploading..." : "Upload .mp4"}
                      </label>
                    </div>
                  </div>
                </div>
                <button
                  id="process-button"
                  onClick={handleProcess}
                  disabled={appState === "processing" || appState === "uploading" || !selectedVideo}
                  className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white text-sm font-medium rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed sm:mt-5 whitespace-nowrap"
                >
                  {appState === "processing" ? "Processing..." : "Start Analysis"}
                </button>
              </div>

              {/* Progress bar */}
              {appState === "processing" && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs text-gray-500 mb-1.5">
                    <span>Frame {stats.frame.toLocaleString()} / {stats.total_frames.toLocaleString()}</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full progress-shimmer rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Alert banners */}
            <AlertBanner warnings={warnings} errors={errors} />

            {/* Stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              <StatCard label="Entries" value={stats.in_count} icon="↓" color="green" subtitle="IN count" />
              <StatCard label="Exits" value={stats.out_count} icon="↑" color="red" subtitle="OUT count" />
              <StatCard label="Male" value={stats.male} icon="♂" color="blue" subtitle="Classified" />
              <StatCard label="Female" value={stats.female} icon="♀" color="amber" subtitle="Classified" />
              <StatCard label="Unknown" value={stats.unknown} icon="?" color="purple" subtitle="Low confidence" />
            </div>

            {/* Flow chart */}
            {timeline.length > 0 && (
              <FlowChart data={timeline} fps={stats.fps} />
            )}

            {/* Video player */}
            <VideoPanel
              videoUrl={outputFile ? getVideoUrl(outputFile) : null}
              timeline={timeline}
              fps={stats.fps}
              totalFrames={stats.total_frames}
            />
          </div>

          {/* Right sidebar */}
          <div className="hidden lg:block">
            <div className="sticky top-24">
              <ModelSidebar />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-4 mt-8">
        <div className="max-w-[1600px] mx-auto px-6 text-center text-xs text-gray-600">
          Temple Head Count System — Computer Vision Analytics Pipeline
        </div>
      </footer>
    </div>
  );
}
