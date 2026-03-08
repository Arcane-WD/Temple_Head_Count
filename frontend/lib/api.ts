const API_BASE = "http://localhost:8000";

export async function fetchVideos(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/videos`);
  const data = await res.json();
  return data.videos;
}

export async function uploadVideo(file: File): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to upload video");
  return data.filename;
}

export async function startProcessing(filename: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to start processing");
  return data.job_id;
}

export interface StatusUpdate {
  status: string;
  frame: number;
  total_frames: number;
  in_count: number;
  out_count: number;
  male: number;
  female: number;
  unknown: number;
  fps: number;
  warnings: Array<{ code: string; message: string; layman: string }>;
  errors: Array<{ code: string; message: string; layman: string }>;
  done: boolean;
}

export function subscribeToStatus(
  jobId: string,
  onUpdate: (data: StatusUpdate) => void,
  onDone: () => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/status/${jobId}`);

  eventSource.onmessage = (event) => {
    const data: StatusUpdate = JSON.parse(event.data);
    onUpdate(data);
    if (data.done) {
      eventSource.close();
      onDone();
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
  };

  return () => eventSource.close();
}

export interface JobResults {
  status: string;
  in_count: number;
  out_count: number;
  male: number;
  female: number;
  unknown: number;
  timeline: Array<{
    frame: number;
    in_count: number;
    out_count: number;
    male: number;
    female: number;
    unknown: number;
  }>;
  warnings: Array<{ code: string; message: string; layman: string }>;
  errors: Array<{ code: string; message: string; layman: string }>;
  output_file: string | null;
}

export async function fetchResults(jobId: string): Promise<JobResults> {
  const res = await fetch(`${API_BASE}/api/results/${jobId}`);
  return res.json();
}

export function getVideoUrl(filename: string): string {
  return `${API_BASE}/api/video/${filename}`;
}
