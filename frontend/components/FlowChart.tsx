"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface TimelinePoint {
  frame: number;
  in_count: number;
  out_count: number;
  male: number;
  female: number;
  unknown: number;
}

interface FlowChartProps {
  data: TimelinePoint[];
  fps: number;
}

function formatTime(frame: number, fps: number): string {
  if (fps <= 0) return `F${frame}`;
  const totalSeconds = Math.floor(frame / fps);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export default function FlowChart({ data, fps }: FlowChartProps) {
  const chartData = data.map((d) => ({
    ...d,
    time: formatTime(d.frame, fps),
  }));

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
        Flow Over Time
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="time"
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
            />
            <YAxis stroke="#6b7280" fontSize={11} tickLine={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #1f2937",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "12px", color: "#9ca3af" }}
            />
            <Line
              type="monotone"
              dataKey="in_count"
              name="Entries (IN)"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="out_count"
              name="Exits (OUT)"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="male"
              name="Male"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 2"
            />
            <Line
              type="monotone"
              dataKey="female"
              name="Female"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 2"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
