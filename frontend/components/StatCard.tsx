interface StatCardProps {
  label: string;
  value: number;
  icon: string;
  color: "blue" | "green" | "amber" | "purple" | "red";
  subtitle?: string;
}

const colorMap = {
  blue: { text: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20", glow: "stat-glow-blue" },
  green: { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20", glow: "stat-glow-green" },
  amber: { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", glow: "stat-glow-amber" },
  purple: { text: "text-purple-400", bg: "bg-purple-500/10", border: "border-purple-500/20", glow: "stat-glow-purple" },
  red: { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20", glow: "stat-glow-red" },
};

export default function StatCard({ label, value, icon, color, subtitle }: StatCardProps) {
  const c = colorMap[color];
  return (
    <div className={`glass-card ${c.glow} p-5 transition-all duration-300 hover:scale-[1.02]`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-400 tracking-wide uppercase">{label}</span>
        <div className={`w-9 h-9 rounded-lg ${c.bg} ${c.border} border flex items-center justify-center`}>
          <span className="text-lg">{icon}</span>
        </div>
      </div>
      <div className={`text-3xl font-bold ${c.text} tabular-nums`}>
        {value.toLocaleString()}
      </div>
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}
