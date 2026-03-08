export default function ModelSidebar() {
  return (
    <aside className="glass-card p-5 space-y-6">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
        Model Information
      </h3>

      {/* Detection Model */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="text-sm font-medium text-gray-200">Detection</span>
        </div>
        <div className="pl-4 space-y-1">
          <p className="text-xs text-gray-400">
            <span className="text-gray-300 font-medium">YOLOv8s</span>
          </p>
          <p className="text-xs text-gray-500">Person detection (class 0)</p>
          <p className="text-xs text-gray-500">Confidence: 0.35 | IoU: 0.65</p>
        </div>
      </div>

      {/* Tracking */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400" />
          <span className="text-sm font-medium text-gray-200">Tracking</span>
        </div>
        <div className="pl-4 space-y-1">
          <p className="text-xs text-gray-400">
            <span className="text-gray-300 font-medium">ByteTrack</span>
          </p>
          <p className="text-xs text-gray-500">Multi-object tracker</p>
          <p className="text-xs text-gray-500">Buffer: 60 frames</p>
        </div>
      </div>

      {/* Gender Classification */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-purple-400" />
          <span className="text-sm font-medium text-gray-200">Classification</span>
        </div>
        <div className="pl-4 space-y-1">
          <p className="text-xs text-gray-400">
            <span className="text-gray-300 font-medium">ConvNeXt-Tiny</span>
          </p>
          <p className="text-xs text-gray-500">Gender classification (ONNX)</p>
          <p className="text-xs text-gray-500">Accuracy: 82.44%</p>
          <p className="text-xs text-gray-500">Votes required: 5</p>
        </div>
      </div>

      {/* Dataset */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-amber-400" />
          <span className="text-sm font-medium text-gray-200">Training Data</span>
        </div>
        <div className="pl-4 space-y-1">
          <p className="text-xs text-gray-400">
            <span className="text-gray-300 font-medium">PA-100K</span>
          </p>
          <p className="text-xs text-gray-500">100,000 pedestrian images</p>
          <p className="text-xs text-gray-500">26 binary attributes</p>
          <p className="text-xs text-gray-500">Outdoor surveillance domain</p>
        </div>
      </div>

      {/* Calibration */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-cyan-400" />
          <span className="text-sm font-medium text-gray-200">Calibration</span>
        </div>
        <div className="pl-4 space-y-1">
          <p className="text-xs text-gray-400">
            <span className="text-gray-300 font-medium">Kinematic PCA</span>
          </p>
          <p className="text-xs text-gray-500">Motion-vector based gate line</p>
          <p className="text-xs text-gray-500">Auto-calibrate enabled</p>
        </div>
      </div>
    </aside>
  );
}
