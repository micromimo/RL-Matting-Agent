import { useState, useRef, useEffect } from "react";
import { ImageCanvas } from "./components/ImageCanvas";
import {
  ControlPanel,
  ModelStatus,
  ProcessingConfig,
  DEFAULT_PROCESSING_CONFIG,
} from "./components/ControlPanel";
import { ModelStatusCard } from "./components/ModelStatusCard";
import { MetricsChart } from "./components/MetricsChart";
import { useRlStepListener } from "./hooks/useRlStepListener";

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function tauriInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) throw new Error("Not running in Tauri");
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

async function tauriSave(opts: { defaultPath?: string; filters?: Array<{ name: string; extensions: string[] }> }): Promise<string | null> {
  if (!isTauri) throw new Error("Not running in Tauri");
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save(opts);
}

function App() {
  const [imagePath, setImagePath] = useState<string | null>(null);
  const { state, history, running, pipelineStages, reset } = useRlStepListener();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [modelChecked, setModelChecked] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [config, setConfig] = useState<ProcessingConfig>({ ...DEFAULT_PROCESSING_CONFIG });

  const enableRlModel = config.enable_rl_model;
  const enableTraditional = config.enable_traditional;
  const setEnableRlModel = (v: boolean) => setConfig({ ...config, enable_rl_model: v });
  const setEnableTraditional = (v: boolean) => setConfig({ ...config, enable_traditional: v });

  const checkModelStatus = async (silent = false) => {
    if (modelChecked && silent) return;
    if (!silent) {
      setModelStatus({ is_available: false, loading: true });
    }
    try {
      const status: ModelStatus = await tauriInvoke("check_model_status_cmd");
      setModelStatus({ ...status, loading: false });
      setModelChecked(true);
    } catch (e) {
      console.error("check_model_status failed:", e);
      setModelStatus({
        is_available: false,
        loading: false,
        error: e instanceof Error ? e.message : String(e),
      });
      setModelChecked(true);
    }
  };

  const handleRefreshModel = () => {
    setModelChecked(false);
    checkModelStatus(false);
  };

  useEffect(() => {
    checkModelStatus(false);
  }, []);

  useEffect(() => {
    if (!isTauri) return;
    let destroyed = false;
    let webviewApi: Awaited<ReturnType<typeof import("@tauri-apps/api/webview").getCurrentWebview>> | null = null;
    let currentZoom = 1.0;
    const MIN_ZOOM = 0.2;
    const MAX_ZOOM = 3.0;
    const ZOOM_STEP = 0.1;

    const setupZoom = async () => {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        webviewApi = getCurrentWebview();
      } catch {
        return;
      }
    };
    setupZoom();

    const applyZoom = async (next: number) => {
      const clamped = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, next));
      currentZoom = clamped;
      try {
        await webviewApi?.setZoom(clamped);
      } catch {
        /* ignore */
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (destroyed) return;
      const isMac = /Mac|iPhone|iPad/.test(navigator.userAgent);
      const mod = isMac ? e.metaKey : e.ctrlKey;
      if (!mod || e.altKey) return;

      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      const isEditable =
        tag === "INPUT" || tag === "TEXTAREA" || !!target?.isContentEditable;

      if (isEditable) return;

      const key = e.key;

      if (key === "=" || key === "+") {
        e.preventDefault();
        void applyZoom(currentZoom + ZOOM_STEP);
      } else if (key === "-" || key === "_") {
        e.preventDefault();
        void applyZoom(currentZoom - ZOOM_STEP);
      } else if (key === "0") {
        e.preventDefault();
        void applyZoom(1.0);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      destroyed = true;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const handleSelectImage = async () => {
    if (isTauri) {
      try {
        const { open } = await import("@tauri-apps/plugin-dialog");
        const path = await open({
          multiple: false,
          filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp", "bmp"] }],
        });
        if (path) {
          setImagePath(path as string);
          reset();
        }
      } catch (e) {
        console.error("open dialog failed:", e);
        fileInputRef.current?.click();
      }
    } else {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImagePath(url);
    reset();

    const filePath = (file as File & { path?: string }).path;
    if (filePath) {
      setImagePath(filePath);
    } else {
      setImagePath(url);
    }

    e.target.value = "";
  };

  const handleStartRl = async () => {
    if (!imagePath) return;
    console.log("Starting RL loop with image path:", imagePath, "config:", config);
    reset();
    try {
      await tauriInvoke("start_rl_loop_cmd", {
        imagePath,
        config,
      });
    } catch (e) {
      console.error("start_rl_loop failed:", e);
      alert("抠图启动失败: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const handleSaveResult = async () => {
    if (!state?.mask_base64) return;
    setSaveLoading(true);
    setSaveMsg(null);
    try {
      if (!isTauri) {
        const byteChars = atob(state.mask_base64);
        const byteNumbers = new Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
          byteNumbers[i] = byteChars.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "image/png" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `matting_result_${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
        setSaveMsg("已下载到浏览器默认位置");
      } else {
        const savePath = await tauriSave({
          defaultPath: `matting_result_${Date.now()}.png`,
          filters: [{ name: "PNG Image", extensions: ["png"] }],
        });
        if (!savePath) {
          setSaveMsg("已取消保存");
          return;
        }
        await tauriInvoke("save_result_cmd", {
          maskBase64: state.mask_base64,
          savePath,
        });
        setSaveMsg(`已保存: ${savePath}`);
      }
    } catch (e) {
      console.error("save_result failed:", e);
      setSaveMsg("保存失败，请重试");
    } finally {
      setSaveLoading(false);
      setTimeout(() => setSaveMsg(null), 3000);
    }
  };

  const isFinished = state?.is_finished ?? false;
  const maskBase64 = state?.mask_base64 ?? null;

  return (
    <div className="w-screen h-screen p-4 pb-0 flex flex-col text-slate-700 select-none gap-3">
      <div className="flex-1 flex gap-5 min-h-0">
        <aside className="w-[340px] shrink-0 flex flex-col gap-4 overflow-y-auto liquid-glass rounded-2xl p-4">
          <div className="liquid-glass rounded-2xl p-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-xl font-bold tracking-wide neon-text-pink">
                  RL Matting Agent
                </h1>
                <p className="text-xs text-slate-600 mt-1">
                  Weakly-Supervised Object Localization&Matting
                </p>
              </div>
              <div className="text-3xl">😈</div>
            </div>
          </div>

          <ModelStatusCard
            modelStatus={modelStatus}
            onRefreshModel={handleRefreshModel}
          />

          {isFinished && maskBase64 ? (
            <div className="liquid-glass-strong rounded-2xl p-4 space-y-3">
              <div className="text-[10px] tracking-[0.3em] uppercase text-slate-500">
                Matting Result
              </div>
              <div className="rounded-xl overflow-hidden bg-slate-100 aspect-square flex items-center justify-center">
                <img
                  src={`data:image/png;base64,${maskBase64}`}
                  alt="matting result preview"
                  className="w-full h-full object-contain"
                  style={{ backgroundImage: "linear-gradient(45deg, #e2e8f0 25%, transparent 25%), linear-gradient(-45deg, #e2e8f0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #e2e8f0 75%), linear-gradient(-45deg, transparent 75%, #e2e8f0 75%)", backgroundSize: "20px 20px", backgroundPosition: "0 0, 0 10px, 10px -10px, -10px 0px" }}
                />
              </div>
              <button
                className="glass-button w-full text-slate-700 hover:text-pink-700 font-semibold flex items-center justify-center gap-2"
                onClick={handleSaveResult}
                disabled={saveLoading}
              >
                <span className="relative z-10">
                  {saveLoading ? "💾 保存中..." : isTauri ? "💾 保存结果" : "💾 下载结果"}
                </span>
              </button>
              {saveMsg && (
                <div className="text-xs text-center text-green-700">{saveMsg}</div>
              )}
            </div>
          ) : null}

          <ControlPanel
            imagePath={imagePath}
            currentAction={state?.action_taken ?? ""}
            currentStep={state?.step ?? 0}
            confidence={state?.confidence ?? 0}
            running={running}
            isFinished={isFinished}
            onSelectImage={handleSelectImage}
            onStartRl={handleStartRl}
            history={history}
            totalSteps={config.rl_max_steps}
            enableRlModel={enableRlModel}
            setEnableRlModel={setEnableRlModel}
            enableTraditional={enableTraditional}
            setEnableTraditional={setEnableTraditional}
            config={config}
            setConfig={setConfig}
            pipelineStages={pipelineStages}
          />

          <MetricsChart history={history} totalSteps={config.rl_max_steps} />
        </aside>

        <main className="flex-1 min-w-0 min-h-0 flex flex-col">
          <ImageCanvas
            imagePath={imagePath}
            bbox={state?.bbox ?? null}
            maskBase64={maskBase64}
            action={state?.action_taken}
            isFinished={isFinished}
            showImage={!isFinished}
          />
        </main>
      </div>

      <footer className="h-6 flex items-center justify-center text-xs text-slate-500 pb-1">
        Powered by <span className="neon-text-pink font-semibold mx-1">Rust</span> & <span className="neon-text-purple font-semibold mx-1">Tauri2</span> & <span className="neon-text-pink font-semibold mx-1">React</span>
      </footer>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp,image/jpg"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  );
}

export default App;
