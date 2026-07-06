import { useEffect, useState } from "react";

export interface RlStepUpdate {
  step: number;
  action_taken: string;
  bbox: { x: number; y: number; width: number; height: number };
  confidence: number;
  is_finished: boolean;
  mask_base64?: string | null;
}

export interface PipelineStage {
  key: string;
  label: string;
  icon: string;
  status: "pending" | "running" | "done";
}

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export function useRlStepListener() {
  const [state, setState] = useState<RlStepUpdate | null>(null);
  const [history, setHistory] = useState<RlStepUpdate[]>([]);
  const [running, setRunning] = useState<boolean>(false);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>([]);

  useEffect(() => {
    if (!isTauri) return;
    let unlistenStep: (() => void) | null = null;
    let unlistenPipeline: (() => void) | null = null;
    let unlistenStage: (() => void) | null = null;

    const setup = async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");
        unlistenStep = await listen<RlStepUpdate>("rl-step-update", (event) => {
          const payload = event.payload;
          setState(payload);
          setHistory((prev) => [...prev, payload].slice(-40));
          if (payload.step === 0 && !running) setRunning(true);
          if (payload.is_finished) setRunning(false);
        });
        unlistenPipeline = await listen<PipelineStage[]>("rl-pipeline-start", (event) => {
          setPipelineStages(event.payload);
        });
        unlistenStage = await listen<PipelineStage>("rl-pipeline-stage", (event) => {
          setPipelineStages((prev) => {
            const next = [...prev];
            const idx = next.findIndex((s) => s.key === event.payload.key);
            if (idx >= 0) {
              next[idx] = event.payload;
            } else {
              next.push(event.payload);
            }
            return next;
          });
        });
      } catch {
        /* ignore */
      }
    };

    setup();

    return () => {
      if (unlistenStep) unlistenStep();
      if (unlistenPipeline) unlistenPipeline();
      if (unlistenStage) unlistenStage();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reset = () => {
    setState(null);
    setHistory([]);
    setRunning(false);
    setPipelineStages([]);
  };

  return { state, history, running, pipelineStages, reset };
}
