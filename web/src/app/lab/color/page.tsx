"use client";

import { useState } from "react";
import {
  runBrightnessContrast,
  runGrayscale,
  runHistogram,
  type HistogramResponse,
  type ImageStep,
  type ProcessImageResponse,
} from "@/lib/api";

// ヒストグラムの棒グラフ描画。256ビンあるので、チャート用ライブラリを
// 導入するほどでもない単純なSVGでその場で描く(このページの中だけで
// 使う想定のため共通コンポーネント化はしていない)。
function HistogramChart({ counts }: { counts: number[] }) {
  const max = Math.max(...counts, 1);
  return (
    <svg
      viewBox="0 0 256 100"
      preserveAspectRatio="none"
      className="h-32 w-full max-w-md text-zinc-700 dark:text-zinc-300"
    >
      {counts.map((count, value) => {
        const height = (count / max) * 100;
        return (
          <rect
            key={value}
            x={value}
            y={100 - height}
            width={1}
            height={height}
            fill="currentColor"
          />
        );
      })}
    </svg>
  );
}

export default function ColorLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [steps, setSteps] = useState<ImageStep[]>([]);
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0);
  const [histogramResult, setHistogramResult] =
    useState<HistogramResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setSteps([]);
    setHistogramResult(null);
    setError(null);
  };

  const runAndShow = async (task: () => Promise<ProcessImageResponse>) => {
    setLoading(true);
    setError(null);
    try {
      const response = await task();
      setSteps(response.steps);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleHistogram = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const response = await runHistogram(file);
      setHistogramResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto flex max-w-4xl flex-col gap-8 p-8">
      <div>
        <h1 className="text-2xl font-semibold">
          Color Lab: Grayscale / Brightness &amp; Contrast
        </h1>
        <p className="text-sm text-zinc-500">
          画像をアップロードして、Grayscale変換や明るさ・コントラスト調整を
          実際に適用して確認できます。
        </p>
      </div>

      <section className="flex flex-col gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="text-sm"
        />
      </section>

      <section className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
        <h2 className="font-medium">Grayscale変換 (ITU-R BT.601)</h2>
        <button
          onClick={() => runAndShow(() => runGrayscale(file!))}
          disabled={!file || loading}
          className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
        >
          Grayscaleに変換
        </button>
      </section>

      <section className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
        <h2 className="font-medium">明るさ・コントラスト調整</h2>
        <label className="flex flex-col gap-1 text-sm">
          alpha (contrast): {alpha.toFixed(2)}
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.05}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          beta (brightness): {beta}
          <input
            type="range"
            min={-150}
            max={150}
            step={1}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
          />
        </label>
        <button
          onClick={() =>
            runAndShow(() => runBrightnessContrast(file!, alpha, beta))
          }
          disabled={!file || loading}
          className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
        >
          明るさ・コントラストを適用
        </button>
      </section>

      <section className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
        <h2 className="font-medium">ヒストグラム</h2>
        <p className="text-xs text-zinc-500">
          Grayscale変換した上で、各ピクセル値(0-255)の出現回数を数える。
        </p>
        <button
          onClick={handleHistogram}
          disabled={!file || loading}
          className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
        >
          ヒストグラムを計算
        </button>
        {histogramResult && (
          <div className="flex flex-wrap items-start gap-6 pt-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={histogramResult.grayscale_image_base64}
              alt="grayscale"
              className="max-w-xs rounded border border-zinc-200 dark:border-zinc-800"
            />
            <HistogramChart counts={histogramResult.histogram} />
          </div>
        )}
      </section>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {steps.length > 0 && (
        <section className="flex flex-wrap gap-6">
          {steps.map((step) => (
            <figure key={step.name} className="flex flex-col gap-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={step.image_base64}
                alt={step.name}
                className="max-w-xs rounded border border-zinc-200 dark:border-zinc-800"
              />
              <figcaption className="max-w-xs text-xs text-zinc-500">
                {step.description}
              </figcaption>
            </figure>
          ))}
        </section>
      )}
    </main>
  );
}
