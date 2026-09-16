"use client";

import { useState } from "react";
import {
  runBrightnessContrast,
  runGrayscale,
  type ImageStep,
  type ProcessImageResponse,
} from "@/lib/api";

export default function ColorLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [steps, setSteps] = useState<ImageStep[]>([]);
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setSteps([]);
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
