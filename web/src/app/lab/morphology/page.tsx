"use client";

import { useState } from "react";
import {
  runClosing,
  runDilate,
  runErode,
  runOpening,
  type ImageStep,
  type ProcessImageResponse,
} from "@/lib/api";

export default function MorphologyLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [t, setT] = useState(128);
  const [steps, setSteps] = useState<ImageStep[]>([]);
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
    <main className="mx-auto flex max-w-5xl flex-col gap-8 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Morphology Lab</h1>
        <p className="text-sm text-zinc-500">
          二値画像(前景ピクセルの座標の集合)に対する集合演算です。
          Erosion(収縮)は構造要素が完全に前景に収まる点だけを残し、
          Dilation(膨張)は構造要素が前景と1点でも重なれば前景にします。
          Opening(収縮→膨張)は小さな突起・ノイズを除去し、
          Closing(膨張→収縮)は小さな穴を埋めます。
          いずれも3x3の正方形構造要素・Grayscale→閾値処理で二値化した
          上で適用します。
        </p>
      </div>

      <section className="flex flex-col gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="text-sm"
        />
        <label className="flex flex-col gap-1 text-sm">
          t (threshold): {t}
          <input
            type="range"
            min={0}
            max={255}
            step={1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
          />
        </label>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => runAndShow(() => runErode(file!, t))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Erosionを適用
          </button>
          <button
            onClick={() => runAndShow(() => runDilate(file!, t))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Dilationを適用
          </button>
          <button
            onClick={() => runAndShow(() => runOpening(file!, t))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Openingを適用
          </button>
          <button
            onClick={() => runAndShow(() => runClosing(file!, t))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Closingを適用
          </button>
        </div>
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
