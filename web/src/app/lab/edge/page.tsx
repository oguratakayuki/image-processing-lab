"use client";

import { useState } from "react";
import {
  runLaplacian,
  runPrewitt,
  runSobel,
  type ImageStep,
  type ProcessImageResponse,
} from "@/lib/api";

export default function EdgeLabPage() {
  const [file, setFile] = useState<File | null>(null);
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
        <h1 className="text-2xl font-semibold">Edge Detection Lab</h1>
        <p className="text-sm text-zinc-500">
          エッジ(輪郭)は輝度の変化率、つまり微分が大きい場所として
          検出できます。Sobel/Prewittはx方向・y方向それぞれの勾配
          Gx, Gyを計算し、その大きさ |∇I| = √(Gx²+Gy²)
          をエッジの強さとして可視化します(Sobelは中心を重視した
          [1,2,1]、Prewittは均等な[1,1,1]で平滑化する違いがあります)。
          Laplacianは2階微分 ∇²I を使い、エッジ位置で符号が反転する
          (零交差)性質を利用します。Gx/Gy/Laplacianはいずれも
          灰色(128)が「変化なし」、明るいほど正、暗いほど負の値を
          表します。
        </p>
      </div>

      <section className="flex flex-col gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="text-sm"
        />
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => runAndShow(() => runSobel(file!))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Sobelエッジ検出を適用
          </button>
          <button
            onClick={() => runAndShow(() => runPrewitt(file!))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Prewittエッジ検出を適用
          </button>
          <button
            onClick={() => runAndShow(() => runLaplacian(file!))}
            disabled={!file || loading}
            className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
          >
            Laplacianエッジ検出を適用
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
