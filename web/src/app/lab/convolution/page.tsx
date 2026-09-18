"use client";

import { useEffect, useState } from "react";
import { fetchKernelPresets, runConvolution, type ImageStep } from "@/lib/api";

const PRESET_LABELS: Record<string, string> = {
  identity: "Identity",
  mean_3x3: "Mean 3x3",
  gaussian_3x3_sigma1: "Gaussian 3x3 (σ=1)",
  sharpen: "Sharpen",
};

// 3x3の単純平均(box blur)カーネルを初期値にする。
// 対称なカーネルなので、畳み込みの「反転」があってもなくても
// 結果が同じになる、最も直感的に効果を確認しやすい例。
const DEFAULT_KERNEL = JSON.stringify(
  [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  null,
  2
);

export default function ConvolutionLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [kernelText, setKernelText] = useState(DEFAULT_KERNEL);
  const [steps, setSteps] = useState<ImageStep[]>([]);
  const [presets, setPresets] = useState<Record<string, number[][]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchKernelPresets()
      .then(setPresets)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setSteps([]);
    setError(null);
  };

  const handleApply = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const kernel = JSON.parse(kernelText) as number[][];
      const response = await runConvolution(file, kernel);
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
        <h1 className="text-2xl font-semibold">Convolution Lab</h1>
        <p className="text-sm text-zinc-500">
          カーネル(JSONの2次元配列)を編集して畳み込みを適用できます。
          畳み込みは「カーネルを180度回転してから、対応する位置の値を
          掛けて足し合わせる」演算です。対称なカーネル(平均・ガウシアン)
          では反転の有無は結果に影響しません。
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
        <h2 className="font-medium">カーネル (kernel)</h2>
        <div className="flex flex-wrap gap-2">
          {Object.entries(presets).map(([name, matrix]) => (
            <button
              key={name}
              onClick={() => setKernelText(JSON.stringify(matrix, null, 2))}
              className="rounded border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              {PRESET_LABELS[name] ?? name}
            </button>
          ))}
        </div>
        <textarea
          value={kernelText}
          onChange={(e) => setKernelText(e.target.value)}
          rows={8}
          className="w-full max-w-sm rounded border border-zinc-300 p-2 font-mono text-xs dark:border-zinc-700 dark:bg-zinc-900"
        />
        <button
          onClick={handleApply}
          disabled={!file || loading}
          className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
        >
          畳み込みを適用
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
