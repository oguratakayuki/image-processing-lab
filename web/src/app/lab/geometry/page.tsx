"use client";

import { useState } from "react";
import {
  runTranslate,
  type ImageStep,
  type ProcessImageResponse,
} from "@/lib/api";

export default function GeometryLabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [tx, setTx] = useState(20);
  const [ty, setTy] = useState(10);
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
        <h1 className="text-2xl font-semibold">Geometry Lab</h1>
        <p className="text-sm text-zinc-500">
          平行移動(Translation)です。出力画像の各点について、
          対応する入力画像上の点を逆算して値を持ってくる
          「逆方向マッピング」で実装しています。画像の端からはみ出た
          部分は切り捨てられ、新しく現れた部分は黒で埋められます。
          色の値には一切手を加えないため、RGB画像のまま変換できます。
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
          tx (横方向の移動量): {tx}
          <input
            type="range"
            min={-100}
            max={100}
            step={1}
            value={tx}
            onChange={(e) => setTx(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          ty (縦方向の移動量): {ty}
          <input
            type="range"
            min={-100}
            max={100}
            step={1}
            value={ty}
            onChange={(e) => setTy(Number(e.target.value))}
          />
        </label>
        <button
          onClick={() => runAndShow(() => runTranslate(file!, tx, ty))}
          disabled={!file || loading}
          className="w-fit rounded bg-foreground px-4 py-2 text-sm text-background disabled:opacity-40"
        >
          平行移動を適用
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
