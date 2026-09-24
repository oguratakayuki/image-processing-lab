"use client";

// カーネル(NxNの数値行列)をグリッドで編集するための汎用コンポーネント。
// Convolution・Edge検出など、複数のトピックで「カーネルを画面上で
// 編集→結果が変わる」という同じ操作が必要になるため、ページ固有の
// 説明文とは分離して共通コンポーネント化している(数学の説明や
// アルゴリズムの実装はページ側/engine側にあり、これは純粋な
// UI部品)。

interface KernelEditorProps {
  kernel: number[][];
  onChange: (kernel: number[][]) => void;
}

const SIZES = [3, 5, 7];

export function KernelEditor({ kernel, onChange }: KernelEditorProps) {
  const size = kernel.length;

  const handleCellChange = (row: number, col: number, value: string) => {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) return;
    // 2次元配列をイミュータブルに更新(行ごとにコピーしてから該当セルだけ書き換える)。
    // Reactの状態更新は「元のオブジェクトを直接書き換えない」のが基本ルール。
    const next = kernel.map((r) => [...r]);
    next[row][col] = parsed;
    onChange(next);
  };

  const handleResize = (newSize: number) => {
    const center = Math.floor(newSize / 2);
    // サイズ変更時は恒等カーネル(中心だけ1)にリセットする。
    const next = Array.from({ length: newSize }, (_, r) =>
      Array.from({ length: newSize }, (_, c) => (r === center && c === center ? 1 : 0))
    );
    onChange(next);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        {SIZES.map((s) => (
          <button
            key={s}
            onClick={() => handleResize(s)}
            className={`rounded border px-2 py-1 text-xs ${
              s === size
                ? "border-foreground bg-foreground text-background"
                : "border-zinc-300 dark:border-zinc-700"
            }`}
          >
            {s}x{s}
          </button>
        ))}
      </div>
      <div
        className="grid w-fit gap-1"
        style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}
      >
        {kernel.map((row, r) =>
          row.map((value, c) => (
            <input
              key={`${r}-${c}`}
              type="number"
              step="any"
              value={value}
              onChange={(e) => handleCellChange(r, c, e.target.value)}
              className="w-14 rounded border border-zinc-300 p-1 text-center text-xs dark:border-zinc-700 dark:bg-zinc-900"
            />
          ))
        )}
      </div>
    </div>
  );
}
