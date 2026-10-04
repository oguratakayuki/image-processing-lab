# 数式⇔コード対応資料

各機能の実装コードが、数式ではどう表現されるかを1対1で対応付けて解説する資料集。[機能一覧（FEATURES.md）](../../FEATURES.md)の各ページから「補足資料」としてリンクされる。

目的は概念の説明ではなく、**「このコードの各行は、数式のどの部分を計算しているのか」**を明確にすることと、最後に具体的な数値での計算例を示すこと。

## 資料一覧

| 資料 | 対象 | 参照元 |
|---|---|---|
| [mean-kernel.md](mean-kernel.md) | `mean_kernel()` | [2-2. カーネルプリセット](../features/2-convolution/2-2-kernel-presets.md) |
| [gaussian-kernel.md](gaussian-kernel.md) | `gaussian_kernel()` | [2-2. カーネルプリセット](../features/2-convolution/2-2-kernel-presets.md) |
| [gaussian-2d-function.md](gaussian-2d-function.md) | `exp(-(x²+y²)/(2σ²))`の項の詳細 | [gaussian-kernel.md](gaussian-kernel.md) |
| [laplacian-kernel.md](laplacian-kernel.md) | `laplacian_kernel()` | [2-2. カーネルプリセット](../features/2-convolution/2-2-kernel-presets.md) |
| [sharpen-kernel.md](sharpen-kernel.md) | `sharpen_kernel()` | [2-2. カーネルプリセット](../features/2-convolution/2-2-kernel-presets.md) |
