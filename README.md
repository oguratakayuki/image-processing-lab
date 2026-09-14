# Image Processing Lab

画像処理エンジニア検定ベーシック相当の知識習得と、画像処理を題材とした数学の学び直しを目的とした個人学習プロジェクト。

「数学的理論 → 数式 → 自前実装 → 実画像への適用 → 可視化」という流れを体験できる学習環境を目指す。

## 構成

このリポジトリは3つの領域に分かれている。

```
engine/   画像処理エンジン本体（Python, NumPy/SciPy）。フレームワーク非依存。学習の本体。
api/      FastAPI。engineをHTTP経由で呼び出す薄いアダプタ層。
web/      Next.js（TypeScript）。学習コンテンツの表示と実験用UI。
```

- **engine** には自前実装のみを置く。OpenCV等の高レベルAPIは `engine/src/imglab_engine/reference/` に隔離し、答え合わせ・性能比較にのみ使用する。
- **api** はビジネスロジックを持たず、engineの呼び出し結果をJSON/画像に変換するだけ。
- **web** は学習コンテンツ（概念・数式・確認問題）と、カーネルエディタやパイプライン可視化などのインタラクティブな実験UIを提供する。

## セットアップ

### engine

```bash
cd engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### api

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`http://localhost:8000/health` で起動確認。

### web

```bash
cd web
npm install
npm run dev
```

`http://localhost:3000` で起動確認。

## 開発方針

- MVPは小さく始める（最初は Grayscale / Histogram / Threshold / Convolution まで）。
- 各トピックは「概念説明 → 数学的背景 → 課題 → 自前実装 → 実験 → 可視化 → テスト → 確認問題」という共通フォーマットで構成する。
- 数学は先出しの座学ではなく、各アルゴリズムを実装する直前に必要な分だけ復習する（Just-in-Time方式）。
- 実装はできる限り自分の手で行い、Claudeはアルゴリズムの説明・数学の解説・課題提示・コードレビュー・テストケース提示・ヒント出しといった学習支援に徹する。
