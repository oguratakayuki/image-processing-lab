# CLAUDE.md

Image Processing Lab — 画像処理エンジニア検定ベーシック相当の知識習得と数学の学び直しを目的とした個人学習プロジェクト。詳細は`README.md`を参照。

## ドキュメント作業時の必読ルール

`FEATURES.md`・`docs/`配下のドキュメントを新規作成・編集する際は、**必ず[docs/DOCUMENTATION_GUIDE.md](docs/DOCUMENTATION_GUIDE.md)に従うこと**。ユーザーから個別に指示がなくても常に適用する。

特に以下は頻繁に関わるため要注意：

- ディレクトリ構成・採番ルール（`N-M`形式）・ファイル命名規則
- 項目詳細ページ／グループ概要ページ／補足資料／`docs/math/`それぞれの定型構成
- **インラインLaTeX数式（`$...$`）は使用禁止**（GitHub上で不安定にしかレンダリングされないことが判明済み）。バッククォートのプレーンテキストかフェンス付きコードブロックを使う
- 数値計算例は必ず実際に計算してから記載する（手計算しない）
- commitメッセージはヒアドキュメントで失敗することがあるため、一時ファイル＋`git commit -F`を使う

## 開発環境

- `engine/`：Python（NumPy/SciPy/Pillow）。自前実装のみ、OpenCVは`engine/src/imglab_engine/reference/`に隔離
- `api/`：FastAPI。engineの薄いアダプタ
- `web/`：Next.js（TypeScript）
- 各ディレクトリのセットアップ・起動コマンドは`README.md`を参照
