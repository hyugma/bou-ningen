# 棒人間 (Bou-Ningen) モーションキャプチャ

YOLOv8 Pose 推定を使用して、Webカメラの映像からリアルタイムで棒人間アニメーションを生成するアプリケーションです。

## 機能
- **リアルタイム変換**: YOLOv8 NanoとMetal Performance Shaders (MPS) を活用し、Mac上で高速に動作します。
- **棒人間スタイル**: 白背景に黒線のシンプルなデザイン。丸い頭と丸みを帯びた関節で親しみやすいスタイルです。
- **マルチプラットフォーム**: ブラウザベースのアプリと、より高速なネイティブアプリの2種類を用意しています。

## 前提条件
- Python 3.10 以上
- `uv` パッケージマネージャー
- macOS (Apple Silicon 推奨) または CUDA対応GPU

## セットアップ

1. **プロジェクトの初期化と依存関係のインストール:**
   ```bash
   uv sync
   ```

## 使い方

### 1. Web アプリケーション (Gradio)
ブラウザ上で動作します。手軽に試したい場合に適しています。

```bash
uv run app.py
```
起動後、ブラウザで `http://localhost:7860` にアクセスしてください。

### 2. ネイティブ アプリケーション (最速)
OpenCVウィンドウを使用し、ブラウザを介さずに直接映像処理を行います。遅延が少なく、最も高速です。

```bash
uv run native_app.py
```
終了するにはキーボードの `q` を押してください。

#### iPhone / 連係カメラの使用
iPhoneをWebカメラとして使用する場合（連係カメラ）、カメラIDを指定して起動できます。通常は `1` または `2` です。

```bash
uv run native_app.py --camera 1
```

> **注意 (macOS):** ネイティブアプリ実行時に「OpenCV: not authorized to capture video」というエラーが出る場合、ターミナル（Terminal, iTerm, VSCodeなど）にカメラへのアクセス権限がない可能性があります。
> 「システム設定 > プライバシーとセキュリティ > カメラ」から、使用しているターミナルアプリを許可してください。

## コンテナでの実行 (Podman / Docker)

コンテナビルドを行うことも可能です。

1. **イメージのビルド:**
   ```bash
   podman build -t bou-ningen .
   ```

2. **コンテナの実行:**
   ```bash
   podman run -p 7860:7860 bou-ningen
   ```
   `http://localhost:7860` でアクセスできます。

## プロジェクト構成
- `app.py`: Webアプリケーション (Gradio)
- `native_app.py`: ネイティブアプリケーション (OpenCV)
- `stickman.py`: 骨格検出データを棒人間に描画するロジック
- `Dockerfile`: コンテナ定義
- `pyproject.toml`: 依存関係定義

## ライセンス
MIT
