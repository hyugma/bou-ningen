# 棒人間 (Bou-Ningen) モーションキャプチャ

**MediaPipe Holistic** を使用して、Webカメラの映像からリアルタイムで表情豊かでユニークな棒人間アニメーションを生成するアプリケーションです。

## 特徴 (Features)

- **AI エンジン**: **MediaPipe Holistic** に移行し、体の動きだけでなく、**指の細かい動き**や**顔の向き**までリアルタイムに追跡します。YOLOv8よりもCPUで高速に動作し、Mac (Apple Silicon) でも快適です。
- **へのへのもへじスタイル**: 顔は日本の伝統的な「へのへのもへじ」スタイルで描画されます。目や口が実際の顔の動きに合わせて動きます。顔の輪郭は「じ」の字（し＋濁点）になっています。
- **手書き風エフェクト**: 設定により、鉛筆で描いたようなラフで味わいのある手書き風ラインに切り替え可能です（デフォルトはOFF）。
- **自動ズーム (Auto-Tracking)**: ユーザーの動きに合わせて常に画面中央に大きく表示するようにカメラが自動追従します。
- **カスタマイズ**: 線の太さやスタイルを自由に調整できます。

## 必要要件 (Prerequisites)

- Python 3.10 以上
- `uv` パッケージマネージャー
- macOS (Apple Silicon 推奨) または PC

## 始め方 (Setup)

1. **プロジェクトの依存関係をインストール:**
   ```bash
   uv sync
   ```

2. **アプリの起動:**

   ### 1. ネイティブアプリ (推奨 - 最速)
   OpenCVウィンドウで高速に動作します。

   # 通常起動
   uv run native_app.py

   # 自動ズーム（追跡）を有効化
   uv run native_app.py --zoom

   # 手書き風スタイルを有効化
   uv run native_app.py --sketch

   # 線の太さを指定 (例: 10)
   uv run native_app.py --thickness 10
   
   # カメラIDを指定 (例: iPhone連係カメラ)
   uv run native_app.py --camera 1
   ```
   **操作:** キーボードの `q` で終了します。

   ### 2. Web アプリケーション (Gradio)
   ブラウザで使用できるインターフェースです。

   ```bash
   uv run app.py
   ```
   起動後、表示されるURL (例: `http://localhost:7860`) にアクセスします。Web UI上で手書きスタイルの切り替えや太さ調整が可能です。

## プロジェクト構成

- `app.py`: Gradioを使用したWeb UIアプリケーション
- `native_app.py`: OpenCVを使用したネイティブアプリケーション
- `stickman.py`: 棒人間の描画ロジック (MediaPipe対応)
- `pyproject.toml`: 依存関係定義

## ライセンス

MIT
