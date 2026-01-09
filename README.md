# 棒人間 (Bou-Ningen) モーションキャプチャ

**MediaPipe Holistic** を使用して、Webカメラの映像からリアルタイムで表情豊かでユニークな棒人間アニメーションを生成するアプリケーションです。

## 特徴 (Features)

- **AI エンジン**: **MediaPipe Holistic** に移行し、体の動きだけでなく、**指の細かい動き**や**顔の向き**までリアルタイムに追跡します。YOLOv8よりもCPUで高速に動作し、Mac (Apple Silicon) でも快適です。
- **へのへのもへじスタイル**: 顔は日本の伝統的な「へのへのもへじ」スタイルで描画されます。目や口が実際の顔の動きに合わせて動きます。顔の輪郭は「じ」の字（し＋濁点）になっています。
- **手書き風エフェクト**: 設定により、鉛筆で描いたようなラフで味わいのある手書き風ラインに切り替え可能です（デフォルトはOFF）。
- **自動ズーム (Auto-Tracking)**: ユーザーの動きに合わせて常に画面中央に大きく表示するようにカメラが自動追従します。
- **仮想カメラ出力 (Virtual Camera)**: Zoomなどのビデオ会議アプリで映像ソースとして使用できます (OBS Studio等の仮想カメラドライバが必要)。
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

   ### 1. ネイティブアプリ
   OpenCVウィンドウで高速に動作します。

   ```bash
   # 通常起動
   uv run native_app.py

   # 自動ズーム（追跡）を有効化
   uv run native_app.py --track

   # Zoomミーティング等で使用する仮想カメラ出力を有効化 (要OBS)
   uv run native_app.py --virtual

   # 線の太さを指定 (例: 10)
   uv run native_app.py --thickness 10
   
   # カメラIDを指定 (例: iPhone連係カメラ)
   uv run native_app.py --camera 1
   ```
   **操作:** キーボードの `q` で終了します。

## プロジェクト構成

- `native_app.py`: OpenCVを使用したネイティブアプリケーション
- `stickman.py`: 棒人間の描画ロジック (MediaPipe対応)
- `stickman_camera.py`: カメラ自動追従ロジック
- `stickman_processor.py`: メイン処理クラス
- `pyproject.toml`: 依存関係定義

## トラブルシューティング & 仮想カメラの使用順序

仮想カメラ (Virtual Camera) が黒い画面になる場合や動作しない場合、以下の手順と順序を確認してください。

1. **前提条件**: OBS Studioがインストールされていること（OBS Virtual Cameraドライバが必要です）。
2. **OBSの起動について**:
   - **通常のOBSアプリを起動する必要はありません。**
   - もしOBSを起動している場合、OBS側の「仮想カメラ開始」ボタンは**押さないでください**。OBSがカメラデバイスを占有し、Pythonアプリからの映像を上書きしてしまう可能性があります。
3. **起動順序**:
   1. Pythonアプリを実行します: `uv run native_app.py --virtual`
   2. コンソールに `Starting Virtual Camera...` と表示され、エラーが出ないことを確認します。
   3. Zoom / Teams / Google Meet などの会議アプリを開きます。
   4. カメラ設定で **「OBS Virtual Camera」** を選択します。
4. **それでも映らない場合**:
   - 会議アプリを再起動してください。
   - `uv run debug_vcam.py` を実行して、色のついたテスト映像が送られるか確認してください。

## ライセンス

MIT
