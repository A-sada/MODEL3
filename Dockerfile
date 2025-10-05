# ベースは軽量な Python
FROM python:3.11-slim

# 基本ツール（ビルドに必要な最低限）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ（MODEL3 の中で動かす）
WORKDIR /app/MODEL3

# 依存ライブラリをまとめて入れる（requirements.txt が無いので直接指定）
# NOTE: torch は CPU 版をインストール。失敗する場合はコメントの代替コマンドを使用。
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    negmas \
    # ---- PyTorch (CPU) ----
    torch

# ↑もし上の torch で失敗する場合（環境によってPyPIのwheelが無い時など）
# RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

# ソースをコピー（親フォルダから MODEL3 丸ごと）
COPY MODEL3 /app/MODEL3

# 出力先をホストにマウントできるようにだけ宣言（任意）
VOLUME ["/app/MODEL3/output_files"]

# ドキュメント用にポート（GUIは使わないので開けなくてもOK）
# EXPOSE 8000

# デフォルト実行。ログや成果物は /app/MODEL3/output_files 以下に出ます
CMD ["python", "VRPTW-main.py"]