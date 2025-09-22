# 軽量な公式Pythonイメージをベースに使用
FROM python:3.9-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /soybean_root_analysis

# 必要なPythonライブラリをインストール
RUN pip install --no-cache-dir opencv-python numpy

# スクリプトを実行するコマンド
# このコマンドは、docker-compose.ymlで上書きできます
CMD ["python", "Calibration.py"]