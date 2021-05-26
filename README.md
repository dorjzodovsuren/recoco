# 概要
このプロジェクトは Python 3.8 以上で動作検証済みです。

## 環境準備の手順
1. Install `virtualenv`:
```
$ pip install virtualenv
```

2. Open a terminal in the project root directory and run:
```
$ virtualenv env
```

3. Then run the command:
```
$ .\env\Scripts\activate
```

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. DB初期化
以下を実行すると test.db が作成されます。
```
$ python prepare_db.py
```

## アプリケーション起動手順

1. 以下でアプリケーションを起動します。
```
$ (env) python app.py
```

初期状態ではポート 5000 番でWebサーバが起動します。
変更する場合は app.run の port 指定を修正します。
```python
if __name__ == "__main__":
    app.run(debug=True, port=<desired port>)
```