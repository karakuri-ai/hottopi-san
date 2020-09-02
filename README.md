# ホットピさん

Slack 内の「ホットな投稿」を検知し，Slack の特定チャンネルに通知する Slackbot 「ホットピさん」

<img src="hottopi-san-icon.jpg" width="30%">

## 事前準備

1. Python3 が動く環境で
```
pip install -r requirements.txt
```
で依存ライブラリをインストールしてください．

2. Slackbotを作成し，以下の Scopes (Bot Token Scopes) を付与してください．
```
channels:history
chat:write
groups:history
groups:read
im:history
mpim:history
usergroups:read
users:read
```
また，ホットな投稿のサーチ対象としたいすべてのチャンネルに，
このボットを招待（アプリを追加）してください．

3. 以下の環境変数を設定してください．
```
HOTTOPI_WORKSPACE_NAME
    Slackのワークスペース名

HOTTOPI_SLACK_BOT_TOKEN
    作成した Slackbot の Bot User OAuth Access Token

HOTTOPI_NOTIFICATION_CHANNEL_ID
    通知する先のチャンネルID
```

## 使い方

```
python batch.py
```
で，過去 24 時間の「ホットな投稿」を標準出力に表示します．
（Slack には通知されません．）

```
python batch.py -s 70 -e 60 -n
```
で過去 70分前-60分前 のホットな投稿を標準出力に表示し，
さらに Slack に通知します．

サーバー上で crontab で
```
*/10 * * * * cd /path/to/this/directory; /path/to/python batch.py -s 70 -e 60 -n 1> log.txt
```
などと指定して定期実行させることで，継続的に監視できます．
スクリプトの実行間隔を（sの値）-（eの値）に一致させておくことで，重複や漏れなくホットな投稿を通知させることができます．


なお，引数の指定方法は
```
python batch.py -h
```
で確認できます．


## 投稿のホット度の算出法

「投稿のホット度」は

- その投稿へのリアクションのつき具合

- その投稿本文に対しての sentiment_analytics による感情分析（利用可能な場合）

を用いて算出しています．後者の詳細は，同名のディレクトリ内の README を参照ください．
