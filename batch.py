"""
1. 最近の投稿を取得
2. ホットな投稿を抽出
3. 抽出されたホット投稿の情報をSlackに通知
"""
import requests
import argparse
import datetime
import time
import os
import numpy as np

parser = argparse.ArgumentParser(description='hoge')
parser.add_argument('--start', '-s', type=int, default=1440, help='ログ取得範囲の始点日時を現在の何分前とするか．デフォルト：1440(分前)')
parser.add_argument('--end', '-e', type=int, default=0, help='ログ取得範囲の終点日時を現在の何分前とするか．デフォルト：0(分前)')
parser.add_argument('--threshold', '-t', type=float, default=8, help='ホットスコアの閾値．デフォルト：8')
parser.add_argument('--notify', '-n', action='store_true', help='通知するか否か．デフォルト：False')
args = parser.parse_args()


now_timestamp = datetime.datetime.now().timestamp()
start_timestamp = now_timestamp - 60 * args.start
end_timestamp = now_timestamp - 60 * args.end
assert start_timestamp <= end_timestamp
threshold = args.threshold
flg_notify = args.notify


WORKSPACE_NAME = os.environ.get("HOTTOPI_WORKSPACE_NAME", "")  # 'xxxxxxxxxxxx'
SLACK_BOT_TOKEN = os.environ.get("HOTTOPI_SLACK_BOT_TOKEN", "")  # 'xoxb-xxxxxxxxxx-xxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxx'
NOTIFICATION_CHANNEL_ID = os.environ.get("HOTTOPI_NOTIFICATION_CHANNEL_ID", "")  # 'xxxxxxxxxxx'


def _check_response_and_parse_json(res):
    """
    Slack からの応答をチェックします．具体的には
        - HTTPステータスがダメなケースを raise します
        - ok が False なケースを raise します
    上記が問題なければ json を dict 形式で返します
    ----
    Args:
        res (requests.get の返り値)
    Returns:
        (dict)
    """
    res.raise_for_status()
    res.encoding = 'utf-8'
    json_data = res.json()
    if not json_data.get('ok', False):
        raise RuntimeError("Response with ok==False.\n Response: {}".format(json_data))
    return json_data


def get_channels(types):
    """
    Args:
        types (str)
            'public_channel' または 'private_channel'
            （Reference によれば 'public_channel,private_channel' と指定すると一括で取れるらしいのだが失敗した）
    Returns:
        (list of dict) 各要素はチャンネル情報．id, name などを要素に含む．
    """
    params = {
        'token': SLACK_BOT_TOKEN,
        'types': types
    }
    res = requests.get('https://slack.com/api/conversations.list', params=params)
    json_data = _check_response_and_parse_json(res)
    return json_data['channels']


def get_history(channel_id, oldest, latest):
    """
    Args:
        channel_id (str)
        oldest (int): 開始日時の UNIX タイムスタンプ
        latest (int): 終了日時の UNIX タイムスタンプ
    Returns:
        (list of dict)
            各要素は一投稿の情報であり，以下の要素を含む
                type (str) 通常の投稿の場合は 'text'
                text (str) 本文．
                reactions (list of dict) リアクション情報．各要素が各絵文字の情報．以下の要素を含む
                    name (str) 絵文字の名前
                    count (int) 得票数
                    users (list of str) 各要素はユーザID.
                ts (str) タイムスタンプ（小数点以下6桁まで）
                channel_id (str) 取得元のチャンネルID
                permalink (str) 投稿へのパーマリンク
    """
    params = {
        'token': SLACK_BOT_TOKEN,
        'channel': channel_id,
        'latest': latest,
        'oldest': oldest,
    }
    res = requests.get('https://slack.com/api/conversations.history', params=params)
    json_data = _check_response_and_parse_json(res)
    messages = json_data['messages']
    for m in messages:
        m['channel_id'] = channel_id
        m['permalink'] = "https://{}.slack.com/archives/{}/p{}".format(
            WORKSPACE_NAME,
            m['channel_id'],
            m['ts'].replace('.', '')
        )
    return messages


def notify(channel_id, text):
    """
    hot な投稿の Slack への通知．
    ----
    Args:
        channel_id (str)
        text (str)
    Returns:
        None
    """
    params = {
        "token": SLACK_BOT_TOKEN,
        "channel": channel_id,
        "text": text
    }
    res = requests.post("https://slack.com/api/chat.postMessage", data=params)
    json_data = _check_response_and_parse_json(res)
    return json_data


class NegaposiEstimator:
    """
    ネガポジ判定用のクラス
    """
    def __init__(self):
        import sys
        self.relpath = "./sentiment_analytics/negapoji_analytics/src"
        sys.path.append(self.relpath)
        from modules.sentiment_analytics import Analytics
        from modules.make_dataloader import Make_DataLoader
        self.negaposi = Analytics(TEXT_pkl_path=self.relpath + "/data/TEXT.pkl")

    def predict(self, text_list):
        """
        与えられたテキストたちのネガポジを判定し，判定結果を返します
        ----
        Args:
            text_list (list of str):
                判定したいテキストの一覧．
        Returns:
            ret (list of float):
                各テキストのポジティブ度を格納したリスト．長さは text_list と同じとなる．
                text_list[i] についての判定結果が ret[i] に格納される．

        Caution:
            クエリ文字列が長すぎる場合などに，内部で呼び出している Analytics class の predict_np 関数が
            判定結果を返さない場合があります．その場合には，この関数では 0.0 を返すようにしています．
        """
        rslt = self.negaposi.predict_np(text_list, save_path=self.relpath + "/weights/20202.pth")
        ret = []
        for i in range(len(text_list)):
            if i in rslt:
                ret.append(rslt[i][1] if rslt[i][0] == 'Positive' else -rslt[i][1])
            else:
                ret.append(0.0)
        return ret


try:
    NPE = NegaposiEstimator()
except:
    NPE = None
    print("WARNING: NegaposiEstimator is not available.")
    pass


def compute_hotness(submission_infos):
    """
    各投稿の hotness を評価する関数
    ----
    Args:
        submission_info (list of dict) 各要素には以下が含まれる
            text (str)
            reactions (list of dict) 各要素に以下が含まれる
                count (int)
    Returns:
        scores (list of float)
    """
    reaction_scores = []
    for info in submission_infos:
        counts = np.array([_.get('count', 0) for _ in info.get('reactions', {})])
        score = np.sum(np.sqrt(counts))
        reaction_scores.append(score)

    if NPE is not None:
        negaposi_scores = NPE.predict([
            info['text'] for info in submission_infos
        ])
    else:
        negaposi_scores = np.zeros(len(submission_infos))

    scores = [r + n for r, n in zip(reaction_scores, negaposi_scores)]
    return scores


if __name__ == '__main__':
    channels = get_channels('public_channel') + get_channels('private_channel')

    submissions = []
    for i, chn in enumerate(channels):
        if chn['is_member']:
            print(chn['name'])
            messages = get_history(chn['id'], start_timestamp, end_timestamp)
            time.sleep(1)
            submissions += messages
            print('    => {} 投稿を抽出'.format(len(messages)))

    # submissions (list of dict) に score 付与
    hotness = compute_hotness(submissions)
    for s, h in zip(submissions, hotness):
        s['hotness'] = h

    # hotness 順でソート
    submissions = sorted(submissions, key=lambda _:_['hotness'], reverse=True)

    if len(submissions) > 0:
        print("{} submissions found.".format(len(submissions)))
        print("max hotness: {}".format(submissions[0]['hotness']))
    else:
        print("no submissions found!")

    for s in submissions:
        if s['hotness'] < threshold:
            break
        print("-------------------------------")
        print(s['text'])
        print("URL: {}".format(s['permalink']))
        print("hotness: {:.1f}".format(s['hotness']))
        print()
        if flg_notify:
            text = "【ホット度{}】{}{}".format(
                # ":hottopi1:" * int(s['hotness']/threshold), 
                int(s['hotness']),
                "ホットなトピックですよ！",
                s['permalink']
            )
            notify(
                NOTIFICATION_CHANNEL_ID,
                text
            )
            time.sleep(3)

