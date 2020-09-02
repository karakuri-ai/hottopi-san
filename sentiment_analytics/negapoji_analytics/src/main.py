
import csv
import os
from pydantic import BaseModel
import time
from typing import List


import pymongo
from pymongo import MongoClient

from bson.objectid import ObjectId
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse


from modules.make_dataloader import Make_DataLoader
from modules.sentiment_analytics import Analytics


app = FastAPI()

datasets_dict, TEXT = Make_DataLoader().make_ds_and_TEXT()
dataloaders_dict = Make_DataLoader().make_dl(datasets_dict=datasets_dict)
net_trained = Analytics().load()


client = MongoClient()
db = client.karakuri_db
mongo = db.nagapoji
mongo.create_index([("input", pymongo.ASCENDING)])
mongo.create_index([("output", pymongo.ASCENDING)])
mongo.create_index([("path", pymongo.ASCENDING)])
mongo.create_index([("status", pymongo.ASCENDING)])
mongo.create_index([("input_date", pymongo.ASCENDING)])


def make_d(input_file, output_file, output_file_path):
    output_file = "analized_" + input_file.filename
    # 2018/10/31 02:37 このような形式で表示
    input_date = time.strftime("%Y/%m/%d %H:%M", time.strptime(time.ctime()))
    objects = {
        "input_file": input_file.filename,
        "output_file": output_file,
        "path": output_file_path,
        "推論数": 0,
        "status": "読み込み中",  # 推論中はdbを更新できないため初期値を10%にしている
        "input_date": input_date,
    }
    return objects


class Job:
    def __init__(self, response_ids, all_output_file_path, files_list):
        self.files_list = files_list
        self.response_ids = response_ids
        self.all_output_file_path = all_output_file_path

    def __call__(self):

        self.analyze_and_write_csv(
            self.all_output_file_path, self.response_ids, self.files_list
        )

    def analyze_and_write_csv(self, all_output_file_path, response_ids, files_list):
        for i in range(len(files_list)):
            res_id = ObjectId(response_ids[i])
            output_file_path = all_output_file_path[i]
            text_list = list(files_list[i].decode("utf-8").splitlines())

            update = mongo.update_one(
                {"_id": res_id}, {"$set": {"推論数": len(text_list)}}
            )
            update = mongo.update_one({"_id": res_id}, {"$set": {"status": "推論中"}})

            np_score_dict = Analytics().predict_np(text_list=text_list)

            # np_score_dict.keys()にはtext_listのindex番号が入ってる。
            # エラーが起きて推論を飛ばした文があるためdictにあるのだけ書き込む
            keyslist = sorted(np_score_dict.keys())

            # 10%ずつmongodbにstatusに更新するための単位
            unit_update_mongo = int(len(keyslist) / 10)

            update = mongo.update_one({"_id": res_id}, {"$set": {"status": "書き込み中"}})

            with open(output_file_path, "w") as f:
                writer = csv.writer(f)
                print(res_id)
                for i in keyslist:
                    np, np_score = np_score_dict[i]
                    l = [text_list[i], np, str(np_score)]
                    writer.writerow(l)

        update = mongo.update_one({"_id": res_id}, {"$set": {"status": "完了"}})

        print("done")


@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    response_ids = []
    all_output_file_path = []
    files_list = []
    for file in files:
        content = await file.read()
        files_list.append(content)
        output_file = "analized_" + file.filename
        output_file_path = os.getcwd() + "/output_files/" + output_file
        all_output_file_path.append(output_file_path)

        res = mongo.insert_one(make_d(file, output_file, output_file_path))
        res_id = res.inserted_id
        response_ids.append(str(res_id))

    t = Job(
        response_ids=response_ids,
        all_output_file_path=all_output_file_path,
        files_list=files_list,
    )

    #推論処理をバックグラウンドへ
    background_tasks.add_task(t)

    return {"response": response_ids}
    


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


class Item(BaseModel):
    id: str
    value: str


class Message(BaseModel):
    message: str


@app.get("/download/{res_id}")  # ,
async def receive_file(res_id: str):
    findOne = mongo.find_one(filter={"_id": ObjectId(res_id)})
    try:
        if findOne["status"] == "完了":
            reponse_file = findOne["path"]
            return FileResponse(reponse_file)
        else:
            return {"status": findOne["status"]}
    except:
        findOne = mongo.find_one(filter={"_id": ObjectId("5e9697d0172e409f7fb7a930")})
        return {"status": findOne["status"]}


@app.get("/parse/{res_id}")  # ,
async def receive_file(res_id: str):
    findOne = mongo.find_one(filter={"_id": ObjectId(res_id)})
    # try:
    if findOne["status"] == "完了":
        reponse_file = findOne["path"]
        response = []
        with open(reponse_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                response.append(row)
        return {"reponse_file": response}

    else:
        return {"status": findOne["status"]}
    # except:
    #     findOne = mongo.find_one(filter={"_id": ObjectId("5e9697d0172e409f7fb7a930")})
    #     return {"status": findOne["status"]}


@app.get("/geturl/{res_id}")  # ,
async def receive_file(res_id: str):
    findOne = mongo.find_one(filter={"_id": ObjectId(res_id)})
    # try:
    if findOne["status"] == "完了":
        return {"download url": findOne}
    else:
        return {"status": findOne["status"]}
    # except:
    #     findOne = mongo.find_one(filter={"_id": ObjectId("5e9697d0172e409f7fb7a930")})
    #     return {"status": findOne["status"]}



#     response_model=Item,
#     responses={
#         404: {"model": Message, "description": "The item was not found"},
#         200: {
#             "description": "Item requested by ID",
#             "content": {
#                 "application/json": {
#                     "example":{"_id" : ObjectId("5e95ae3cf010ba8e5c577eb6"),
#                     "input_file" : "all_sentence_make_for_vocab.csv",
#                     "output_file" : "analized_all_sentence_make_for_vocab.csv",
#                     "path" : "/Users/ogurayuki/OldDesk/karakuri/NegaPojiTransformer/negapoji_analytics/src/output_files/analized_all_sentence_make_for_vocab.csv",
#                     "推論数" : 2813,
#                     "status" : "100%",
#                     "input_date" : "2020/04/14 21:36"
#                     }
#                 }
#             },
#         },
#     },
# )
# async def receive_file(res_id: str):
#     findOne = mongo.find_one(filter={'_id':ObjectId(res_id)})
#     if findOne['status']=='100%':
#         reponse_file=findOne['path']
#         return findOne, FileResponse(reponse_file)
#     else:
#         return {"status" : findOne['status'] + " 実行中"}
