import argparse
import os,re
import json,jsonlines
import numpy as np
from utils import *
from tqdm import tqdm
from Loading import LAMM_EVAL_3D
import random
import sys
sys.path.insert(0,'/media/kou/Data1/htc/FastChat/fastchat/serve')
from torch.utils.data import DataLoader, Dataset
import openai
openai.base_url = 'xxxxxxx'
openai.api_key = 'xxxxxx'

from openai import OpenAI

client = OpenAI(
    base_url="xxxxxx",
    api_key="xxxxxx"
)

Class_ALL = [
    "alarmclock",
    "apple",
    "armchair",
    "baseballbat",
    "basketball",
    "bed",
    "book",
    "boots",
    "bottle",
    "bowl",
    "box",
    "bread",
    "butterknife",
    "candle",
    "cart",
    "cellphone",
    "chair",
    "cloth",
    "clothesdryer",
    "coffeemachine",
    "coffeetable",
    "countertop",
    "creditcard",
    "cup",
    "desk",
    "desklamp",
    "desktop",
    "diningtable",
    "dishsponge",
    "dogbed",
    "doorway",
    "dresser",
    "dumbbell",
    "egg",
    "faucet",
    "floorlamp",
    "fork",
    "fridge",
    "garbagebag",
    "garbagecan",
    "houseplant",
    "kettle",
    "keychain",
    "knife",
    "ladle",
    "laptop",
    "laundryhamper",
    "lettuce",
    "microwave",
    "mug",
    "newspaper",
    "ottoman",
    "painting",
    "pan",
    "papertowelroll",
    "pen",
    "pencil",
    "peppershaker",
    "pillow",
    "plate",
    "plunger",
    "pot",
    "potato",
    "remotecontrol",
    "safe",
    "saltshaker",
    "shelvingunit",
    "sidetable",
    "sink",
    "soapbar",
    "soapbottle",
    "sofa",
    "spatula",
    "spoon",
    "spraybottle",
    "statue",
    "stool",
    "tabletopdecor",
    "teddybear",
    "television",
    "tennisracket",
    "tissuebox",
    "toaster",
    "toilet",
    "toiletpaper",
    "tomato",
    "tvstand",
    "vacuumcleaner",
    "vase",
    "washingmachine",
    "watch",
    "window",
    "winebottle"
]

# def get_completion(prompt, model="gpt-3.5-turbo"):
#     prompt = 'hello'
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=1,#this is the degree of randomness of the model'soutput
#     )
#     return response.choices[0].message["content"]
#
# print(completion('hello'))

def Navigation(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    panish = 1    #每个节点的损失约束,前1000个
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['positions']
        # if len(gt_objects)>=20:
        #     continue
        text = pred['text']
        points = parse_bbox_3d_Nav(text)
        cnt += 1
        len_a = len(gt_objects)
        len_b = len(points)
        if len_a < len_b:
            short = gt_objects
            long = points
        else:
            short = points
            long = gt_objects
        difference = cal_path_3d(short,long)
        if difference < panish*len(gt_objects):
            score += 1

    print(score / cnt)

# #多目标分类
# def grounding3d_eval(dataset, pred_data, thres=0.25):
#     score = 0
#     cnt = 0
#     scene_num = 0
#     Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
#     new_classdict = {}
#     for i in Detection:
#         flag =0
#         key = list(i.keys())[0]
#         if int(key) < 460 or int(key) > 500:
#             continue
#
#         new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in ["cabinet","bed","chair","sofa","diningtable","doorway","window","shelf", "painting","countertop","desk","fridge","toilet","sink","garbagecan"]]
#         new_classdict[key] = new_classlist
#
#
#     test = 1
#     collet_pred = []
#     if test:
#         one = {"id":'460'}
#         text = []
#         for i in pred_data:
#             if one["id"] == i['id']:
#                 text.append(i['text'])
#             else:
#                 one["text"] = text
#                 text = [i['text']]
#                 collet_pred.append(one)
#                 one = {"id": i['id']}
#         one["text"] = text
#         collet_pred.append(one)
#
#         pred_data = collet_pred
#
#     for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
#         text = pred['text']
#         meta_pre = gt['object']
#         cnt += len(text)
#         box_gt = new_classdict[gt['id']]
#
#         for pred,obj in zip(text,meta_pre):
#             cnt+=1
#             if obj['name'] in pred:
#                 for real_box in box_gt:
#                     if obj['name'] in real_box['name'].lower():
#                         iou = cal_aro_3d(obj['BoundingBox'], real_box["BoundingBox"])
#                         if iou>0.5:
#                             score += 1
#                             continue
#
#         # object_names = re.findall(r':(\w+)!', text)
#         # assert gt["id"] == pred["id"]
#         # preding = Pred[gt["id"]]
#         # pred_name = [i['name'] for i in preding]
#         # gt_name = [i['name'] for i in gt_objects]
#         # pred_box = [i['BoundingBox'] for i in preding]
#         # cnt += len(pred_box)  # gt_objects,pred_box
#         # # 按顺序多目标分类
#         # for pred,obj in zip(pred_name,object_names):
#         #     if pred==obj:
#         #         score += 1
#         #         break
#         #只做目标是否检测到
#         # for gt_info in gt_objects:
#         #     if gt_info['name'] in pred_name and gt_info['name'] in object_names:
#         #         class_box = [b for i,b in zip(pred_name,pred_box) if i == gt_info['name']]
#         #         cnt += len(class_box)-1
#         #         for index, point in enumerate(class_box):
#         #             iou = cal_aro_3d(gt_info['BoundingBox'], point)
#         #             if iou > thres:
#         #                 score += 1
#         #                 break
#         scene_num += 1
#     print(scene_num,score / cnt)

def grounding3d_eval(dataset, pred_data, thres=0.25):
    score = 0
    cnt = 0
    scene_num = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt["object"]
        text = pred['text']
        points = parse_bbox_3d_Vis(text)
        # if len(gt_objects) > 10:
        #     continue
        # if len(points) > 10:
        #     continue
        cnt += len(points)  # gt_objects,points
        for object_info in gt_objects:

            # if (not (object_info['name'].lower() in text.lower())):
            #     continue
            if (not (object_info['label'] in text.lower())):
                continue
            for index, point in enumerate(points):
                object_info['bbox'][:3] = (np.asarray(object_info['bbox'][:3])+np.asarray(object_info['bbox'][3:]))/2
                # point[:3] = np.asarray((point[0],point[2],point[1]))
                # point[:2] = np.asarray(point[:2])
                iou = cal_in_3d(object_info['bbox'], point)

                # object_info['BoundingBox'][:3] = (np.asarray(object_info['BoundingBox'][:3]) + np.asarray(
                #     object_info['BoundingBox'][3:])) / 2
                # iou = cal_in_3d(object_info['BoundingBox'], point)
                if iou > thres:
                    score += 1
                    break
        scene_num += 1
    print(scene_num,score / cnt)


#直接对Detection专家检测
# def grounding3d_eval(dataset, pred_data, thres=0.5):
#     score = 0
#     cnt = 0
#     metadata = json.load(open("/media/kou/Data1/htc/LAMM/data/metadata/Detection.json"))
#     for gt, pred in tqdm(zip(dataset, metadata), ncols=40):
#         gt_objects = gt["object"]
#         text = metadata[gt["id"]]
#         if len(gt_objects) > 30:
#             continue
#         text_name = [i["name"] for i in text]
#         # text = parse_bbox_3d_Vis(text)
#         cnt += len(points)#gt_objects,points
#         for object_info in gt_objects:
#             if not (object_info['name'].lower() in text_name):
#                 continue
#             # if not (object_info['label'] in text):
#             #     continue
#             for index, point in enumerate(text):
#                 #iou = cal_iou_3d(object_info['bbox'], point)
#                 iou = cal_aro_3d(object_info['BoundingBox'], point['BoundingBox'])
#                 if iou > thres:
#                     score += 1
#                     break
#     print(score / cnt)

def point2box(points):
    x_max = 0
    z_max = 0
    x_min = 100
    z_min = 100

    for i in points:
        if x_max < i['x']:
            x_max = i['x']
        if z_max < i['z']:
            z_max = i['z']
        if x_min > i['x']:
            x_min = i['x']
        if z_min > i['z']:
            z_min = i['z']
    y_mid = round(points[0]['y'],3)
    y_mid = y_mid/2
    h = y_mid*2

    x_mid = round((x_min + x_max)/2,3)
    z_mid = round((z_min + z_max) / 2,3)

    l = round(x_max - x_min,3)
    w = round(z_max - z_min,3)
    answer = [x_mid,z_mid,y_mid,l,w,h]
    return answer

def Rgrounding3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox_3d_Vis(text)
        cnt += len(bboxes)#gt_objects,bboxes
        # for object_info in gt_objects:
        # if not classification_acc(gt_objects['label'], text):
        #     continue
        # for bbox in bboxes:

        for object_info in gt_objects:
            #判断房间分类
            if (not classification_acc(object_info['label'], text)) and (not (object_info['label'].lower() in text.lower())):
                continue
            for index, point in enumerate(bboxes):
                iou = cal_iou_3d(object_info['bbox'], point)
                if iou > 0.5:
                    score += 1
                    break
    print(score / cnt)


def Vgrounding3d_eval(dataset, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
        gt_objects = gt['object']
        text = pred['text']
        bboxes = parse_bbox_3d_Vis(text)
        cnt += 1#gt_objects,bbox

        if cnt % 500 == 0:
            print(cnt,' : ',score / cnt)

        # for object_info in gt_objects:
        # if not classification_acc(gt_objects['label'], text):
        #     continue
        # for bbox in bboxes:
        if len(bboxes) != 1:
            continue
        if len(bboxes[0]) != 6:
            continue
        iou = cal_aro_3d(gt_objects, bboxes[0])
        # if iou > 0:
        #     print(iou)
        if iou > thres:
            score += 1

    print(score / cnt)

# def Vgrounding3d_eval(dataset, pred_data, thres=0.5):
#     score = 0
#     cnt = 0
#
#     Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
#     new_classdict = {}
#     for i in Detection:
#         flag =0
#         key = list(i.keys())[0]
#         if int(key) < 460 or int(key) >= 500:
#             continue
#         new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
#         new_classdict[key] = new_classlist
#
#     for gt, pred in tqdm(zip(dataset, pred_data), ncols=40):
#         gt_objects = gt['object']
#         text = pred['text']
#         cnt += 1  # gt_objects,bbox
#         match = re.search(r'obj(\d+)', text)
#         gtBox = new_classdict[gt['id']]
#         # 直接读obj
#         if match:
#             pre_num = int(match.group(1))
#             gtBox = gtBox[pre_num]['BoundingBox']
#             bboxes = gtBox
#
#         #直接读xyz
#         bboxes = parse_bbox_2d_Vis(text)
#         if len(bboxes):
#             bboxes = bboxes[0]
#         else:
#             continue
#
#         # 提取点2的xyz坐标
#         point2_xyz = gt_objects[:3]
#
#         # 计算两点之间的欧几里得距离
#         distance = math.sqrt(
#             (point2_xyz[0] - bboxes[0]) ** 2 + (point2_xyz[1] - bboxes[1]) ** 2 + (point2_xyz[2] - bboxes[2]) ** 2)
#
#         # 判断距离是否小于或等于1
#         if distance <= 1:
#             score += 1
#
#     print(score / cnt)

def grounding3d(dataset, pred_data):
    # Vgrounding3d_eval(dataset, pred_data, thres=0.25)
    Vgrounding3d_eval(dataset, pred_data, thres=0.5)

CHOICE = ['A', 'B', 'C', 'D', 'E', 'F']         # 6 choices in total

def VG_plus_acc(dataset,pred_data):
    import re
    score = 0.0
    testnum = 0
    pred_bbox = json.load(open("/media/kou/Data1/htc/LAMM/data/metadata/Detection.json"))
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        #gt_choice = gt['obj_num']   #误区,detection的num不是metadata的num
        pred_text = pred['text']
        pre_box = pred_bbox[gt['id']]
        match = re.search(r'obj(\d+)', pred_text)

        # if match:
        #     pre_num = int(match.group(1))
        #     if pre_num>=len(pre_box):
        #         continue
        #     pre_box = pre_box[pre_num]["BoundingBox"]
        #     iou = cal_iou_3d(gt['bbox'], pre_box)
        #     if iou>0.5:
        #         tmp_score = 1

        #直接读xyz
        bboxes = parse_bbox_2d_Vis(pred_text)
        if len(bboxes):
            bboxes = bboxes[0]
        else:
            continue

        # 提取点2的xyz坐标
        point2_xyz = gt['bbox'][:3]

        # 计算两点之间的欧几里得距离
        distance = math.sqrt(
            (point2_xyz[0] - bboxes[0]) ** 2 + (point2_xyz[1] - bboxes[1]) ** 2 + (point2_xyz[2] - bboxes[2]) ** 2)

        # 判断距离是否小于或等于1
        if distance <= 1:
            score += 1
        # if match and int(match.group(1)) == gt_choice:
        #     # 返回匹配到的数字部分
        #     gtbox = gt["bbox"]
        #     pre_box = pre_box[int(match.group(1))]["BoundingBox"]
        #     iou = cal_iou_3d(gtbox, pre_box)
        #     if iou>0.5:
        #         tmp_score = 1

        score += tmp_score
        testnum += 1
    print('vision: {}'.format(score / testnum))

def VQAvisionacc(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W')
    pattern_2 = re.compile(r'option [A-F]')
    pattern_3 = re.compile(r'\([A-F]\)')
    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    testnum = 0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        # if len(res_1) != 0:
        #     if check_option(res_1, gt_char):
        #         tmp_score = 1.0
        # elif len(res_2) != 0:
        #     if check_pattern2(res_2, gt_char):
        #         tmp_score = 1.0
        # elif len(res_3) != 0:
        #     if check_option(res_3, gt_char):
        #         tmp_score = 1.0
        # elif check_text(pred_text, gt['gt_choices'], gt_choice):
        #     tmp_score = 1.0
        if check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
            # print(testnum ,":", gt["sentences"])
            # print(pred['text'])
            # print("####################################################")
        score += tmp_score
        testnum += 1
    print('vision: {}'.format(score / testnum))

def Positoinacc(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-D]\)?\W|the answer is \(?[A-D]\)?\W')
    pattern_2 = re.compile(r'\([A-D]\)')
    pattern_3 = re.compile(r'[A-D]')

    def check_text(text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-2]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    testnum = 0

    # from cli import vicuna
    # from inference import chat
    # model, tokenizer, chatio = vicuna()
    answer_save = {}
    for gt, pred,index in tqdm(zip(dataset, pred_data,range(len(dataset)))):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        pred_text = pred['text']
        pred_text = pred_text
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)

        prompt = "Accurately understand positional information firstly, then determine whether the following two sentences express the same or different positional relationship. Be as concise as possible, the same or different\n"
        # input_text = prompt + "Sentence1: " + gt["sentences"][5:] + "\nSentence2: " + gt["sentences"][5:]
        input_text = prompt+"Sentence1: "+ gt["sentences"][5:]+"\nSentence2: "+pred_text
        # answer = chat(input_text,model, tokenizer,chatio)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
        answer = completion.choices[0].message.content
        # print(completion.choices[0].message.content)
        if "true" in answer.lower() or "same" in answer.lower():
            tmp_score = 1.0
            if "not" in answer.lower():
                tmp_score = 0
        with open("/media/kou/Data1/htc/LAMM/answers/Gpt_results/positionrelation.jsonl",
                  'a') as f:
            f.write(json.dumps({tmp_score:answer}) + "\n")
            f.flush()

        score += tmp_score
        testnum += 1
    print('vision: {}'.format(score / testnum))
    
def Counting(dataset,pred_data):
    import re
    pattern_1 = re.compile(r'The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W')
    pattern_2 = re.compile(r'ANSWER: [A-F]')
    pattern_3 = re.compile(r'\([A-F]\)')
    TwoEnglish = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                  '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
                  '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen',
                  '19': 'nineteen', '20': 'twenty'}
    def check_text(text, choices, gt_id):
        text = text.lower()
        if str(choices[gt_id]) not in text and TwoEnglish[str(choices[gt_id])] not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if str(choice) in text:
                return False
        return True
    def check_option(res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    score = 0.0
    for gt, pred in tqdm(zip(dataset, pred_data)):
        tmp_score = 0
        gt_choice = gt['gt_choice']
        gt_char = CHOICE[gt_choice]
        answer = gt["gt_choices"][gt["gt_choice"]]
        pred_text = pred['text']
        pred_num = re.findall(r'\d+(?:\.\d+)?', pred_text)
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if len(res_1) != 0:
            if check_option(res_1, gt_char):
                tmp_score = 1.0
        elif len(res_2) != 0:
            if check_pattern2(res_2, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif len(res_3) != 0:
            if check_option(res_3, gt_char):
                tmp_score = 1.0
        elif len(pred_num)==1 and str(answer)==pred_num[0]:
            tmp_score = 1.0
        elif check_text(pred_text, gt['gt_choices'], gt_choice):
            tmp_score = 1.0
        score += tmp_score
    print('vision: {}'.format(score / len(dataset)))

dataset2evalfunc = {
    'Detection': grounding3d_eval,
    'MyData': VQAvisionacc,
    'ScanRefer': grounding3d,
    'ScanQA_multiplechoice': VQAvisionacc,
    'Counting': Counting,
    'Classification': VQAvisionacc,
    'PositionRelation':Positoinacc,
    'VisualGrounding':grounding3d,
    'Navigation':Navigation,
    'RoomDetection':Rgrounding3d_eval,
    "VisualGrounding_plus":VG_plus_acc
}

def collate_fn(batch):
    res = dict()
    keys = batch[0].keys()
    for key in keys:
        res[key] = [data[key] for data in batch]
    return res

if __name__ == "__main__":
    root_path = '/media/kou/Data1/'
    # root_path = 'G:\event\htc/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Lamm")#Lamm,Mydata
    parser.add_argument("--task-name", default="VisualGrounding")#Detection,Counting,Classification,PositionRelation
                                                                # VisualGrounding,RoomDetection,Navigation
                                                                #VisualGrounding_plus
    parser.add_argument('--answer-file', default=root_path+r"htc/LAMM_v0/answers")
    parser.add_argument('--base-data-path', default=root_path+r"htc/MYDATA/BenchMark/Task/Task_Reconstruct/Test")
    # parser.add_argument('--base-data-path', default=root_path+r"htc/MYDATA/BenchMark/Task/Test")
    args = parser.parse_args()
   
    dataset_name = args.dataset_name
    task_name = args.task_name
    dataset = LAMM_EVAL_3D(args.base_data_path,
                           dataset_name,
                           task_name
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False,
                            collate_fn=collate_fn)
    dataset = dataloader.dataset

    eval_func = dataset2evalfunc[task_name]


    # if args.answer_file.endswith('.jsonl'):
    # if task_name == 'Navigation':
    #     jonal = r'G:\event\htc\LAMM\answers\Navigation\Navigation_Mydata.jsonl'
    # if task_name == 'VG':
    #     jonal = r'G:\event\htc\LAMM\answers\VG\VG_Mydata.jsonl'
    # if task_name == 'Counting':
    #     jonal = r'G:\event\htc\LAMM\answers\Counting\Counting.jsonl'
    # if task_name == 'Class':
    #     jonal = r'G:\event\htc\LAMM\answers\Class\Class.jsonl'
    if task_name == 'Navigation':#PositionRelation
        jonal = r'/media/kou/Data1/htc/LAMM_v1/answers/Navigation_Finetune_600/Navigation_Mydata.jsonl'
        pred_data = jsonlines.Reader(open(jonal))
    elif task_name == 'VisualGrounding_plus':#PositionRelation
        jonal = r'/media/kou/Data1/htc/LAMM_v1/answers/VG_Finetune_600/Navigation_Mydata.jsonl'
        pred_data = jsonlines.Reader(open(jonal))
    elif task_name == 'Classification' or 1:
        file_ext = '.jsonl'
        file_name = task_name  + file_ext
        # args.answer_file = os.path.join(args.answer_file, file_name)
        # args.answer_file = "/media/kou/Data1/htc/LAMM_v0/answers/answer/Detection_Mydata.jsonl"
        args.answer_file = "/media/kou/Data1/htc/LAMM_v0/answers/answer/VisualGrounding_Mydata.jsonl"
        pred_data = jsonlines.Reader(open(args.answer_file, 'rb'))
    elif args.answer_file.endswith('.json'):
        pred_data = json.load(open(args.answer_file,'rb'))
    else:
        file_ext = '.json'
        file_name = task_name  + '_'+args.dataset_name+file_ext
        args.answer_file = os.path.join(args.answer_file,task_name, file_name)
        pred_data = json.load(open(args.answer_file, 'rb'))
    print(f'Eval [{args.answer_file}] on {dataset_name}')
    eval_func(dataset, pred_data)
