import sys
import math
import os
import json
import numpy as np
from random import sample
import random
import re
import math
import open3d as o3d
import multiprocessing
import jsonlines

dirpath = os.path.split(os.path.realpath(__file__))[0]

def getdir(p0,p1):
    x0,y0,z0 = p0[:3]
    x1, y1, z1 = p1[:3]
    dis = 0.5
    if abs(y0-y1)>dis and abs(x0-x1)<dis and abs(z0-z1)<dis:#上下
        if y0 > y1:
            return random.randint(240,269),0
        elif y0 < y1:
            return random.randint(210,239),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)>dis and abs(z0-z1)<dis:#左右
        if x0 > x1:
            return random.randint(60,89),0
        elif x0 < x1:
            return random.randint(30,59),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)<dis and abs(z0-z1)>dis:#前后
        if z0 > z1:
            return random.randint(90,119),0
        elif z0 < z1:
            return random.randint(120,149),0
    elif abs(y0-y1)<2*dis and abs(x0-x1)>dis and abs(z0-z1)>dis:#斜方向
        if z0 > z1 and x0 > x1:
            return random.randint(150,179),0#a在b左前方
        elif z0 > z1 and x0 < x1:
            return random.randint(180,209),0#a在b右前方
        elif z0 < z1 and x0 > x1:
            return random.randint(180,209),1#b在a左后方
        elif z0 < z1 and x0 < x1:
            return random.randint(150,179),1#b在a右后方
    else:
        return 0,-1

def getrandom(x,y):
    return random.randint(x, y)

def getsinglejson(src_id,id,pcl_path,newchat,task_type):
    """
    :param src_id:
    :param id:
    :param pcl_path:
    :param newchat:
    :return:
    """
    singlejson = {}
    singlejson["src_id"] = str(src_id)
    singlejson["id"] = str(id)
    singlejson["pcl"] = pcl_path
    singlejson["conversations"] = newchat
    singlejson["task_type"] = task_type
    singlejson["src_dataset"] = "Mydata"
    return singlejson

def getVGanswer(GT,Answer,Question):

    out = ''
    answer = ''
    for index,list in enumerate(GT):
        name = list['name']
        value = list['BoundingBox']
        x = round(value[0],1)
        y = round(value[1], 1)
        z = round(value[2], 1)
        random_number = random.randint(30, 59)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{C}", "(obj"+str(index)+"):"+name, temp)
        temp = re.sub(r"{P}", str([x,y,z]), temp)
        answer = temp
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        out = Question[random_number]
        out = re.sub(r"{C}", name, out)

    return answer,out

def getcouting(GT,Answer):
    answer = ''
    for list in GT:
        name, num = list
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{C}", name, temp)
        temp = re.sub(r"{N}", str(num), temp)
        answer += temp

    return answer

def getroom(GT,Answer):
    answer = ''
    # N = len(GT)
    # random_number = random.randint(30, 59)
    # random_number = str(random_number)
    # temp = Answer[random_number]
    # answer = re.sub(r"{N}", str(N), temp)
    for list in GT:
        R, P = list
        random_number = random.randint(0, 29)
        random_number = str(random_number)
        temp = Answer[random_number]
        temp = re.sub(r"{R}", R, temp)
        temp = re.sub(r"{P}", str(P), temp)
        answer = answer + temp

    return answer

def chouyang(GT,Answer):
    answer = ''
    _, obj0 = GT[getrandom(0, len(GT) - 1)]
    _, obj1 = GT[getrandom(0, len(GT) - 1)]
    p0 = str(tuple(list(obj0.values())[0]))
    p1 = str(tuple(list(obj1.values())[0]))
    while (obj0 == obj1):
        _,obj1 = GT[getrandom(0, len(GT) - 1)]
    head0 = Answer[str(getrandom(0, 29))]
    head1 = Answer[str(getrandom(0, 29))]
    head0 = re.sub(r"{C}", list(obj0.keys())[0], head0)
    head0 = re.sub(r"{P}", p0, head0)
    head1 = re.sub(r"{C}", list(obj1.keys())[0], head1)
    head1 = re.sub(r"{P}", p1, head1)
    answer = head0 + head1
    return answer,obj0,obj1

def chouyang_train(simgt,Answer):
    class_list = list(simgt.keys())
    class_num = random.sample(range(0,len(class_list)),2)
    answer = ''
    for a,b in zip(class_num[0::2],class_num[1::2]):
        class_a = class_list[a]
        class_b = class_list[b]
        pos_a = simgt[class_a]['BoundingBox']
        pos_b = simgt[class_b]['BoundingBox']

        direaction, flag = getdir(pos_a, pos_b)

        if flag == -1:
            continue
        if flag == 1:
            temp = class_a
            class_a = class_b
            class_b = temp
        dir = Answer[str(direaction)]
        answer = re.sub(r"C1", class_a, dir)
        answer = re.sub(r"C2", class_b, answer)

    return answer,direaction,class_a,class_b,flag

def chouyang_test(class_list,position,Answer):

    class_num = random.sample(range(0,len(class_list)),2)
    answer = ''
    for a,b in zip(class_num[0::2],class_num[1::2]):
        class_a = class_list[a]
        class_b = class_list[b]
        pos_a = position[class_a]
        pos_b = position[class_b]


        direaction, flag = getdir(pos_a, pos_b)

        if flag == -1:
            continue
        if flag == 1:
            temp = class_a
            class_a = class_b
            class_b = temp
        dir = Answer[str(direaction)]
        answer = re.sub(r"C1", class_a, dir)
        answer = re.sub(r"C2", class_b, answer)

    return answer,direaction,class_a,class_b,flag

def get_answer(Answer,flag,direaction,class_a,class_b):
    false = []
    if direaction>=30 and direaction<=149:      #前后左右
        if direaction>=30 and direaction<=89:
            fa = Answer[str(random.randint(90, 119))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(120, 149))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
        else:
            fa = Answer[str(random.randint(30, 59))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(60, 89))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)

    if direaction>=150 and direaction<=209:      #斜向

        if direaction>=150 and direaction<=179:
            fa = Answer[str(random.randint(180, 209))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(180, 209))]
            fa = re.sub(r"C1", class_a, fa)
            fa = re.sub(r"C2", class_b, fa)
            false.append(fa)
        else:
            fa = Answer[str(random.randint(150, 179))]
            fa = re.sub(r"C1", class_b, fa)
            fa = re.sub(r"C2", class_a, fa)
            false.append(fa)
            fa = Answer[str(random.randint(150, 179))]
            fa = re.sub(r"C1", class_a, fa)
            fa = re.sub(r"C2", class_b, fa)
            false.append(fa)
    fa = Answer[str(direaction)]  # 反向答案
    fa = re.sub(r"C1", class_b, fa)
    fa = re.sub(r"C2", class_a, fa)
    false.append(fa)
    random.shuffle(false)
    return false

def unique_name_mask(lst):
    name_count = {}  # 用于记录每个name出现的次数

    # 统计每个name的出现次数
    for item in lst:
        name = item.get('name', '')  # 获取字典中的name值
        if name:
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1

    # 创建布尔类型的掩码列表，标记唯一存在的name元素
    mask = []
    for item in lst:
        name = item.get('name', '')
        if name:
            if name_count[name] == 1:
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(False)  # 如果字典中没有name键，则默认为False

    return mask


def get_testPosition(answer,false_list,count,template, class_a, class_b,id):
    result = {}
    result["question_id"] = count

    result["pcl"] = template['pcl'][:-7]+str(id)+'.npy'

    result["id"] = str(id)
    result["src_dataset"] = "Mydata"

    answer_pos = random.randint(0,3)
    result["gt_choice"] = answer_pos

    false_list.insert(answer_pos, answer)
    gt_choices = false_list
    result["gt_choices"] = false_list

    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) "}
    result["sentences"] = answer_query[str(answer_pos)] + answer

    query = "What is the positional relationship of "+class_a+" and "+class_b+"?"+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + str(i)
    result["query"] = query

    return result

def getPosrealtion(GT,Answer):#GT,pos,Answer
    answer,direaction,class_a,class_b,flag = chouyang_train(GT,Answer)
    # answer, obj0, obj1,_,_ = chouyang_test(GT, pos,Answer)
    # direaction,flag = getdir(tuple(list(obj0.values())[0]),tuple(list(obj1.values())[0]))
    num = 0
    while(flag == -1):#重来
        #print("重新抽样")
        if num>=100000:
            return 0
        num += 1
        answer,direaction,class_a,class_b,flag = chouyang_train(GT,Answer)

    return answer,class_a,class_b

def Train_PositionRelation():
    outjson = []
    Q, A, GT, result = filepath('PositionRelation')
    Question, Answer, GT = loading(Q, A, GT)

    for id in range(0, 460):
        GTid = str(id)
        simpleGT = GT[str(id)]
        for x in range(4):
            answer,class_a,class_b = getPosrealtion(simpleGT, Answer)
            random_number = random.randint(0, 29)
            random_number = str(random_number)
            question = Question[random_number]
            question = re.sub("{C1}", class_a, question)
            question = re.sub("{C2}", class_b, question)
            pcl_path = "scene/"+GTid+".npy"
            conversations = [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]
            singlejson = getsinglejson(str(id), str(x), pcl_path, conversations, "PositionRelation")

            outjson.append(singlejson)

        print(id,"OK")

    return result, outjson

def get_testClass(gt,num):
    result = {}
    src_id = re.sub(r"_.*","",gt)

    classname = re.sub(r".*\d_", "", gt)
    classname = re.sub(r"\d.*", "", classname)
    result["question_id"] = num
    result["pcl"] = "Objects/"+gt+".npy"

    result["id"] = src_id
    result["src_dataset"] = "Mydata"

    path = "/media/kou/Data3/htc/dataset/Object/my_names.json"
    ALL = json.load(open(path,'r'))
    random_number = random.sample(range(0,len(ALL)),5)
    random_class = [ALL[i] for i in random_number]
    while (classname in random_class):
        random_number = random.sample(range(0, len(ALL)), 5)
        random_class = [ALL[i] for i in random_number]

    answer = random.randint(0,5)        #答案序号
    result["gt_choice"] = answer

    gt_choices = random_class.insert(answer, classname)#答案列表
    gt_choices = random_class
    result["gt_choices"] = gt_choices
    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) ","4":" (E) ","5":" (F) "}
    # result["sentences"] = answer_query[str(answer)] + classname

    # Q = root + "/BenchMark/Task/Template/Q_Classification.json"
    # q = open(Q, 'r')
    # Question = json.load(q)
    # Question = Question[str(random.randint(0,29))]
    Question = "What's the 3D point cloud about?"
    query = Question+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + i
    result["query"] = query
    result["sentences"] = answer_query[str(answer)]+classname
    # result["query"] = Question
    # result["sentences"] = classname
    return result

def get_trainClass(name):
    classnum = re.sub(r"_.*","",name)
    classname = re.sub(r".*\d_", "", name)
    classname = re.sub(r"\d.*", "", classname)

    Q = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Classification.json"
    q = open(Q, 'r')
    Question = json.load(q)
    Question = Question[str(random.randint(0, 29))]
    A = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/A_Classification.json"
    a = open(A, 'r')
    Answer = json.load(a)
    Answer = Answer[str(random.randint(0, 29))]
    Answer = re.sub("{C}", classname, Answer)

    #物体对齐数据集
    Answer = classname

    conversation = [{"from": "human", "value": Question},{"from": "gpt","value": Answer}]
    pcl_path = "Objects/" + name+".npy"

    single = getsinglejson(classnum, classnum, pcl_path, conversation,"Classification3d")

    return single

def get_testCounting(gtnum,index,objclass,num):
    result = {}
    result["question_id"] = index + 1
    result["pcl"] = "scene/"+str(gtnum)+'.npy'


    result["id"] = str(gtnum)
    result["src_dataset"] = "Mydata"

    path = "G:\event\htc\MYDATA\BenchMark\Task\GT\Counting.json"

    random_number = random.sample(range(1,10),5)

    while (num in random_number):
        random_number = random.sample(range(1, 10), 5)

    answer_pos = random.randint(0,5)
    result["gt_choice"] = answer_pos

    random_number.insert(answer_pos, num)
    gt_choices = random_number
    result["gt_choices"] = gt_choices

    answer_query = {"0":" (A) ","1":" (B) ","2":" (C) ","3":" (D) ","4":" (E) ","5":" (F) "}
    result["sentences"] = answer_query[str(answer_pos)] + str(num)

    query = "How many "+objclass+" are in the scene?"+" \n Options: "
    for index,i in enumerate(gt_choices):
        query = query + answer_query[str(index)] + str(i)
    result["query"] = query

    return result

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

def filepath(Class):
    root = "/media/kou/Data1/htc/MYDATA"
    # root = "/media/cvlab/Data/htc/MYDATA"
    Q = root + "/BenchMark/Task/Template/Q_"+Class+".json"
    A = root + "/BenchMark/Task/Template/A_"+Class+".json"
    GT = root + "/BenchMark/Task/GT/"+Class+".json"

    if Class == "Detection":
        A = root + "/BenchMark/Task/Template/A_" + "VisualGrounding" + ".json"
        GT = root + "/BenchMark/Task/GT/" + "Detection" + ".json"

    if Class == "VisualGrounding":
        GT = root + "/BenchMark/Task/GT/"+"VisualGrounding"+".json"
    if Class == "PositionRelation":
        GT = root + "/BenchMark/Task/GT/"+"VisualGrounding"+".json"

    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    return Q,A,GT,result

def loading(Q,A,GT):
    q = open(Q, 'r')
    Question = json.load(q)
    a = open(A, 'r')
    Answer = json.load(a)
    gt = open(GT, 'r')
    GT = json.load(gt)
    return Question,Answer,GT

# ## SCaption and SVQA
# for one in GT:
#     one['query'] = "Generate 5 round Q&A conversation of the given point cloud."
#     # one['sentences'] = one['sentences'][5:]
#     outjson.append(one)

class_mapping = {
    "cabinet": 0,
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "diningtable": 4,
    "doorway": 5,
    "window": 6,
    "shelf": 7,
    "painting": 8,
    "countertop": 9,
    "desk": 10,
    # "curtain": 11,  #
    "fridge": 12,
    # "showercurtrain": 13,  #
    "toilet": 14,
    "sink": 15,
    # "bathtub": 16,  #
    "garbagecan": 17,
}

def getanswerDe(oneGT,Answer):
    answer = ''
    new_gt = []
    for i in oneGT:
        if not len(i):
            continue
        name, bbox = i['name'], i['BoundingBox']
        if name.lower() not in class_mapping.keys():
            continue
        new_gt.append(i)
    oneGT = new_gt

    if len(oneGT)>=10:
        class_num = random.randint(2, 10)
        random_list = random.sample(range(0, len(oneGT)), class_num)
        oneGT = [oneGT[i] for i in random_list]

    i = 0
    box = []
    for one in oneGT:
        name, bbox = one['name'],one['BoundingBox']
        answer += f"(obj{str(i)}):{name.lower()}! "
        box.append(bbox)
        i += 1
        # obj = random.randint(0,29)
        # obj = Answer[str(obj)]
        # singleans = re.sub("{C}", name, obj)
        # singleans = re.sub("{P}", str(bbox), singleans)
        # answer += singleans
    return answer,box

def getanswerRoomDe(oneGT,Answer):
    answer = ''
    for name,bbox in oneGT.items():
        bbox = point2box(bbox)
        obj = random.randint(0,29)
        obj = Answer[str(obj)]
        singleans = re.sub("{R}", name, obj)
        singleans = re.sub("{P}", str(bbox), singleans)
        answer += singleans
    return answer



# Test Navigation  ##
def Test_Navigation():
    countnum = 0
    outjson = []
    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    GT_path = r"/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Navigation.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Navigation.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():

        if int(key) <= 459 or int(key) >= 500:
            continue
        # oneGT = list(value.values())
        for one in value:
            for name,positions in one.items():
                singlejson = {}
                countnum += 1
                singlejson["positions"] = positions

                ans_num = random.randint(0,29)
                que = Question[str(ans_num)]
                que = re.sub(r"{C}", name, que)
                que = re.sub(r"{P}", str(positions[0]), que)
                singlejson["question"] = que
                singlejson["query"] = que
                singlejson["question_id"] = countnum
                pcl_path = "scene/" + str(key) + '.npy'
                singlejson["pcl"] = pcl_path
                singlejson["id"] = str(key)
                singlejson["src_dataset"] = "Mydata"
                outjson.append(singlejson)

            print(key,"OK")
    return result, outjson

## Test RoomDetection  ##
def Test_RoomDetection():
    outjson = []
    result = "G:\event\htc\MYDATA\BenchMark\Task\Task_Reconstruct\Temp.json"
    countnum = 0
    GT_path = "G:\event\htc\MYDATA\BenchMark\Task\GT\RoomDetection.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/MYDATA/BenchMark/Task/Template_v0\Q_RoomDetection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():

        if int(key) <= 459 or int(key) >= 500:
            continue
        # oneGT = list(value.values())
        object = []
        for label,one in value.items():
            one = point2box(one)
            object.append({'label':label,'bbox':one})
        singlejson = {}
        countnum += 1
        # answer =one['BoundingBox']
        singlejson["object"] = object

        ans_num = random.randint(30,59)
        que = Question[str(ans_num)]
        # que = re.sub(r"{C}", one["name"], que)
        singlejson["question"] = que
        singlejson["query"] = que + " Please locate rooms' position with the coordinate of center x, y, z and its length, width and height,represented as (x,y,z,l,w,h)"
        singlejson["question_id"] = countnum
        pcl_path = "scene/" + str(key) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
#
# Test VisualGrounding  ##
def Test_VisualGrounding():
    countnum = 0
    outjson = []
    result = "G:\event\htc\MYDATA\BenchMark\Task\Task_Reconstruct\Temp.json"
    Onlyobj = r"G:\event\htc\MYDATA\BenchMark\Task\GT\Counting.json"
    oo = open(Onlyobj,"r")
    Only = json.load(oo)
    GT_path = "G:\event\htc\MYDATA\BenchMark\Task\GT\VisualGrounding.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Question_path = "/MYDATA/BenchMark/Task/Template_v0\Q_VisualGrounding.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)

    for key,value in GT.items():
        if int(key) <= 459 or int(key) >= 500:
            continue
        oneGT = list(value.values())
        for one in oneGT:
            if one["name"].lower() in [name for name,num in Only[int(key)].items() if num == 1]:
                pass
            else:
                continue
            countnum += 1
            singlejson = {}
            answer =one['BoundingBox']
            singlejson["object"] = answer

            ans_num = random.randint(0,29)
            que = Question[str(ans_num)]
            que = re.sub(r"{C}", one["name"], que)
            singlejson["question"] = que
            singlejson["query"] = que + " Please locate its position with the coordinate of center x, y, z and its length, width and height."
            singlejson["question_id"] = countnum
            pcl_path = "scene/" + str(key) + '.npy'
            singlejson["pcl"] = pcl_path
            singlejson["id"] = str(key)
            singlejson["src_dataset"] = "Mydata"
            outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
#



# Train RoomDetection  ##
def Train_RoomDetection():
    outjson = []
    Q, A, GT, result = filepath('RoomDetection')
    Question, Answer, GT = loading(Q, A, GT)
    for key,value in GT.items():
        singlejson = {}
        if int(key) > 459:
            continue

        answer = getanswerRoomDe(value,Answer)
        ans_num = random.randint(30, 59)
        question = Question[str(ans_num)]

        conversations = [{"from": "human","value":question},{"from": "gpt","value":answer}]
        pcl_path = "scene/" + str(key) + '.npy'

        singlejson = getsinglejson(str(key), str(key), pcl_path, conversations, "RoomDetection3d")
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson
## Test Detection  ##
def Multi_Classification():
    countnum = 0
    result = "/media/kou/Data1/htc/MYDATA/Task/Task_Reconstruct/Temp.json"
    outjson = []
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"
    GT = open(GT_path,'r')
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Detection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)
    for oneline in jsonlines.Reader(GT):
        for key1, value1 in oneline.items():
            key, value = key1,value1
        singlejson = {}
        if int(key) >= 500 or int(key)<460:
            continue
        countnum += 1
        obj = []
        for one in value:
            name = one["name"]
            box = one["BoundingBox"]
            if name.lower() not in class_mapping.keys():
                continue
            obj.append({'name':name.lower(),'BoundingBox':box})

        que_num = random.randint(0,29)
        question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers."
        singlejson['question_id'] = countnum
        class_num = random.randint(2,5)
        random_list = random.sample(range(0, len(obj)), class_num)
        obj = [obj[i] for i in random_list]
        singlejson['object'] = obj
        singlejson["query"] = question
        pcl_path = "scene/" + str(key) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson

def Test_Detection():
    countnum = 0
    result = "/media/kou/Data1/htc/MYDATA/Task/Task_Reconstruct/Temp.json"
    outjson = []
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"
    GT = open(GT_path,'r')
    Question_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_Detection.json"
    Question = open(Question_path, 'r')
    Question = json.load(Question)
    for oneline in jsonlines.Reader(GT):
        for key1, value1 in oneline.items():
            key, value = key1,value1
        singlejson = {}
        if int(key) >= 500 or int(key)<460:
            continue
        countnum += 1
        obj = []
        for one in value:
            name = one["name"]
            box = one["BoundingBox"]
            if name.lower() not in class_mapping.keys():
                continue
            obj.append({'name':name.lower(),'BoundingBox':box})

        que_num = random.randint(0,29)
        question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers."
        singlejson['question_id'] = countnum
        singlejson['object'] = obj
        singlejson["query"] = question
        pcl_path = "scene/" + str(key) + '.npy'
        singlejson["pcl"] = pcl_path
        singlejson["id"] = str(key)
        singlejson["src_dataset"] = "Mydata"
        outjson.append(singlejson)

        print(key,"OK")
    return result, outjson

# ## Test Pos Relation ##
def Test_PositionRelation():
    Count = 1
    outjson = []
    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    root = "/media/kou/Data1/htc/MYDATA"
    GT_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Counting.json"
    GT = open(GT_path, 'r')
    GT = json.load(GT)
    Answer_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/A_PositionRelation.json"
    Answer = open(Answer_path, 'r')
    Answer = json.load(Answer)
    for id in range(460, 500):

        Testpos = root + "/BenchMark/Task/Test/RoomDetection.json"
        Tp = open(Testpos, 'r')
        Tp = json.load(Tp)
        if id in [int(tp['id'] )for tp in Tp]:
            pass
        else:
            continue
        template = Tp[0]

        pos = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Relationship.json" #所有存在的物体
        pos = open(pos, 'r')
        pos = json.load(pos)

        position = {}

        class_list = [key for key, value in GT[id].items() if value == 1]   #所有数量为1的物体
        classlist = []
        pos = pos[str(id)]
        pos = list(pos.values())
        for values in pos:
            if sum(1 for name in pos if list(name.keys())[0] == list(values.keys())[0]) == 1:     #判断源文件json文件中物体数量是否大于1
                pass
            else:
                continue
            for x,y in values.items():
                x = x.lower()
            if x in class_list:
                values = {x:y}
                classlist.append(x)
                position.update(values)

        for x in range(5):
            answer,direaction,class_a,class_b ,flag= chouyang_test(classlist,position,Answer)

            if (answer == 0 or (direaction>=210 and direaction<=269) or direaction == 0) :
                print("跳过",x,id)
                continue

            false_list = get_answer(Answer, flag, direaction, class_a, class_b)
            if len(false_list)==1:
                print(direaction,false_list)

            singlejson = get_testPosition(answer,false_list,Count,template, class_a, class_b,id)

            outjson.append(singlejson)
            Count += 1

        print(id,"OK")
    return result, outjson
#
# Test Counting ##
def Test_Counting():
    Q_num = 0
    outjson = []
    Q, A, GT, result = filepath('Counting')
    Question, Answer, GT = loading(Q, A, GT)
    for i, single in enumerate(GT):
        if i<460:
            continue
        gt = single
        """
        抽取相应数量的物体，数量==value
        对数量为value的物体抽样
        """
        # single = {key: value for key, value in single.items() if value == 3}
        if len(single)>=8:
            random_number = random.sample(range(0, len(single)), 2)
        # elif len(single)>=4:
        #     random_number = random.sample(range(0, len(single)), 4)
        elif len(single)>=3:
            random_number = random.sample(range(0, len(single)), 3)
        elif len(single)>=2:
            random_number = random.sample(range(0, len(single)), 2)
        elif len(single)>=1:
            random_number = random.sample(range(0, len(single)), 1)
        else:
            continue
        random_class = [list(single)[x] for x in random_number]
        for objclass in random_class:
            single_out = get_testCounting(i, Q_num, objclass, gt[objclass])
            outjson.append(single_out)
            Q_num += 1
            print(i)
    return result, outjson

## Train Classification ##
def Train_Classification():

    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Temp.json"
    outjson = []
    All = json.load(open("/media/kou/Data3/htc/dataset/Object/my_train.json"))
    for path in All[:4000]:
        single_out = get_trainClass(path)
        outjson.append(single_out)
    # for root, dirs, files in os.walk("H:\Objects_drc"):
    #     for file in files:
    #         print(file)
    #         single_out = get_trainClass(file)
    #         outjson.append(single_out)

    return result,outjson

## Test_Classification ##
def Test_Classification():
    outjson = []
    result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/1.json"
    num = 0
    All = json.load(open("/media/kou/Data3/htc/dataset/Object/my_test.json"))
    for path in All:
        num += 1
        single_out = get_testClass(path,num)
        outjson.append(single_out)
    return result, outjson

#Train Navigation
def Train_Navigation():
    outjson = []
    Q, A, GT, result = filepath('Navigation')
    Question, Answer, GT = loading(Q, A, GT)

    for id in range(0, 460):
        GTid = str(id)
        if not GTid in list(GT.keys()):
            continue
        ALLpath = GT[GTid]
        for index,path in enumerate(ALLpath):
            singlejson = getNV(path,GTid,index,Answer,Question)
            # if singlejson == 0:
            #     print("跳过",id)
            #     continue
            outjson.append(singlejson)
        print(id,"OK")

    return result, outjson

def getNV(path,GTid,index,Answer,Question):
    random_number = random.randint(0, 29)
    random_number = str(random_number)
    Answer = Answer[random_number]
    Answer = re.sub("{P}", str(list(path.values())[0]), Answer)
    random_number = random.randint(0, 29)
    random_number = str(random_number)
    Question = Question[random_number]
    Question = re.sub("{C}", str(list(path.keys())[0]), Question)
    Question = re.sub("{P}", str(list(path.values())[0][0]), Question)

    pcl_path = "scene/" + str(GTid) + ".npy"
    converstions = [{"from": "human", "value": Question}, {"from": "gpt", "value": Answer}]
    singlejson = getsinglejson(str(GTid), str(index), pcl_path, converstions, "Navigation")

    return singlejson

""" 
Counting的训练数据
"""
def Train_Counting():
    Q, A, GT, result = filepath('Counting')
    Question, Answer, GT = loading(Q, A, GT)
    outjson = []

    for id in range(0, 460):
        GTid = str(id)
        newchat = []

        oneGT = list(GT[id].items())
        # round = math.floor(len(oneGT)/1)+1
        round = len(oneGT)
        for i in range(round):
            answer = getcouting(oneGT[i :(i + 1) ], Answer)
            # answer = getcouting(oneGT[i*5:(i+1)*5], Answer)
            # answer = re.sub(r"{C}", name, Answer[random_number])

            random_number = random.randint(0, 29)
            random_number = str(random_number)
            question = Question[random_number]
            name ,_ = oneGT[i :(i + 1) ][0]
            question = re.sub(r"{C}", name, question)
            newchat = [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]

            pcl_path = "scene/" + str(id) + '.npy'
            singlejson = getsinglejson(str(id),str(i),pcl_path,newchat,"Counting")

            outjson.append(singlejson)
        print(id,"OK")
    return result, outjson[0:17189:3]

# Train Detection // 只保留前25个物体
def Train_Detection():
    choosen_scene = json.load(open("/media/kou/Data1/htc/LAMM/data/meta_file/choosenscene.json"))
    outjson = []
    Q, A, GT, result = filepath('Detection')
    Question, Answer, _ = loading(Q, A, Q)
    with open(GT, 'rb') as f:
        for item in jsonlines.Reader(f):
            if int(list(item.keys())[0])< 500 or int(list(item.keys())[0])>= 10000:
                continue
            # if list(item.keys())[0] not in choosen_scene:
            #     continue
            GTid = list(item.keys())[0]
            oneGT = list(item.values())[0]
            # oneGT = list(oneGT.items())

            que_num = random.randint(0,29)
            question = Question[str(que_num)]+" With obj0:name0, obj1:name1... form of answers. "
            answer,box = getanswerDe(oneGT,Answer)
            if '20' in answer:
                continue
            conversations = [{"from": "human","value": question},{"from": "gpt","value":answer}]
            pcl_path = "scene/" + GTid + '.npy'

            singlejson = getsinglejson(str(GTid),str(GTid),pcl_path,conversations,"Detection3d")
            singlejson["conversations"] = conversations
            singlejson["box"] = box

            outjson.append(singlejson)
            print(GTid,"OK")

    return result, outjson

## *VisualGrounding* ##
"""
每个场景取两个物体组成3W*2个对话
剔除部分场景，空场景和物体不够
"""
def Train_VisualGrounding():
    outjson = []
    countnum = 0
    Q, A, GT, result = filepath('VisualGrounding')
    Question, Answer, _ = loading(Q, A, Q)
    choosen_scene = json.load(open("/media/kou/Data1/htc/LAMM/data/meta_file/choosenscene.json"))

    new_classdict={}
    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    for i in Detection:
        key = list(i.keys())[0]
        flag = 0
        if key == '26230' or key == '8213' or (int(key) <= 499) or (key not in choosen_scene):
            continue
        if int(list(i.keys())[0]) < 460:
            continue
        for new_class in i[key]:
            if len(new_class)==0:
                flag = 1
        if flag==1:
            continue
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
        new_classdict[key] = new_classlist

    GT = json.load(open(GT, 'r'))
    for key,value in GT.items():
        GTid = key
        oneGT = value

        class_name = list(oneGT)
        if GTid == '26230' or GTid == '8213' or (int(GTid) <= 499) or (GTid not in choosen_scene):
            continue
        if len(oneGT)==0:        #忽略没有物体的场景
            continue
        if len(oneGT)<2:        #忽略没有物体的场景
            num = [0]
        else:
            num = [random.randint(0, len(oneGT) - 1) for _ in range(2)]
        oneGT = [oneGT[class_name[i]]for i in num]
        if not new_classdict.get(key):
            continue
        obj_num = list(new_classdict[key])
        obj_name = [i['name'] for i in obj_num]
        for i in range(len(oneGT)):
            singlejson = {}
            # singlejson["question_id"] = countnum
            countnum += 1
            answer,question0 = getVGanswer(oneGT[i :(i + 1)], Answer,Question)
            name = oneGT[i :(i + 1)][0]['name']
            if name in obj_name:
                index = obj_name.index(name)
            else:
                continue
            answer = "It is (obj"+str(index) + ") at "+str([round(oneGT[i :(i + 1)][0]["BoundingBox"][0],1),round(oneGT[i :(i + 1)][0]["BoundingBox"][1],1),round(oneGT[i :(i + 1)][0]["BoundingBox"][2],1)])+"."

            conversations = [{"from": "human", "value": question0}, {"from": "gpt", "value": answer}]
            if len(answer)==0:
                continue

            pcl_path = "scene/" + GTid + '.npy'

            singlejson = getsinglejson(str(GTid), str(i), pcl_path, conversations, "VisualGrounding3d")

            outjson.append(singlejson)
        print(GTid,"OK")
    return result, outjson

def crop_point_cloud(point_cloud, bounding_box):
    """
    根据给定的bounding box切割点云场景的一部分。

    Args:
        point_cloud (open3d.geometry.PointCloud): 已加载的点云对象。
        bounding_box (list): 包含[x, y, z, l, w, h]形式的bounding box参数。
            其中 (x, y, z) 是边界框的中心坐标，l 是边界框的长度，w 是宽度，h 是高度。

    Returns:
        open3d.geometry.PointCloud: 切割出的点云对象，如果边界框内没有点云，则返回 None。
    """
    # 解析bounding box参数
    x, y, z, l, w, h = bounding_box

    # 计算边界框的范围
    min_bound = np.array([x - l/2, y - w/2, z - h/2])
    max_bound = np.array([x + l/2, y + w/2, z + h/2])

    # 获取点云数据
    points = np.asarray(point_cloud.points)

    # 筛选出在边界框内的点
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    cropped_points = points[mask]

    if len(cropped_points) == 0:
        return None
    else:
        # 创建新的点云对象并返回
        cropped_point_cloud = o3d.geometry.PointCloud()
        cropped_point_cloud.points = o3d.utility.Vector3dVector(cropped_points)
        return cropped_point_cloud

def check_point_cloud_in_boxes(point_cloud, bounding_boxes):
    """
    根据给定的 bounding boxes 判断对应位置是否存在点云。

    Args:
        point_cloud (open3d.geometry.PointCloud): 已加载的点云对象。
        bounding_boxes (list): 包含多个 [x, y, z, l, w, h] 形式的 bounding box 列表。

    Returns:
        list: 包含布尔值的 mask 列表，每个元素表示对应位置是否存在点云。
    """
    # 获取点云数据
    points = np.asarray(point_cloud.points)

    # 初始化 mask 列表
    mask = []

    # 提取所有 bounding box 的参数
    bbox_params = np.array([bbox['BoundingBox'] for bbox in bounding_boxes])

    # 计算所有 bounding box 的范围
    min_bounds = bbox_params[:, :3] - bbox_params[:, 3:] / 2
    max_bounds = bbox_params[:, :3] + bbox_params[:, 3:] / 2

    # 判断 bounding box 区域内是否存在点云
    for min_bound, max_bound in zip(min_bounds, max_bounds):
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        has_points = np.any(indices)
        mask.append(has_points)

    return mask





"""
进一步处理Detection,删除场景中不存在的物体
"""

#检查drc文件转存ply失败的文件
def check(i):
    try:
        # print(i)
        scene_path = "/media/kou/Data3/htc/scene/" + str(i) + ".ply"
        scene = o3d.io.read_point_cloud(scene_path)
        # print(i)
    except Exception  as e:
        print(i)
        print(e)


def process_item(i):
    if i%1000 == 0:
        print(str(i), 'over')
    # 删除场景中不存在的物体
    sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/Detection.json", 'r')
    try:
        for item in jsonlines.Reader(sen):
            if str(i) == list(item.keys())[0]:
                # print(i,list(item.keys())[0],'over')
                scene_path = "/media/kou/Data3/htc/scene/" + str(i) + ".ply"
                scene = o3d.io.read_point_cloud(scene_path)
                items = []
                for one in item[str(i)]:
                    if one:
                       items.append(one)
                if len(items):
                    mask = check_point_cloud_in_boxes(scene,items)
                    pro_item = [item for item, m in zip(item[str(i)], mask) if m]
                else:
                    pro_item = item
                pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/pro_Detection.json", 'a')
                pro_sen.write(json.dumps({str(i):pro_item}) + "\n")
                pro_sen.flush()
                return
    except Exception as e:
        print(e,"bug in ",str(i))

# 给detection排序
def sort_item(i):

    sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/pro_Detection.json", 'r')

    for item in jsonlines.Reader(sen):
        if str(i) != list(item.keys())[0]:
            continue
        else:
            pro_item = item[str(i)]

            pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/PRO_Detection.json", 'a')
            pro_sen.write(json.dumps({str(i):pro_item}) + "\n")
            pro_sen.flush()

            return



# pro_sen = open(dirpath + "/../MYDATA/BenchMark/Task/GT/Detection.json", 'a')
#
# for i in range(0,30):
#     scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection/Detection" + str(i) + ".json"
#     sce = open(scene_path,"r")
#     for item in jsonlines.Reader(sce):
#         pro_sen.write(json.dumps(item) + "\n")

# for i in range(0,30000):
#     if str(i) not in exsiting:
#         print(i)

#
# exsiting = []
# for i in range(0,30):
#     scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection" +".json"
#     sce = open(scene_path,"r")
#     for item in jsonlines.Reader(sce):
#         exsiting.append(list(item.keys())[0])
"""
预先定义
"""
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

Detection_class = ["cabinet","bed","chair","sofa","diningtable","doorway","window","shelf", "painting","countertop","desk","fridge","toilet","sink","garbagecan"]

# ## Scene GPT Test:SVQA,Relation,SCaption  ##
# import jsonlines
#
# outjson = []
# for i in range(460,500):
#     gt={}
#
#     # SCaption
#     # gt['query'] = "Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge."
#     # gt["task_type"] = "Caption"
#     # SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # gt["task_type"] = "VQA"
#
#     # Relation
#     gt['query'] = "Analyze the relationship of two object in the given scene point cloud. Generate a relation explanation."#For example,a dining table and a bowl are used for dining.Cupboard can be used to store plates.Remember relation explanation must be about two things and the object mentioned must be various and in the given scene point cloud."
#     gt["task_type"] = "Relation"
#
#     gt['id'] = i
#     gt['pcl'] = "scene/"+str(i)+".npy"
#     gt['src_dataset'] = "Mydata"
#     outjson.append(gt)
#





## Object GPT Test:SVQA,Relation,SCaption  ##
# import jsonlines
# GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/ClassificationCaption/Train_class/Classification.jsonl"))
# outjson = []
# for gt in GT:
#     #SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # Relation
#     gt['query'] = "Analyze the relationship of two object in the given scene point cloud.Generate a relation explanation.For example,a dining table and a bowl are used for dining.Cupboard can be used to store plates.Remember relation explanation must be about two things and the object mentioned must be various and in the given scene point cloud."
#     # SCaption
#     gt['id'] = 'O'+gt['id'][1:]
#     # gt['query'] = "Write a detailed caption by classifying and describing different rooms in 150-200 words, illustrating their types, appearance and other information such as functionalities, usages, daily-life knowledge."
#     outjson.append(gt)



## Object GPT Train:SVQA,Relation,SCaption  ##
# import jsonlines
# import re
# GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/Classification/Classification.jsonl"))
# GT = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/ClassificationCaption/Train_class/Classification.jsonl"))
# outjson = []
#
# for gt in GT:
#     #SVQA
#     # gt['query'] = "Generate 5 single-round Q&As about 5 different object in rooms,considering diverse aspects like usage,material,belonged rooms and daily-life knowledge."
#     # SCaption
#     if len(gt["text"])<=550:
#         continue
#     name = re.sub(r".*\_", "", gt['id'])
#     name = re.sub(r"\d+.*", "", name)
#     gt['pcl'] = 'O'+gt['id'][1:]+".npy"
#     gt["conversations"] = [{"from": "human",
#                             "value":f"Describe this {name} as detailed as possible, as if the {name} is right in front of you."},
#                            {
#                                "from": "gpt",
#                                "value": gt["text"]
#                            }
#     ]
#     gt['task_type'] = 'DescriptionObj3d'
#     # gt["conversations"] = [{"from": "human",
#     #                         "value":f"You need to create three question-and-answer pairs centered around the {name}, ensuring that the context is interconnected. Format your response as a list,[Q1,A1,Q2,A2,Q3,A3]"},
#     #                        {
#     #                            "from": "gpt",
#     #                            "value": gt["text"]
#     #                        }
#     # ]
#     # gt['task_type'] = 'ConversationObj3d'
#     gt['src_dataset'] = "Mydata"
#     outjson.append(gt)


############################ 扩展的VG Test ##
def VG_Test():
    Description = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/VQA/VQA.jsonl"))
    VGobj = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/VisualGrounding.json"))
    outjson = []
    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    new_classdict = {}

    for i in Detection:
        key = list(i.keys())[0]
        if int(list(i.keys())[0]) < 460 or int(list(i.keys())[0])>=500:
            continue
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]#检测数据集小
        # new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Detection_class]
        new_classdict[key] = new_classlist
    for i in Description:
        gt = {}
        vg = VGobj[i['pcl'][:3]]
        ovnamelist= [name for name,other in vg.items()]
        gt["pcl"] = "scene/"+i["pcl"][:-4]+".npy"
        gt["id"] = i["pcl"][:-4]
        question = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_VisualGrounding.json"))
        gt["query"] = "Which is "+ i["text"][0].lower()+i["text"][1:]
        scene = new_classdict[i["pcl"][:-4]]
        gt["obj_num"] = None
        for index,box in enumerate(scene):
            if box["BoundingBox"] == i["box"]:
                gt["obj_num"] = index
                gt["bbox"] = i["box"]
        if gt["obj_num"]==None:
            continue
        if  gt["obj_num"]>20:
            continue
        outjson.append(gt)

    return outjson
# outjson = VG_Test()



# 扩展的VG Train ##
def VG_Train():
    Description = jsonlines.Reader(open("/media/kou/Data1/htc/PointLLM/Results/VG_description/Detection_description.jsonl"))
    outjson = []

    Detection = jsonlines.Reader(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json"))
    new_classdict = {}
    for i in Detection:
        key = list(i.keys())[0]
        if int(list(i.keys())[0]) >= 460:
            continue
        new_classlist = [new_class for new_class in i[key] if new_class["name"].lower() in Class_ALL]
        new_classdict[key] = new_classlist
    for i in Description:
        gt = {}
        gt["pcl"] = "scene/"+i["pcl"][:-4]+".npy"
        gt["id"] = i["pcl"][:-4]
        question = json.load(open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/Template/Q_VisualGrounding.json"))
        query = "In all objects, tell me which is "+ i["text"][0].lower()+i["text"][1:]
        scene = new_classdict[i["pcl"][:-4]]
        for index,box in enumerate(scene):
            if box["BoundingBox"] == i["box"]:
                obj_num = index
        if obj_num>20:
            continue
        gt["conversations"]= [
            {
                "from": "human",
                "value": query
            },
            {
                "from": "gpt",
                "value": "It is (obj"+str(obj_num) +")."
            }
        ]
        gt["task_type"] = "VisualGrounding3d"
        outjson.append(gt)
    return outjson
# outjson = VG_Train()





"""
训练Instruction tuning data
"""
#Classification
# result, outjson = Train_Classification()
#Counting
# result, outjson = Train_Counting()
#Detection
result, outjson = Train_Detection()
#VisualGrounding
# result, outjson = Train_VisualGrounding()
#RoomDetection
# result, outjson = Train_RoomDetection()
#Navigation
# result, outjson = Train_Navigation()
#PositionRelation
# result, outjson = Train_PositionRelation()

"""
测试Instruction tuning data
"""
# #Classification
# result, outjson = Test_Classification()
# #Counting
# result, outjson = Test_Counting()
#Detection
# result, outjson = Test_Detection()
#Multi_Classification
# result, outjson = Multi_Classification()
#VisualGrounding
# result, outjson = Test_VisualGrounding()
#RoomDetection
# result, outjson = Test_RoomDetection()
#Navigation
# result, outjson = Test_Navigation()
#PositionRelation
# result, outjson = Test_PositionRelation()
result = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/temp.json"
with open(result, 'w') as f:
    # 把列表写入到文件里，转换成json格式
    json.dump(outjson, f, indent=4)
#


#
# exsiting = []
#
# scene_path = "/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/pro_Detection.json"
# sce = open(scene_path,"r")
# for item in jsonlines.Reader(sce):
#     exsiting.append(list(item.keys())[0])
#
# #多线程控制
#
# max_processes = 10
# pool = multiprocessing.Pool(processes=max_processes)
#
# num_jobs = 30000  # 总共要执行的任务数
#
# for i in range(0, num_jobs):
#     if str(i) in exsiting:
#         continue
#     #     # process_item(i)
#     # 启动一个新进程来执行 worker_function
#     pool.apply_async(process_item, args=(i,))
#
# # 关闭进程池，不再接受新任务
# pool.close()
#
# # 等待所有进程完成
# pool.join()
# print("所有进程已完成")
