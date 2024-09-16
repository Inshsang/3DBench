import numpy as np
import math
from ai2thor.controller import Controller
from PIL import Image
import os
import re
import prior
import random
import datetime
from functools import reduce
import json
from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)
import matplotlib.pyplot as plt
import jsonlines
import multiprocessing
import open3d as o3d

def to_rad(th):
    return th*math.pi / 180

dirpath = os.path.split(os.path.realpath(__file__))[0]

lock = multiprocessing.Lock()

def getintri(fovw,fovh,width,height):
    # Convert fov to focal length
    focal_length = 0.5 * width / math.tan(to_rad(fovw / 2))
    #focal_length_h = 0.5 * height / math.tan(to_rad(fovh / 2))
    #focal_length = 0.5 * width / math.tan(to_rad(fov / 2))
    # camera intrinsics
    fx, fy, cx, cy = (focal_length, focal_length, width / 2, height / 2)

    return fx, fy, cx, cy

def vertical_to_horizontal_fov(
    vertical_fov_in_degrees: float, height: float, width: float
):
    assert 0 < vertical_fov_in_degrees < 180
    aspect_ratio = width / height
    vertical_fov_in_rads = (math.pi / 180) * vertical_fov_in_degrees
    return (
        (180 / math.pi)
        * math.atan(math.tan(vertical_fov_in_rads * 0.5) * aspect_ratio)
        * 2
    )

def getobj(obj,objnum):
    objlist = []

    obj = [i['objectType'] for i in obj]

    for i in obj:
        if obj.count(i) > 1:
            continue
        else:
            objlist.append(i)

    if len(objlist)>=objnum:
        objlist = random.sample(objlist,objnum)

    return objlist

def control(house,id,controller,sence):
    #fovw = vertical_to_horizontal_fov(fovh,width,height)
    controller.reset(
        scene=house
    )
    event = controller.step(action="Pass")
    id = str(id)
    # sence = event.metadata
    # path = '/media/cvlab/Data/htc/' +   'metadata1.json'
    # with open(path, 'w') as f:
    #     # 把列表写入到文件里，转换成json格式
    #     json.dump(sence, f, indent=4)

    flag = ['openable',
            'sliceable',
            'canBeUsedUp',
            'cookable',
            'dirtyable',
            'canFillWithLiquid',
            'breakable',
            'toggleable']
    save = ['openness',
            'isSliced',
            'isUsedUp',
            'isCooked',
            'isDirty',
            'isFilledWithLiquid',
            'isBroken',
            'isToggled']

    room = {}
    for i in house['rooms']:
        a = i['id']
        b = i['roomType']
        room[a] = b


    sig_object = {}
    for object in event.metadata['objects']:
        if not object['objectOrientedBoundingBox']:
            continue
        objdict = {}
        objdict['name'] = object['assetId']
        objdict['material'] = object['salientMaterials']
        #print('name' + ":" + str(object['name']))
        objdict['objectOrientedBoundingBox'] = object['objectOrientedBoundingBox']
        for key,value in object.items():
            if key in flag and value:
                position = flag.index(key)
                #print(save[position]+":"+str(object[save[position]]))
                objdict[save[position]] = object[save[position]]
        sig_object[object['name']] = objdict
    sig_object['room'] = room
    sence[id] = sig_object
    if (int(id)+1)%500 ==0:
        # 打开一个文件，用来存储json数据
        path = '/media/cvlab/Data/htc/metadata1.json'
        with open(path, 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(sence, f, indent=4)
            sence = {}
    now = datetime.datetime.now()

    print(str(now)[11:19]+":" ,"GTsaved:",id)
    return controller, sence

def getVisualGrounding(house,id,controller,sen):
    # fovw = vertical_to_horizontal_fov(fovh,width,height)
    # controller.reset(
    #     scene=house
    # )
    # event = controller.step(action="Pass")
    # id = str(id)
    f = open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/pro_Detection.json", 'r')
    sen = {}
    for scene in jsonlines.Reader(f):
        allobject = {}
        id = list(scene.keys())[0]
        scene = list(scene.values())[0]
        dellist = []

        if not len(scene):
            sen[id] = allobject
            continue
        else:
            for object in scene:
                #删除场景中多次出现的同类物体
                if not len(object):
                    continue
                sig_objdict = {}
                sig_objdict['name'] = object['name']
                sig_objdict['BoundingBox'] = object['BoundingBox']
                if object['name'] in list(allobject.keys()):
                    dellist.append(object['name'])
                    continue
                allobject[object['name']] = sig_objdict

            for delone in set(dellist):
                del allobject[delone]

                # print('name' + ":" + str(object['name']))
        sen[id] = allobject

        with open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/VisualGrounding.json", 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(sen, f, indent=4)

        now = datetime.datetime.now()
        print(str(now)[11:19] + ":", "GTsaved:", id)
    return controller, sen

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


    # 提取所有 bounding box 的参数
    bbox_params = np.asarray(bounding_boxes)

    # 计算所有 bounding box 的范围
    min_bound = bbox_params[:3] - bbox_params[3:] / 2
    max_bound = bbox_params[:3] + bbox_params[3:] / 2

    # 判断 bounding box 区域内是否存在点云
    indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    has_points = np.any(indices)

    return has_points

def corner2bbox(corners):
    x_min = min(corner[0] for corner in corners)
    y_min = min(corner[1] for corner in corners)
    z_min = min(corner[2] for corner in corners)
    x_max = max(corner[0] for corner in corners)
    y_max = max(corner[1] for corner in corners)
    z_max = max(corner[2] for corner in corners)

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    z = (z_min + z_max) / 2

    w = x_max - x_min
    h = y_max - y_min
    l = z_max - z_min

    return [x, y, z, w, h, l]


def getDetection(house,id,controller,sen,**kwargs):
    # fovw = vertical_to_horizontal_fov(fovh,width,height)
    controller.reset(
        scene=house
    )
    event = controller.step(action="Pass")
    id = str(id)

    sig_object = []
    # try:
    if len(event.metadata['objects']) == 0:
        print(id,'lengh=0')
        sen.write(json.dumps({str(id): sig_object}) + "\n")
        sen.flush()
        return controller, sen
    for idx,object in enumerate(event.metadata['objects']):
        if not object['axisAlignedBoundingBox']:
            continue
        objdict = {}

        if object['axisAlignedBoundingBox']['cornerPoints']:
            bbox = corner2bbox(object['axisAlignedBoundingBox']['cornerPoints'])
            grounding = [round(num, 2) for num in bbox]
            objdict['BoundingBox'] = grounding
            objdict['name'] = object['objectType']
        else:
            objdict = {}
        # scene = o3d.io.read_point_cloud("/media/kou/Data3/htc/scene/"+id+".ply")
        # mask = check_point_cloud_in_boxes(scene, grounding)
        # if not mask:
        #     continue

        sig_object.append(objdict)
    sen.write(json.dumps({str(id): sig_object}) + "\n")
    sen.flush()
    # except Exception as e:
    #     print(e)


    now = datetime.datetime.now()
    # print(str(now)[11:19] + ":", "GTsaved:", id)
    return controller, sen

def getRoomDetection(house,id,controller,out):
    # fovw = vertical_to_horizontal_fov(fovh,width,height)
    controller.reset(
        scene=house
    )
    event = controller.step(action="Pass")
    id = str(id)

    room = {}
    height = house['proceduralParameters']['lights'][-1]['position']['y']+0.22
    if height<=2:
        print(height,"asdfasdfasfdasfsadfas")
    for i in house['rooms']:
        a = i['roomType']
        b = i['floorPolygon']
        for my_dict in b:
            my_dict["y"] = height
        room[a] = b
    out[id] = room
    if (int(id) + 1) % 1 == 0:
        # 打开一个文件，用来存储json数据
        path = '/media/cvlab/data1/htc/MYDATA/BenchMark/Task/GT/RoomDetection.json'
        with open(path, 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(out, f, indent=4)
    now = datetime.datetime.now()

    print(str(now)[11:19] + ":", "GTsaved:", id)
    return controller, out

def getRelation(house,id,controller,sence):
    controller.reset(
        scene=house
    )
    event = controller.step(action="Pass")
    id = str(id)

    sig_object = {}
    i = 0
    for object in event.metadata['objects']:
        if not object['objectOrientedBoundingBox']:
            continue
        objdict = {}
        # print('name' + ":" + str(object['name']))
        center = object['axisAlignedBoundingBox']["center"]
        size = object['axisAlignedBoundingBox']["size"]
        grounding = [center['x'], center['y'], center['z']]
        grounding = [round(num, 2) for num in grounding]
        objdict[object['objectType']] = grounding
        sig_object[str(i)] = objdict
        i += 1

    sence[id] = sig_object
    if (int(id) + 1) % 500 == 0:
        # 打开一个文件，用来存储json数据
        path = '/media/cvlab/Data/htc/MYDATA/BenchMark/Task/GT/Relationship.json'
        with open(path, 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(sence, f, indent=4)
            sence = {}
    now = datetime.datetime.now()

    print(str(now)[11:19] + ":", "GTsaved:", id)
    return controller, sence

def getpath(positions,event,controller,objnum,flag):
    flag = flag -1

    if flag == 0:
        return 0
    path = []
    objlist = getobj(event.metadata['objects'],objnum)
    for position in positions:
        for obj in objlist:
            try:
                way = get_shortest_path_to_object_type(
                    controller=controller,
                    object_type=obj,
                    initial_position=position
                )
            except Exception as e:
                #print('flag:',flag)
                objnum = 1
                path = getpath(positions, event, controller,objnum,flag)
                return path
            xy = []
            for i in way:
                xy.append((round(i['x'],2),round(i['z'],2)))
            path.append({obj:xy})
    return path

def draw(path,map):

    for oneobj in path[0:10]:
        answer_x = []
        answer_y = []
        positions = list(oneobj.values())[0]
        for position in positions:
            x ,y = position
            answer_x.append(300/13.155*x)
            answer_y.append(300/13.155*y)
        plt.plot(answer_x, answer_y)
        plt.imshow(map)
    plt.show()

def getPath(house,id,controller,sence):

    id = str(id)
    print(id)
    controller.reset(
        scene=house
    )
    event = controller.step(action="Pass")
    positions = controller.step(dict(action="GetReachablePositions")).metadata["actionReturn"]
    if len(positions)>=20:
        positions = random.sample(positions, 20)#20%5
        objnum = 5          #positions*objnum
    else:
        print("positions:",len(positions))
        objnum = 1
    path = getpath(positions,event,controller,objnum,flag = 100)
    if path == 0:
        return controller, sence
    # draw(path,map)
    sence[id] = path
        # 打开一个文件，用来存储json数据

    if (int(id)+1)%100 == 0:
        with open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Navigation.json", 'w') as f:
            # 把列表写入到文件里，转换成json格式
            json.dump(sence, f, indent=4)
        now = datetime.datetime.now()
        print(str(now)[11:19] + ":", "GTsaved:", id)

    return controller, sence

width = 50
height = 50
fovh = 120
gridsize = 0.1

import argparse

def cloude(i):
    parser = argparse.ArgumentParser()
    #data = prior.load_dataset("procthor-10k")
    #parser.add_argument("-begin", type=int, help="an optional integer argument")
    parser.add_argument("-i", type=int, help="an optional integer argument")
    args = parser.parse_args()

    #dataset = np.load("/media/cvlab/Data/htc/dataset.npy",allow_pickle=True).item()
    dataset = prior.load_dataset("procthor-10k")

    house = dataset["train"][0]
    with Controller(
            gpu_device=5,
            platform='CloudRendering',
            quality='Low',
        agentMode="default",
        scene=house,
        gridSize=gridsize,
        width=width,
        height=height,
        fieldOfView=fovh) as controller:

        #num = args.i
        num = 1


        """
        位置
        """
        # VisualGrounding
        # sen = open(dirpath+"/../MYDATA/BenchMark/Task/GT/VisualGrounding.json", 'r')
        # sen = json.load(sen)

        # # Detection
        # data = {}
        # file_path = dirpath+"/../MYDATA/BenchMark/Task/GT/Detection/Detection"+str(i)+".json"
        # with open(file_path, "w") as json_file:
        #     json.dump(data, json_file, indent=4)
        # sen = open(dirpath+"/../MYDATA/BenchMark/Task/GT/Detection/Detection"+str(i)+".json", 'w')
        # Navigation
        sen = open("/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Navigation.json", 'r')
        sen = json.load(sen)
        # RoomDetection
        # sen = open("/media/cvlab/data1/htc/MYDATA/BenchMark/Task/GT/RoomDetection.json", 'r')
        # sen = json.load(sen)
        print('beginning' + str(i))
        for id in range(500*i,500*(i+1)):

            file = dirpath+"/../myjson/("+str(id)+").json"
            filepath = os.path.join(dirpath, file)
            # if not os.path.exists(filepath):
            #     continue
            f = open(filepath, 'r')
            house = json.load(f)

            now = datetime.datetime.now()
            controller, sen = getPath(house,id,controller,sen)

            """
            # controller, sen = getVisualGrounding(house,id,controller,sen) no need
            controller, sen = getDetection(house, id, controller, sen)
            controller, sen = getPath(house,id,controller,sen)
            controller, sen = getRoomDetection(house,id,controller,sen)
            """
        return



if __name__ == "__main__":
    cloude(0)
    # max_processes = 5
    # pool = multiprocessing.Pool(processes=max_processes)
    #
    # num_jobs = 5  # 总共要执行的任务数
    #
    # for i in range(0, num_jobs):
    #
    #     # 启动一个新进程来执行 worker_function
    #     pool.apply_async(cloude, args=(i,))
    #
    # # 关闭进程池，不再接受新任务
    # pool.close()
    #
    # # 等待所有进程完成
    # pool.join()
    print("所有进程已完成")



