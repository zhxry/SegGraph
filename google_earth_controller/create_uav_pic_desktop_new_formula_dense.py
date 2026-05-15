import pandas as pd
import pyautogui
import time
import math
import os.path
import time
import numpy as np
from PIL import Image
# from selenium import webdriver
import json
import random
from rsi_base_info import *

def restart_google_earth(name='',dir=''):
    time.sleep(3)
    os.system("taskkill /im googleearth.exe")
    time.sleep(3)
    pyautogui.click(819,600)  # 关闭欢迎页面
    time.sleep(5)
    os.startfile(r"D:\\UAV-navigation\\project\\UAV_3d_v2\\main_conroller_desktop.kml")

    
    
    time.sleep(15)
    
    pyautogui.hotkey('ctrl', 'alt', 's')
    time.sleep(2)
    # 点击地图选项 # 338,112
    pyautogui.click(338,112)
    time.sleep(1)
    # 点击四个选项 
    
    # 352,183
    # 326,214
    # 321,259
    # 349,288
    # 关闭 标题和说明、图例、比例、罗盘

    #关闭 标题和说明
    pyautogui.click(352,183)
    time.sleep(1)
    # 关闭 图例
    pyautogui.click(326,214)
    time.sleep(1)
    # 关闭 比例
    pyautogui.click(321,259)
    time.sleep(1)
    # 关闭 罗盘
    pyautogui.click(349,288)
    time.sleep(1)
    # 点击地图中央
    pyautogui.click(974,522)
    time.sleep(0.5)
    # 拖动地图左侧界限 252,475  ->  497,482
    pyautogui.moveTo(252,475)
    pyautogui.dragTo(497,482, duration=2)
    time.sleep(0.5)


    pyautogui.click(974,522)  # 点击地图中央
    pyautogui.moveTo(1070,119)  # 点击保存图片按钮
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(987,1048)  # 点击更改截图名称
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')  # 全选现有名称
    pyautogui.press('backspace')  # 删除现有名称
    pyautogui.write(name, interval=0.1)  # 输入新名称(每个字符间隔0.25秒)
    # click 1148,1026
    # 1122,1078
    # # 改到png
    # pyautogui.click(953,1074)
    # time.sleep(0.5)
    # pyautogui.click(938,1134)
    # time.sleep(0.5)
    pyautogui.moveTo(2040, 1156)  # 点击保存图像按钮
    pyautogui.click()
    path = dir + name + '.jpg'
    timewait = 0
    time.sleep(1)
    
    while os.path.exists(path) is False:
        print('正在下载：', path)
        # print('等待下载完成,已等待{}秒'.format(timewait))
        timewait += 2
        time.sleep(2)
    time.sleep(1)

def compute_target_latlon_from_norm(
    block_x, block_y,
    x_norm, y_norm,
    RSI_SIZE, lat_orig, lon_orig,
    lat_per_pixel, lon_per_pixel,
    PATCH_SIZE=256,
    BLOCK_SIZE=512,
    UNI_PIXEL=128
):
    start_x = block_x * PATCH_SIZE
    start_y = block_y * PATCH_SIZE

    x_block = x_norm * UNI_PIXEL + PATCH_SIZE
    y_block = y_norm * UNI_PIXEL + PATCH_SIZE
    x_rsi = start_x + x_block
    y_rsi = start_y + y_block
    lat = lat_orig - ((y_rsi - RSI_SIZE / 2) * lat_per_pixel)
    lon = lon_orig + ((x_rsi - RSI_SIZE / 2) * lon_per_pixel)
    return lat, lon

 
RSI_SIZE = 4096  
# lat_orig = 35.67091338738739
# lon_orig = 139.69289911300856
rsi_stem = rsi_name[:-4]
rsi_json_path = "./maps/c8_25pp_4096bc/1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city.json"  
drsi = rsijson2info(rsi_json_path)
lat_per_pixel = drsi['lat_per_pixel']
lon_per_pixel = drsi['lng_per_pixel']
lat_orig = drsi['lat']
lon_orig = drsi['lng']




def convert_path(path):

    path = path.replace('\\', '/')
    path = path.replace(' ', '_')
    return path.split('/')[-1].split('.')[0]  




def generate_pic(lat=53.947228,lon=-1.155475,height=100,heading=0,name='',dir='C:/Users/Zhy/Desktop/数据集/',distance_w=36,distance_h=36,output_size_w=256,output_size_h=256):
    dataset_type = 'city'
    path = dir + name + '.jpg'
    # output_size_w = 256
    # output_size_h = 256
    center_lat = lat
    center_lon = lon

    data={
        "image":name+".jpg", 
        "lat":lat,  
        "lng":lon,
        "height_meter":distance_h,
        "width_meter":distance_w,
        "height_pixel": output_size_h,
        "width_pixel": output_size_w,
        "coorsys": 4326,
        "lat_per_pixel":distance_w/(output_size_w*111000),
        "lng_per_pixel":distance_h/(output_size_h*111000)/ math.cos(center_lat / 180 * math.pi),
        "latm_per_pixel":distance_w/output_size_w, 
        "lngm_per_pixel":distance_h/output_size_h,
        "left_mid":(center_lat,center_lon-distance_w/(2*111000)/ math.cos(center_lat / 180 * math.pi)),
        "right_mid":(center_lat,center_lon+distance_w/(2*111000)/ math.cos(center_lat / 180 * math.pi)),
        "top_mid":(center_lat+distance_h/(2*111000),center_lon),
        "bottom_mid":(center_lat-distance_h/(2*111000),center_lon),
        "alt":height,
        "time":20260406, 

    }
    with open(dir+name+".json","w") as f:
        json.dump(data,f,indent=4)
    if os.path.exists(path):
        print('已存在：', path)
        img = Image.open(dir + name + '.jpg')
        w, h = img.size
        target_w, target_h = output_size_w, output_size_h
        mid_x, mid_y = w // 2, h // 2
        half_w, half_h = target_w // 2, target_h // 2
        left = mid_x - half_w
        top = mid_y - half_h
        right = mid_x + half_w
        bottom = mid_y + half_h
        cropped = img.crop((left, top, right, bottom))
        cropped.save(dir + name + '.jpg')
        return

    # continue
    
    print('正在下载：', path)
    # time.sleep(5)
    # 构建kml文件

    tilt = 0
    template = '''<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
            <Document>
            <Camera>
            <longitude>{center_lon}</longitude>
            <latitude>{center_lat}</latitude>
            <altitude>{height}</altitude>
            <heading>{heading}</heading>
            <tilt>{tilt}</tilt>
            <roll>0</roll>
            <altitudeMode>relativeToGround</altitudeMode>
            </Camera>
            </Document>
            </kml>
            '''
    kml_content = template.format(center_lon=lon, center_lat=lat, height=height, heading=heading, tilt=tilt)
    kml_path = 'camera_view.kml'
    with open(kml_path, 'w') as kml_file:
        kml_file.write(kml_content)

    print('正在切换视角')
    time.sleep(8)

    # 3.16公里
    pyautogui.click(974,522)  # 点击地图中央
    pyautogui.moveTo(1070,119)  # 点击保存图片按钮
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(987,1048)  # 点击更改截图名称
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')  # 全选现有名称
    pyautogui.press('backspace')  # 删除现有名称
    pyautogui.write(name, interval=0.1)  # 输入新名称(每个字符间隔0.25秒)
    pyautogui.moveTo(2040, 1156)  # 点击保存图像按钮
    pyautogui.click()
    # time.sleep(150)
    timewait = 0
    time.sleep(1)
    while os.path.exists(path) is False:
        print('等待下载完成,已等待{}秒'.format(timewait))
        timewait += 2
        time.sleep(2)
    time.sleep(1)
    img = Image.open(dir + name + '.jpg')
    w, h = img.size
    target_w, target_h = output_size_w, output_size_h
    mid_x, mid_y = w // 2, h // 2
    half_w, half_h = target_w // 2, target_h // 2
    left = mid_x - half_w
    top = mid_y - half_h
    right = mid_x + half_w
    bottom = mid_y + half_h
    cropped = img.crop((left, top, right, bottom))
    cropped.save(dir + name + '.jpg')

# === 计算经纬度 ===
def solve_formula(
    height=None,
    distance_h=None,
    output_size_h=None,
    resolution=None,
):


    params = {
        "height": height,
        "distance_h": distance_h,
        "output_size_h": output_size_h,
        "resolution": resolution,
    }

    none_keys = [k for k, v in params.items() if v is None]
    if len(none_keys) != 1:
        raise ValueError("Exactly one parameter must be None.")

    K = 204.5 * 256 / (64 * 935)

    if height is None:
        return distance_h * K * resolution / output_size_h

    if distance_h is None:
        return height * output_size_h / (K * resolution)

    if output_size_h is None:
        return distance_h * K * resolution / height

    if resolution is None:
        return height * output_size_h / (distance_h * K)

if __name__ == "__main__":
    map_name = '37bc'
    dir_="D:\\UAV-navigation\\project\\UAV_3d_v2\\xry_disa_data\\Turkey-earthquake-2023-2\\pre\\"
    # restart_google_earth(name=f'init_{random.randint(1,9999)}',dir=dir_)
    # dir_="D:\\UAV-navigation\\project\\UAV_3d_v2\\c1_254k_34bc_b15_s100_v3d\\"
    # df = pd.read_csv("p1_254k_37bc_b15_s100.csv")
    count = 0
    output_size_w = 256  # 8192x4650
    output_size_h = 256  # 8192x4650
    # print('image_width:', latm_per_pixel* output_size_w)
    # print('image_height:', lngm_per_pixel* output_size_h)
    distance_w = 64
    distance_h = 64
    height = solve_formula(
        distance_h=distance_h,
        output_size_h=output_size_h,
        resolution=935,
    )
    lat_num = 30
    lon_num = 30    
    # print((lat_num-1)/2)
    # stophere = stophere
    pic_diff_meter = 64 
    lat_orig = 36.230616

    lon_orig = 36.164287
    center_lat = lat_orig
    center_lon = lon_orig
    # print(center_lat, center_lon)
    # print(lat_per_pixel, lon_per_pixel)
    # stophere = stophere
    lat_diff = pic_diff_meter / 111000
    # center_lat = lat_st
    lat_st = center_lat - (lat_num-1) / 2 * lat_diff

    # height = 1000*distance_h*3.16*4650*935/(output_size_h*2050*8192)
    #     height = distance_h*204.5*256*Resolution/(output_size_h*64*935)   
    paths = []
    print(lat_num, lon_num)
    for i in range(lat_num):
        center_lat = round(lat_st + i * lat_diff, 6)
        # lon_st = center_lon - (lon_num - 1) / 2 * distance_w / 111000 / math.cos(center_lat / 180 * math.pi)
        lon_diff = pic_diff_meter / 111000 / math.cos(center_lat / 180 * math.pi)
        lon_st = lon_orig - (lon_num - 1) / 2 * lon_diff
        # print(lat_st, lon_st)
        # print(lat_diff, lon_diff)
        # stophere = stophere
        for j in range(lon_num):
            
            center_lon = round(lon_st + j * lon_diff, 6)

            name = str(center_lat) + "_" + str(center_lon)+ "_" +str(int(distance_w)) + "_" + str(int(distance_h)) + "_" + str(4326) + "_city"
            path = name + '.jpg'
            generate_pic(
                lat=center_lat,
                lon=center_lon,
                height=height,  
                heading=0,  
                name=name,
                distance_h=distance_h,
                distance_w=distance_w,
                output_size_h=output_size_h,
                output_size_w=output_size_w,
                dir=dir_
            )
    
    # df['target_patch_3d'] = paths
    # df.to_csv("metadata_lkj_31_3d.csv", index=False)
    
        # print(f"生成图片：{name}_v3d.jpg")
        # count += 1
        # if count > 5:
        #     break



