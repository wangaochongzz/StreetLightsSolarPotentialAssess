# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

import pandas as pd

if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = True
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = r"D:\paper\02_论文和图\路灯识别和定位流程图\yolov5"
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   dir_save_excel_name 指定了将预测结果保存到当前工作路径下的excel名字

    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "D:\\paper\\02_论文和图\\路灯识别和定位流程图\\img\\"
    dir_save_path = "D:\\paper\\02_论文和图\\路灯识别和定位流程图\\yolov5\\"
    dir_save_csv_name = "D:\\paper\\02_论文和图\\路灯识别和定位流程图\\yolov5\\Images.csv"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1./(time.time()-t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (
                fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' +
              str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        # 在当前工作路径下建立一个新的excel存放预测结果
        # writer = pd.ExcelWriter(dir_save_excel_name)
        # 创建几个列表存储预测图片的id和预测后结果left、right
        ids = []
        names = []
        lefts = []
        rights = []
        tops = []
        bottoms = []
        lats = []
        lons = []
        photoTimes = []
        elevations = []
        northRotations = []
        bearings = []
        classes = []
        areas = []

        # 半张全景图预测
        # LorRs = []

        # 读取当前文件夹中的所有图片名字存进img_names列表,如["001.jpg","002.jpg"]
        img_names = os.listdir(dir_origin_path)
        # 遍历图片名列表，并以进度条的形式显示，img_name = "001.jpg"
        for img_name in tqdm(img_names):
            # 判断将当前的图片名是否是.jpg等形式结尾
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                # 读取当前图片的路径,img_path="(dir_origin_path)predict/img_01/001.jpg"
                image_path = os.path.join(dir_origin_path, img_name)
                # 读取当前图片
                image = Image.open(image_path)
                # 返回已经被添加预测框后的图片r_image和预测的left、right(一张图可能有多个)
                predicted_ids = []
                predicted_names = []
                predicted_lats = []
                predicted_lons = []
                predicted_photoTimes = []
                predicted_elevations = []
                predicted_northRotations = []
                predicted_lefts = []
                predicted_rights = []
                predicted_tops = []
                predicted_bottoms = []
                predicted_bearings = []
                predicted_classes = []
                predicted_areas = []

                # 半张全景图预测
                # predicted_LorRs = []

                # 进行预测
                r_image, predicted_classes, predicted_lefts, predicted_rights, predicted_tops, predicted_bottoms = yolo.detect_image(
                    image)
                # 将预测的predicted_id、predicted_lefts、predicted_rights结果添加到ids、lefts、rights列表中
                # 截取当前图片名字上的信息
                img_name_splitted = img_name.split("_")
                id = img_name_splitted[0]
                name = img_name
                lat = img_name_splitted[1]
                lon = img_name_splitted[2]
                photoTime = img_name_splitted[3]
                elevation = img_name_splitted[4]
                northRotation = img_name_splitted[5].rstrip(".jpg")

                # 半张全景图预测 如果对全景图预测就将下面一行注释掉
                # LorR = img_name_splitted[6].rstrip(".jpg")

                predicted_count = len(predicted_lefts)
                for count in range(0, predicted_count):
                    predicted_ids.append(id)
                    predicted_names.append(name)
                    predicted_lats.append(lat)
                    predicted_lons.append(lon)
                    predicted_photoTimes.append(photoTime)
                    predicted_elevations.append(elevation)
                    predicted_northRotations.append(northRotation)
                    predicted_areas.append((float(predicted_rights[count])-float(predicted_lefts[count]))*(
                        float(predicted_bottoms[count])-float(predicted_tops[count])))

                    # 半张全景图预测 如果对全景图预测就将下面一行注释掉
                    # predicted_LorRs.append(LorR)

                    # 半张全景图的预测公式 如果对全景图预测就将下面四行注释掉

                    # if(LorR=="L"):
                    #     predicted_bearings.append(((((float(predicted_lefts[count])+float(predicted_rights[count]))/2.0+256) * 0.17578125) + 360 - float(northRotation)) % 360)
                    # else:
                    #     predicted_bearings.append(((((float(predicted_lefts[count])+float(predicted_rights[count]))/2.0+1280) * 0.17578125) + 360 - float(northRotation)) % 360)

                    # 全景图的预测公式 如果对全景图预测就将下面一行打开 本结果是个粗略结果
                    predicted_bearings.append((((float(predicted_lefts[count])+float(
                        predicted_rights[count]))/2.0 * 0.17578125) + 360 - float(northRotation)) % 360)

                ids.extend(predicted_ids)
                names.extend(predicted_names)
                lats.extend(predicted_lats)
                lons.extend(predicted_lons)
                photoTimes.extend(predicted_photoTimes)
                elevations.extend(predicted_elevations)
                northRotations.extend(predicted_northRotations)

                classes.extend(predicted_classes)
                lefts.extend(predicted_lefts)
                rights.extend(predicted_rights)
                tops.extend(predicted_tops)
                bottoms.extend(predicted_bottoms)
                areas.extend(predicted_areas)
                bearings.extend(predicted_bearings)

                # 半张全景图图预测 如果对全景图预测就将下面一行注释掉
                # LorRs.extend(predicted_LorRs)

                # 保存预测图片的路径(dir_save_path)"predict/img_01_out/"不存在就创建一个
                # 将预测过的新图片保存到dir_save_path文件夹中
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(
                    ".jpg", ".png")), quality=95, subsampling=0)
        # 将获取的ids,lefts,rights列表数据存进字典dict中 如果对全景图预测就使用下面的那个dict（即删除掉结尾的,"LorR":LorRs ）
        # dict = {"id":ids,"lat":lats,"lon":lons,"photoTime":photoTimes,"elevation":elevations,"northRotation":northRotations,
        #         "left":lefts,"right":rights,"bearing":bearings,"LorR":LorRs}
        dict = {"lat": lats, "lon": lons, "bearing_test": bearings, "name": names, "id": ids, "photoTime": photoTimes, "elevation": elevations, "northRotation": northRotations,
                "class": classes, "left": lefts, "right": rights, "top": tops, "bottom": bottoms, "area": areas}
        # 将字典转换为dataframe格式的数据data
        df = pd.DataFrame(dict)
        # 将data存进开始建立的excel中
        # data.to_excel(writer, "sheet1", startcol=0, index=False)
        # 保存并关闭excel
        # writer.close()
        # 保存为 CSV 文件
        df.to_csv(dir_save_csv_name, index=False)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
