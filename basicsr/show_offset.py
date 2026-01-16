import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_offset(image, all_offset, deform_kernel_size=3, position_shift=None, step=None, plot_area=2):
    """
    当deform_kernel_size=3时
    position_shift可取[-1,0,1],[-2,0,2]

    当deform_kernel_size=3时
    position_shift可取[-2,-1,0,1,2]

    当deform_kernel_size决定position_shift的维度，position_shift的数值大小决定视觉上偏移点的离散程度

    :param image: 原始分辨率图像 np(h*w*3)
    :param all_offset: 从第0层到第len()-1层所有的偏移 第0层的偏移特征图作为底图 以可形变核尺寸3,3层卷积为例[(b*18*(h/4)*(w/4)),(b*18*(h/2)*(w/2)),(b*18*h*w)]
    :param deform_kernel_size: 可形变卷积核尺寸  deform_kernel_size与position_shift维度要对应
    :param position_shift: 规则偏移
    :param step: 隔step个像素可视化一次 step[0]代表h方向  step[1]代表w方向
    :param plot_area: 每个点显示为(2*plot_area+1)*(2*plot_area+1)的区域
    :return:
    """
    if step is None:
        step = [2, 2]
    if position_shift is None:
        position_shift = [-1, 0, 1]
    assert deform_kernel_size == len(position_shift)  # deform_kernel_size与position_shift维度要对应
    plt.figure()
    for h in range(0, all_offset[0].shape[2], step[0]):  # 以第0层的偏移特征图作为底图
        for w in range(0, all_offset[0].shape[3], step[1]):
            source_h = np.round(h * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 低分辨率特征图上采样点坐标放缩回原始分辨率坐标
            source_w = np.round(w * image.shape[1] / all_offset[0].shape[3]).astype('i')
            if source_h < plot_area or source_w < plot_area or source_h >= image.shape[0] - plot_area or source_w >= image.shape[1] - plot_area:
                continue  # 原始分辨率坐标越界检测
            image_copy = np.copy(image)
            target_points = [np.array([h, w])]  # 当前层 低分辨率特征图上采样点坐标 在第0层只有一个点，第i-1层的偏移点，是第i层的采样点
            offset_points = []  # 第0层采样点在第len(all_offset)层的所有偏移点 常见数量9*9*9=729
            # 获取所有偏移点坐标
            ''''
            当前层偏移点作为下一层采样点的时候，要不要根据当前层与下一层特征分辨率做放缩
            我觉得是需要的，但可形变卷积作者没这么做，我也先不这么做 2022-03-26-20:01
            不，我觉得必须进行放缩 2022-03-26-20:06
            当前层偏移点作为下一层采样点，下一层的偏移量是相对于下一层特征图分辨率来说，如果下一层特征图与当前层特征图分辨率不同，那采样点应当做缩放 2022-03-27-13:39
            '''
            for level in range(len(all_offset)):  # 遍历所有层的偏移，一般是三层
                offset_points.clear()  # 记录本层所有偏移点坐标前，把上一层偏移点清空
                for target_point in target_points:  # 遍历当前层采样点
                    target_point = np.round(target_point)  # 四舍五入到整数
                    if target_point[0] < 0 or target_point[1] < 0 or target_point[0] > all_offset[level].shape[2] - 1 or target_point[1] > all_offset[level].shape[3] - 1:
                        continue  # 越界检测 当前层采样点坐标不在当前层特征分辨率地图内
                    shift_target_points = []  # 采样点对应的规则偏移点，常见数量9、25
                    # 获取采样点对应的规则偏移点
                    for i in range(deform_kernel_size * deform_kernel_size):
                        shift_target_point_h = target_point[0] + position_shift[i // deform_kernel_size]  # 常见式子[i//3]或[i//5]
                        shift_target_point_w = target_point[1] + position_shift[i % deform_kernel_size]  # 常见式子[i%3]或[i%5]
                        if shift_target_point_h < 0 or shift_target_point_w < 0 or shift_target_point_h > all_offset[level].shape[2] - 1 or shift_target_point_w > all_offset[level].shape[3] - 1:
                            continue  # 越界检测 规则偏移点坐标不在当前层特征分辨率地图内
                        shift_target_points.append(np.array([shift_target_point_h, shift_target_point_w]).astype('f'))  # 规则采样点

                    offset = np.squeeze(all_offset[level][:, :, int(target_point[0]), int(target_point[1])])  # 当前层采样点对应的(1,18,1,1)偏移量
                    # 规则偏移点+偏移量=最终的偏移点坐标
                    for i in range(len(shift_target_points)):
                        shift_target_points[i][0] += offset[2 * i]
                        shift_target_points[i][1] += offset[2 * i + 1]
                    offset_points = offset_points + shift_target_points
                target_points = offset_points[:]  # 必须要用深拷贝，如果用浅拷贝，待会offset_points.clear()又清空数据
            # 给偏移点上色
            for offset_point in offset_points:
                y = np.round((offset_point[0] + 0.5) * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 从第0层坐标放缩回原始分辨率坐标
                x = np.round((offset_point[1] + 0.5) * image.shape[1] / all_offset[0].shape[3]).astype('i')
                if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
                    continue  # 越界检测
                y = min(y, image.shape[0] - plot_area - 1)
                x = min(x, image.shape[1] - plot_area - 1)
                y = max(y, plot_area)
                x = max(x, plot_area)
                image_copy[y - plot_area:y + plot_area + 1, x - plot_area:x + plot_area + 1, :] = np.tile(np.reshape([255, 0, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
            # 给采样点上色
            image_copy[source_h - plot_area:source_h + plot_area + 1, source_w - plot_area:source_w + plot_area + 1, :] = np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
            plt.axis("off")
            plt.imshow(image_copy)
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()


def show_offset_test():
    image = Image.open("D:\\dataset\\GOPRO_Large\\test\\GOPR0384_11_00\\blur\\000001.png")
    image = np.array(image)
    offset1 = np.random.random(size=(1, 18, 180, 320)) * 10
    offset2 = np.random.random(size=(1, 18, 360, 640)) * 10
    offset3 = np.random.random(size=(1, 18, 720, 1280)) * 10
    all_offset = [offset1, offset2, offset3]
    show_offset(image=image, all_offset=all_offset, position_shift=[-2, 0, 2])  # 测试成功


def get_offset_image(image, all_offset, deform_kernel_size=3, position_shift=None, step=None, plot_area=2):
    """
    当deform_kernel_size=3时
    position_shift可取[-1,0,1],[-2,0,2]

    当deform_kernel_size=3时
    position_shift可取[-2,-1,0,1,2]

    当deform_kernel_size决定position_shift的维度，position_shift的数值大小决定视觉上偏移点的离散程度

    :param image: 原始分辨率图像 np(h*w*3)
    :param all_offset: 从第0层到第len()-1层所有的偏移 第0层的偏移特征图作为底图 以可形变核尺寸3,3层卷积为例[(b*18*(h/4)*(w/4)),(b*18*(h/2)*(w/2)),(b*18*h*w)]
    :param deform_kernel_size: 可形变卷积核尺寸  deform_kernel_size与position_shift维度要对应
    :param position_shift: 规则偏移
    :param step: 隔step个像素可视化一次 step[0]代表h方向  step[1]代表w方向
    :param plot_area: 每个点显示为(2*plot_area+1)*(2*plot_area+1)的区域
    :return:
    """
    if step is None:
        step = [20, 20]
    if position_shift is None:
        position_shift = [-1, 0, 1]
    assert deform_kernel_size == len(position_shift)  # deform_kernel_size与position_shift维度要对应
    # plt.figure()
    image_copy = np.copy(image)
    for h in range(0, all_offset[0].shape[2], step[0]):  # 以第0层的偏移特征图作为底图
        for w in range(0, all_offset[0].shape[3], step[1]):
            source_h = np.round(h * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 低分辨率特征图上采样点坐标放缩回原始分辨率坐标
            source_w = np.round(w * image.shape[1] / all_offset[0].shape[3]).astype('i')
            if source_h < plot_area or source_w < plot_area or source_h >= image.shape[0] - plot_area or source_w >= image.shape[1] - plot_area:
                continue  # 原始分辨率坐标越界检测
            # image_copy = np.copy(image)
            target_points = [np.array([h, w])]  # 当前层 低分辨率特征图上采样点坐标 在第0层只有一个点，第i-1层的偏移点，是第i层的采样点
            offset_points = []  # 第0层采样点在第len(all_offset)层的所有偏移点 常见数量9*9*9=729
            # 获取所有偏移点坐标
            ''''
            当前层偏移点作为下一层采样点的时候，要不要根据当前层与下一层特征分辨率做放缩
            我觉得是需要的，但可形变卷积作者没这么做，我也先不这么做 2022-03-26-20:01
            不，我觉得必须进行放缩 2022-03-26-20:06
            当前层偏移点作为下一层采样点，下一层的偏移量是相对于下一层特征图分辨率来说，如果下一层特征图与当前层特征图分辨率不同，那采样点应当做缩放 2022-03-27-13:39
            '''
            for level in range(len(all_offset)):  # 遍历所有层的偏移，一般是三层
                offset_points.clear()  # 记录本层所有偏移点坐标前，把上一层偏移点清空
                for target_point in target_points:  # 遍历当前层采样点
                    target_point = np.round(target_point)  # 四舍五入到整数
                    if target_point[0] < 0 or target_point[1] < 0 or target_point[0] > all_offset[level].shape[2] - 1 or target_point[1] > all_offset[level].shape[3] - 1:
                        continue  # 越界检测 当前层采样点坐标不在当前层特征分辨率地图内
                    shift_target_points = []  # 采样点对应的规则偏移点，常见数量9、25
                    # 获取采样点对应的规则偏移点
                    for i in range(deform_kernel_size * deform_kernel_size):
                        shift_target_point_h = target_point[0] + position_shift[i // deform_kernel_size]  # 常见式子[i//3]或[i//5]
                        shift_target_point_w = target_point[1] + position_shift[i % deform_kernel_size]  # 常见式子[i%3]或[i%5]
                        if shift_target_point_h < 0 or shift_target_point_w < 0 or shift_target_point_h > all_offset[level].shape[2] - 1 or shift_target_point_w > all_offset[level].shape[3] - 1:
                            continue  # 越界检测 规则偏移点坐标不在当前层特征分辨率地图内
                        shift_target_points.append(np.array([shift_target_point_h, shift_target_point_w]).astype('f'))  # 规则采样点

                    offset = np.squeeze(all_offset[level][:, :, int(target_point[0]), int(target_point[1])])  # 当前层采样点对应的(1,18,1,1)偏移量
                    # 规则偏移点+偏移量=最终的偏移点坐标
                    for i in range(len(shift_target_points)):
                        # shift_target_points[i][0] += offset[2 * i]
                        # shift_target_points[i][1] += offset[2 * i + 1]
                        # shift_target_points[i][1] += offset[i]  # 这才是正解！！
                        # shift_target_points[i][0] += offset[i + (deform_kernel_size * deform_kernel_size)]
                        shift_target_points[i][1] -= offset[i]  # 这才是正解！！ x-u、y-v
                        shift_target_points[i][0] -= offset[i + (deform_kernel_size * deform_kernel_size)]
                    offset_points = offset_points + shift_target_points
                target_points = offset_points[:]  # 必须要用深拷贝，如果用浅拷贝，待会offset_points.clear()又清空数据
            # 给偏移点上色
            for offset_point in offset_points:
                y = np.round((offset_point[0] + 0.5) * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 从第0层坐标放缩回原始分辨率坐标
                x = np.round((offset_point[1] + 0.5) * image.shape[1] / all_offset[0].shape[3]).astype('i')
                if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
                    continue  # 越界检测
                y = min(y, image.shape[0] - plot_area - 1)
                x = min(x, image.shape[1] - plot_area - 1)
                y = max(y, plot_area)
                x = max(x, plot_area)
                image_copy[y - plot_area:y + plot_area + 1, x - plot_area:x + plot_area + 1, :] = np.tile(np.reshape([255, 0, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
            # 给采样点上色
            image_copy[source_h - plot_area:source_h + plot_area + 1, source_w - plot_area:source_w + plot_area + 1, :] = np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
    return image_copy


def get_offset_weight_image(image, all_offset, weight, deform_kernel_size=3, position_shift=None, step=None, plot_area=2):
    """
    当deform_kernel_size=3时
    position_shift可取[-1,0,1],[-2,0,2]

    当deform_kernel_size=3时
    position_shift可取[-2,-1,0,1,2]

    当deform_kernel_size决定position_shift的维度，position_shift的数值大小决定视觉上偏移点的离散程度

    :param image: 原始分辨率图像 np(h*w*3)
    :param all_offset: 从第0层到第len()-1层所有的偏移 第0层的偏移特征图作为底图 以可形变核尺寸3,3层卷积为例[(b*18*(h/4)*(w/4)),(b*18*(h/2)*(w/2)),(b*18*h*w)]
    :param weight:
    :param deform_kernel_size: 可形变卷积核尺寸  deform_kernel_size与position_shift维度要对应
    :param position_shift: 规则偏移
    :param step: 隔step个像素可视化一次 step[0]代表h方向  step[1]代表w方向
    :param plot_area: 每个点显示为(2*plot_area+1)*(2*plot_area+1)的区域
    :return:
    """
    if step is None:
        step = [20, 20]
    if position_shift is None:
        position_shift = [-1, 0, 1]
    assert deform_kernel_size == len(position_shift)  # deform_kernel_size与position_shift维度要对应
    # plt.figure()
    image_copy = np.copy(image)
    for h in range(0, all_offset[0].shape[2], step[0]):  # 以第0层的偏移特征图作为底图
        for w in range(0, all_offset[0].shape[3], step[1]):
            source_h = np.round(h * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 低分辨率特征图上采样点坐标放缩回原始分辨率坐标
            source_w = np.round(w * image.shape[1] / all_offset[0].shape[3]).astype('i')
            if source_h < plot_area or source_w < plot_area or source_h >= image.shape[0] - plot_area or source_w >= image.shape[1] - plot_area:
                continue  # 原始分辨率坐标越界检测
            # image_copy = np.copy(image)
            target_points = [np.array([h, w])]  # 当前层 低分辨率特征图上采样点坐标 在第0层只有一个点，第i-1层的偏移点，是第i层的采样点
            offset_points = []  # 第0层采样点在第len(all_offset)层的所有偏移点 常见数量9*9*9=729
            # 获取所有偏移点坐标
            ''''
            当前层偏移点作为下一层采样点的时候，要不要根据当前层与下一层特征分辨率做放缩
            我觉得是需要的，但可形变卷积作者没这么做，我也先不这么做 2022-03-26-20:01
            不，我觉得必须进行放缩 2022-03-26-20:06
            当前层偏移点作为下一层采样点，下一层的偏移量是相对于下一层特征图分辨率来说，如果下一层特征图与当前层特征图分辨率不同，那采样点应当做缩放 2022-03-27-13:39
            '''
            for level in range(len(all_offset)):  # 遍历所有层的偏移，一般是三层
                offset_points.clear()  # 记录本层所有偏移点坐标前，把上一层偏移点清空
                for target_point in target_points:  # 遍历当前层采样点
                    target_point = np.round(target_point)  # 四舍五入到整数
                    if target_point[0] < 0 or target_point[1] < 0 or target_point[0] > all_offset[level].shape[2] - 1 or target_point[1] > all_offset[level].shape[3] - 1:
                        continue  # 越界检测 当前层采样点坐标不在当前层特征分辨率地图内
                    shift_target_points = []  # 采样点对应的规则偏移点，常见数量9、25
                    # 获取采样点对应的规则偏移点
                    for i in range(deform_kernel_size * deform_kernel_size):
                        shift_target_point_h = target_point[0] + position_shift[i // deform_kernel_size]  # 常见式子[i//3]或[i//5]
                        shift_target_point_w = target_point[1] + position_shift[i % deform_kernel_size]  # 常见式子[i%3]或[i%5]
                        if shift_target_point_h < 0 or shift_target_point_w < 0 or shift_target_point_h > all_offset[level].shape[2] - 1 or shift_target_point_w > all_offset[level].shape[3] - 1:
                            continue  # 越界检测 规则偏移点坐标不在当前层特征分辨率地图内
                        shift_target_points.append(np.array([shift_target_point_h, shift_target_point_w]).astype('f'))  # 规则采样点

                    offset = np.squeeze(all_offset[level][:, :, int(target_point[0]), int(target_point[1])])  # 当前层采样点对应的(1,18,1,1)偏移量
                    # 规则偏移点+偏移量=最终的偏移点坐标
                    for i in range(len(shift_target_points)):
                        # shift_target_points[i][0] += offset[2 * i]
                        # shift_target_points[i][1] += offset[2 * i + 1]
                        # shift_target_points[i][1] += offset[i]  # 这才是正解！！
                        # shift_target_points[i][0] += offset[i + (deform_kernel_size * deform_kernel_size)]
                        shift_target_points[i][1] -= offset[i]  # 这才是正解！！ x-u、y-v
                        shift_target_points[i][0] -= offset[i + (deform_kernel_size * deform_kernel_size)]
                    offset_points = offset_points + shift_target_points
                target_points = offset_points[:]  # 必须要用深拷贝，如果用浅拷贝，待会offset_points.clear()又清空数据
            # 给偏移点上色
            # for offset_point in offset_points:
            for i in range(len(offset_points)):
                y = np.round((offset_points[i][0] + 0.5) * image.shape[0] / all_offset[0].shape[2]).astype('i')  # 从第0层坐标放缩回原始分辨率坐标
                x = np.round((offset_points[i][1] + 0.5) * image.shape[1] / all_offset[0].shape[3]).astype('i')
                if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
                    continue  # 越界检测
                y = min(y, image.shape[0] - plot_area - 1)
                x = min(x, image.shape[1] - plot_area - 1)
                y = max(y, plot_area)
                x = max(x, plot_area)
                image_copy[y - plot_area:y + plot_area + 1, x - plot_area:x + plot_area + 1, :] = np.tile(np.reshape([255, 0, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
            # 给采样点上色
            image_copy[source_h - plot_area:source_h + plot_area + 1, source_w - plot_area:source_w + plot_area + 1, :] = np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2 * plot_area + 1, 2 * plot_area + 1, 1))
    return image_copy


def get_offset_image_test():
    image = Image.open("D:\\dataset\\GOPRO_Large\\test\\GOPR0384_11_00\\blur\\000001.png")
    image = np.array(image)
    offset1 = np.random.random(size=(1, 18, 180, 320)) * 10
    offset2 = np.random.random(size=(1, 18, 360, 640)) * 10
    offset3 = np.random.random(size=(1, 18, 720, 1280)) * 10
    # all_offset = [offset1, offset2, offset3]
    all_offset = [offset1]
    image = get_offset_image(image=image, all_offset=all_offset, position_shift=[-2, 0, 2])  # 测试成功
    # from Utils.utils import show_image
    # show_image(image)


if __name__ == '__main__':
    # show_offset_test()
    get_offset_image_test()
