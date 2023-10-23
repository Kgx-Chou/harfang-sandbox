# from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


def draw_plt(val, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name, color='darkorange')
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('reward.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def draw_dif(file_name, dif, dir):
    plt.figure()
    plt.plot(range(len(dif)), dif, color='darkorange')
    plt.xlabel('step')
    plt.ylabel('dif')
    file_name = os.path.join(dir, file_name)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


# def draw_pos(file_name, self_pos, oppo_pos, dir):
#     ax = plt.figure().add_subplot(projection='3d')
#     self_pos = np.array(self_pos)
#     oppo_pos = np.array(oppo_pos)

#     num_points = len(self_pos)

#     colors = plt.cm.jet(np.linspace(0,1,num_points))

#     for i in range(num_points - 1):
#         if i != i-2:
#             ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=colors[i], label='self')
#             ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=colors[i], label='oppo')
#         else:
#             ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=colors[i], label='self')
#             ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=colors[i], label='oppo')
    
#     plt.legend()
#     file_name = os.path.join(dir, file_name)
#     plt.savefig(file_name, bbox_inches='tight')

def draw_pos(file_name, self_pos, oppo_pos, fire, dir):
    ax = plt.figure().add_subplot(projection='3d')
    self_pos = np.array(self_pos)
    oppo_pos = np.array(oppo_pos)

    num_points = len(self_pos)

    self_colors = plt.cm.Blues(np.linspace(0.2, 1, num_points))
    oppo_colors = plt.cm.Reds(np.linspace(0.2, 1, num_points))

    for i in range(num_points - 1):
        ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=self_colors[i], label='self', zorder=1)
        ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=oppo_colors[i], label='oppo', zorder=1)

    for i in fire:
        ax.plot(self_pos[i:i+1,0], self_pos[i:i+1,2], self_pos[i:i+1,1], color='yellow', zorder=2)

    file_name = os.path.join(dir, file_name)

    plt.title('追击图', fontsize=28)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()

def draw_pos2(file_name, self_pos, oppo_pos, dir):
    ax = plt.figure().add_subplot(projection='3d')
    self_pos = np.array(self_pos)
    oppo_pos = np.array(oppo_pos)

    num_points = len(self_pos)

    self_colors = plt.cm.Blues(np.linspace(0.2, 1, num_points))
    oppo_colors = plt.cm.Reds(np.linspace(0.2, 1, num_points))

    for i in range(num_points - 1):
        ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=self_colors[i], label='self')
        ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=oppo_colors[i], label='oppo')


    file_name = os.path.join(dir, file_name)

    plt.title('追击图', fontsize=28)
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()


def plot_dif(dif, lock, fire, dir, name):
    # 创建一个新图形
    plt.figure(figsize=(10, 6))

    # 绘制 `dif` 曲线
    plt.plot(dif, color='blue', linewidth=2, marker='None')  # 用'None'禁用点的绘制


    # 着色 `lock` 区间为黄色
    for i in range(len(lock)):
        plt.axvspan(lock[i], lock[i]+1, facecolor='yellow', alpha=0.5)

    # 标记 `fire` 对应的点为红色并增大
    for fire_step in fire:
        plt.scatter(fire_step, dif[fire_step], color='red', s=40)

    plt.xticks(fontsize=10)  # 设置x轴标签的字体大小
    plt.yticks(fontsize=10)  # 设置y轴标签的字体大小     
    plt.xlabel('步数', fontsize=16)
    plt.ylabel('距离', fontsize=16)
    # plt.legend()
    plt.title('距离图', fontsize=20)
    plt.grid(True)
    dir = os.path.join(dir, name)
    plt.savefig(dir, dpi=300)
    plt.close()

def plot_dif2(dif, lock, missile, fire, dir, name):
    # 创建一个新图形
    plt.figure(figsize=(10, 7))

    # 绘制 `dif` 曲线
    plt.plot(dif, color='blue', linewidth=3, marker='None', zorder=2)  # 用'None'禁用点的绘制


    # 着色 `lock` 区间为淡黄色
    for i in range(len(lock)):
        plt.axvspan(lock[i], lock[i]+1, facecolor='#FFDFBF', alpha=0.6, zorder=1)

    # 着色 `missile` 区间为淡蓝色
    for j in range(len(missile)):
        plt.axvspan(missile[j], missile[j]+1, facecolor='#ACD8E6', alpha=0.6, zorder=1)

    # 标记 `fire` 对应的点为红色并增大
    for fire_step in fire:
        plt.scatter(fire_step, dif[fire_step], color='red', s=45, zorder=3)

    plt.xticks(fontsize=18)  # 设置x轴标签的字体大小
    plt.yticks(fontsize=18)  # 设置y轴标签的字体大小     
    plt.xlabel('步数', fontsize=25)
    plt.ylabel('距离', fontsize=25)
    # plt.legend()
    plt.title('距离图', fontsize=28)
    plt.grid(True)
    dir = os.path.join(dir, name)
    plt.savefig(dir, dpi=300)
    plt.close()

if __name__ == "__main__":
    tensorboard_path = '/Users/wenyongyan/Downloads/logs/events.out.tfevents.1687995824.DESKTOP-6S5E44I.284.0'
    val_name = 'Training/Episode Reward'
    val = read_tensorboard_data(tensorboard_path, val_name)
    draw_plt(val, val_name)
