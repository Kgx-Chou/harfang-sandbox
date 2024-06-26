
# Smart Dogfight:Integrating Imitation Learning and Reinforcement Learning


## 依赖
1. 安装`Harfang3D sandbox`的[Release版本](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/tag/v1.3.0)或[源代码](https://github.com/harfang3d/dogfight-sandbox-hg2)，推荐安装源代码版本，这样可以自行更改环境的端口
2. 安装本代码所需依赖，可以在[官网requirements.txt](https://github.com/harfang3d/dogfight-sandbox-hg2/blob/main/Agent/requirements.txt)的基础上补充安装，也可以通过以下命令直接安装
    ~~~bash
    pip install -r requirements.txt
    ~~~

## 生成专家数据

### 基于规则
> 飞行规则自定
* 运行`demo.py`将会生成专家数据的csv文件
* video</br>[![rule-based expert demo](https://github.com/zrc0622/harfang-sandbox/blob/master/pictures/1(1).jpg)](https://www.youtube.com/watch?v=i6DAneyneh8 "rule-based expert demo") 

### 基于AI
* 运行`demo_AI.py`将会生成专家数据的csv文件
* video</br>[![AI-based expert demo](https://github.com/zrc0622/harfang-sandbox/blob/master/pictures/2(1).jpg)](https://www.youtube.com/watch?v=uQKoI0rQC2k "AI-based expert demo")
