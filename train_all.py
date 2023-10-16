#IMPORTS
from Network.TD3 import Agent as TD3Agent
from Network.ROT import Agent as ROTAgent
from Network.BC import Agent as BCAgent
from read_data import read_data
from ReplayMemory import *
import numpy as np
import time
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from HarfangEnv_GYM import *
import dogfight_client as df

import datetime
import os
from pathlib import Path
import csv

from plot import draw_dif, draw_pos, plot_dif
import argparse

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

def main(config):
    print(torch.cuda.is_available())

    agent_name = config.agent
    model_name = config.model_name
    port = config.port
    rot_type = config.type
    if_up_sample = config.upsample
    bc_weight = config.bc_weight

    df.connect("10.241.58.126", port) #TODO:Change IP and PORT values

    start = time.time() #STARTING TIME
    df.disable_log()

    # PARAMETERS
    trainingEpisodes = 6000
    validationEpisodes = 25 # 100
    explorationEpisodes = 200 # 200

    Test = False
    if Test:
        render = False
    else:
        render = True
        
    df.set_renderless_mode(render)
    df.set_client_update_mode(True)

    bufferSize = (10**6)
    gamma = 0.99
    criticLR = 1e-3
    actorLR = 1e-3
    tau = 0.005
    checkpointRate = 25 # 100
    highScore = -math.inf
    successRate = -math.inf
    batchSize = 128
    maxStep = 6000
    validatStep = 6000
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 14 # gai
    actionDim = 4 # gai
    useLayerNorm = True

    data_dir = './expert_data/expert_data_ai2.csv'
    data_folder_dir = './expert_data'
    expert_states, expert_actions = read_data(data_dir)
    print(expert_states.shape)
    print(expert_actions.shape)

    name = "Harfang_GYM"


    #INITIALIZATION
    env = HarfangEnv()

    if agent_name == 'ROT':
        agent = ROTAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight, if_up_sample)
        model_dir = './model/ROT/' + model_name
    elif agent_name == 'BC':
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)
        model_dir = './model/BC/' + model_name
    elif agent_name == 'TD3':
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, if_up_sample)
        model_dir = './model/TD3/' + model_name

    if not Test:
        start_time = datetime.datetime.now()
        dir = Path.cwd() # 获取工作区路径
        log_dir = str(dir) + "\\" + model_name + "\\" + "log\\" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute) # tensorboard文件夹路径
        plot_dir = log_dir + "\\" + "plot"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validatStep=validatStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=model_dir,rot_type=rot_type,upsample=if_up_sample)
        env.save_parameters_to_txt(log_dir)

        writer = SummaryWriter(log_dir)
    
    arttir = 1
    # agent.loadCheckpoints(f"Agent0_", model_dir) # 使用未添加导弹的结果进行训练
    # agent.loadCheckpoints(f"Agent10_score-984.673179037746", model_dir) # 使用未添加导弹的结果进行训练
    # 从500开始，550之后的（不包括550）使用此数据

    if not Test:
        if agent_name == 'BC':
            print('agent is BC')
            for episoed in range(2000):
                for step in range(maxStep):
                    bc_loss = agent.train_actor()
                    writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
                now = time.time()
                seconds = int((now - start) % 60)
                minutes = int(((now - start) // 60) % 60)
                hours = int((now - start) // 3600)
                print('Episode: ', episode+1, 'RunTime: ', hours, ':',minutes,':', seconds)

                # validateion
                if (((episode + 1) % 100) == 0):
                    success = 0
                    valScores = []
                    dif = []
                    self_pos = []
                    oppo_pos = []
                    for e in range(validationEpisodes):

                        dif1=[]
                        fire=[]
                        lock=[]
                        state = env.reset()
                        totalReward = 0
                        done = False
                        for step in range(validatStep):
                            if not done:
                                action = agent.chooseActionNoNoise(state)
                                n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward   = env.step_test(action)
                                state = n_state
                                totalReward += reward
                                
                                dif1.append(env.loc_diff)
                                if iffire:
                                    fire.append(step)
                                if locked:
                                    lock.append(step)
                                
                                if e == validationEpisodes - 1:
                                    dif.append(env.loc_diff)
                                    self_pos.append(env.get_pos())
                                    oppo_pos.append(env.get_oppo_pos())

                                if step is validatStep - 1:
                                    break

                            elif done:
                                if 500 < env.Plane_Irtifa < 10000: # 改
                                # if env.Ally_target_locked == True:
                                    
                                    # with open('./dif/dif{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['dif'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], dif1))  # 将列表的每个元素写入CSV行
                                    # with open('./dif/fire{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['step'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], fire))  # 将列表的每个元素写入CSV行
                                    # with open('./dif/lock{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['step'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], lock))  # 将列表的每个元素写入CSV行
                                    plot_dif(dif1, fire, lock, f'my_sdif_{arttir}.png')

                                    success += 1
                                break

                        valScores.append(totalReward)

                    if mean(valScores) > highScore or success/validationEpisodes > successRate or arttir%10 == 0:
                        if mean(valScores) > highScore: # 总奖励分数
                            highScore = mean(valScores)
                            agent.saveCheckpoints("Agent{}_score{}".format(arttir, highScore), model_dir)
                            draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                            draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir) 
                            plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')

                        elif success / validationEpisodes > successRate: # 追逐成功率
                            successRate = success / validationEpisodes
                            agent.saveCheckpoints("Agent{}_successRate{}".format(arttir, successRate), model_dir)
                            draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                            draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir)
                            plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')
                
                    arttir += 1

                    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes)
                    writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
                    writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
        if agent_name == 'ROT':
            print(f'agent is ROT, ROT type is {rot_type}')
            # RANDOM EXPLORATION
            print("Exploration Started")
            for episode in range(explorationEpisodes):
                state = env.reset()
                done = False
                for step in range(maxStep):
                    if not done:
                        action = env.action_space.sample()                

                        n_state,reward,done, info, stepsuccess = env.step(action)
                        # print(n_state)
                        if step is maxStep-1:
                            done = True
                        agent.store(state,action,n_state,reward,done,stepsuccess)
                        state=n_state

                        if done:
                            break
                sys.stdout.write("\rExploration Completed: %.2f%%" % ((episode+1)/explorationEpisodes*100))
            sys.stdout.write("\n")

            print("Training Started")
            scores = []
            trainsuccess = []
            for episode in range(trainingEpisodes):
                state = env.reset()
                totalReward = 0
                episode_step = 0
                done = False
                fire = False

                if rot_type == 'linear':
                    bc_weight_now = bc_weight - episode/1000
                    if bc_weight_now <= 0:
                        bc_weight_now = 0
                elif rot_type == 'fixed':
                    bc_weight_now = bc_weight
                elif rot_type == 'soft':
                    bc_weight_now = 100

                for step in range(maxStep):
                    if not done:
                        action = agent.chooseAction(state)
                        n_state,reward,done, info, stepsuccess = env.step(action)

                        if step is maxStep - 1:
                            episode_step = step
                            break

                        agent.store(state, action, n_state, reward, done, stepsuccess) # n_state 为下一个状态
                        if stepsuccess:
                            print('success')
                        state = n_state
                        totalReward += reward

                        if agent.buffer.fullEnough(agent.batchSize):
                            critic_loss, actor_loss, bc_loss, rl_loss, bc_fire_loss = agent.learn(bc_weight_now)
                            writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                            writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                            writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
                            writer.add_scalar('Loss/RL_Loss', rl_loss, step + episode * maxStep)     
                            writer.add_scalar('Loss/BC_Fire_Loss', bc_fire_loss, step + episode * maxStep)
                            
                    elif done:
                        if 500 < env.Plane_Irtifa < 10000: # 改
                        # if env.Ally_target_locked == True:
                            fire = True
                            episode_step = step
                        break
                    
                scores.append(totalReward)
                if fire:
                    trainsuccess.append(1)
                else:
                    trainsuccess.append(0)
                writer.add_scalar('Training/Episode Reward', totalReward, episode)
                writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
                writer.add_scalar('Training/Average Step Reward', totalReward/episode_step, episode)
                writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)               
                
                now = time.time()
                seconds = int((now - start) % 60)
                minutes = int(((now - start) // 60) % 60)
                hours = int((now - start) // 3600)
                print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, \
                    ' FinalReward: %.2f' % totalReward, \
                    ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                    'RunTime: ', hours, ':',minutes,':', seconds, 'BC_weight: ', bc_weight_now)
                    
                #VALIDATION
                if (((episode + 1) % checkpointRate) == 0):
                    success = 0
                    valScores = []
                    dif = []
                    self_pos = []
                    oppo_pos = []
                    for e in range(validationEpisodes):

                        dif1=[]
                        fire=[]
                        lock=[]
                        state = env.reset()
                        totalReward = 0
                        done = False
                        for step in range(validatStep):
                            if not done:
                                action = agent.chooseActionNoNoise(state)
                                n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward   = env.step_test(action)
                                state = n_state
                                totalReward += reward
                                
                                dif1.append(env.loc_diff)
                                if iffire:
                                    fire.append(step)
                                if locked:
                                    lock.append(step)
                                
                                if e == validationEpisodes - 1:
                                    dif.append(env.loc_diff)
                                    self_pos.append(env.get_pos())
                                    oppo_pos.append(env.get_oppo_pos())

                                if step is validatStep - 1:
                                    break

                            elif done:
                                if 500 < env.Plane_Irtifa < 10000: # 改
                                # if env.Ally_target_locked == True:
                                    
                                    # with open('./dif/dif{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['dif'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], dif1))  # 将列表的每个元素写入CSV行
                                    # with open('./dif/fire{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['step'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], fire))  # 将列表的每个元素写入CSV行
                                    # with open('./dif/lock{}.csv'.format(dif1[-1]), 'w', newline='') as file:
                                    #     writer1 = csv.writer(file)
                                    #     writer1.writerow(['step'])  # 写入列标题
                                    #     writer1.writerows(map(lambda x: [x], lock))  # 将列表的每个元素写入CSV行
                                    plot_dif(dif1, fire, lock, f'my_sdif_{arttir}.png')

                                    success += 1
                                break

                        valScores.append(totalReward)

                    if mean(valScores) > highScore or success/validationEpisodes > successRate or arttir%10 == 0:
                        if mean(valScores) > highScore: # 总奖励分数
                            highScore = mean(valScores)
                            agent.saveCheckpoints("Agent{}_score{}".format(arttir, highScore), model_dir)
                            draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                            draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir) 
                            plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')

                        elif success / validationEpisodes > successRate: # 追逐成功率
                            successRate = success / validationEpisodes
                            agent.saveCheckpoints("Agent{}_successRate{}".format(arttir, successRate), model_dir)
                            draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                            draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir)
                            plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')
                
                    arttir += 1

                    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes)
                    writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
                    writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
        
        elif agent_name == 'TD3':
            print('agent is TD3')
            if not Test:
                # RANDOM EXPLORATION
                print("Exploration Started")
                for episode in range(explorationEpisodes):
                    state = env.reset()
                    done = False
                    for step in range(maxStep):
                        if not done:
                            action = env.action_space.sample()                

                            n_state,reward,done, info, stepsuccess = env.step(action)
                            # print(n_state)
                            if step is maxStep-1:
                                done = True
                            agent.store(state,action,n_state,reward,done,stepsuccess)
                            state=n_state

                            if done:
                                break
                    sys.stdout.write("\rExploration Completed: %.2f%%" % ((episode+1)/explorationEpisodes*100))
                sys.stdout.write("\n")

                print("Training Started")
                scores = []
                trainsuccess = []
                for episode in range(trainingEpisodes):
                    state = env.reset()
                    totalReward = 0
                    episode_step = 0
                    done = False
                    fire = False # 表示是否成功
                    for step in range(maxStep):
                        if not done:
                            action = agent.chooseAction(state)
                            n_state,reward,done, info , stepsuccess= env.step(action)

                            if step is maxStep - 1:
                                episode_step = step
                                break

                            agent.store(state, action, n_state, reward, done, stepsuccess) # n_state 为下一个状态
                            if stepsuccess:
                                print('success')
                            state = n_state
                            totalReward += reward

                            if agent.buffer.fullEnough(agent.batchSize):
                                critic_loss, actor_loss = agent.learn()
                                writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                                writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                                
                        elif done:
                            if 500 < env.Plane_Irtifa < 10000: # 改
                            # if env.Ally_target_locked == True:
                                fire = True
                                episode_step = step
                            break
                        
                    scores.append(totalReward)
                    if fire:
                        trainsuccess.append(1)
                    else:
                        trainsuccess.append(0)
                    writer.add_scalar('Training/Episode Reward', totalReward, episode)
                    writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
                    writer.add_scalar('Training/Average Step Reward', totalReward/episode_step, episode)
                    writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
                    
                    now = time.time()
                    seconds = int((now - start) % 60)
                    minutes = int(((now - start) // 60) % 60)
                    hours = int((now - start) // 3600)
                    print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, \
                        ' FinalReward: %.2f' % totalReward, \
                        ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
                        'RunTime: ', hours, ':',minutes,':', seconds)
                        
                    #VALIDATION
                    if (((episode + 1) % checkpointRate) == 0):
                        success = 0
                        valScores = []
                        dif = []
                        self_pos = []
                        oppo_pos = []
                        for e in range(validationEpisodes):

                            dif1=[]
                            fire=[]
                            lock=[]
                            state = env.reset()
                            totalReward = 0
                            done = False
                            for step in range(validatStep):
                                if not done:
                                    action = agent.chooseActionNoNoise(state)
                                    n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward   = env.step_test(action)
                                    state = n_state
                                    totalReward += reward

                                    dif1.append(env.loc_diff)
                                    if iffire:
                                        fire.append(step)
                                    if locked:
                                        lock.append(step)
                                    
                                    if e == validationEpisodes - 1:
                                        dif.append(env.loc_diff)
                                        self_pos.append(env.get_pos())
                                        oppo_pos.append(env.get_oppo_pos())

                                    if step is validatStep - 1:
                                        break

                                elif done:
                                    if 500 < env.Plane_Irtifa < 10000: # 改
                                    # if env.Ally_target_locked == True:
                                        
                                        # with open('./dif/sdif{}.csv'.format(fire[0]), 'w', newline='') as file:
                                        #     writer1 = csv.writer(file)
                                        #     writer1.writerow(['dif'])  # 写入列标题
                                        #     writer1.writerows(map(lambda x: [x], dif1))  # 将列表的每个元素写入CSV行
                                        # with open('./dif/sfire{}.csv'.format(fire[0]), 'w', newline='') as file:
                                        #     writer1 = csv.writer(file)
                                        #     writer1.writerow(['step'])  # 写入列标题
                                        #     writer1.writerows(map(lambda x: [x], fire))  # 将列表的每个元素写入CSV行
                                        # with open('./dif/slock{}.csv'.format(fire[0]), 'w', newline='') as file:
                                        #     writer1 = csv.writer(file)
                                        #     writer1.writerow(['step'])  # 写入列标题
                                        #     writer1.writerows(map(lambda x: [x], lock))  # 将列表的每个元素写入CSV行
                                        plot_dif(dif1, fire, lock, f'my_sdif_{arttir}.png')

                                        success += 1
                                    break
                            
                            # with open('./dif/tdif{}{}.csv'.format(fire[0],dif1[-1]), 'w', newline='') as file:
                            #     writer1 = csv.writer(file)
                            #     writer1.writerow(['dif'])  # 写入列标题
                            #     writer1.writerows(map(lambda x: [x], dif1))  # 将列表的每个元素写入CSV行
                            # with open('./dif/tfire{}{}.csv'.format(fire[0],dif1[-1]), 'w', newline='') as file:
                            #     writer1 = csv.writer(file)
                            #     writer1.writerow(['step'])  # 写入列标题
                            #     writer1.writerows(map(lambda x: [x], fire))  # 将列表的每个元素写入CSV行
                            # with open('./dif/tlock{}{}.csv'.format(fire[0],dif1[-1]), 'w', newline='') as file:
                            #     writer1 = csv.writer(file)
                            #     writer1.writerow(['step'])  # 写入列标题
                            #     writer1.writerows(map(lambda x: [x], lock))  # 将列表的每个元素写入CSV行
                            
                            valScores.append(totalReward)

                        if mean(valScores) > highScore or success/validationEpisodes > successRate or arttir%10 == 0:
                            if mean(valScores) > highScore: # 总奖励分数
                                highScore = mean(valScores)
                                agent.saveCheckpoints("Agent{}_score{}".format(arttir, highScore))
                                draw_dif(f'dif_{arttir}.png', dif, plot_dir)
                                draw_pos(f'pos_{arttir}.png', self_pos, oppo_pos, plot_dir) 
                                plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')

                            elif success / validationEpisodes > successRate: # 追逐成功率
                                successRate = success / validationEpisodes
                                agent.saveCheckpoints("Agent{}_successRate{}".format(arttir, successRate))
                                draw_dif(f'dif_{arttir}.png', dif, plot_dir)
                                draw_pos(f'pos_{arttir}.png', self_pos, oppo_pos, plot_dir)
                                plot_dif(dif1, fire, lock, f'my_dif_{arttir}.png')
                        
                        arttir += 1

                        print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes)
                        writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
                        writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
    else:
        success = 0
        validationEpisodes = 1000
        dif=[]
        fire=[]
        lock=[]
        for e in range(validationEpisodes):
            state = env.reset()
            totalReward = 0
            done = False
            print('before state: ', state)
            for step in range(validatStep):
                if not done:
                    action = agent.chooseActionNoNoise(state)
                    n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward   = env.step_test(action)
                    dif.append(env.loc_diff)
                    if iffire:
                        fire.append(step)
                    if locked:
                        lock.append(locked)
                    if action[3]>0:
                        print(step)
                        print('reward:', reward)
                        print('action: ', action)
                        print('next state: ', n_state)
                        print('before missile: ' , beforeaction, '  if fire: ', iffire, '   after missile: ', afteraction, '    locked', locked)
                        print("+"*15)
                        if locked == True:
                            done = True
                        else:
                            break
                    if step is validatStep - 1:
                        print(totalReward)
                        break

                    state = n_state
                    totalReward += reward
                elif done:
                    if 500 < env.Plane_Irtifa < 10000: # 改
                        success += 1
                        print(success)
                    break
                with open('dif.csv', 'w', newline='') as file:
                    writer1 = csv.writer(file)
                    writer1.writerow(['dif'])  # 写入列标题
                    writer1.writerows(map(lambda x: [x], dif))  # 将列表的每个元素写入CSV行
                with open('fire.csv', 'w', newline='') as file:
                    writer1 = csv.writer(file)
                    writer1.writerow(['step'])  # 写入列标题
                    writer1.writerows(map(lambda x: [x], fire))  # 将列表的每个元素写入CSV行
                with open('lock.csv', 'w', newline='') as file:
                    writer1 = csv.writer(file)
                    writer1.writerow(['setp'])  # 写入列标题
                    writer1.writerows(map(lambda x: [x], lock))  # 将列表的每个元素写入CSV行

            # print('Test  Reward:', totalReward)
        print('Success Ratio:', success / validationEpisodes)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ROT') # 代理：ROT、TD3
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--type', type=str, default='linear') # ROT type：linear、fixed、soft
    parser.add_argument('--upsample', action='store_true')
    parser.add_argument('--bc_weight', type=float, default=1)
    parser.add_argument('--model_name', type=str, default=None)
    main(parser.parse_args())

# 线性rot
# python train_all.py --agent ROT --port 12345 --type linear --upsample --bc_weight 1 --model_name lrot_1 
# 
# 固定rot
# python train_all.py --agent ROT --port 12345 --type fixed --upsample --bc_weight 0.5 --model_name frot_1 
# 
# 软rot
# python train_all.py --agent ROT --port 12345 --type soft --upsample --model_name srot_1 
#
# td3
# python train_all.py --agent TD3 --port 12345 --model_name td3_1
# 
# BC
# python train_all.py --agent BC --port 12345 --upsample  --model_name bc_1 
