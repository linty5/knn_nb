import re
from numba import jit
import numpy as np
from collections import Counter
import time
import csv
import math

def one_hot():       #将经过处理的列表转换为各式矩阵
    word_list = []    #创建一个空列表，用来存储所有的单词
    emotion_list = [] #创建一个空列表，用来存储所有的情绪
    emotion_p_list = []
    emotion_bo_p_list = []
    word_sum = 0
    row_num = 0
    for row in train_set_reader:
        if row_num!=0:
            de_dst = str(row[0])  #将列表转换为字符串
            de_la_list = de_dst.split() #通过空格分割该字符串
            if row[1] not in emotion_list:        #将没出现过的情绪添入列表
                emotion_list.append(row[1])
            for j in range(0, len(de_la_list)): #将没出现过的单词添入列表
                if de_la_list[j] not in word_list :
                    word_list.append(de_la_list[j])
                else :
                    continue
        row_num += 1
    one_hot_matrix = np.zeros((row_num, len(word_list)+1))#创建one-hot矩阵
    emotion_p_matrix = np.zeros((len(emotion_list), len(word_list)+1))
    emotion_bo_p_matrix = np.zeros((len(emotion_list), len(word_list) + 1))
    # 一列为一个单词的情绪概率
    for i in range(0,len(emotion_list)):
        emotion_p_list.append(0)
        emotion_bo_p_list.append(0)
    rocount = 0
    for row2 in train_set_reader2:
        if rocount != 0:
            de_dst = str(row2[0])
            de_la_list = de_dst.split()
            ep = emotion_list.index(row2[1])#取出一个情绪，在情绪列表中定位他的位置为ep
            one_hot_matrix[rocount-1][len(word_list)] = ep
            emotion_bo_p_list[ep] += 1
            emotion_bo_p_matrix[ep][len(word_list)] += 1
            emotion_bo_p_matrix[int(one_hot_matrix[rocount - 1][len(word_list)])][ep] += 1
            for j in range(0, len(de_la_list)):
                p = word_list.index(de_la_list[j])#取出一个单词，在单词列表中定位他的位置为p
                word_sum += 1
                emotion_p_list[ep] += 1
                one_hot_matrix[rocount-1][p]=1          #one-hot矩阵对应位置为1
                emotion_p_matrix[ep][len(word_list)] += 1
                emotion_p_matrix[int(one_hot_matrix[rocount-1][len(word_list)])][p] += 1
        rocount += 1
    for i in range(0,len(emotion_list)):
        emotion_p_list[i] = emotion_p_list[i]/word_sum
        emotion_bo_p_list[i] = emotion_bo_p_list[i]/(rocount-1)
        #print(emotion_p_list[i])

    for j in range(0,len(word_list)):
        for i in range(0, len(emotion_list)):
            #print("分子： ",emotion_p_matrix[i][j])
            #print("分母1： ",emotion_p_matrix[i][len(word_list)])
            #print("分母2： ",len(word_list))
            emotion_p_matrix[i][j] = (emotion_p_matrix[i][j]+0.4)/(emotion_p_matrix[i][len(word_list)]+len(word_list))
            emotion_bo_p_matrix[i][j] = (emotion_bo_p_matrix[i][j] + 1) / (emotion_bo_p_matrix[i][len(word_list)] + 2)
            #print(emotion_p_matrix[i][j])

    np.savetxt(pathnine, one_hot_matrix, fmt="%d", delimiter=" ")  # 将one-hot矩阵写入文件
    return row_num, word_list, one_hot_matrix, emotion_list,emotion_p_matrix,emotion_p_list,emotion_bo_p_matrix,emotion_bo_p_list

def nb_cla_duoxiangshi():
    socount = 0
    r_answer = 0
    de_emotion_p_matrix = np.ones((len(emotion_list)+2,row_num-1))
    # 每列为一个测试句子的情绪概率
    de_answer = []
    for row3 in validation_set_reader:
        if socount!=0:
            de_dst = str(row3[0])
            de_answer.append(str(row3[1]))
            de_la_list = de_dst.split()
            for j in range(0, len(de_la_list)):
                if de_la_list[j] in word_list:
                    vp = word_list.index(de_la_list[j])     #取出一个单词，在单词列表中定位他的位置为p
                    for k in range(0,len(emotion_list)):
                        de_emotion_p_matrix[k][socount-1]*=emotion_p_matrix[k][vp]
                    de_emotion_p_matrix[len(emotion_list)][socount - 1]=0
            for k in range(0,len(emotion_list)):
                de_emotion_p_matrix[k][socount-1] *= emotion_bo_p_list[k]
                if de_emotion_p_matrix[k][socount-1]>de_emotion_p_matrix[len(emotion_list)][socount-1]:
                    de_emotion_p_matrix[len(emotion_list)+1][socount - 1] = k
                    de_emotion_p_matrix[len(emotion_list)][socount - 1] =de_emotion_p_matrix[k][socount-1]
            print("预测： ",emotion_list[int(de_emotion_p_matrix[len(emotion_list)+1][socount - 1])],"答案： ",de_answer[socount-1])
            if emotion_list[int(de_emotion_p_matrix[len(emotion_list)+1][socount - 1])] == de_answer[socount-1]:
                r_answer += 1
        socount += 1
    print("共：",socount-1,"正确率:",r_answer/(socount-1))

def nb_cla_bonuli():
    socount = 0
    r_answer = 0
    de_emotion_bo_p_matrix = np.ones((len(emotion_list)+2,row_num-1))
    # 每列为一个测试句子的情绪概率
    de_answer = []
    for row3 in validation_set_reader:
        sign_list = []
        if socount!=0:
            de_dst = str(row3[0])
            de_answer.append(str(row3[1]))
            de_la_list = de_dst.split()
            for j in range(0, len(de_la_list)):
                if de_la_list[j] in word_list:
                    vp = word_list.index(de_la_list[j])     #取出一个单词，在单词列表中定位他的位置为p
                    sign_list.append(vp)
                    for k in range(0,len(emotion_list)):
                        de_emotion_bo_p_matrix[k][socount-1]*=emotion_bo_p_matrix[k][vp]
                        de_emotion_bo_p_matrix[len(emotion_list)][socount - 1]=0
            for j in range(0, len(word_list)):
                if j not in sign_list:
                    for k in range(0,len(emotion_list)):
                        de_emotion_bo_p_matrix[k][socount-1]*=(1-emotion_bo_p_matrix[k][j])
                        #print(de_emotion_bo_p_matrix[k][socount-1])
                else :
                    print(j)
            for k in range(0,len(emotion_list)):
                de_emotion_bo_p_matrix[k][socount-1] *= emotion_bo_p_list[k]
                #print(de_emotion_bo_p_matrix[k][socount-1])
                if de_emotion_bo_p_matrix[k][socount-1]>de_emotion_bo_p_matrix[len(emotion_list)][socount-1]:
                    de_emotion_bo_p_matrix[len(emotion_list)+1][socount - 1] = k
                    de_emotion_bo_p_matrix[len(emotion_list)][socount - 1] =de_emotion_bo_p_matrix[k][socount-1]
            print("预测： ",emotion_list[int(de_emotion_bo_p_matrix[len(emotion_list)+1][socount - 1])],"答案： ",de_answer[socount-1])
            if emotion_list[int(de_emotion_bo_p_matrix[len(emotion_list)+1][socount - 1])] == de_answer[socount-1]:
                r_answer += 1
                #print("HELLO")
        socount += 1
    print("共：",socount-1,"正确率:",r_answer/(socount-1))

if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/train_set.csv'
    pathone = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/one_hot.txt'
    pathtwo = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/validation_set.csv'
    paththree = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/test.csv'
    pathfour = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/knnemotion.txt'
    pathfive = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/testone.csv'
    pathsix = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/tfidf.txt'
    pathseven = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/tf.txt'
    patheight = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/trainone.csv'
    pathnine = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/classification_dataset/tr_one_hot.txt'
    train_set_reader = csv.reader(open(train_set,encoding='utf-8'))
    train_set_reader2 = csv.reader(open(train_set, encoding='utf-8'))
    train_set_reader3 = csv.reader(open(train_set, encoding='utf-8'))
    validation_set_reader = csv.reader(open(pathtwo, encoding='utf-8'))
    validation_set_reader2 = csv.reader(open(pathtwo, encoding='utf-8'))
    row_num, word_list, one_hot_matrix, emotion_list,emotion_p_matrix,emotion_p_list,emotion_bo_p_matrix,emotion_bo_p_list = one_hot()
    nb_cla_duoxiangshi()
    end_time = time.time()
    print(end_time - start_time)
