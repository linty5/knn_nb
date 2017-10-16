import re
from numba import jit
import numpy as np
from collections import Counter
import time
import csv
import math

def tf_idf():       #将经过处理的列表转换为各式矩阵
    word_list = []    #创建一个空列表，用来存储所有的单词
    emotion_list = [] #创建一个空列表，用来存储所有的情绪
    word_sum = 0
    row_num = 0
    for row in train_set_reader:
        if row_num == 0:
            for i in range(1,len(row)):
                emotion_list.append(row[i])
        if row_num!=0:
            de_dst = str(row[0])  #将列表转换为字符串
            de_la_list = de_dst.split() #通过空格分割该字符串
            for j in range(0, len(de_la_list)): #将没出现过的单词添入列表
                if de_la_list[j] not in word_list :
                    word_list.append(de_la_list[j])
                else :
                    continue
        row_num += 1

    one_hot_matrix = np.zeros((row_num-1, len(word_list)+len(emotion_list)))#创建one-hot矩阵

    rocount = 0
    for row2 in train_set_reader2:
        if rocount != 0:
            de_dst = str(row2[0])
            de_la_list = de_dst.split()
            for i in range(1,len(row)):
                one_hot_matrix[rocount-1][len(word_list)+i-1] = row2[i]
            for j in range(0, len(de_la_list)):
                p = word_list.index(de_la_list[j])#取出一个单词，在单词列表中定位他的位置为p
                word_sum += 1
                one_hot_matrix[rocount-1][p]=1          #one-hot矩阵对应位置为1
        rocount += 1

    TF_matrix = np.zeros((row_num-1, len(word_list)+len(emotion_list)+1))        #创建TF矩阵
    hcount = 0
    for row in train_set_reader3:
        if hcount!=0:
            de_position = []
            de_dst = str(row[0])  #将列表转换为字符串
            de_la_list = de_dst.split() #通过空格分割该字符串
            for i in range(1,len(row)):
                TF_matrix[hcount-1][len(word_list)+i-1] = row[i]
            for j in range(0, len(de_la_list)):
                p = word_list.index(de_la_list[j])
                TF_matrix[hcount-1][p] += 1
                if p not in de_position:
                    de_position.append(p)
                if j == len(de_la_list)-1:
                    for k in de_position:
                        TF_matrix[hcount-1][len(word_list)+len(emotion_list)]+=TF_matrix[hcount-1][k]
        hcount += 1

    np.savetxt(pathseven, TF_matrix, fmt="%f", delimiter=" ")  # 将one-hot矩阵写入文件
    np.savetxt(pathnine, one_hot_matrix, fmt="%d", delimiter=" ")  # 将one-hot矩阵写入文件
    return row_num, word_list, one_hot_matrix, emotion_list,TF_matrix,word_sum

def nb_reg():
    print("HELLO")
    socount = 0
    mycount = 0
    for row4 in validation_set_reader:
        mycount += 1
    my_answer = np.zeros((mycount-1,len(emotion_list)))
    for row3 in validation_set_reader2:
        if socount!=0:
            print(socount)
            de_dst = str(row3[0])
            de_la_list = de_dst.split()
            dd_sum = 0
            for em in range(0, len(emotion_list)):
                for k in range(0, row_num - 1):
                    de_sum = 1
                    for j in range(0, len(de_la_list)):
                        if de_la_list[j] in word_list:
                            vp = word_list.index(de_la_list[j])     #取出一个单词，在单词列表中定位他的位置为p
                            de_sum = de_sum * (TF_matrix[k][vp]+0.007)/(word_sum+TF_matrix[k][len(word_list)+len(emotion_list)])
                    de_sum = de_sum * TF_matrix[k][len(word_list)+em]
                    my_answer[socount-1][em] += de_sum
                #print("em",em)
                dd_sum += my_answer[socount - 1][em]
            for em in range(0, len(emotion_list)):
                my_answer[socount - 1][em] /= dd_sum
                #print(my_answer[socount - 1][em])
        socount += 1
    np.savetxt(pathtenl, my_answer, fmt="%f", delimiter=',')

if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/train_set.csv'
    pathone = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/one_hot.txt'
    pathtwo = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/validation_set.csv'
    paththree = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/test.csv'
    pathfour = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/knnemotion.txt'
    pathfive = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/testone.csv'
    pathsix = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/tfidf.txt'
    pathseven = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/tf.txt'
    patheight = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/trainone.csv'
    pathnine = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/tr_one_hot.txt'
    pathten = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/answer.txt'
    pathtenl = 'E:/B,B,B,BBox/大三上/人工智能/lab2/DATA/regression_dataset/myanswer.csv'
    train_set_reader = csv.reader(open(train_set,encoding='utf-8'))
    train_set_reader2 = csv.reader(open(train_set, encoding='utf-8'))
    train_set_reader3 = csv.reader(open(train_set, encoding='utf-8'))
    validation_set_reader = csv.reader(open(pathtwo, encoding='utf-8'))
    validation_set_reader2 = csv.reader(open(pathtwo, encoding='utf-8'))
    row_num, word_list, one_hot_matrix, emotion_list,TF_matrix,word_sum = tf_idf()
    nb_reg()
    end_time = time.time()
    print(end_time - start_time)
