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
    de_lie = np.zeros((len(word_list),2)) #创建一个二维矩阵，行数为单词的种数，列数为2
    #列的第一位存放出现了该单词的文章总数，也就是one-hot矩阵中列的数值总数，第二位为idf值

    one_hot_matrix = np.zeros((row_num-1, len(word_list)+1))#创建one-hot矩阵
    rocount = 0
    for row2 in train_set_reader2:
        if rocount != 0:
            de_dst = str(row2[0])
            de_la_list = de_dst.split()
            ep = emotion_list.index(row2[1])#取出一个情绪，在情绪列表中定位他的位置为ep
            one_hot_matrix[rocount-1][len(word_list)] = ep
            for j in range(0, len(de_la_list)):
                p = word_list.index(de_la_list[j])#取出一个单词，在单词列表中定位他的位置为p
                if one_hot_matrix[rocount-1][p] == 0:   #如果one-hot矩阵还没有他
                   de_lie[p][0] += 1
                one_hot_matrix[rocount-1][p]=1          #one-hot矩阵对应位置为1
        rocount += 1


    for i in range(0, len(word_list)):
        de_lie[i][1] = math.log((row_num-1)/(1+de_lie[i][0]),2)  #计算idf值
    TF_matrix = np.zeros((row_num-1, len(word_list)))        #创建TF矩阵
    TF_IDF_matrix = np.zeros((row_num-1, len(word_list)+1))    #创建TF-IDF矩阵，多出一列用来计有多少单词
    hcount = 0
    for row in train_set_reader3:
        if hcount!=0:
            de_position = []
            de_dst = str(row[0])  #将列表转换为字符串
            de_la_list = de_dst.split() #通过空格分割该字符串
            count = 0
            for j in range(0, len(de_la_list)):
                p = word_list.index(de_la_list[j])
                TF_matrix[hcount-1][p] += 1
                if p not in de_position:
                    de_position.append(p)
                count += 1
                if j == len(de_la_list)-1:
                    for k in de_position:
                        TF_matrix[hcount-1][k] /= count
                        TF_IDF_matrix[hcount - 1][k] = TF_matrix[hcount-1][k]*de_lie[k][1]
                        TF_IDF_matrix[hcount - 1][len(word_list)]+=1
        hcount += 1

    np.savetxt(pathsix, TF_IDF_matrix, fmt="%f", delimiter=" ")
    np.savetxt(pathnine, one_hot_matrix, fmt="%d", delimiter=" ")  # 将one-hot矩阵写入文件
    return row_num, word_list, one_hot_matrix, emotion_list,TF_IDF_matrix,de_lie

def knn_cla():
    socount = 0
    k_test_k = []
    k_test_right = []
    # de_lie = np.zeros((len(word_list),2)) #创建一个二维矩阵，行数为单词的种数，列数为2
    # #列的第一位存放出现了该单词的文章总数，也就是one-hot矩阵中列的数值总数，第二位为idf值
    de_one_hot_matrix = np.zeros((row_num-1, len(word_list)))  # 创建one-hot矩阵
    de_one_hot_matrix_emotion = []
    de_answer = []
    for row3 in validation_set_reader:
        if socount!=0:
            de_dst = str(row3[0])
            de_answer.append(str(row3[1]))
            de_la_list = de_dst.split()
            for j in range(0, len(de_la_list)):
                if de_la_list[j] in word_list:
                    vp = word_list.index(de_la_list[j])     #取出一个单词，在单词列表中定位他的位置为p
                    # if de_one_hot_matrix[socount - 1][vp] == 0:  # 如果one-hot矩阵还没有他
                    #     de_lie[vp][0] += 1
                    de_one_hot_matrix[socount-1][vp]=1      #one-hot矩阵对应位置为1
        socount += 1
    # for i in range(0, len(word_list)):
    #     de_lie[i][1] = math.log((row_num-1)/(1+de_lie[i][0]),2)
    de_TF_matrix = np.zeros((socount, len(word_list)))  # 创建TF矩阵
    de_TF_IDF_matrix = np.zeros((socount, len(word_list)+1))    #创建TF-IDF矩阵，多出一列用来计有多少单词
    scount = 0
    for row4 in validation_set_reader2:
        if scount!=0:
            de_position = []
            de_dst = str(row4[0])  #将列表转换为字符串
            de_la_list = de_dst.split() #通过空格分割该字符串
            count = 0
            for j in range(0, len(de_la_list)):
                if de_la_list[j] in word_list:
                    p = word_list.index(de_la_list[j])
                    de_TF_matrix[scount-1][p] += 1
                    if p not in de_position:
                        de_position.append(p)
                    count += 1
                if j == len(de_la_list)-1:
                    for k in de_position:
                        de_TF_matrix[scount-1][k] /= count
                        de_TF_IDF_matrix[scount - 1][k] = de_TF_matrix[scount - 1][k] * de_lie[k][1]
                        de_TF_IDF_matrix[scount - 1][len(word_list)] += 1
        scount += 1
    np.savetxt(pathone, de_one_hot_matrix, fmt="%d",delimiter=" ")#将one-hot矩阵写入文件
    np.savetxt(pathseven, de_TF_matrix, fmt="%f", delimiter=" ")
    knn_compare = []
    knn_emotion = []
    knn_temp = []
    knn_sum = []
    d_k = 15#int(row_num ** 0.5)
    k = d_k
    k_test_k.append(k)
    print("k: ", k)
    r_answer = 0
    for i in range(0, socount - 1):
        print("i: ",i)
        if juli == 1:
            for df in range(0, row_num - 1):
                dsum = 0
                same_count = 0
                for j in range(0, len(word_list)):
                    dsum += abs(de_TF_IDF_matrix[i][j]-TF_IDF_matrix[df][j])
                    #print(de_TF_IDF_matrix[i][j],"  ",TF_IDF_matrix[df][j])
                    if de_TF_IDF_matrix[i][j]!=0 and TF_IDF_matrix[df][j] != 0:
                        same_count += 1
                if same_count!= 0:
                    dsum = dsum+(de_TF_IDF_matrix[i][len(word_list)]+TF_IDF_matrix[i][len(word_list)])/(same_count*2)
                if same_count == 0:
                    dsum = dsum + 99
                #print(dsum)
                knn_sum.append(dsum)
                knn_emotion.append(emotion_list[int(one_hot_matrix[df][len(word_list)])])
        if juli == 2:
            for df in range(0, row_num - 1):
                dsum = 0
                for j in range(0, len(word_list)-1):
                    dsum += (de_TF_IDF_matrix[i][j]-TF_IDF_matrix[df][j])*(de_TF_IDF_matrix[i][j]-TF_IDF_matrix[df][j])
                dsum = dsum ** 0.5
                knn_sum.append(dsum)
                knn_emotion.append(emotion_list[int(one_hot_matrix[df][len(word_list)])])
        if juli == 3:
            for df in range(0, row_num - 1):

                dsum = 0
                same_count = 0
                for j in range(0, len(word_list)):
                    dsum += abs(de_TF_IDF_matrix[i][j]-TF_IDF_matrix[df][j])
                    if de_TF_IDF_matrix[i][j]!=0 and TF_IDF_matrix[df][j] != 0:
                        same_count += 1
                if same_count!= 0:
                    dsum = dsum+(de_TF_IDF_matrix[i][len(word_list)]+TF_IDF_matrix[i][len(word_list)])/(same_count*2)
                if same_count == 0:
                    dsum = dsum + 99

                dot_product = 0.0
                normA = 0.0
                normB = 0.0
                for a, b in zip(de_TF_IDF_matrix[i], TF_IDF_matrix[df]):
                    dot_product += a * b
                    normA += a ** 2
                    normB += b ** 2
                if (normA * normB) == 0:
                    cossum = 0.0001
                if (normA * normB) != 0:
                    cossum = 100*(dot_product / ((normA * normB) ** 0.5))/dsum
                #print(cossum)
                #print(dot_product / ((normA * normB) ** 0.5))
                knn_sum.append(cossum)
                knn_emotion.append(emotion_list[int(one_hot_matrix[df][len(word_list)])])
        Inf = 10000
        if juli == 1 or juli == 2:
            for bi in range(k):
                knn_temp.append(knn_sum.index(min(knn_sum)))
                knn_compare.append(knn_emotion[knn_sum.index(min(knn_sum))])
                knn_sum[knn_sum.index(min(knn_sum))] = Inf
        Inf = -10000
        if juli == 3:
            for bi in range(k):
                knn_temp.append(knn_sum.index(max(knn_sum)))
                knn_compare.append(knn_emotion[knn_sum.index(max(knn_sum))])
                knn_sum[knn_sum.index(max(knn_sum))] = Inf
        if de_answer[i] == Counter(knn_compare).most_common(1)[0][0]:
            r_answer += 1
        de_one_hot_matrix_emotion.append(Counter(knn_compare).most_common(1)[0][0])
        knn_compare.clear()
        knn_emotion.clear()
        knn_temp.clear()
        knn_sum.clear()
    k_test_right.append(r_answer)
    print(r_answer/(socount-1))
    f = open(pathfour, "w")
    for of in range(0, len(de_one_hot_matrix_emotion)):
        f.write(de_one_hot_matrix_emotion[of]+'\n')
    f.close()
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
    row_num, word_list, one_hot_matrix, emotion_list, TF_IDF_matrix,de_lie = tf_idf()
    juli = 3
    knn_cla()
    end_time = time.time()
    print(end_time - start_time)
