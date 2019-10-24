import csv
import numpy as np
import os

"""Read in the test parameters from solution.csv"""
solutionFile = open("solution.csv", "rt", encoding='ascii')
reader = csv.reader(solutionFile)
ID_Digit = 0
pages = 0
choice1 = []
choice1_score = []
choice2 = []
choice2_score = []
choice3 = []
choice3_score = []
blank1 = []
blank2 = []
blank3 = []


def get_score_item(item):
    result = []
    if (len(item) > 1):
        for i, c in enumerate(item):
            if i != 0 and i != 1 and c != '':
                result.append(c)
        return result
    return result


def get_score_value(item):
    result = []
    count = 0
    for i in item:
        if (i != ''):
            count += 1
        else:
            break
    if count == 1:
        return result
    elif count == 2:
        result = [int(item[1])]
    elif count == 3:
        result = [int(item[1]), int(item[2])]
    else:
        print("Input error, too many columes for choice score")
        exit(0)
    return result


def get_blank_score(item):
    result = []
    for i, c in enumerate(item):
        if i != 0 and c != '':
            result.append(c)
    return result


for item in reader:
    if reader.line_num == 1:
        ID_Digit = int(item[1])
    elif reader.line_num == 2:
        pages = int(item[1])
    elif reader.line_num == 4:
        choice1 = get_score_item(item)
    elif reader.line_num == 5:
        choice1_score = get_score_value(item)
    elif reader.line_num == 6:
        blank1 = get_blank_score(item)
    elif reader.line_num == 7:
        choice2 = get_score_item(item)
    elif reader.line_num == 8:
        choice2_score = get_score_value(item)
    elif reader.line_num == 9:
        blank2 = get_blank_score(item)
    elif reader.line_num == 10:
        choice3 = get_score_item(item)
    elif reader.line_num == 11:
        choice3_score = get_score_value(item)
    elif reader.line_num == 12:
        blank3 = get_blank_score(item)

grade_score = []
grade_score_pair = []
class_average = []
class_stage = []
for f in os.listdir('./stats'):
    if f.split('.')[-1] == 'csv':
        filename = f.split('.')[0]
        new_csv = open('./stats/class/'+f, 'w', newline='',encoding='utf-8-sig')
        writer = csv.writer(new_csv)

        csv_file = csv.reader(open('./stats/'+f, 'r'))
        rows = [row for row in csv_file]

        writer.writerow([filename + '班统计表'])
        final_score = [s[-1] for s in rows]
        final_score = [int(s) for s in final_score[1:]]

        average = round(np.mean(np.array(final_score)),2)
        class_average.append((int(filename),average,len(final_score)))

        grade_score += final_score
        for s in final_score:
            grade_score_pair.append((s,int(filename)))
        # Calculate the full score of this exam
        full = 0
        for s in blank1[1:]:
            full += int(s)
        for s in blank2[1:]:
            full += int(s)
        for s in blank3[1:]:
            full += int(s)
        if len(choice1)>0:
            full += choice1_score[0]*len(choice1)
        if len(choice2)>0:
            full += choice2_score[0]*len(choice2)
        if len(choice3)>0:
            full += choice3_score[0]*len(choice3)

        writer.writerow(['满分值',str(full),'班级总人数',str(len(final_score)),'平均分',round(np.mean(np.array(final_score)),2),'最高分',max(final_score),'最低分',min(final_score),'中位分',np.median(np.array(final_score))])

        for i in range(1,len(choice1)+len(choice2)+len(choice3)+len(blank1)-1+len(blank2)-1+len(blank3)-1+1):
            output = [str(i)]
            c1_index = i*2-1
            c2_index = i*2
            #print(c2_index)
            if i <= len(choice1):
                if i == 0 + 1:
                    writer.writerow([])
                    label = ['题号', '题型', '满分', '平均分', '得分率%', 'A项人数', 'B项人数', 'C项人数', 'D项人数', '空白人数', '正确选项']
                    writer.writerow(label)
                index = i
                output.append('选择题')
                output.append(choice1_score[0])
                column_answer = [row[c1_index] for row in rows]
                column_score = [row[c2_index] for row in rows]
                tmp = []
                for k in range(1,len(column_score)):
                    if len(column_score[k]) > 0 and column_score[k][0] != 'N':
                        tmp.append(int(column_score[k]))
                output.append(round(np.mean(np.array(tmp)), 2))
                output.append(round(100 * np.mean(np.array(tmp)) / choice1_score[0], 2))
                num_A = 0
                num_B = 0
                num_C = 0
                num_D = 0
                num_blank = 0
                for ans in column_answer[1:]:
                    if len(ans) == 0:
                        num_blank += 1
                    if len(ans) > 0 and ans[0] != 'N':
                        if 'A' in ans:
                            num_A += 1
                        if 'B' in ans:
                            num_B += 1
                        if 'C' in ans:
                            num_C += 1
                        if 'D' in ans:
                            num_D += 1
                output += [num_A, num_B, num_C, num_D, num_blank, choice1[index - 1]]
            elif i <= len(choice1)+len(blank1)-1:
                if i == len(choice1)+1:
                    labeled = False
                index = i - len(choice1)
                output.append('主观题')
                output.append(int(blank1[index]))
                column_score = []
                for k in range(1,len(rows)):
                    if len(rows[k][c2_index])>0 and rows[k][c2_index][0] != 'N':
                        column_score.append(int(rows[k][c2_index]))
                column_score = np.array(column_score)
                # print(column_score)
                average = np.mean(column_score)
                percent = average / int(blank1[index])
                output.append(round(average,2))
                output.append(round(percent,2))
                # print(average,percent)
                if blank1[0] == 'A':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank1[index]))
                    zero_num = np.sum(column_score == 0)
                    # print(full_num,zero_num)
                    output.append(full_num)
                    output.append(zero_num)
                elif blank1[0] == 'B':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','满分-1人数','满分-2人数','满分-3人数','满分-4人数','满分-5人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank1[index]))
                    full_num_1 = np.sum(column_score == int(blank1[index]) - 1)
                    full_num_2 = np.sum(column_score == int(blank1[index]) - 2)
                    full_num_3 = np.sum(column_score == int(blank1[index]) - 3)
                    full_num_4 = np.sum(column_score == int(blank1[index]) - 4)
                    full_num_5 = np.sum(column_score == int(blank1[index]) - 5)
                    zero_num = np.sum(column_score == 0)
                    output += [full_num,full_num_1,full_num_2,full_num_3,full_num_4,full_num_5,zero_num]
            elif i <= len(choice1)+len(blank1)-1+len(choice2):
                if i == len(choice1)+len(blank1)-1+1:
                    writer.writerow([])
                    label = ['题号','题型','满分','平均分','得分率%','A项人数','B项人数','C项人数','D项人数','空白人数','正确选项']
                    writer.writerow(label)
                index = i - (len(choice1)+len(blank1)-1)
                output.append('选择题')
                output.append(choice2_score[0])
                column_answer = [row[c1_index] for row in rows]
                column_score = [row[c2_index] for row in rows]
                tmp = []
                for k in range(1, len(column_score)):
                    if len(column_score[k]) > 0 and column_score[k][0] != 'N':
                        tmp.append(int(column_score[k]))
                output.append(round(np.mean(np.array(tmp)), 2))
                output.append(round(100 * np.mean(np.array(tmp)) / choice2_score[0], 2))
                num_A = 0
                num_B = 0
                num_C = 0
                num_D = 0
                num_blank = 0
                for ans in column_answer[1:]:
                    if len(ans) == 0:
                        num_blank += 1
                    if len(ans) > 0 and ans[0] != 'N':
                        if 'A' in ans:
                            num_A += 1
                        if 'B' in ans:
                            num_B += 1
                        if 'C' in ans:
                            num_C += 1
                        if 'D' in ans:
                            num_D += 1
                output += [num_A,num_B,num_C,num_D,num_blank,choice2[index-1]]
            elif i <= len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1:
                if i == len(choice1)+len(blank1)-1+len(choice2)+1:
                    labeled = False
                index = i - (len(choice1)+len(blank1)-1+len(choice2))
                output.append('主观题')
                output.append(int(blank2[index]))
                column_score = []
                for k in range(1, len(rows)):
                    if len(rows[k][c2_index]) > 0 and rows[k][c2_index][0] != 'N':
                        column_score.append(int(rows[k][c2_index]))
                column_score = np.array(column_score)
                # print(column_score)
                average = np.mean(column_score)
                percent = average / int(blank2[index])
                output.append(round(average,2))
                output.append(round(percent,2))
                # print(average,percent)
                if blank2[0] == 'A':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank2[index]))
                    zero_num = np.sum(column_score == 0)
                    # print(full_num,zero_num)
                    output.append(full_num)
                    output.append(zero_num)
                elif blank2[0] == 'B':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','满分-1人数','满分-2人数','满分-3人数','满分-4人数','满分-5人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank2[index]))
                    full_num_1 = np.sum(column_score == int(blank2[index]) - 1)
                    full_num_2 = np.sum(column_score == int(blank2[index]) - 2)
                    full_num_3 = np.sum(column_score == int(blank2[index]) - 3)
                    full_num_4 = np.sum(column_score == int(blank2[index]) - 4)
                    full_num_5 = np.sum(column_score == int(blank2[index]) - 5)
                    zero_num = np.sum(column_score == 0)
                    output += [full_num,full_num_1,full_num_2,full_num_3,full_num_4,full_num_5,zero_num]
            elif i <= len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1+len(choice3):
                if i == len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1+1:
                    writer.writerow([])
                    label = ['题号','题型','满分','平均分','得分率%','A项人数','B项人数','C项人数','D项人数','空白人数','正确选项']
                    writer.writerow(label)
                index = i - (len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1)
                output.append('选择题')
                output.append(choice3_score[0])
                column_answer = [row[c1_index] for row in rows]
                column_score = [row[c2_index] for row in rows]
                tmp = []
                for k in range(1, len(column_score)):
                    if len(column_score[k]) > 0 and column_score[k][0] != 'N':
                        tmp.append(int(column_score[k]))
                output.append(round(np.mean(np.array(tmp)), 2))
                output.append(round(100 * np.mean(np.array(tmp)) / choice3_score[0], 2))
                num_A = 0
                num_B = 0
                num_C = 0
                num_D = 0
                num_blank = 0
                for ans in column_answer[1:]:
                    if len(ans) == 0:
                        num_blank += 1
                    if len(ans) > 0 and ans[0] != 'N':
                        if 'A' in ans:
                            num_A += 1
                        if 'B' in ans:
                            num_B += 1
                        if 'C' in ans:
                            num_C += 1
                        if 'D' in ans:
                            num_D += 1
                output += [num_A,num_B,num_C,num_D,num_blank,choice3[index-1]]
            elif i <= len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1+len(choice3)+len(blank3)-1:
                if i == len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1+len(choice3)+1:
                    labeled = False
                index = i - (len(choice1)+len(blank1)-1+len(choice2)+len(blank2)-1+len(choice3))
                output.append('主观题')
                output.append(int(blank3[index]))
                column_score = []
                for k in range(1, len(rows)):
                    if len(rows[k][c2_index]) > 0 and rows[k][c2_index][0] != 'N':
                        column_score.append(int(rows[k][c2_index]))
                column_score = np.array(column_score)
                # print(column_score)
                average = np.mean(column_score)
                percent = average / int(blank3[index])
                output.append(round(average,2))
                output.append(round(percent,2))
                # print(average,percent)
                if blank3[0] == 'A':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank3[index]))
                    zero_num = np.sum(column_score == 0)
                    # print(full_num,zero_num)
                    output.append(full_num)
                    output.append(zero_num)
                elif blank3[0] == 'B':
                    if not labeled:
                        writer.writerow([])
                        label = ['题号','题型','满分','平均分','得分率%','满分人数','满分-1人数','满分-2人数','满分-3人数','满分-4人数','满分-5人数','0分人数']
                        writer.writerow(label)
                        labeled = True
                    full_num = np.sum(column_score == int(blank3[index]))
                    full_num_1 = np.sum(column_score == int(blank3[index]) - 1)
                    full_num_2 = np.sum(column_score == int(blank3[index]) - 2)
                    full_num_3 = np.sum(column_score == int(blank3[index]) - 3)
                    full_num_4 = np.sum(column_score == int(blank3[index]) - 4)
                    full_num_5 = np.sum(column_score == int(blank3[index]) - 5)
                    zero_num = np.sum(column_score == 0)
                    output += [full_num,full_num_1,full_num_2,full_num_3,full_num_4,full_num_5,zero_num]
            writer.writerow(output)
        writer.writerow([])
        output = ['班级分数段统计', '最高分', '满分(' + str(full) + ')', str(full - 1) + '-' + str(full - 10), str(full - 11) + '-' + str(full - 20), str(full - 21) + '-' + str(full - 30)]
        writer.writerow(output)
        output = ['']
        output.append(max(final_score))
        highest = 0
        high1_10 = 0
        high11_20 = 0
        high21_30 = 0
        for s in final_score:
            if s == full:
                highest += 1
            elif s <= full - 1 and s >= full - 10:
                high1_10 += 1
            elif s <= full - 11 and s >= full - 20:
                high11_20 += 1
            elif s <= full - 21 and s >= full - 30:
                high21_30 += 1
        output += [highest, high1_10, high11_20, high21_30]
        class_stage.append((int(filename),output))
        writer.writerow(output)
new_csv = open('./stats/grade/grade_stats.csv', 'w', newline='')
writer = csv.writer(new_csv)
writer.writerow(['年级统计表'])
temp = ['满分值',full,'年级总人数',len(grade_score),'平均分',np.mean(np.array(grade_score)),'最高分',max(grade_score),'最低分',min(grade_score),'中位数',np.median(np.array(grade_score))]
writer.writerow(temp)
writer.writerow([''])

temp = ['班级','班级人数','总平均分']
writer.writerow(temp)
def takeFirstReverse(elem):
    return -elem[0]

def takeFirst(elem):
    return elem[0]

class_average.sort(key=takeFirst)
class_stage.sort(key=takeFirst)
grade_score_pair.sort(key=takeFirst)
for c,avg,l in class_average:
    temp = [c,l,avg]
    writer.writerow(temp)
writer.writerow([''])
writer.writerow(['班级分数段统计', '最高分', '满分(' + str(full) + ')', str(full - 1) + '-' + str(full - 10), str(full - 11) + '-' + str(full - 20), str(full - 21) + '-' + str(full - 30)])
for c,t in class_stage:
    writer.writerow(t)
writer.writerow([''])
writer.writerow(['年级排序统计','年级前10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','年级前60','年级前100'])

num_all = len(grade_score)
for i in range(1,len(class_average)+1):
    temp = [i]
    num_1_10 = 0
    for j in range(0,10):
        if j<num_all:
            if grade_score_pair[j][1] == i:
                num_1_10 += 1
    num_11_20 = 0
    for j in range(10,20):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_11_20 += 1
    num_21_30 = 0
    for j in range(20,30):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_21_30 += 1
    num_31_40 = 0
    for j in range(30,40):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_31_40 += 1
    num_41_50 = 0
    for j in range(40,50):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_41_50 += 1
    num_51_60 = 0
    for j in range(50,60):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_51_60 += 1
    num_61_70 = 0
    for j in range(60,70):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_61_70 += 1
    num_71_80 = 0
    for j in range(70,80):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_71_80 += 1
    num_81_90 = 0
    for j in range(80,90):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_81_90 += 1
    num_91_100 = 0
    for j in range(90, 100):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_91_100 += 1
    temp = [i,num_1_10,num_11_20,num_21_30,num_31_40,num_41_50,num_51_60,num_61_70,num_71_80,num_81_90,num_91_100]
    temp.append(num_1_10+num_11_20+num_21_30+num_31_40+num_41_50+num_51_60)
    temp.append(num_1_10+num_11_20+num_21_30+num_31_40+num_41_50+num_51_60+num_61_70+num_71_80+num_81_90+num_91_100)
    writer.writerow(temp)

writer.writerow(['年级排序统计2','年级后10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','年级后60','年级后100'])
grade_score_pair.sort(key=takeFirstReverse)
num_all = len(grade_score)
for i in range(1,len(class_average)+1):
    temp = [i]
    num_1_10 = 0
    for j in range(0,10):
        if j<num_all:
            if grade_score_pair[j][1] == i:
                num_1_10 += 1
    num_11_20 = 0
    for j in range(10,20):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_11_20 += 1
    num_21_30 = 0
    for j in range(20,30):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_21_30 += 1
    num_31_40 = 0
    for j in range(30,40):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_31_40 += 1
    num_41_50 = 0
    for j in range(40,50):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_41_50 += 1
    num_51_60 = 0
    for j in range(50,60):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_51_60 += 1
    num_61_70 = 0
    for j in range(60,70):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_61_70 += 1
    num_71_80 = 0
    for j in range(70,80):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_71_80 += 1
    num_81_90 = 0
    for j in range(80,90):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_81_90 += 1
    num_91_100 = 0
    for j in range(90, 100):
        if j < num_all:
            if grade_score_pair[j][1] == i:
                num_91_100 += 1
    temp = [i,num_1_10,num_11_20,num_21_30,num_31_40,num_41_50,num_51_60,num_61_70,num_71_80,num_81_90,num_91_100]
    temp.append(num_1_10+num_11_20+num_21_30+num_31_40+num_41_50+num_51_60)
    temp.append(num_1_10+num_11_20+num_21_30+num_31_40+num_41_50+num_51_60+num_61_70+num_71_80+num_81_90+num_91_100)
    writer.writerow(temp)
# print(output)



