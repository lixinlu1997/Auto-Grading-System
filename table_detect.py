import numpy as np
import argparse
import cv2
import os
from PIL import Image
from mypdf2img import convertPDF
from mypdf2img import halfImage
import csv
import torchvision.transforms as transforms
from train import predict
import shutil

def print_dimension(img):
    print("image shape: " + "h=" +
          str(img.shape[0]) + ", w=" +
          str(img.shape[1]) + ", d=" +
          str(img.shape[2]))


def cv2_show(img):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(img, ratio):
    """ height is the reference
        ratio have to be float """
    dimension = (int(img.shape[1]/ratio), int(img.shape[0]/ratio))   # (w,h)
    # print("resizing at: " + str(dimension))
    # print(" with ratio: " + str(ratio))
    new_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    return new_img


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, off):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    # print('test')
    # print(rect)
    offset = off
    (tl, tr, br, bl) = rect
    tl[0] -= offset
    tl[1] -= offset
    tr[0] += offset
    tr[1] -= offset
    br[0] += offset
    br[1] += offset
    bl[0] -= offset
    bl[1] += offset
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    # print(rect)
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # print(dst)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped, rect, dst



def transfer_img(image, debug_model):
    orig = image.copy()
    # ratio = float(image.shape[0]) / 500
    ratio = 1
    image = resize(image, ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 50, 20, 40)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if (gray[i][j] > 220):
                gray[i][j] = 255
            else:
                gray[i][j] = 0


    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 0, 50)
    if debug_model:
        cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
        cv2.imshow("edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    im, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    contoursFinded = []
    index = 1
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        screen_cnt = approx
        if len(approx) == 4:
            warped, rect_points, dst_points = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio, -8)
            if debug_model:
                cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
                cv2.imshow("edged", warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(dst_points[2][1])
            if dst_points[2][1] >= 40 and dst_points[2][1] <= 60:
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                contoursFinded.append((warped, rect_points[0][1]*5+rect_points[0][0]))
        index += 1
    contoursFinded = sorted(contoursFinded, key=lambda contoursFinded: contoursFinded[1])
    return contoursFinded

def find_block(image, debug_model):
    orig = image.copy()
    ratio = 1
    image = resize(image, ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray,None,50,20,40)
    if debug_model:
        cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
        cv2.imshow("edged", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if (gray[i][j] > 220):
                gray[i][j] = 255
            else:
                gray[i][j] = 0

    # print(type(gray))

    # print(gray == gray1)

    # exit(0)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 0, 50)
    if debug_model:
        cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
        cv2.imshow("edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    im, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    blocksFinded = []
    index = 1
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        screen_cnt = approx
        if len(approx) == 4:
            warped, rect_points, dst_points = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio, 10)
            if dst_points[2][1] >= 80 and dst_points[2][1] <= 120:
                if debug_model:
                    cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
                    cv2.imshow("edged", warped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(dst_points[2][1])
                blocksFinded.append((screen_cnt, rect_points[0][1]*5+rect_points[0][0]))
        index += 1
    blocksFinded = sorted(blocksFinded, key=lambda contoursFinded: contoursFinded[1])
    return blocksFinded

def process(content):
    pre = content.split(':')[0]
    if len(content.split(':')) == 1:
        result = content.split(' ')[1].replace(' ', '').replace('\n', '')
        return 0, result
    elif pre.find('nswe') != -1:
        index = int(pre.split(' ')[0].replace(' ', ''))
        result = content.split(':')[1].replace(' ', '').replace('\n', '')[-1]
        return index, result


def cut_image(image, num, offset=3):
    width, height = image.size
    item_width = int(width/num)
    image = image.resize((item_width*num, height))
    box_list = []
    count=0
    for j in range(0, 1):
        for i in range(0, num):
            count+=1
            box=(i*item_width, j*item_width, (i+1)*item_width, (j+1)*item_width)
            box_list.append(box)
    # print(count)
    image_list=[image.crop(box) for box in box_list]
    new_image_list = []
    for im in image_list:
        w, h = im.size
        box = (offset, offset, w-offset, h-offset)
        new_im = im.crop(box)
        new_image_list.append(new_im)
    return new_image_list


def preprocess(img):
    temp = np.array(img)
    x, y = temp.shape
    #print(x,y)
    for i in range(x):
        for j in range(y):
            if (temp[i][j] > 50):
                temp[i][j] = 255
            else:
                temp[i][j] = 0
    return Image.fromarray(temp)


def recognize_number(digit):
    temp = digit.resize((28, 28))
    # print(np.array(temp))
    temp = Image.fromarray(255 - np.array(temp))
    # temp = preprocess(temp)
    temp.show()
    trans = transforms.Compose([transforms.ToTensor()])
    temp = trans(temp)
    temp = temp.view(1, 1, 28, 28)
    return predict(temp)

def empty_converted():
    shutil.rmtree('./converted')
    os.mkdir('./converted')


def main(args):
    folder = args["path"]
    path_origin = "./" + folder + "/"
    debug_model = args["debug"]
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
        if (len(item)>1):
            for i,c in enumerate(item):
                if i!=0 and i!= 1 and c != '':
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
        if count==1:
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
        for i,c in enumerate(item):
            if i!= 0 and c != '':
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

    if args["debug"]:
        print(ID_Digit)
        print(choice1)
        print(choice2)
        print(choice3)
        print(choice1_score)
        print(choice2_score)
        print(choice3_score)
        print(blank1)
        print(blank2)
        print(blank3)
    total_table = ID_Digit+len(choice1)+len(choice2)+len(choice3)+2*len(blank1)-2+2*len(blank2)-2+2*len(blank3)-2
    if debug_model:
        print(total_table)
    actual_table = 0
    """Read the location of blocks from template"""
    empty_converted()
    convertPDF('template.pdf', pages)
    converted_path = "./converted/"
    for folder in sorted(os.listdir(converted_path)):
        if folder[0] == '.':
            continue
        for i, img in enumerate(sorted(os.listdir(os.path.join(converted_path, folder)))):
            halfImage(os.path.join(os.path.join(converted_path, folder), img), i)
            os.remove(os.path.join(os.path.join(converted_path, folder), img))
    for folder in sorted(os.listdir(converted_path)):
        if folder[0] == '.':
            continue
        print("Converting the template exam...")
        allBlocks = []
        for im in sorted(os.listdir(os.path.join(converted_path, folder))):
            image = cv2.imread(os.path.join(os.path.join(converted_path, folder), im))
            # print_dimension(image)
            tables = find_block(image, debug_model)
            actual_table += len(tables)
            # print(tables)
            allBlocks.append(tables)
    if debug_model:
        print(actual_table)
    print("actual table: ", actual_table)
    print("theoretical table: ", total_table)
    if actual_table != total_table:
        print("Template detect wrong numbers of table!")
        exit(0)
    """Convert PDF files into image files"""
    first_row = ['Student ID']
    for i in range(1,len(choice1)+len(choice2)+len(choice3)+len(blank1)-1+len(blank2)-1+len(blank3)-1+1):
        first_row += [str(i)]
        first_row += [str(i)+"_Score"]
    first_row += ['Final Score']

    for f in os.listdir(path_origin):
        empty_converted()
        if f[0] == '.':
            continue
        filename = f.split('.')[0]
        PDFpath = os.path.join(path_origin, f)
        convertPDF(PDFpath,pages)
        try:
            print(filename + " loaded!")
        except:
            print("No PDF file in the directory!")
            exit(0)

        csvfile = open('./stats/' + filename + '.csv', 'w', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(first_row)

        converted_path = "./converted/"
        for folder in sorted(os.listdir(converted_path)):
            if folder[0] == '.':
                continue
            for i, img in enumerate(sorted(os.listdir(os.path.join(converted_path, folder)))):
                # print(i,img)
                halfImage(os.path.join(os.path.join(converted_path, folder), img), i)
                os.remove(os.path.join(os.path.join(converted_path, folder), img))
        for folder in sorted(os.listdir(converted_path)):
            if folder[0] == '.':
                continue
            print("Converting the " + folder + "th exam...")
            output = [folder]
            allTables = []
            # print(sorted(os.listdir(os.path.join(converted_path, folder))))
            for i, im in enumerate(sorted(os.listdir(os.path.join(converted_path, folder)))):
                image = cv2.imread(os.path.join(os.path.join(converted_path, folder), im))
                # cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
                # cv2.imshow("edged", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                screen_cnts = allBlocks[i]
                for screen_cnt in screen_cnts:
                    warped, _, _ = four_point_transform(image, screen_cnt[0].reshape(4, 2), 20)
                    warped_ = transfer_img(warped, debug_model=debug_model)
                    if len(warped_)==0:
                        # cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
                        # cv2.imshow("edged", warped)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        warped_ = np.array([[-1,-1],[-1,-1]])
                    else:
                        warped_ = warped_[0][0]
                    allTables += [warped_]
            output = []
            choice_corr = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            student_ID = ''
            if debug_model:
                print(len(allTables))
            for table in allTables:
                if table[0][0] == -1:
                    continue
                else:
                    table = cv2.fastNlMeansDenoising(table, None, 40, 7, 21)
                    for i in range(table.shape[0]):
                        for j in range(table.shape[1]):
                            if (table[i][j] < 80):
                                table[i][j] = 0
                            elif (table[i][j] > 200):
                                table[i][j] = 255
                            else:
                                table[i][j] -= 80
                    if debug_model:
                        cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
                        cv2.imshow("edged", table)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            i = 0
            for table in allTables:
                # print(table.size)
                if_empty = False
                if table[0][0] == -1:
                    if_empty = True
                if i < ID_Digit:
                    if if_empty:
                        i += 1
                        if i == ID_Digit:
                            output.append(student_ID)
                        continue
                    # image = Image.fromarray(table)
                    # divide_method(table,3,8)
                    # image_list = cut_image(image, 8)
                    # content = ''
                    temp = Image.fromarray(table)
                    temp = temp.resize((28, 28))
                    # print(np.array(temp))
                    temp = Image.fromarray((255 - np.array(temp)) * 1.0 / 255.0)
                    # temp = preprocess(temp)
                    # print(np.array(temp))
                    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                    temp = trans(temp)
                    temp = temp.view(1, 1, 28, 28)
                    student_ID += str(predict(temp))
                    i += 1
                    if i == ID_Digit:
                        output.append(student_ID)
                elif i < ID_Digit + len(choice1):
                    if if_empty:
                        output.append("Not Detected!")
                        output.append("Not Detected!")
                        i += 1
                        continue
                    index = i - ID_Digit
                    result = ''
                    image = Image.fromarray(table)
                    image_list = cut_image(image, 4)
                    for choice_index, choice in enumerate(image_list):
                        temp = choice.resize((20, 20))
                        if np.array(temp).sum() < 75000:
                            result += choice_corr[choice_index]
                    output.append(result)
                    score = 0
                    if len(choice1_score) == 1:
                        if result == choice1[index]:
                            score = choice1_score[0]
                    else:
                        all_in = True
                        ans_len = len(choice1[index])
                        for t in range(len(result)):
                            if result[t] not in choice1[index]:
                                all_in = False
                        if all_in and len(result) == ans_len:
                            score = choice1_score[0]
                        elif all_in and len(result) != ans_len:
                            score = choice1_score[1]
                        else:
                            score = 0
                    output.append(str(score))
                    i += 1
                elif i < ID_Digit + len(choice1) + 2 * (len(blank1) - 1):
                    index = i - ID_Digit - len(choice1)
                    remainder = (index) % 2
                    quotient = (index) // 2
                    if if_empty:
                        if remainder == 0:
                            output.append("Not Detected!")
                            output.append("Not Detected!")
                        else:
                            if blank1[0] == 'A':
                                if len(output[-1]) > 0 and output[-1][0] == 'N':
                                    continue
                                else:
                                    output[-1] = "Not Detected!"
                                    output.append("Not Detected!")
                            else:
                                output[-1] = "Not Detected!"
                                output[-2] = "Not Detected!"
                        i += 1
                        continue
                    if (blank1[0] == 'A'):
                        if remainder == 0:
                            output.append('')
                        if remainder == 1:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            temp = Image.fromarray(table)
                            temp = temp.resize((20, 20))
                            if np.array(temp).sum() < 100000:
                                output.append(str(blank1[quotient + 1]))
                            else:
                                output.append('0')
                    else:
                        temp = Image.fromarray(table)
                        temp = temp.resize((20, 20))
                        if np.array(temp).sum() > 100000:
                            d = ''
                        else:
                            temp = Image.fromarray(table)
                            temp = temp.resize((28, 28))
                            temp = Image.fromarray((255 - np.array(temp)) * 1.0 / 255.0)
                            trans = transforms.Compose(
                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                            temp = trans(temp)
                            temp = temp.view(1, 1, 28, 28)
                            d = str(predict(temp))
                        if remainder == 0:
                            output.append('')
                            #d = str(predict(temp))
                            output.append(d)
                        else:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            x = output[-1]
                            x = x + d
                            if (int(x) > int(blank1[quotient + 1])):
                                output[-1] = blank1[quotient + 1]
                            else:
                                output[-1] = x
                    i += 1
                elif i < ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2):
                    if if_empty:
                        output.append("Not Detected!")
                        output.append("Not Detected!")
                        i += 1
                        continue
                    index = i - (ID_Digit + len(choice1) + 2 * (len(blank1) - 1))
                    result = ''
                    image = Image.fromarray(table)
                    image_list = cut_image(image, 4)
                    for choice_index, choice in enumerate(image_list):
                        temp = choice.resize((20, 20))
                        # temp.show()
                        # print(choice_index,choice_corr[choice_index], np.array(temp).sum())
                        if np.array(temp).sum() < 80000:
                            result += choice_corr[choice_index]
                    output.append(result)
                    score = 0
                    if len(choice2_score) == 1:
                        if result == choice2[index]:
                            score = choice2_score[0]
                    else:
                        all_in = True
                        ans_len = len(choice2[index])
                        for t in range(len(result)):
                            if result[t] not in choice2[index]:
                                all_in = False
                        if all_in and len(result) == ans_len:
                            score = choice2_score[0]
                        elif all_in and len(result) != ans_len:
                            score = choice2_score[1]
                        else:
                            score = 0
                    output.append(str(score))
                    i += 1
                elif i < ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2) + 2 * (len(blank2) - 1):
                    index = i - (ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2))

                    remainder = (index) % 2
                    quotient = (index) // 2
                    if if_empty:
                        if remainder == 0:
                            output.append("Not Detected!")
                            output.append("Not Detected!")
                        else:
                            if blank2[0] == 'A':
                                if len(output[-1]) > 0 and output[-1][0] == 'N':
                                    continue
                                else:
                                    output[-1] = "Not Detected!"
                                    output.append("Not Detected!")
                            else:
                                output[-1] = "Not Detected!"
                                output[-2] = "Not Detected!"
                        i += 1
                        continue
                    if (blank2[0] == 'A'):
                        if remainder == 0:
                            output.append('')
                        if remainder == 1:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            temp = Image.fromarray(table)
                            temp = temp.resize((20, 20))
                            if np.array(temp).sum() < 100000:
                                output.append(str(blank2[quotient + 1]))
                            else:
                                output.append('0')
                    else:
                        temp = Image.fromarray(table)
                        temp = temp.resize((20, 20))
                        if np.array(temp).sum() > 100000:
                            d = ''
                        else:
                            temp = Image.fromarray(table)
                            temp = temp.resize((28, 28))
                            temp = Image.fromarray((255 - np.array(temp)) * 1.0 / 255.0)
                            trans = transforms.Compose(
                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                            temp = trans(temp)
                            temp = temp.view(1, 1, 28, 28)
                            d = str(predict(temp))
                        if remainder == 0:
                            output.append('')
                            #d = str(predict(temp))
                            output.append(d)
                        else:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            x = output[-1]
                            x = x + d
                            if (int(x) > int(blank2[quotient + 1])):
                                output[-1] = blank2[quotient + 1]
                            else:
                                output[-1] = x
                    i += 1
                elif i < ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2) + 2 * (len(blank2) - 1) + len(
                        choice3):
                    # print(i,len(choice3))
                    if if_empty:
                        output.append("Not Detected!")
                        output.append("Not Detected!")
                        i += 1
                        continue
                    index = i - (ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2) + 2 * (len(blank2) - 1))
                    result = ''
                    image = Image.fromarray(table)
                    image_list = cut_image(image, 4)
                    for choice_index, choice in enumerate(image_list):
                        temp = choice.resize((20, 20))
                        # temp.show()
                        # print(choice_index,choice_corr[choice_index], np.array(temp).sum())
                        if np.array(temp).sum() < 80000:
                            result += choice_corr[choice_index]
                    output.append(result)
                    score = 0
                    if len(choice3_score) == 1:
                        if result == choice3[index]:
                            score = choice3_score[0]
                    else:
                        all_in = True
                        ans_len = len(choice3[index])
                        for t in range(len(result)):
                            if result[t] not in choice3[index]:
                                all_in = False
                        if all_in and len(result) == ans_len:
                            score = choice3_score[0]
                        elif all_in and len(result) != ans_len:
                            score = choice3_score[1]
                        else:
                            score = 0
                    output.append(str(score))
                    i += 1
                else:
                    index = i - (ID_Digit + len(choice1) + 2 * (len(blank1) - 1) + len(choice2) + 2 * (
                                len(blank2) - 1) + len(choice3))
                    remainder = (index) % 2
                    quotient = (index) // 2
                    if if_empty:
                        if remainder == 0:
                            output.append("Not Detected!")
                            output.append("Not Detected!")
                        else:
                            if blank3[0] == 'A':
                                if len(output[-1]) > 0 and output[-1][0] == 'N':
                                    continue
                                else:
                                    output[-1] = "Not Detected!"
                                    output.append("Not Detected!")
                            else:
                                output[-1] = "Not Detected!"
                                output[-2] = "Not Detected!"
                        i += 1
                        continue
                    if (blank3[0] == 'A'):
                        if remainder == 0:
                            output.append('')
                        if remainder == 1:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            temp = Image.fromarray(table)
                            temp = temp.resize((20, 20))
                            if np.array(temp).sum() < 100000:
                                output.append(str(blank3[quotient + 1]))
                            else:
                                output.append('0')
                    else:
                        temp = Image.fromarray(table)
                        temp = temp.resize((20, 20))
                        # print(np.array(temp).sum())
                        if np.array(temp).sum() > 101500:
                            d = ''
                        else:
                            temp = Image.fromarray(table)
                            temp = temp.resize((28, 28))
                            temp = Image.fromarray((255 - np.array(temp)) * 1.0 / 255.0)
                            trans = transforms.Compose(
                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                            temp = trans(temp)
                            temp = temp.view(1, 1, 28, 28)
                            d = str(predict(temp))
                        if remainder == 0:
                            output.append('')
                            # d = str(predict(temp))
                            output.append(d)
                        else:
                            if len(output[-1])>0 and output[-1][0] == 'N':
                                i += 1
                                continue
                            x = output[-1]
                            x = x + d
                            if (int(x) > int(blank3[quotient + 1])):
                                output[-1] = blank3[quotient + 1]
                            else:
                                output[-1] = x
                    i += 1
            # print(answer)
            score = 0
            for idx, val in enumerate(output):
                if (idx > 0 and idx % 2 == 0):
                    if val[0] != 'N':
                        score += int(val)
            output.append(str(score))
            if debug_model:
                print(output)
            writer.writerow(output)
            print("Completed !")

    """Create a csv file"""



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to the folder of images to be scanned")
    ap.add_argument("--debug", action='store_true')
    args = vars(ap.parse_args())
    main(args)
