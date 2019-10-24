import os
from pdf2image import convert_from_path
import cv2
import shutil

def name(num):
    if num < 10:
        return "00" + str(num)
    elif num < 100:
        return "0" + str(num)
    else:
        return str(num)

def convertPDF(filename, page_num):
    pages = convert_from_path(filename)
    foldername = filename.split('/')[-1].split('.')[0]
    if not os.path.exists('converted/'):
        os.makedirs('converted/')

    for i, page in enumerate(pages):
        if i%page_num == 0:
            if not os.path.exists('converted/'+name(int(i/page_num))+'/'):
                os.makedirs('converted/'+name(int(i/page_num))+'/')
            else:
                shutil.rmtree('converted/'+name(int(i/page_num))+'/')
                os.makedirs('converted/' + name(int(i / page_num)) + '/')
        page.save('converted/'+name(int(i/page_num))+'/_'+str(i%page_num)+'.jpg', 'JPEG')

def halfImage(image,index):
    """This function is used to divide each image into 2 half. (left and right)"""
    img = cv2.imread(image)
    _,width,_ = img.shape
    first = img[:, 0:int(width/2)-1]
    second = img[:,int(width/2):width-1]

    # cv2.imshow("first",first)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Unmute the following 2 lines if you are on mac
    split = image.split('/')
    upperFolder = split[0]+'/'+split[1]+'/'+split[2]

    # Unmute the following line if you are on windows
    # upperFolder = image.split('\\')

    cv2.imwrite(upperFolder+'/'+name(int(index*2))+'.jpg',first)
    cv2.imwrite(upperFolder+'/'+name(int(index*2+1))+'.jpg',second)