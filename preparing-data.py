
# coding: utf-8

# In[1]:


import xml.etree.ElementTree
import csv
import os
from random import shuffle


# In[2]:


# get_ipython().magic(u'run utils/utilities.py')
# get_ipython().magic(u'run HAR-CNN-GPU.py')
import sys
import CNN_GPU
sys.path.append('utils')
import utilities

# In[3]:


class_label_vn = [u"Cổ tay",u"Cổ chân",u"Bả vai",u"Xoay người",u"Xoay đầu gối",u"Đi bộ",u"Chạy",u"Đá bóng",u"Đạp xe",u"Đánh răng",u"Rửa tay", u"Lau bàn", u"Quét nhà", u"Nạo",u"Thái",u"Trộn",u"Lên cầu thang",u"Xuống cầu thang"]
# class_label=["wrist","ankle","shoulder","haunch","knee","walking","running","kicking","cycling","brushing","washing_hand","wipe","sweep_floor","peel","slice","mixing","upstair","downstair"]


# In[4]:


DATASET_ROOT = './datasets/PTIT'
DATASET_NORM = DATASET_ROOT + '/normalized'
DATASET_TRAIN = DATASET_NORM + '/train'
DATASET_TEST = DATASET_NORM + '/test'
WINDONW_OVERLAP = 0.5
WINDOWN_OVERLAP_SIZE = WINDONW_OVERLAP * WINDOWN_SIZE
ANNO_FILE = 'anno.eaf'
WAX3_FILE = 'wax3.csv'
GEARS2_FILE = 'gears2.csv'


# In[5]:


train_window_cnt = {}
test_window_cnt = {}
total_window = 0


# In[6]:


def getAnno(anno_path):
    timestamp = {}
    annotation = []
    time_range = []
    root_ele = xml.etree.ElementTree.parse(anno_path).getroot()
    for time_slot in root_ele.iter('TIME_SLOT'):
        att = time_slot.attrib
        timestamp[att['TIME_SLOT_ID']]  = att['TIME_VALUE']
    for anno in root_ele.find('TIER').iter('ANNOTATION'):
        alig_anno = anno.find('ALIGNABLE_ANNOTATION')
        anno_text = alig_anno.find('ANNOTATION_VALUE').text.strip()
        if anno_text not in class_label:
            continue
        startTs = timestamp[alig_anno.attrib['TIME_SLOT_REF1']]
        endTs = timestamp[alig_anno.attrib['TIME_SLOT_REF2']]
#         annotation[anno_text] = {'start': startTs, 'end': endTs}
        annotation.append(anno_text)
        time_range.append({'start': int(startTs), 'end': int(endTs)})
    return annotation, time_range

def getMilisecond(s):
    try:
        hours, minutes, seconds = (["0", "0"] + s.split(":"))[-3:]
        hours = int(hours)
        minutes = int(minutes)
        seconds = float(seconds)
        miliseconds = int(3600000 * hours + 60000 * minutes + 1000 * seconds)
        return miliseconds
    except:
#         print "format exception " + s
        return 0

def getTotalWindowSize(len):
    total_size = (len // WINDOWN_SIZE) * 2 - 1 + (len % WINDOWN_SIZE) // WINDOWN_OVERLAP_SIZE
    return int(total_size)


# In[7]:


def exportData(dir_path, target_path, is_training=True):
    global total_window
    anno_file = dir_path + '/' + ANNO_FILE
    sensor_data_path = target_path + '/sensor'
    x_watch_acc_file = sensor_data_path + '/x_watch_acc.txt'
    y_watch_acc_file = sensor_data_path + '/y_watch_acc.txt'
    z_watch_acc_file = sensor_data_path + '/z_watch_acc.txt'
    x_watch_gyr_file = sensor_data_path + '/x_watch_gyr.txt'
    y_watch_gyr_file = sensor_data_path + '/y_watch_gyr.txt'
    z_watch_gyr_file = sensor_data_path + '/z_watch_gyr.txt'
    x_sensor_acc_file = sensor_data_path + '/x_sensor_acc.txt'
    y_sensor_acc_file = sensor_data_path + '/y_sensor_acc.txt'
    z_sensor_acc_file = sensor_data_path + '/z_sensor_acc.txt'
    class_file = target_path + '/class.txt'
    annotation, time_range = getAnno(anno_file)
    num_anno = len(annotation)
    gears2_data = [[] for anno in annotation];
    wax3_data = [[] for anno in annotation];
    start_annotation = 0
    wax3_file = dir_path + '/' + WAX3_FILE
    gears2_file = dir_path + '/' + GEARS2_FILE
    with open(gears2_file, 'r') as gears2_csv, open(wax3_file, 'r') as wax3_csv:
        gears2_csv_reader = csv.reader(gears2_csv, delimiter=',')
        wax3_csv_reader = csv.reader(wax3_csv, delimiter=',')
        for row in gears2_csv_reader:
            ts = getMilisecond(row[0].strip())
            for i in range(num_anno):
                if ts >= time_range[i]['start'] and ts < time_range[i]['end']:
                    x_acc = float(row[1].strip())
                    y_acc = float(row[2].strip())
                    z_acc = float(row[3].strip())
                    x_gyr = float(row[4].strip())
                    y_gyr = float(row[5].strip())
                    z_gyr = float(row[6].strip())
                    gears2_data[i].append([ts, x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr])
                    
        for row in wax3_csv_reader:
            ts = getMilisecond(row[0].strip())
            for i in range(num_anno):
                if ts >= time_range[i]['start'] and ts < time_range[i]['end']:
                    x_acc = float(row[1].strip())
                    y_acc = float(row[2].strip())
                    z_acc = float(row[3].strip())
                    wax3_data[i].append([ts, x_acc, y_acc, z_acc])
         
    for i in range(num_anno):
        num_windows = getTotalWindowSize(len(gears2_data[i]))
        startWindow = 0
#         print num_windows
        for j in range(num_windows):
            windowSliced = gears2_data[i][startWindow:startWindow+WINDOWN_SIZE]
            if(len(windowSliced) < WINDOWN_SIZE):
                break
            startWindow += WINDOWN_SIZE
            startTs = windowSliced[0][0]
            endTs = windowSliced[WINDOWN_SIZE - 1][0]
            equivWax3Data = []
            # get data from wax3 and group with gears2
            for k in range(len(wax3_data[i])):
                wax3Ts = wax3_data[i][k][0]
                if(wax3Ts > endTs):
                    break
                if wax3Ts >= startTs and wax3Ts <= endTs:
                    equivWax3Data.append(wax3_data[i][k])
            zero_arr = [0 for zit in range(WINDOWN_SIZE)]
            while len(equivWax3Data) < WINDOWN_SIZE:
                equivWax3Data.append(zero_arr);
            equivWax3Data = equivWax3Data[0:WINDOWN_SIZE]
#             print (startTs, endTs, len(windowSliced), len(equivWax3Data))
            #export windows to file
            x_watch_acc = y_watch_acc = z_watch_acc = "";
            x_watch_gyr = y_watch_gyr = z_watch_gyr = "";
            x_sensor_acc = y_sensor_acc = z_sensor_acc = "";
#             if len(windowSliced) > 150 or len(equivWax3Data) > 150:
#                 print(i, j, len(windowSliced), len(equivWax3Data))
            for k in range(WINDOWN_SIZE):
                x_watch_acc = x_watch_acc + " " + str(windowSliced[k][1])
                y_watch_acc = y_watch_acc + " " + str(windowSliced[k][2])
                z_watch_acc = z_watch_acc + " " + str(windowSliced[k][3])
                
                x_watch_gyr = x_watch_gyr + " " + str(windowSliced[k][4])
                y_watch_gyr = y_watch_gyr + " " + str(windowSliced[k][5])
                z_watch_gyr = z_watch_gyr + " " + str(windowSliced[k][6])
                
                x_sensor_acc = x_sensor_acc + " " + str(equivWax3Data[k][1])
                y_sensor_acc = y_sensor_acc + " " + str(equivWax3Data[k][2])
                z_sensor_acc = z_sensor_acc + " " + str(equivWax3Data[k][3])
                

            with open(x_watch_acc_file, "a") as fw:
                fw.write(x_watch_acc + "\n")
            with open(y_watch_acc_file, "a") as fw:
                fw.write(y_watch_acc + "\n")
            with open(z_watch_acc_file, "a") as fw:
                fw.write(z_watch_acc + "\n")
                
            with open(x_watch_gyr_file, "a") as fw:
                fw.write(x_watch_gyr + "\n")
            with open(y_watch_gyr_file, "a") as fw:
                fw.write(y_watch_gyr + "\n")
            with open(z_watch_gyr_file, "a") as fw:
                fw.write(z_watch_gyr + "\n")
                
            with open(x_sensor_acc_file, "a") as fw:
                fw.write(x_sensor_acc + "\n")
            with open(y_sensor_acc_file, "a") as fw:
                fw.write(y_sensor_acc + "\n")
            with open(z_sensor_acc_file, "a") as fw:
                fw.write(z_sensor_acc + "\n")
            with open(class_file, "a") as fw:
                fw.write(str(class_label_int[annotation[i]]) + "\n")
        train_window_cnt[class_label_vn[int(class_label_int[annotation[i]])]]+=num_windows
        total_window+=num_windows;
#     return wax3_data
        
# exportData('./datasets/PTIT/001/in/', DATASET_TRAIN)


# In[8]:


def prepareTrainTestFile(trainDir, testDir, name="default"):
    for dirPath in trainDir:
#         print dirPath
        exportData(DATASET_ROOT + '/' + dirPath + '/in', DATASET_TRAIN, True)
        exportData(DATASET_ROOT + '/' + dirPath + '/out', DATASET_TRAIN, True)
    for dirPath in testDir:
#         print dir
        exportData(DATASET_ROOT + '/' + dirPath + '/in', DATASET_TEST, False)
        exportData(DATASET_ROOT + '/' + dirPath + '/out', DATASET_TEST, False)


# In[9]:


def prepareTempTrainTestFormat(test_dir):
    allDir = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011","013", "014"]
    trainDir = []
    for cdir in allDir:
        if cdir not in test_dir:
            trainDir.append(cdir)
    global train_window_cnt
    global test_window_cnt
    global total_train_window
    total_train_window = 0
    train_window_cnt = dict.fromkeys(class_label_vn, 0)
    test_window_cnt = dict.fromkeys(class_label_vn, 0)
    prepareTrainTestFile(trainDir, test_dir)
    print ("training window count ", train_window_cnt)
    print ("test window count ", test_window_cnt)


# In[10]:


def randomTrainFile():
    records = None
    random_dir = DATASET_NORM + "_random/train"
    files = [DATASET_TRAIN + "/" + "class.txt"]
    files_random  = [random_dir + "/" + "class.txt"]
    for channel in CHANNEL_LIST:
        filePath = DATASET_TRAIN + "/sensor/" + channel + ".txt"
        files.append(filePath)
        files_random.append(random_dir + "/sensor/" + channel + ".txt")
    for file in files:
        print "read file " + file
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            if records == None:
#                 print "create records"
                records = [[] for _ in range(len(lines))]
            for idx, line in enumerate(lines):
                records[idx].append(line)
#                 print (idx, line, records[idx][0])
#             return
#     print "record len %d" % len(records)
    shuffle(records)
    total10 = total0 = 0;
    for file_idx, file in enumerate(files_random):
#         print "write file " + file
        with open(file, "a") as wf:
            for record in records:
#                 print record[file_idx]
                wf.write(record[file_idx] + "\n")
    print(total0, total10)


# In[11]:


# !rm -rf datasets/PTIT/normalized*
# #!mkdir -p datasets/PTIT/normalized/{train,test}/sensor
# !mkdir -p datasets/PTIT/normalized/train/sensor
# !mkdir -p datasets/PTIT/normalized/test/sensor
# !mkdir -p datasets/PTIT/normalized_random/train/sensor
# !mkdir -p datasets/PTIT/normalized_random/test/sensor
# prepareTempTrainTestFormat()
# randomTrainFile()


# In[14]:


for cdir in ["014"]:
    os.system(u'rm -rf datasets/PTIT/normalized*')
    #!mkdir -p datasets/PTIT/normalized/{train,test}/sensor
    os.system(u'mkdir -p datasets/PTIT/normalized/train/sensor')
    os.system(u'mkdir -p datasets/PTIT/normalized/test/sensor')
    os.system(u'mkdir -p datasets/PTIT/normalized_random/train/sensor')
    os.system(u'mkdir -p datasets/PTIT/normalized_random/test/sensor')
    prepareTempTrainTestFormat([cdir])
    randomTrainFile()
    print train()
