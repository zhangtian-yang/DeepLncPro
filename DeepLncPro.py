#!/usr/bin/env python
# -*- coding:utf-8-*-
# author: ZhangTianyang time:2022/5/1 QQ:980557945 e-mail:Tianyang.Zhang819@outlook.com
# ----------------------------------------------------------------------------
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F


def predict(input, model, device):
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        pred = output.detach().cpu().numpy().reshape(output.shape[0])
        return pred


def load_dataset(data_path):
    base = 'ATGC'
    DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
                   0.76847598772376],
            'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811,
                   0.32686549630327577],
            'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
                   0.5476708282666789],
            'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
                   0.6891727614587756],
            'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
                   0.7467063799220581],
            'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332],
            'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172,
                   0.8400043540595654],
            'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
                   0.3501900760908136],
            'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
                   0.45903777008667923],
            'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868,
                   0.32686549630327577],
            'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
                   0.6083143907016332]}
    name, sequence = [], []
    with open(data_path, 'rt') as f:
        lines = f.readlines()
        if lines[0][0] != '>':
            print('Please check the input file format!')
            sys.exit()
        flagx = 0
        seq = ''
        for line in lines:
            if line[0] == '>':
                name.append(line[1:].strip())
                if flagx == 1:
                    sequence.append(seq)
                    seq = ''
                flagx = 1
            else:
                seq = seq + line.strip()
        sequence.append(seq)
        namex, locationx, sequencex = [], [], []
        for i in range(len(name)):
            for j in range(len(sequence[i]) - 180):
                namex.append(name[i])
                locationx.append(str(j + 1) + ' to ' + str(j + 181))
                sequencex.append(sequence[i][j:181 + j])
    data = []
    for n in sequencex:
        data.append(encoding(n, base, DPCP))
    datax = np.asarray([i for i in data], dtype=np.float32)
    datax = torch.from_numpy(datax)
    return namex, locationx, sequencex, datax


def encoding(sequence, base, DPCP):
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 13])
    for i in range(matrix_lenth):
        for j in range(4):
            if sequence[i] == base[j]:
                matrix[i, j] = np.float32(1)
            else:
                matrix[i, j] = np.float32(0)
        if sequence[i] == base[0]:
            matrix[i, 4] = np.float32(1)
            matrix[i, 5] = np.float32(1)
            matrix[i, 6] = np.float32(1)
        elif sequence[i] == base[1]:
            matrix[i, 4] = np.float32(0)
            matrix[i, 5] = np.float32(1)
            matrix[i, 6] = np.float32(0)
        elif sequence[i] == base[2]:
            matrix[i, 4] = np.float32(1)
            matrix[i, 5] = np.float32(0)
            matrix[i, 6] = np.float32(0)
        elif sequence[i] == base[3]:
            matrix[i, 4] = np.float32(0)
            matrix[i, 5] = np.float32(0)
            matrix[i, 6] = np.float32(1)
    for i in range(matrix_lenth - 1):
        couple = sequence[i] + sequence[i + 1]
        properties = DPCP[couple]
        for m in range(6):
            matrix[i, 7 + m] += np.float32(properties[m] / 2)
            matrix[i + 1, 7 + m] += np.float32(properties[m] / 2)
    return np.transpose(matrix)


class DeepncPro(nn.Module):
    def __init__(self, configs):
        super(DeepncPro, self).__init__()
        self.sequence_lenth = configs.sequence_lenth
        self.input_channel = configs.input_channel
        self.out_channel1 = configs.out_channel1
        self.out_channel2 = configs.out_channel2
        self.filter_size = configs.filter_size
        self.filter_size2 = configs.filter_size2
        self.stride = configs.stride
        self.fc1_size = configs.fc1_size
        self.fc2_size = configs.fc2_size
        self.pool_1 = configs.pool_1
        self.pool_2 = configs.pool_2
        self.fc2 = configs.fc2

        flatten_size = (self.sequence_lenth - self.filter_size) // self.stride + 1
        if configs.pool_1:
            flatten_size = (flatten_size - self.filter_size2) // self.stride + 1
        flatten_size = (flatten_size - self.filter_size2) // self.stride + 1
        if configs.pool_2:
            flatten_size = (flatten_size - self.filter_size2) // self.stride + 1
        self.flatten_size = flatten_size

        self.conv1 = nn.Conv1d(in_channels=self.input_channel, out_channels=self.out_channel1,
                               kernel_size=self.filter_size, stride=self.stride)
        self.max_pool1 = nn.MaxPool1d(kernel_size=self.filter_size2, stride=self.stride)
        self.conv2 = nn.Conv1d(self.out_channel1, self.out_channel2, self.filter_size2, self.stride)
        self.max_pool2 = nn.MaxPool1d(self.filter_size2, self.stride)
        self.liner1 = nn.Linear(self.out_channel2 * self.flatten_size, self.fc1_size)
        self.liner2 = nn.Linear(self.fc1_size, self.fc2_size)
        if configs.fc2:
            self.liner3 = nn.Linear(self.fc2_size, 1)
        else:
            self.liner3 = nn.Linear(self.fc1_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.pool_1:
            x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        if self.pool_2:
            x = self.max_pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.liner1(x))
        if self.fc2:
            x = F.relu(self.liner2(x))
        x = self.liner3(x)
        return torch.sigmoid(x)


def prediction_process(input_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('model/DeepLncPro.pkl', map_location='cpu')
    name_list, location_list, sequence_list, data_list = load_dataset(input_file)
    pre_list = predict(data_list, model, device)
    return [name_list, location_list, sequence_list, pre_list]

def write_outputFile(output_list,outputFile,threshold):
    f = open(outputFile, 'w', encoding="utf-8")
    name_list = output_list[0]
    location_list = output_list[1]
    sequence_list = output_list[2]
    pre_list = output_list[3]
    prediction_list = [1 if i > threshold else 0 for i in pre_list]
    out1 = open("js/out1.txt", "r")
    out2 = open("js/out2.txt", "r")
    s = out1.read()
    f.write(s)
    out1.close()
    for i in range(len(name_list)):
        f.write('<tr>'+'\n')
        f.write('<td>'+str(i+1)+'</td>'+'\n')
        f.write('<td>'+name_list[i]+'</td>'+'\n')
        f.write('<td>' + location_list[i]+'</td>'+'\n')
        f.write('<td>' +str(pre_list[i]) +'</td>'+'\n')
        f.write('<td>' + str(prediction_list[i])+'</td>'+'\n')
        f.write('<td>' +sequence_list[i] +'</td>'+'\n')
        f.write('</tr>'+'\n')
    s = out2.read()
    f.write(s)
    out2.close()
    f.close()
def parse_args():
    description = "DeepncPro is able to identify the promoter of non-coding RNA in Human and Mouse.\n" \
                  "Example: python DeepLncPro.py -i example.txt -o output.html -s Human -ts 0.5"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--inputFile', help='-i example.txt (The input file is a complete Fasta format sequence.)')
    parser.add_argument('-o', '--outputFile', help='-o output.html (Results are saved under results folder.)')
    parser.add_argument('-s', '--species', help='-s Human/Mouse (Choose one from two species to use.)')
    parser.add_argument('-ts', '--threshold', help='-ts ')
    args = parser.parse_args()
    return args
def preprocess(inputFile,outputFile,species,threshold):
    output_list = prediction_process(inputFile)
    write_outputFile(output_list, outputFile, threshold)


if __name__ == '__main__':
    args = parse_args()
    preprocess(args.inputFile, args.outputFile, args.species, float(args.threshold))
