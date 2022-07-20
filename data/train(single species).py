import argparse
import csv
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torchsummary import summary
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
from sklearn import preprocessing
import weblogo
import shelve
import math
import numpy as np
import random
import copy

import datetime

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device:{torch.cuda.get_device_name(0)}")

"""
load_dataset
============
"""


def make_dir(species, seq_len, encoding_type, proj_name=None):
    global proj_dir
    global motif_result_dir
    global result_file_path
    global best_model_path
    if proj_name and isinstance(proj_name, str):
        proj_dir = './%s/%s_%d_%d' % (proj_name, species, seq_len, encoding_type)
    else:
        proj_dir = './DeepncPro_projects/' + species + '_' + str(seq_len) + '_' + str(encoding_type)
    if os.path.exists(proj_dir):
        for i in range(999):
            if not os.path.exists(proj_dir + '_' + str(i + 1)):
                proj_dir = proj_dir + '_' + str(i + 1)
                break
    os.makedirs(proj_dir)
    motif_result_dir = proj_dir + '/motif_analysis'
    result_file_path = proj_dir + '/result.txt'
    best_model_path = proj_dir + '/best_model.pkl'


def load_dataset(train_data_path, validation_data_path, test_data_path, batch_size=32, encoding=7):
    """Create dataset and dataloader."""
    global train_dataset
    global validation_dataset
    global test_dataset
    global train_dataloader
    global validation_dataloader
    global test_dataloader
    global motif_dataloader
    global motif_seq
    global base
    global DPCP
    global encoding_functions
    global input_c

    base = 'ATGC'
    numbers_input_c = {1: 4, 2: 3, 3: 6, 4: 7, 5: 10, 6: 9, 7: 13}
    input_c = numbers_input_c[encoding]
    encoding_functions = {1: first_encoding, 2: second_encoding, 3: third_encoding, 4: fourth_encoding,
                          5: fifth_encoding, 6: sixth_encoding, 7: seventh_encoding}

    # AA,AT,AG,AC,TA,TT,TG,TC,GA,GT,GG,GC,CA,CT,CG,CC
    # Twist, Tilt, Roll, Shift, Slide, Rise
    # DPCP = {'AA': [0.062976, 0.502398, 0.266277, 1.586862, 0.110915, -0.109294],
    #         'AT': [1.070587, 0.215313, 0.621313, -1.019244, 2.512808, 1.170689],
    #         'AG': [0.782698, 0.358856, 0.088759, 0.678674, -0.240954, -0.623090],
    #         'AC': [1.502421, 0.502398, 0.798831, 0.125863, 1.288914, 1.044493],
    #         'TA': [-1.232525, -2.368448, -0.443795, -2.243325, -1.510744, -1.389278],
    #         'TT': [0.062976, 0.502398, -3.284081, 1.586862, 0.110915, -0.109294],
    #         'TG': [-1.376469, -1.363652, -0.266277, -0.861298, -0.623421, -1.254068],
    #         'TC': [-0.080969, 0.502398, 0.266277, 0.125863, -0.393941, 0.710977],
    #         'GA': [-0.080969, 0.502398, 0.266277, 0.125863, -0.393941, 0.710977],
    #         'GT': [1.502421, 0.502398, 0.798831, 0.125863, 1.288914, 1.044493],
    #         'GG': [0.062976, 1.076567, 0.088759, 0.560214, -0.822304, 0.242250],
    #         'GC': [-0.080969, 0.215313, 1.331384, -0.347974, 0.646369, 1.585331],
    #         'CA': [-1.376469, -1.363652, -0.266277, -0.861298, -0.623421, -1.254068],
    #         'CT': [0.782698, 0.358856, 0.088759, 0.678674, -0.240954, -0.623090],
    #         'CG': [-1.664358, -1.220110, -0.443795, -0.821812, -0.286850, -1.389278],
    #         'CC': [0.062976, 1.076567, 0.088759, 0.560214, -0.822304, 0.242250]}

    # Normalization
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

    # Create data object.
    train_data = open_data_path(train_data_path, encoding, base)
    validation_data = open_data_path(validation_data_path, encoding, base)
    test_data = open_data_path(test_data_path, encoding, base)
    motif_data, motif_seq = merge_positive_data(train_data_path, validation_data_path, test_data_path, encoding)

    # Create dataset object.
    train_dataset = MyDataset(train_data)
    validation_dataset = MyDataset(validation_data)
    test_dataset = MyDataset(test_data)
    motif_dataset = MyDataset(motif_data)

    # Create dataloader object.
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100000, shuffle=True,
                                                        pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100000, shuffle=False,
                                                  pin_memory=True)
    motif_dataloader = torch.utils.data.DataLoader(dataset=motif_dataset, batch_size=100000, shuffle=False)


def open_data_path(data_path, encoding, base):
    all_data = []
    encoding_function = encoding_functions.get(encoding, default_encoding)
    with open(data_path, 'rt') as data:
        next(data)
        lines = csv.reader(data, delimiter='\t')
        for line in lines:
            all_data.append([encoding_function(line[0], base), [int(line[1])]])
    return all_data


def merge_positive_data(train_data, validation_data, test_data, encoding):
    motif_data, motif_seq = [], []
    encoding_function = encoding_functions.get(encoding, default_encoding)

    # Train.
    with open(train_data, 'rt') as data:
        next(data)
        lines = csv.reader(data, delimiter='\t')
        for line in lines:
            if int(line[1]) == 1:
                motif_data.append([encoding_function(line[0], base), [int(line[1])]])
                motif_seq.append(line[0])

    # Validation.
    with open(validation_data, 'rt') as data:
        next(data)
        lines = csv.reader(data, delimiter='\t')
        for line in lines:
            if int(line[1]) == 1:
                motif_data.append([encoding_function(line[0], base), [int(line[1])]])
                motif_seq.append(line[0])

    # Test.
    with open(test_data, 'rt') as data:
        next(data)
        lines = csv.reader(data, delimiter='\t')
        for line in lines:
            if int(line[1]) == 1:
                motif_data.append([encoding_function(line[0], base), [int(line[1])]])
                motif_seq.append(line[0])
    return motif_data, motif_seq


def first_encoding(sequence, base):
    '''One-hot encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.empty([matrix_lenth, 4])
    for i in range(matrix_lenth):
        for j in range(4):
            if sequence[i] == base[j]:
                matrix[i, j] = np.float32(1)
            else:
                matrix[i, j] = np.float32(0)
    return np.transpose(matrix)


def second_encoding(sequence, base):
    '''NCP encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 3])
    for i in range(matrix_lenth):
        if sequence[i] == base[0]:
            matrix[i, 0] = np.float32(1)
            matrix[i, 1] = np.float32(1)
            matrix[i, 2] = np.float32(1)
        elif sequence[i] == base[1]:
            matrix[i, 0] = np.float32(0)
            matrix[i, 1] = np.float32(1)
            matrix[i, 2] = np.float32(0)
        elif sequence[i] == base[2]:
            matrix[i, 0] = np.float32(1)
            matrix[i, 1] = np.float32(0)
            matrix[i, 2] = np.float32(0)
        elif sequence[i] == base[3]:
            matrix[i, 0] = np.float32(0)
            matrix[i, 1] = np.float32(0)
            matrix[i, 2] = np.float32(1)
    return np.transpose(matrix)


def third_encoding(sequence, base):
    '''DPCP encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 6])
    for i in range(matrix_lenth - 1):
        couple = sequence[i] + sequence[i + 1]
        properties = DPCP[couple]
        for m in range(6):
            matrix[i, m] += np.float32(properties[m] / 2)
            matrix[i + 1, m] += np.float32(properties[m] / 2)
    return np.transpose(matrix)


def fourth_encoding(sequence, base):
    '''One-hot+NCP encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 7])
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
    return np.transpose(matrix)


def fifth_encoding(sequence, base):
    '''One-hot+DPCP encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 10])
    for i in range(matrix_lenth):
        for j in range(4):
            if sequence[i] == base[j]:
                matrix[i, j] = np.float32(1)
            else:
                matrix[i, j] = np.float32(0)
    for i in range(matrix_lenth - 1):
        couple = sequence[i] + sequence[i + 1]
        properties = DPCP[couple]
        for m in range(6):
            matrix[i, 4 + m] += np.float32(properties[m] / 2)
            matrix[i + 1, 4 + m] += np.float32(properties[m] / 2)
    return np.transpose(matrix)


def sixth_encoding(sequence, base):
    '''NCP+DPCP encoding.'''
    matrix_lenth = len(sequence)
    matrix = np.zeros([matrix_lenth, 9])
    for i in range(matrix_lenth):
        if sequence[i] == base[0]:
            matrix[i, 0] = np.float32(1)
            matrix[i, 1] = np.float32(1)
            matrix[i, 2] = np.float32(1)
        elif sequence[i] == base[1]:
            matrix[i, 0] = np.float32(0)
            matrix[i, 1] = np.float32(1)
            matrix[i, 2] = np.float32(0)
        elif sequence[i] == base[2]:
            matrix[i, 0] = np.float32(1)
            matrix[i, 1] = np.float32(0)
            matrix[i, 2] = np.float32(0)
        elif sequence[i] == base[3]:
            matrix[i, 0] = np.float32(0)
            matrix[i, 1] = np.float32(0)
            matrix[i, 2] = np.float32(1)
    for i in range(matrix_lenth - 1):
        couple = sequence[i] + sequence[i + 1]
        properties = DPCP[couple]
        for m in range(6):
            matrix[i, 3 + m] += np.float32(properties[m] / 2)
            matrix[i + 1, 3 + m] += np.float32(properties[m] / 2)
    return np.transpose(matrix)


def seventh_encoding(sequence, base):
    '''One-hot+NCP+DPCP encoding.'''
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


def default_encoding(sequence, base):
    print('Wrong encoding!')
    sys.exit()
    return 0


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, xy=None):
        self.x_data = np.asarray([el[0] for el in xy], dtype=np.float32)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def model_metrics(labels, pred):
    '''Calculate evaluation metrics.'''
    tn, fp, fn, tp = metrics.confusion_matrix(labels, pred).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    return acc, sn, sp, mcc


"""
argument
============
"""


def deepncpro(args):
    global encoding
    global seq_length
    global species
    global base_type
    # if base_type == 'RNA':
    #     motif_sequences = [sequence.replace('T', 'U') for sequence in motif_sequences]
    # elif base_type == 'DNA':
    #     motif_sequences = [sequence.replace('U', 'T') for sequence in motif_sequences]
    global tomtom_directory
    global stride
    encoding = args.encoding
    seq_length = args.seq_length
    species = args.species


def parsing_args():
    parser = argparse.ArgumentParser(description='DeepncRro')

    # parser.add_argument('--xx', type=str, default='xx', help='xx')
    parser.add_argument('--encoding', type=str, default='all', help='Feature encoding strategy:onehot, NCP, DPCP.')
    parser.add_argument('--seq_length', type=int, default='221', help='Sequence length.')
    parser.add_argument('--species', type=str, default='h', help='Species:h, m.')

    args = parser.parse_args()
    return args


"""
motif_analysis
============
"""


def motif_analysis(model_path, motif_dataloader, motif_sequences, motif_directory, tomtom_directory):
    if not os.path.exists(motif_directory):
        os.makedirs(motif_directory)
    model = torch.load(model_path)
    model = model.to(device)
    for layer in model.parameters():
        layer1_parameters = layer.detach().cpu().numpy()
        break
    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(motif_dataloader):
            data = data.to(device)
            layer1_out = model.conv1(data).detach().cpu().numpy()
            break

    motif_extraction(motif_sequences, layer1_parameters, layer1_out, motif_directory, tomtom_directory)


def meme(motif_sequences, motif_directory):
    '''Format Specification
    The minimal MEME format contains following sections:
    1.Version (required)
    2.Alphabet (recommended)
    3.Strands (optional)
    4.Background frequencies (recommended)
    5.Motifs (required)
    '''
    global meme_file_path
    meme_file_path = motif_directory + '/meme_file.txt'
    meme_file = open(meme_file_path, 'w')
    nucleotide = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Version
    print('MEME version 4', file=meme_file)
    print('', file=meme_file)

    # Alphabet
    print('ALPHABET= ACGT', file=meme_file)
    print('', file=meme_file)
    print('Background letter frequencies:', file=meme_file)

    # Background frequencies
    nucleotide_counts = [0] * 4
    for i in range(len(motif_sequences)):
        for j in motif_sequences[i]:
            try:
                nucleotide_counts[nucleotide[j]] += 1
            except KeyError:
                pass
    nucleotide_all = float(sum(nucleotide_counts))
    frequencies = [nucleotide_counts[i] / nucleotide_all for i in range(4)]
    print('A %.4f C %.4f G %.4f T %.4f' % tuple(frequencies), file=meme_file)
    print('', file=meme_file)
    return meme_file


def filters_heatmap(filter_matrix, filters_directory, n):
    out_path = '%s/filter%d_heatmap.pdf' % (filters_directory, n)
    param_range = abs(filter_matrix).max()
    sns.set(font_scale=2)
    plt.figure(figsize=(filter_matrix.shape[1], 4))
    sns.heatmap(filter_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range, vmax=param_range)
    new_d = plt.gca()
    new_d.set_xticklabels(range(1, filter_matrix.shape[1] + 1))
    new_d.set_yticklabels('ATGC', rotation='horizontal')  # , size=10)
    plt.savefig(out_path)
    plt.close()


def filters_weblogo(filter_out, filters_lenth, motif_sequences, filters_directory, n):
    out_path = '%s/filter%d_weblogo' % (filters_directory, n)
    all_out = np.ravel(filter_out)
    all_out_mean = all_out.mean()
    all_out_norm = all_out - all_out_mean
    bounds = 0.65 * all_out_norm.max() + all_out_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_path, 'w')
    filter_count = 0
    for i in range(filter_out.shape[0]):
        for j in range(filter_out.shape[1]):
            if filter_out[i, j] > bounds:
                high_activation_value_seq = motif_sequences[i][j:j + filters_lenth]
                print('>%d_%d' % (i, j), file=filter_fasta_out)
                print(high_activation_value_seq, file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()

    if filter_count > 0:
        weblogo_f = open('%s.fa' % out_path, 'r')
        weblogo_seqs = weblogo.read_seq_data(weblogo_f)
        weblogo_data = weblogo.LogoData.from_seqs(weblogo_seqs)
        weblogo_options = weblogo.LogoOptions()
        weblogo_options.fineprint = ''
        weblogo_format = weblogo.LogoFormat(weblogo_data, weblogo_options)
        weblogo_eps = weblogo.eps_formatter(weblogo_data, weblogo_format)
        with open('%s.eps' % out_path, 'wb') as weblogo_f:
            weblogo_f.write(weblogo_eps)


def filters_pwm(filters_directory, n):
    ''' Make a PWM for the filter by sequences with high activation values.'''
    filter_fasta_out = '%s/filter%d_weblogo.fa' % (filters_directory, n)
    nucleotide = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    pwm_counts = []
    nsites = 4  # pseudocounts
    for line in open(filter_fasta_out):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0] * 4))
            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nucleotide[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.25] * 4)
    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j] / float(nsites) for j in range(4)])
    return np.array(pwm_freqs), nsites - 4


def info_content(pwm, bg_gc=0.415):
    ''' Compute PWM information content.'''
    pseudoc = 1e-9
    bg_pwm = [1 - bg_gc, 1 - bg_gc, bg_gc, bg_gc]
    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j] * np.log2(bg_pwm[j]) + pwm[i][j] * np.log2(pseudoc + pwm[i][j])
    return ic


def meme_add(meme_out, n, filter_pwm, nsites):
    ''' Print a filter to the growing MEME file
    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    '''
    ic_start = 0
    ic_end = filter_pwm.shape[0] - 1
    if ic_start < ic_end:
        print('MOTIF filter%d' % n, file=meme_out)
        print('letter-probability matrix: alength= 4 w= %d nsites= %d' % (ic_end - ic_start + 1, nsites), file=meme_out)
        for i in range(ic_start, ic_end + 1):
            print('%.4f %.4f %.4f %.4f' % tuple(filter_pwm[i]), file=meme_out)
        print('', file=meme_out)


def motif_and_protein(database):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_and_protein = {}
    for line in open(database, 'r'):
        reads = line.split()
        if len(reads) > 0 and reads[0] == 'MOTIF':
            motif_and_protein[reads[1]] = reads[2]
    return motif_and_protein


def name_filters(filters_number, tomtom_output, database):
    ''' Name the filters using Tomtom matches.
    Attrs:
        filters_number (int) : total number of filters
        tomtom_output (str) : filename of Tomtom output table.
        database (str) : filename of MEME db
    Returns:
        filter_names [str] :
    '''
    # name by number
    filter_names = ['f%d' % fi for fi in range(filters_number)]

    # name by protein
    if tomtom_output and database:
        print(tomtom_output, database)
        motif_protein = motif_and_protein(database)

        # hash motifs and q-value's by filter
        filter_motifs = {}
        f_tomtom = open(tomtom_output, 'r')
        next(f_tomtom)
        lines = f_tomtom.readline()
        for line in lines:
            reads = line.split()
            if reads == []:
                break
            fi = int(reads[0][6:])
            motif_id = reads[1]
            qval = float(reads[5])
            filter_motifs.setdefault(fi, []).append((qval, motif_id))
        f_tomtom.close()

        # assign filter's best match
        for fi in filter_motifs:
            top_motif = sorted(filter_motifs[fi])[0][1]
            filter_names[fi] += '_%s' % motif_protein[top_motif]
    return np.array(filter_names)


def filter_motif(param_matrix):
    nts = 'ACGT'
    motif_list = []
    for v in range(param_matrix.shape[1]):
        max_n = 0
        for n in range(1, 4):
            if param_matrix[n, v] > param_matrix[max_n, v]:
                max_n = n
        if param_matrix[max_n, v] > 0:
            motif_list.append(nts[max_n])
        else:
            motif_list.append('N')
    return ''.join(motif_list)


def plot_score_density(f_scores, out_pdf):
    sns.set(font_scale=1.3)
    plt.figure()
    sns.distplot(f_scores, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()
    return f_scores.mean(), f_scores.std()


def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence
    filter_seqs = filter_outs.mean(axis=2)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)
    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)
    hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)
    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(filter_seqs[:, seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False,
                   vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    # out_png = out_pdf[:-2] + 'ng'
    # plt.savefig(out_png, dpi=300)
    plt.close()


def motif_extraction(motif_sequences, layer1_parameters, layer1_out, motif_directory, tomtom_directory, stride=1):
    global weblogo_opts
    weblogo_opts = '-X NO --fineprint ""'
    weblogo_opts += ' -C "#CB2026" A A'
    weblogo_opts += ' -C "#34459C" C C'
    weblogo_opts += ' -C "#FBB116" G G'
    weblogo_opts += ' -C "#0C8040" T T'
    filters = []
    for filter in layer1_parameters:
        # normalized, scale = preprocess_data(x)
        # normalized = normalized.T
        # normalized = normalized/normalized.sum(axis=1)[:,None]
        mean_filter = filter - np.mean(filter, axis=0)
        filters.append(mean_filter)
    filters = np.array(filters)

    # ???
    filters_number = filters.shape[0]
    filters_lenth = filters.shape[2]
    filters_ic = []
    meme_file = meme(motif_sequences, motif_directory)
    filters_directory = motif_directory + '/filters'
    if not os.path.exists(filters_directory):
        os.makedirs(filters_directory)
    for n in range(filters_number):
        # print('Filter %d' % i)

        ##########################################################################################################
        # filters_heatmap(filters[n, :, :], filters_directory, n)
        filters_weblogo(layer1_out[:, n, :], filters_lenth, motif_sequences, filters_directory, n)
        n_filters_pwm, nsites = filters_pwm(filters_directory, n)
        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(n_filters_pwm))
            # add to the meme motif file
            meme_add(meme_file, n, n_filters_pwm, nsites)
    meme_file.close()


    # subprocess.call(tomtom_directory + ' -dist pearson -thresh 0.05 -oc %s/tomtom_result %s/meme_file.txt %s' % (
    #     motif_directory, motif_directory, 'motif_database/JASPAR2018_CORE_vertebrates_non-redundant.meme'),
    #                 shell=True)


"""
Basic variable
============
"""


# is_hidden = False
# conv_layers = 1
# batch_size=32
# Embepochs=100
# Embsize=50
# kmer_len=3
# stride=1
# motif_num=16
# motif_len=24
# dilation = 1
# epoch = 5
# learning_rate = 0.01


class Config:
    '''Initialization, loading and saving of configuration.'''

    def __init__(self, sequence_lenth, encoding, batch_size):
        self.batch_size = batch_size
        self.sequence_lenth = sequence_lenth
        input_size_dic = {1: 4, 2: 3, 3: 6, 4: 7, 5: 10, 6: 9, 7: 13}
        self.encoding = encoding
        self.input_channel = input_size_dic[encoding]
        self.out_channel1 = 64
        self.out_channel2 = 32
        self.filter_size = 10
        self.filter_size2 = 10
        self.stride = 1
        self.fc1_size = 64
        self.fc2_size = 32
        self.pool_1 = False
        self.pool_2 = False
        self.fc2 = False
        self.lr = 0.001

    def log_sampler(self, x_down, x_up):
        sample = np.random.uniform(low=0, high=1)
        sample_log = 10 ** ((math.log10(x_up) - math.log10(x_down)) * sample + math.log10(x_down))
        return sample_log

    def initialize(self):
        out_channel1_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
        self.out_channel1 = random.choice(out_channel1_list)
        out_channel2_list = range(8, self.out_channel1 + 1, 8)
        self.out_channel2 = random.choice(out_channel2_list)
        pool_1_list = [False]
        self.pool_1 = random.choice(pool_1_list)
        pool_2_list = [False]
        self.pool_2 = random.choice(pool_2_list)
        fc2_list = [False,True]
        self.fc2 = random.choice(fc2_list)
        stride_list = [1]
        self.stride = random.choice(stride_list)
        self.lr = self.log_sampler(0.0001, 0.01)
        fc1_size_list = range(16, 129, 8)
        self.fc1_size = random.choice(fc1_size_list)
        fc2_size_list = range(8, self.fc1_size, 8)
        self.fc2_size = random.choice(fc2_size_list)
        num_layer = 1
        if self.pool_1:
            num_layer += 1
        if self.pool_2:
            num_layer += 1
        filter_size_list = range(10, 31)
        self.filter_size = random.choice(filter_size_list)
        range_filter2 = ((self.sequence_lenth - self.filter_size) // num_layer) + 1
        filter_size_list2 = range(3, range_filter2)
        self.filter_size2 = random.choice(filter_size_list2)

    def load_config(self, config_name, config_db):
        try:
            db = shelve.open(config_db, flag='w', writeback=True)
            if self.batch_size != db[config_name + '_batch_size']:
                print('batch_size mismatch.')
            if self.sequence_lenth != db[config_name + '_sequence_lenth']:
                print('sequence_lenth mismatch.')
            if self.encoding != db[config_name + '_encoding']:
                print('encoding mismatch.')
            self.input_channel = db[config_name + '_input_channel']
            self.out_channel1 = db[config_name + '_out_channel1']
            self.out_channel2 = db[config_name + '_out_channel2']
            self.filter_size = db[config_name + '_filter_size']
            self.filter_size = db[config_name + '_filter_size2']
            self.stride = db[config_name + '_stride']
            self.fc1_size = db[config_name + '_fc1_size']
            self.fc2_size = db[config_name + '_fc2_size']
            self.pool_1 = db[config_name + '_pool_1']
            self.pool_2 = db[config_name + '_pool_2']
            self.fc2 = db[config_name + '_fc2']
            self.lr = db[config_name + '_lr']
            db.close()
        except:
            print('Error loading configuration.')

    def save_config(self, config_name, config_db):
        try:
            db = shelve.open(config_db, flag='w', writeback=True)
            db[config_name + '_sequence_lenth'] = self.sequence_lenth
            db[config_name + '_encoding'] = self.encoding
            db[config_name + '_input_channel'] = self.input_channel
            db[config_name + '_batch_size'] = self.batch_size
            db[config_name + '_out_channel1'] = self.out_channel1
            db[config_name + '_out_channel2'] = self.out_channel2
            db[config_name + '_filter_size'] = self.filter_size
            db[config_name + '_filter_size2'] = self.filter_size2
            db[config_name + '_stride'] = self.stride
            db[config_name + '_fc1_size'] = self.fc1_size
            db[config_name + '_fc2_size'] = self.fc2_size
            db[config_name + '_pool_1'] = self.pool_1
            db[config_name + '_pool_2'] = self.pool_2
            db[config_name + '_fc2'] = self.fc2
            db[config_name + '_lr'] = self.lr
            db.close()
        except:
            print('Error saving configuration.')


class Record:
    '''Appending, printing and saving of result records.'''

    def __init__(self, species, seq_len, encoding_type):
        self.species = species
        self.seq_len = seq_len
        self.encoding_type = encoding_type
        self.records = []

    def append(self, v_acc, metric_list, model, config):
        '''metric_list:[species,seq_len,encoding_type,best_validation_acc,acc, sn, sp, mcc, auc]'''
        self.records.append([v_acc, metric_list, model, config])
        self.records.sort(reverse=True)

    def print(self, num_print):
        len_records = len(self.records)
        print('\nspecies：%s\tlenth：%d\tencoding：%d' % (self.species, self.seq_len, self.encoding_type))
        print('name\tvalid\ttest')
        if len_records < num_print:
            print('Not enough records.')
        else:
            for i in range(num_print):
                print('best%d\t%.4f\t%.4f' % (i, self.records[i][1][3], self.records[i][1][4]))

    def save(self, result_path, project_path, database_path):
        len_records = len(self.records)
        if len_records >= 5:
            len_records = 5
        for i in range(len_records):
            torch.save(self.records[i][2], project_path + '/bestmodel_' + str(i) + '.pkl')
            self.records[i][3].save_config(
                '%s_%d_%d_best%d' % (self.records[i][1][0], self.records[i][1][1], self.records[i][1][2], i),
                database_path)
        with open(result_path, 'a') as f:
            f.write('>%s_%d_%d\n' % (self.species, self.seq_len, self.encoding_type))
            f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('name', 'acc_v', 'acc_t', 'sn', 'sp', 'mcc', 'auc',))
            for i in range(len_records):
                f.write('best%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                    i, self.records[i][1][3], self.records[i][1][4], self.records[i][1][5], self.records[i][1][6],
                    self.records[i][1][7], self.records[i][1][8],))

    def save_old(self, result_path_old):
        with open(result_path_old, 'a') as f:
            f.write('%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (self.records[0][1][0], self.records[0][1][1],
                                                                          self.records[0][1][2], self.records[0][1][3],
                                                                          self.records[0][1][4], self.records[0][1][5],
                                                                          self.records[0][1][6], self.records[0][1][7],
                                                                          self.records[0][1][8],))


"""
Model
============
"""


class DeepncPro(nn.Module):
    """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for one_epoch in range(100):
    """

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


"""
测试函数
============
"""


def all_proc(species, seq_len, encoding_type):
    batch_size = 16
    train_data_path = './DataSet/%sbp/%s_train_%s.csv' % (str(seq_len), species, str(seq_len))
    validation_data_path = './DataSet/%sbp/%s_validation_%s.csv' % (str(seq_len), species, str(seq_len))
    test_data_path = './DataSet/%sbp/%s_test_%s.csv' % (str(seq_len), species, str(seq_len))
    tomtom_dir = './meme-5.3.3/src/tomtom'

    # data
    make_dir(species, seq_len, encoding_type, None)
    load_dataset(train_data_path, validation_data_path, test_data_path, batch_size=batch_size, encoding=encoding_type)
    record = Record(species, seq_len, encoding_type)

    # train model
    for try_time in range(500):

        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        best_validation_acc = 0
        if False:
            print('Try time:', try_time + 1)

        # hyperparametric initialization of model
        config = Config(seq_len, encoding_type, batch_size)
        config.initialize()
        model = DeepncPro(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for one_epoch in range(50):
            if False:
                print('one_epoch', one_epoch)

            # train
            model.train()
            for i, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                target = target.to(device)
                # forward pass
                output = model(data)
                loss = F.binary_cross_entropy(output, target)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # test
            if one_epoch < 100:
                with torch.no_grad():
                    model.eval()
                    validation_acc = []
                    test_acc = []
                    for i, (data, target) in enumerate(validation_dataloader):
                        data = data.to(device)
                        target = target.to(device)
                        output = model(data)
                        pred = output.detach().cpu().numpy().reshape(output.shape[0])
                        labels = target.cpu().numpy().reshape(output.shape[0])
                        validation_acc.append(metrics.accuracy_score(labels, pred.round()))

                    # Select the best result from the accuracy of the validation set.
                    mean_validation_acc = np.mean(validation_acc)
                    # print(mean_validation_acc)
                    if mean_validation_acc > best_validation_acc:
                        for i, (data, target) in enumerate(validation_dataloader):
                            data = data.to(device)
                            target = target.to(device)
                            output = model(data)
                            pred = output.detach().cpu().numpy().reshape(output.shape[0])
                            labels = target.cpu().numpy().reshape(output.shape[0])
                            test_acc.append(metrics.accuracy_score(labels, pred.round()))
                            acc, sn, sp, mcc = model_metrics(labels, pred.round())
                            auc = metrics.roc_auc_score(labels, pred)
                        mean_test_acc = np.mean(test_acc)
                        best_validation_acc = mean_validation_acc
                        best_model = copy.deepcopy(model)
        # save records
        record.append(round(best_validation_acc, 4),
                      [species, seq_len, encoding_type, round(best_validation_acc, 4), round(acc, 4), round(sn, 4),
                       round(sp, 4), round(mcc, 4), round(auc, 4)], best_model, config)

    # save results
    record.print(5)
    record.save('./result_best1-5.txt', proj_dir, './config/config.db')
    record.save_old('./result.txt')


def all_process(species_list, seq_len_list, encoding_type_list):
    for choice_species in species_list:
        for choice_seq_len in seq_len_list:
            for choice_encoding_type in encoding_type_list:
                all_proc(choice_species, choice_seq_len, choice_encoding_type)


def just_motif_analysis_all(species, seq_len, encoding_type):
    for bestmodel_index in range(5):
        motif_directory = './DeepncPro_projects/%s_%s_%s/motif_analysis_%d' % (
            species, str(seq_len), str(encoding_type), bestmodel_index)
        tomtom_directory = 'tomtom'
        model_path = './DeepncPro_projects/%s_%s_%s/bestmodel_%d.pkl' % (
            species, str(seq_len), str(encoding_type), bestmodel_index)
        batch_size = 32
        train_data_path = './DataSet/%sbp/%s_train_%s.csv' % (str(seq_len), species, str(seq_len))
        validation_data_path = './DataSet/%sbp/%s_validation_%s.csv' % (str(seq_len), species, str(seq_len))
        test_data_path = './DataSet/%sbp/%s_test_%s.csv' % (str(seq_len), species, str(seq_len))
        # make_dir(species, seq_len, encoding_type)
        load_dataset(train_data_path, validation_data_path, test_data_path, encoding=encoding_type)
        motif_analysis(model_path, motif_dataloader, motif_seq, motif_directory, tomtom_directory)


def just_motif_analysis(species, seq_len, encoding_type,model_path='xxx'):
    motif_directory = './motif_analysis'
    tomtom_directory = 'xxx'
    # model_path = './DeepncPro_projects_2021.12.03一层卷积/model/' + species + '_' + str(seq_len) + '_' + str(encoding_type) + '/bestmodel_0.pkl'
    batch_size = 32
    train_data_path = './DataSet/%sbp/%s_train_%s.csv' % (str(seq_len), species, str(seq_len))
    validation_data_path = './DataSet/%sbp/%s_validation_%s.csv' % (str(seq_len), species, str(seq_len))
    test_data_path = './DataSet/%sbp/%s_test_%s.csv' % (str(seq_len), species, str(seq_len))
    # tomtom_path = './meme-5.3.3/src/tomtom'
    # make_dir(species, seq_len, encoding_type)
    load_dataset(train_data_path, validation_data_path, test_data_path, encoding=encoding_type)
    motif_analysis(model_path, motif_dataloader, motif_seq, motif_directory, tomtom_directory)


if __name__ == '__main__':
    # args = parsing_args()
    # deepncpro(args)
    # motif分析
    #just_motif_analysis('m', 181, 7,'D:\\Project\\deepnc\\DDD\\bestmodel_0.pkl')
    # just_motif_analysis('m', 61, 7)

    # 2*7*7 all_process
    all_process(['m', 'h'], [61, 101, 141, 181, 221, 261, 301], [1, 2, 3, 4, 5, 6, 7])
    # for xxxxx in range(3):
    #     all_process(['m','h'], [181], [5,7])
