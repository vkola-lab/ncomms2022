import csv
import random
import os


def balanced_split(groups, path):
    part1, part2, part3 = read_csv_dict("all.csv", groups)
    random.shuffle(part1)
    filed = set([])
    partitions = [[] for i in range(5)]
    count_ADD_0 = [0 for i in range(5)]
    count_ADD_1 = [0 for i in range(5)]
    count_COG_0 = [0 for i in range(5)]
    count_COG_1 = [0 for i in range(5)]
    count_COG_2 = [0 for i in range(5)]
    for case in part1:
        if case['filename'] not in filed and case['ADD'] != '':
            filed.add(case['filename'])
            if case['ADD'] == '0':
                idx = count_ADD_0.index(min(count_ADD_0))
                count_ADD_0[idx] += 1
                count_COG_2[idx] += 1
                partitions[idx].append(case)
            elif case['ADD'] == '1':
                idx = count_ADD_1.index(min(count_ADD_1))
                count_ADD_1[idx] += 1
                count_COG_2[idx] += 1
                partitions[idx].append(case)
    for case in part1:
        if case['filename'] not in filed and case['COG'] != '':
            filed.add(case['filename'])
            if case['COG'] == '0':
                idx = count_COG_0.index(min(count_COG_0))
                count_COG_0[idx] += 1
                partitions[idx].append(case)
            elif case['COG'] == '1':
                idx = count_COG_1.index(min(count_COG_1))
                count_COG_1[idx] += 1
                partitions[idx].append(case)
            elif case['COG'] == '2':
                idx = count_COG_2.index(min(count_COG_2))
                count_COG_2[idx] += 1
                partitions[idx].append(case)
    print(count_ADD_0)
    print(count_ADD_1)
    print(count_COG_0)
    print(count_COG_1)
    print(count_COG_2)

    if not os.path.exists(path): os.mkdir(path)

    for way in range(5):
        subpath = path + 'cross{}/'.format(way)
        if not os.path.exists(subpath): os.mkdir(subpath)
        train = []
        for i in range(5):
            if i != way and i != (way-1+5) % 5:
                train.extend(partitions[i])
        write(train, subpath + 'train.csv')
        write(partitions[way-1], subpath + 'valid.csv')
        write(partitions[way], subpath + 'test.csv')
        write(part2, subpath + 'exter_test.csv')
        write(part3, subpath + 'OASIS.csv')

        # new section
        for fold in range(4):
            subpath = path + 'cross{}/fold{}/'.format(way, fold)
            if not os.path.exists(subpath): os.mkdir(subpath)
            write(partitions[way], subpath + 'test.csv')
            write(part2, subpath + 'exter_test.csv')
            write(part3, subpath + 'OASIS.csv')
            write(partitions[way-1-fold], subpath + 'valid.csv')
            train = []
            for j in range(4):
                if j != way and j != (way - 1 - fold + 5) % 5:
                    train.extend(partitions[j])
            write(train, subpath + 'train.csv')


def read_csv_dict(csv_table, groups):
    content1, content2, content3 = [], [], []
    with open(csv_table, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            found_in_group = False
            for group in groups:
                if group in row['path']:
                    found_in_group = True
                    content1.append(row)
            if not found_in_group:
                content2.append(row)
            if 'OASIS' in row['path']:
                content3.append(row)
    return content1, content2, content3


def write(content, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        column_names = ['path', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER']
        spamwriter.writerow(column_names)
        for row in content:
            spamwriter.writerow([row[col_name] if col_name in row else '' for col_name in column_names])


if __name__ == "__main__":
    # ['NACC', 'ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']
    balanced_split(['NACC'], './')
