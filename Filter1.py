import os


def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def handle(csv_file_list, dest):
        # 先清空文件
    with open(dest, 'w') as f:
        pass
    f_out = open(dest, 'a')
    f_out.write("station,year,month,tavg,tmin,tmax,prcp,wspd,pres,tsun\n")

    def operate(csv_file_path, dest):
        filename = os.path.basename(csv_file_path)
        station_id = os.path.splitext(filename)[0]

        with open(csv_file_path, 'r') as file:
            for line in file:
                i = line.strip()
                f_out.write(station_id + "," + i + '\n')

    for path in csv_file_list:
        operate(path, dest)
    f_out.close()


# 指定文件夹路径
folder_path = 'dataset/monthly'
dest = 'dataset/concat.csv'
csv_files = traverse_folder(folder_path)

handle(csv_files, dest)
