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
    f_out.write("station,year,month,tavg,tmin,tmax,prcp\n")

    # 生成要求年份的所有字符串
    start = 1977
    end = 2021

    # 应该是类似这样的数据： 1990,1,
    date_strings = [f"{year},{month}," for year in range(
        start, end) for month in range(1, 13)]

    def operate(csv_file_path, dest):
        filename = os.path.basename(csv_file_path)
        station_id = os.path.splitext(filename)[0]

        with open(csv_file_path, 'r') as file:
            content = file.read().strip()

        if all(date_string in content for date_string in date_strings):
            for line in content.splitlines():
                line = line.strip()
                parts = line.split(',')[:-3]

                if parts[-1].strip() == '':
                    pass  # return False
                line = ','.join(parts)
                f_out.write(station_id + "," + line + '\n')
            return True

    count = 0
    for path in csv_file_list:
        if operate(path, dest) is True:
            count += 1

    print(f'{count} stations meet critera from {start} to {end-1}')
    f_out.close()


# 指定文件夹路径
folder_path = 'dataset/monthly'
dest = 'dataset/concat-filter.csv'
csv_files = traverse_folder(folder_path)

handle(csv_files, dest)
