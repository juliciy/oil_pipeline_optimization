# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

def parse_project_structure(structure):
    lines = structure.split('\n')
    folders = []
    files = []

    for line in lines:
        if '──' in line:
            # Extract the path
            path = line.split('── ')[1]
            if path.endswith('/'):
                # It's a folder
                folders.append(path[:-1])
            else:
                # It's a file
                files.append(path)

    return folders, files


def generate_dir_and_file_lists(text):
    lines = text.strip().split('\n')
    dir_list = []  # 存放完整路径的目录列表
    file_list = []  # 存放完整路径的文件列表
    current_path = []  # 当前路径，初始化为空列表

    for line in lines:
        # 移除行首的装饰性字符和行尾的注释
        line_content = line.split('#')[0].strip()
        clean_line = line_content.replace('│', '').replace('├──', '').replace('└──', '').strip()

        if not clean_line:  # 忽略空行
            continue

        # 通过缩进级别确定目录层级
        level = (len(line_content) - len(clean_line)) // 4

        # 更新当前路径以匹配当前行的层级，确保正确反映目录层级
        current_path = current_path[:level]  # 适当地缩短或保持 current_path 的长度

        if clean_line.endswith('/'):  # 目录
            dir_name = clean_line.rstrip('/')  # 移除末尾的斜杠
            current_path.append(dir_name)  # 更新当前路径
            full_dir_path = '/'.join(current_path)
            dir_list.append(full_dir_path)  # 将目录的完整路径添加到列表
        else:  # 文件
            file_name = clean_line
            full_file_path = '/'.join(current_path + [file_name])  # 构建文件的完整路径
            file_list.append(full_file_path)  # 将文件的完整路径添加到列表

    return dir_list, file_list

def create_structure(folders, files):
    # 创建所有目录
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Directory created: {folder}")
        else:
            print(f"Directory already exists: {folder}")

    # 创建所有文件和目录（对于以斜杠结尾的路径）
    for file_or_dir in files:
        # 移除"#"及其后面的内容，并去除多余空格
        path = file_or_dir.split('#')[0].strip()
        if not path:  # 如果处理后的路径为空，则跳过
            continue

        if path.endswith('/'):  # 如果路径以斜杠结尾，视为目录
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory created: {path}")
            else:
                print(f"Directory already exists: {path}")
        else:  # 视为文件
            if not os.path.exists(path):
                # 确保文件所在的目录存在
                directory = os.path.dirname(path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Directory created for file: {directory}")

                # 创建文件
                with open(path, 'w') as f:
                    pass  # 创建一个空文件
                print(f"File created: {path}")
            else:
                print(f"File already exists: {path}")

def remove_comments_and_leading_spaces(text):
    lines = text.split('\n')  # 按行分割文本
    processed_lines = []  # 用于存储处理后的行

    for line in lines:
        # 如果行中存在"#"符号，仅保留"#"之前的内容，并去除尾部空格
        if "#" in line:
            line = line.split("#")[0].rstrip()
        # 添加处理后的行到列表中
        processed_lines.append(line)

    # 将处理后的行重新组合为字符串并返回
    return '\n'.join(processed_lines)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open("project_truct.txt", "r", encoding='utf-8') as file:
        directory_structure = file.read()

    # 去掉注释
    directory_structure = remove_comments_and_leading_spaces(directory_structure)

    # 打开目标txt文件用于写入
    with open('project_truct_temp.txt', 'w', encoding='utf-8') as destination_file:
        destination_file.write(directory_structure)

    # 解析“项目文本结构”
    folders, files = generate_dir_and_file_lists(directory_structure)

    print("Folders to create:")
    for folder in folders:
        print(folder)
    print("\nFiles to create:")
    for file in files:
        print(file)

    # 根据解析的路径，创建文件夹和文件
    create_structure(folders, files)



