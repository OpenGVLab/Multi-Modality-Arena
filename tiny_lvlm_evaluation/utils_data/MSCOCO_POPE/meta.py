# 打开原始JSON文件和新的JSON文件
with open('/home/mengfanqing/NIPS_benchmark/holistic_evaluation-main2/MSCOCO_POPE/coco_pope_adversarial.json', 'r') as f1, open('/home/mengfanqing/NIPS_benchmark/holistic_evaluation-main2/MSCOCO_POPE/coco_pope_adversarial1.json', 'w') as f2:
    # 对于原始JSON文件中的每一行数据
    for line in f1:
        # 去掉末尾可能存在的换行符
        line_strip = line.strip()
        # 给每一行末尾加上逗号
        if line_strip.endswith('}'):
            line_strip += ','
        # 将修改后的行写入新的JSON文件中
        f2.write(line_strip + '\n')