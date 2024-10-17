"""
这是文档分析测试
"""

import os
from paddleocr import PPStructure, save_structure_res
from ppocr.utils.utility import check_and_read
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
from copy import deepcopy


def test_cwd():
    cwd = os.getcwd()
    print(cwd)



def test_pp_structure():
    engine = PPStructure(show_log=True, lang="ch", revovery=True)
    img_path = ""
    pdf_path = ''
    output_path = r"output\\pdf_structure"
    # 测试  自己获取pdf图片
    images, flag_gif, flag_pdf = check_and_read(pdf_path)
    pdf_name = os.path.basename(pdf_path)[:-4]
    all_res = []
    index_list = [3, 7, 8, 9, 11, 13, 15, 16]
    for index, image in enumerate(images):
        # if index not in index_list:
        #     continue
        result = engine(image, img_idx=index)
        save_structure_res(result, output_path, pdf_name, index)
        if result:
            height, width, channels = image.shape
            result_cp = deepcopy(result)
            # 排序
            result_sorted = sorted_layout_boxes(result_cp, width)
            all_res += result_sorted
    # 转到docx进行版面恢复
    convert_info_docx(images, all_res, output_path, pdf_name)
