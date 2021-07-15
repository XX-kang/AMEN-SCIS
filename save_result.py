# -*- coding: utf-8 -*-
import xlsxwriter
import xlrd
from xlutils.copy import copy


# 生成excel文件
def generate_excel(path,expenses,param):

    workbook = xlrd.open_workbook(path)
    #print(workbook.sheet_names())
    new_workbook = copy(wb = workbook)
    worksheet = new_workbook.get_sheet(0)
    table = workbook.sheets()[0]
    nrows = table.nrows
    #ncols = table.ncols
    #row = 1
    col = 0
    worksheet.write(nrows, col, str(param))
    worksheet.write(nrows, col + 1, str(expenses['Overall Acc: \t']))
    worksheet.write(nrows, col + 2, str(expenses['Precision : \t']))
    worksheet.write(nrows, col + 3, str(expenses['Recall: \t']))
    worksheet.write(nrows, col + 4, str(expenses['F1-score : \t']))
    print(str(expenses['Overall Acc: \t']))
    if str(param) == "boosting":
        nrows += 1
        worksheet.write(nrows + 1, col, str("    "))
    # for item in (expenses):
    #     # 使用write方法，指定数据格式写入数据
    #     worksheet.write(nrows+1, col, str(param))
    #     print(expenses[item])
    #     print(type(item['Overall Acc: \t']))
    #     print(str(item['Overall Acc: \t']))
    #     worksheet.write(nrows+1, col + 1, str(item['Overall Acc: \t']))
    #     worksheet.write(nrows+1, col + 2, item['Precision : \t'])
    #     worksheet.write(nrows+1, col + 3, item['Recall: \t'])
    #     worksheet.write(nrows+1, col + 4, item['F1-score : \t'])
    #
    #     nrows += 1
    new_workbook.save(path)


if __name__ == '__main__':
    workbook = xlsxwriter.Workbook('./result.xlsx')
    rec_data = {}
    generate_excel(rec_data,"branch3" )