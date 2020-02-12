### 用于数据预处理
import pandas as pd
import numpy as py

def get_x_data():
    # 数据导入
    x_data = pd.DataFrame(pd.read_csv('Methylation.csv',header=0,index_col=66,low_memory=False))

    # 去除重复的数据
    x_data = x_data.drop_duplicates(keep='first')

    # 以病人的id作为列索引.从第2行开始读取
    x_id = x_data.iloc[1:]

    # 装置
    x_reshape=x_id.T

    return x_reshape



def get_y_data_and_join(x_data):
    y_target = pd.DataFrame(pd.read_csv('drug_sensity_num.csv',usecols=[4,5]))
    # 标签数据处理函数
    def fun(x):
        if(x=='高度敏感'):
            return 3
        if(x=='中度敏感'):
            return 2
        if(x=='低度敏感'):
            return 1
        else:
            return 0

    # 去掉名字和首尾的0
    def delete_name(x): 
        lists = x.split(' ')

        return lists[0]

    # 敏感度(sensity) 中文-数值 替换
    y_target['sensity'] = y_target['sensity'].apply(lambda x: fun(x))

    # 只保留病例的id，去掉名字
    y_target['name'] = y_target['name'].apply(lambda x: delete_name(x))

    
    # 设置合并index
    y_target.set_index(["name"],inplace=True)
 

    total_data=x_data

    for drug_num in range (24):
        
        # 取出对应药物的药敏数据  #0 柔红霉素 
        new_column_name='sensity{}'.format(drug_num)

        
        y_target_single = y_target.iloc[drug_num::24]
 
        y_target_single.rename({"sensity":new_column_name},axis='columns',inplace=True)

        print(y_target_single.columns.values)

        #表合并
        total_data = total_data.join(y_target_single)
        
        #031号病人重复
        special_lable=y_target_single.loc['031',new_column_name]
 
        #处理空值
        total_data = total_data.fillna(value=0)

        # 处理重复病人
        total_data.loc['031.1',new_column_name]=special_lable

        # 保存 #0 号药物： 0:30  1:24  2:9  3:3
        print(drug_num,' done')

    total_data = total_data.T

    total_data.to_csv('process_data_T.csv')

if __name__ == "__main__":

    # 获取x_data
    x_data=get_x_data()
    # 取出标签数据
    

    get_y_data_and_join(x_data)

