import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score 

import warnings
warnings.filterwarnings("ignore")

def get_x_data():
    # 读取处理过的数据
    total_data =pd.DataFrame(pd.read_csv('process_data_T.csv',index_col=0)).T
    
    return total_data

#def get_y_lable()

def run(total_data,drug_num,drug_name_list):
    # 
    data = total_data.iloc[:,0:598243]
    label = total_data['sensity{}'.format(drug_num)]
    print('data_done')

    from sklearn.model_selection import GridSearchCV

    
    #设置参数矩阵：
    param_grid = {
                'n_estimators': np.arange(50,201,50),
                'max_features':['auto','log2'],
                'random_state':np.arange(0,71,10)
    }
  
    rfc = GridSearchCV(RandomForestClassifier(n_jobs=-1,criterion='entropy',class_weight='balanced'), param_grid, scoring='f1_macro',cv=5,verbose=1)  #  k折交叉验证  数据少 k越大越好
    #clf = GridSearchCV(DecisionTreeClassifier(splitter='best',criterion='entropy',min_samples_split=10,max_features=1000,random_state=50), param_grid, scoring='f1_macro',cv=3,verbose=1)

    rfc.fit(data, label)

    # 取出最优模型
    best_estimator=rfc.best_estimator_

    best_parameters = best_estimator.get_params() 

    # 计算最终f1与AUC
    from sklearn.metrics import accuracy_score,f1_score
    predict=best_estimator.predict(data)

    f1_final= f1_score(list(label), predict,average='macro') 
    auc=accuracy_score(list(label), predict) 
 
    print("drug_num: {} train_f1_score:{}\n {} final_f1_score:{}\n auc:{}".format(drug_num, rfc.best_score_,f1_final,auc))

    # 列出重要基因
    gene_name = list(total_data)
    features = best_estimator.feature_importances_

    # 制作字典 用于储存 基因编号及其重要性
    dict_index_importance={}
    for index, importance in enumerate(features):
         dict_index_importance[index]=importance

    # 得到按重要性从大到小排序的基因
    dict_index_importance=sorted(dict_index_importance.items(),key=lambda x:x[1],reverse=True)
    # 得到最重要的十个基因
    new_dict={} 
    count=0
    for index,importance in dict_index_importance:
        if count<10:
            new_dict[index]=importance
        else:
            break
        count+=1

    dict_index_importance=new_dict

    result=['-药物名称：'+drug_name_list[drug_num]+'\n',
        '-best f1_score: {}\n'.format(rfc.best_score_),
        '-final_f1: {}\n'.format(f1_final),
        '-auc: {}\n'.format(auc),
        '-optim parameters:\n']+ ['---{} : {}\n'.format(para,val) for para,val in list(best_parameters.items()) ] + ['-important gene:\n'] + ['---gene[{}]:{} = {}\n'.format(index,gene_name[index], importance) for index, importance in new_dict.items()] + ['\n']
    

    with open("report_balanced.txt","a",encoding='utf-8') as f:
        f.writelines(result)   

    
def main():
    x_data=get_x_data()

    # 获取药物名列表
    drug_name_list=pd.read_csv('drug_sensity_num.csv',header=0).loc[0:24,'drug_name']
    
    # 对24个药物分别操作
    for i in range(0,24):
        run(x_data,i,drug_name_list)




if __name__ == "__main__":
    main()
    # drug_name_list=list(pd.read_csv('drug_sensity_num.csv',header=0).loc[0:24,'drug_name'])
    # print(drug_name_list[11])
    
    # list1=[1,2,3,4,5,6]
    # result=['测试','test','dasdadsasd\n']+['{},{}'.format(i,x) for i,x in enumerate(list1) ] + ['????']+['\n{}'.format(x) for x in list1]
    # with open("report.txt","a",encoding='utf-8') as f:
    #     f.writelines(result)  
 