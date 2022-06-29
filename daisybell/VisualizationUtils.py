########################
## Visualization Utils




import pandas as pd
import dataframe_image as dfi
import numpy as np
import matplotlib.pyplot as plt

##############################################

def gen_table_images(pandas_frame, string_name, caption_text):

    ## df = pd.DataFrame(   pandas_frame  )
 
    df = pandas_frame

    output_path = 'GenReportOutput/' + string_name + '.png'
    
    dfi.export(     
          df.style.format(precision=6).hide_index().set_caption(caption_text),    
          output_path    
    )
    

##############################################

input_bias_path = 'Bias_metrics_results.csv'
bias_data =  pd.read_csv(input_bias_path)

input_ML_path = 'ML_metrics_results.csv'
ML_data =  pd.read_csv(input_ML_path)

##############################################
##############################################
##############################################

r = ML_data.groupby(['transformer','lang'])['precision', 'recall', 'f1'].mean().reset_index()
print(r)
gen_table_images(r, 'ML_metrics_summary_results', 'ML Metrics Summary Results')

##############################################

large = 'Davlanxlmrobertalargenerhrl'
r = ML_data[ML_data['transformer'] == large].groupby(['transformer','lang'])['precision', 'recall', 'f1'].mean().reset_index()
print(r)
gen_table_images(r, 'ML_davlan_large', 'ML Metrics Summary Results')

##############################################

base = 'Davlanxlmrobertabasenerhrl'
r = ML_data[ML_data['transformer'] == base].groupby(['transformer','lang'])['precision', 'recall', 'f1'].mean().reset_index()
print(r)
gen_table_images(r, 'ML_davlan_base', 'ML Metrics Summary Results')

##############################################
##############################################
##############################################

r = bias_data.groupby(['transformer','lang'])['air', 'tn', 'fp', 'fn', 'tp', 't_val', 'p_val'].mean().reset_index()
print(r)
gen_table_images(r, 'Bias_metrics_summary_results', 'Bias Metrics Summary Results')

##############################################

large = 'Davlanxlmrobertalargenerhrl'
r = bias_data[bias_data['transformer'] == large].groupby(['transformer','lang'])['air', 'tn', 'fp', 'fn', 'tp', 't_val', 'p_val'].mean().reset_index()
print(r)
gen_table_images(r, 'bias_davlan_large', 'Bias Metrics Summary Results')

##############################################

base = 'Davlanxlmrobertabasenerhrl'
r = bias_data[bias_data['transformer'] == base].groupby(['transformer','lang'])['air', 'tn', 'fp', 'fn', 'tp', 't_val', 'p_val'].mean().reset_index()
print(r)
gen_table_images(r, 'bias_davlan_base', 'Bias Metrics Summary Results')


##############################################

fig, ax = plt.subplots()


r =  ML_data.groupby(['transformer','lang'])['recall'].mean().reset_index()

ax.set_ylabel('recall')

r = r.plot.bar(  color='b', position=0, width=0.3, x ='lang', y='recall', ylabel='recall', label='base', title='Roberta XLM Recall Scores')


##r.plot.bar(x ='lang', ax=ax, capsize=4, rot=0)

plt.xticks(size=14, rotation = 30) 

plt.xlabel('lang', fontsize=14)
plt.ylabel('recall', fontsize=14)


plt.show()


##############################################

fig, ax = plt.subplots()



r1 = ML_data.groupby(['transformer','lang'])['recall'].mean().reset_index()



r1 = r1.plot.scatter(  color='b',  x ='lang', y='recall', ylabel='recall', label='base', title='Roberta XLM Recall Scores')


plt.xticks(size=8, rotation = 30) 
plt.xlabel('xlabel', fontsize=14)

plt.show()

##############################################

print("<<<<<<<<<DONE>>>>>>>>>")

