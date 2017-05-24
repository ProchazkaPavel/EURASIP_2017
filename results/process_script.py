import pandas as pd

#l = ['scheme_b_HW_A1', 'scheme_b_HW_A', 'scheme_b_NS_A', 'scheme_b_TL_A', 'scheme_b_HW_B','scheme_b_NS_B']
l = ['scheme_a_HW_A1', 'scheme_a_HW_A', 'scheme_a_NS_A', 'scheme_a_TL_A','scheme_a_HW_B','scheme_a_NS_B', 'scheme_a_NS_B1', 'scheme_a_TL_B', 'scheme_a_TL_B1', 'scheme_a_HW_B1']

#writer = ExcelWriter('scheme_b.xlsx')
writer = pd.ExcelWriter('scheme_a.xlsx')
for i in l:
  df = pd.read_csv(i,header=1,sep=',') 
  df.to_excel(writer,i)

writer.save()

