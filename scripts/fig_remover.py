infile = open('../Const_design.tex', 'r')
outfile = open('../Const_design_final.tex', 'w')
fig_list = open('fig_list', 'w')
for line in infile:
    if not 'includegraphics' in line:
        outfile.write(line)
    else:
        st_ind = line.find('fig')
        act_fig = line[st_ind:-2]+'\n'
        fig_list.write(act_fig)
infile.close()
outfile.close()
fig_list.close()
