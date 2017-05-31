infile = open('../Const_design.tex', 'r')
outfile_fig = open('../all_figures.tex', 'w')
outfile_tab = open('../all_tables.tex', 'w')
extract_block_fig = False
extract_block_tab = False
for line in infile:
    if ('begin{figure}' in line) or ('begin{figure*}' in line):
        extract_block_fig = True
    if 'begin{table}' in line:
        extract_block_tab = True
    if extract_block_fig:
        outfile_fig.write(line)
    if extract_block_tab:
        outfile_tab.write(line)
    if ('end{figure}' in line) or ('end{figure*}' in line):
        extract_block_fig = False
        outfile_fig.write("\n\n")
    if 'end{table}' in line:
        extract_block_tab = False
        outfile_tab.write("\n\n")
infile.close()
outfile_fig.close()
outfile_tab.close()
