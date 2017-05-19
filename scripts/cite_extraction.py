import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

infile = open('Const_design.tex', 'r')
#outfile = open('citations', 'w')
cits = []
bibfiles = []
for line in infile:
    if 'cite' in line:
        while True:
            ind = line.find('cite')
            if ind >= 0:
                line = line[(ind + 4):]
                i_e = line.find('}')
                items = line[1:i_e]
                cits += items.split(',')
                line = line[(i_e + 1):]
                #outfile.write("\n\n")
            else:
                break
    if 'bibliography{' in line:
        ind = line.find('{')
        i_e = line.find('}')
        items = line[(ind+1):i_e]
        bibfiles += items.split(',')


infile.close()

cits_uniq = []
for c in cits:
    if not c in cits_uniq:
        cits_uniq.append(c)

for i, c in enumerate(cits_uniq):
    print i, c

res_dir = []
for c in cits_uniq:
    for f in bibfiles:
        with open(f+'.bib') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
            if c in bib_database.entries_dict.keys():
                res_dir.append(bib_database.entries_dict[c])
                break

db = BibDatabase()
db.entries = res_dir

writer = BibTexWriter()
with open('bib/res_bib.bib', 'w') as bibfile:
    bibfile.write(writer.write(db))

        
#outfile.close()
