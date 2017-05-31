all:
	latex Const_design.tex && bibtex Const_design && makeindex Const_design.nlo -s nomencl.ist -o Const_design.nls && latex Const_design.tex && latex Const_design.tex && dvipdf Const_design.dvi
	latex Const_design_final.tex && bibtex Const_design_final && makeindex Const_design_final.nlo -s nomencl.ist -o Const_design_final.nls && latex Const_design_final.tex && latex Const_design_final.tex && dvipdf Const_design_final.dvi
	cd response_letter && make

clean:
	rm Const_design.[abdln]* nomencl.ist 
	rm response_letter/response_letter.[adl]*


