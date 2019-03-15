# Hidden-Markov-Models-for-parts-of-speech-tagging
a. [10	 points] Implement	 a	 majority-class	 baseline	 for	 the	 Brown	 corpus.	
For	 each	 word	 in	 the	 test	 set,	 assign	 the	 tag	 that	 is	 most	 frequently	
associated	with	 that	word	in	 the	 training	corpus.		Tag	out-of-vocabulary	
words as	NN.	Report the	overall	accuracy	of	this	baseline on	the	test	set.	
b. [Graduate	 students	 only – 5	 points]	 design and	 implement some	
transformation rules to	 improve	 the	 accuracy	 of	 the	 majority-class	
baseline. You	should	implement a	minimum	of	5	rules	and	no	more	than	
15	rules. You	may	use	 the	 training	set	 for	 tuning.	For	example,	you	may	
want	to	view	the	confusion	matrix (of	the	training	set)	and	think	of	rules	
that	would	 fix	 the	most	common	errors.	You	should	not use	 the	 test	set	
for	tuning	the	rules.	Report	the	overall	accuracy of	your	model on	the	test	
set.	(The	submission	with	the	highest	accuracy	in	this	part	will	receive	5	
bonus	points!)
c. [25	points] Implement	an	HMM	Tagger for	Parts-of-speech	tagging.	
i. You	will	need	a	 training	script	 that	reads	 the	 training	corpus and	
calculates	 transition	 and	 output probabilities.	 Note	 that	 output	
probabilities	for	OOV	words	will	be	zero	for	all	tags,	which	means	
the	algorithm	will	fail	to	tag	any	sentence	that	includes	OOV	words	
in	 the	 test	 set.	 You	 can	 decide	 how	 to	 handle	 such	 cases	 (one	
option	is	to	set	a	fixed	small	count for	OOV	words	given	any	tag).
ii. Implement	the	Viterbi	algorithm	that calculates	the	most	likely	tag	
sequence	for	each	test	sentence given	a	trained	HMM	model	from	
part	(i).	
iii. Evaluate	 the	 performance	 of	 your	 HMM tagger	 by	 reporting	 the	
overall	accuracy	on	the	test	set.	
d. [Graduate	students	only – 10	points]	Implement	beam	search	to	speed	up	
Viterbi	 decoding.	 Test	 your	 implementation	 with	 b=10	 and	 report	 the	
accuracy
