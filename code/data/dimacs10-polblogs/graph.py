with open('out.dimacs10-polblogs', 'r') as fin:
    with open('edgelist.txt', 'w') as fout:
        for i, lines in enumerate(fin):
            if i>0:
                lines = lines.split('\t')
                string = str(int(lines[0])-1) + ' ' + str(int(lines[1])-1) + '\n'
                fout.write(string)
