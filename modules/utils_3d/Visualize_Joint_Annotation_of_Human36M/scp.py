import os
if __name__ == '__main__':
    f=open('/Users/Guo/Desktop/unmet.txt')
    f_w=open('/Users/Guo/Desktop/Sunmet.sh','w')
    line=f.readline()
    while line:
        line = line.strip('\n')
        x=line.split()[-1]
        print(x)
        w='sudo apt-get autoremove ' + x
        line=f.readline()
        if '(' in x or ')' in x:
            continue
        f_w.write(w+'\n')
    f.close()
    f_w.close()