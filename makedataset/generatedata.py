import random
import math
import numpy as np
import time

TRANSET_SIZE = 4000
TESTSET_SIZE = 100
GROUPPLAN_NUM = 10

def plan_to_vector_tree():#actual+log train diff dic
    fi1 = open("planset.txt", 'r')
    fo1 = open("planvector.txt", 'w')

    dic =   {'Parallel Seq Scan':'0,0,0,0,0', 'Seq Scan':'0,0,0,0,1', 'Bitmap Index Scan':'0,0,0,1,0', 'Index Scan':'0,0,0,1,1', 
            'Bitmap Heap Scan':'0,0,1,0,0', 'Merge Join':'0,0,1,0,1', 'Parallel Hash Join':'0,0,1,1,0', 'Hash Join':'0,0,1,1,1', 
            'Nested Loop':'0,1,0,0,0','Partial HashAggregate':'0,1,0,0,1', 'Finalize HashAggregate':'0,1,0,1,0','HashAggregate':'0,1,0,1,1', 
            'Index Only Scan':'0,1,1,0,0', 'Finalize GroupAggregate':'0,1,1,0,1', 'Partial GroupAggregate':'0,1,1,1,0', 'GroupAggregate':'0,1,1,1,1', 
            'Group':'1,0,0,0,0', 'Materialize':'1,0,0,0,1', 'Gather Merge':'1,0,0,1,0', 'Gather':'1,0,0,1,1', 'Limit':'1,0,1,0,0', 
            'Parallel Hash':'1,0,1,0,1', 'Hash':'1,0,1,1,0', 'Incremental Sort':'1,0,1,1,1', 'Sort':'1,1,0,0,0', 'Memoize':'1,1,0,0,1', 'Finalize Aggregate':'1,1,0,1,0'
            , 'Partial Aggregate':'1,1,0,1,1', 'Aggregate':'1,1,1,0,0', 'BitmapOr':'1,1,1,0,1'}
    key = ""
    nodelist = []
    maxrows = 0
    minrows = 99999999
    for line in fi1:
        str_in = ""
        str_out = ""
        if "actual time=" in line:
            str_in = line.split(" (")
            # str_in=[' Hash Join ', 'cost=11543.49..41153.16 rows=16149235 width=158)', 'actual time=73.046..1547.523 rows=16143826 loops=1)\n']
            kind = str_in[0]
            width = str_in[1].split(" ")
            width = width[2].replace("width=", "")
            width = width.replace(")", "")
            cost_row_str = str_in[1].split(" ")
            cost = cost_row_str[0].replace("cost=", "")
            row = str_in[1].split(" ")
            row = row[1].replace("rows=", "")
            if maxrows < int(row):
                maxrows = int(row)
            if minrows > int(row):
                minrows = int(row)
            location = ""
            #kind & location
            tag = 0
            for item in dic:
                if item in kind:
                    key = key + item + "," + str(row) + ","
                    str_out = str_out + dic[item] + ","
                    tag = 1
                    break
            if tag == 0:
                str_out = str_out + "1,1,1,1,1,"
            
            kind = kind.split(item)
            location = kind[0]
            location = location.replace("->  ", "")
            n = location.count(" ")
            temp = int(n / 6)
            if temp == 0:
                if n == 1:
                    location = "1"
                else:
                    location = "2"
            else:
                temp = temp + 2
                location = str(temp)
            #cost row width
            cost_temp = cost.split("..")
            # cost_1 = float(cost_temp[0])
            cost_2 = round(math.log10(float(cost_temp[1])%10000000000 + 0.000001), 2)
            # cost_f = str((cost_2 - cost_1))
            cost_f = str(cost_2)
            str_out =  str_out + row + "," + width +  "," + location + " " #cost -> width
            nodelist.append(str_out)
        if "Execution Time:" in line:
            for item in nodelist:
                node = ""
                item = item.split(",")
                item[5] = str((math.log(int(item[5])+1)-math.log(int(minrows)+1))/(math.log(int(maxrows)+1)-math.log(int(minrows)+1)))
                for num in item:
                    node = node + num + " "
                fo1.write(node[:-1])
            str_in = line.split(" ")
            key = key.replace(" ", "")
            str_out = str_in[3] + " " + str(key) + "\n"
            fo1.write("|" + str_out)
            key = ""
            nodelist = []
            maxrows = 0
            minrows = 99999999
        if "||||" in line:
            fo1.write("|||\n")

    fi1.close()
    fo1.close()

def generatedata_to_transformer_tree_random(fp1, fp2, fp3, size):#make temp_data, train_x and train_y
    fi1 = open(fp1, "r")
    fo1 = open(fp2, "w")
    fo2 = open(fp3, "w")
    planset = fi1.read()
    planset = planset.split("\n")
    planset = planset[:-1]
    dic = []
    no = []
    for i in range(0, size):
        plans = []
        num = 0
        while True:
            random.seed(time.time())
            tempplan = planset[random.randint(0, len(planset)-1)]
            if tempplan not in plans:
                plans.append(tempplan)
                num += 1
            if num == 10:
                break
        dic.clear()
        no.clear()
        for plan in plans:
            plan = plan.split(" |")
            y = plan[-1].split(" ")
            y = float(y[0])
            dic.append(str(plan[0]) + "|" + str(y))
        
        tempmin = 999999
        tempmax = 0
        for key_1 in dic:
            key_1 = key_1.split("|")

            if float(key_1[1]) > tempmax:
                tempmax = float(key_1[1])
            if float(key_1[1]) < tempmin:
                tempmin = float(key_1[1])

        for key_2 in dic:
            key_2 = key_2.split("|")
            temptime = float(key_2[1])
            flag = (temptime - tempmin) / (tempmax - tempmin)
            # if flag <= 0.05:
            #     no.append(0)
            # elif flag <= 0.25:
            #     no.append(1)
            # elif flag <= 0.5:
            #     no.append(2)
            # elif flag > 0.5:
            #     no.append(3)
            no.append(flag)
            
        for x, key in zip(no, dic):#write to train_*.txt
            key = key.split("|")
            key = str(key[0])
            n = len(key.split(" "))
            for i in range(n,512):#padding to 512
                key = key + " 0" 
            fo1.write(key + "\n")
            # for z in range(0, 4):
            #     if z == x:
            #         fo2.write("1 ")
            #     else:
            #         fo2.write("0 ")
            fo2.write(str(x)+"\n")
        fo2.write("\n")
        fo1.write("\n")
    fi1.close()
    fo1.close()
    fo2.close()

def generate_npz_tree(setsize, plannum, fp1, fp2, fp3):
    num_x = np.zeros((setsize, plannum, 64, 8), dtype = float)#216
    num_y = np.full((setsize, plannum, 1), 0, dtype = float)#216
    num_z = np.zeros((setsize, plannum, 192, 1), dtype=int)#216
    num_mask = np.full((setsize, plannum), False, dtype = bool)
    fi1 = open(fp1, "r")
    fi2 = open(fp2, "r")

    i_x = 0
    j_x = 0
    k_x = 0
    l_x = 0
    for line in fi1:
        if "\n" != line:
            num_mask[k_x][j_x] = True
            line = line.replace("\n", "")
            line = line.split(" ")
            i_x = 0
            l_x = 0
            for item in line:
                num_x[k_x][j_x][i_x][l_x] = float(item)
                l_x = l_x + 1
                if l_x == 8:
                    i_x = i_x + 1
                    l_x = 0
            j_x = j_x + 1
        else:
            k_x = k_x + 1
            j_x = 0

    i_y = 0
    j_y = 0
    k_y = 0

    for line in fi2:
        if line != "\n":
            line = line.replace(" \n", "")
            # line = line.split(" ")

            k_y = 0
            num_y[j_y][i_y][k_y] = float(line)
            # for item in line:
            #     if k_y == 4:
            #         break
            #     num_y[j_y][i_y][k_y] = float(item)
            #     k_y = k_y + 1
            # num_y[j_y][i_y][k_y] = float(1)
            i_y = i_y + 1

        else:
            i_y = 0
            j_y = j_y + 1

    for i in range(0, setsize):#
        for j in range(0, plannum):
            l = 0
            for k in range(0, 64):
                if num_x[i][j][k][7] < num_x[i][j][k+1][7]:
                    num_z[i][j][l][0] = k
                    num_z[i][j][l+1][0] = k + 1
                    flag = 0
                    for m in range(k+2, 64):
                        if num_x[i][j][k+1][7] == num_x[i][j][m][7]:
                            flag = 1
                            num_z[i][j][l+2][0] = m
                            break
                    if flag == 0:
                        num_z[i][j][l+2][0] = 63
                    l = l + 3
                if num_x[i][j][k][7] == 0:
                    break
            for m in range(l, 192):
                num_z[i][j][m][0] = 63

    
    np.savez(fp3, tx = num_x, ty = num_y, tz = num_z, mask = num_mask)
    fi1.close()
    fi2.close()


def make_dataset_tree():
    # plan_to_vector_tree()
    # generatedata_to_transformer_tree_random("planvector.txt", "train_x.txt", "train_y.txt", TRANSET_SIZE)
    # generatedata_to_transformer_tree_random("planvector.txt", "test_x.txt", "test_y.txt", TESTSET_SIZE)
    generate_npz_tree(TRANSET_SIZE, GROUPPLAN_NUM, "train_x.txt", "train_y.txt", "train.npz")
    generate_npz_tree(TESTSET_SIZE, GROUPPLAN_NUM, "test_x.txt", "test_y.txt", "test.npz")


def main():
    make_dataset_tree()

if __name__ == "__main__":
    main()