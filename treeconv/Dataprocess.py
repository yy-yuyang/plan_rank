import numpy as np
import math
import random
import time

class dataprocess():
    def __init__(self, plan):

        self.dic = {'Parallel Seq Scan':'0,0,0,0,0', 'Seq Scan':'0,0,0,0,1', 'Bitmap Index Scan':'0,0,0,1,0', 'Index Scan':'0,0,0,1,1', 
            'Bitmap Heap Scan':'0,0,1,0,0', 'Merge Join':'0,0,1,0,1', 'Parallel Hash Join':'0,0,1,1,0', 'Hash Join':'0,0,1,1,1', 
            'Nested Loop':'0,1,0,0,0','Partial HashAggregate':'0,1,0,0,1', 'Finalize HashAggregate':'0,1,0,1,0','HashAggregate':'0,1,0,1,1', 
            'Index Only Scan':'0,1,1,0,0', 'Finalize GroupAggregate':'0,1,1,0,1', 'Partial GroupAggregate':'0,1,1,1,0', 'GroupAggregate':'0,1,1,1,1', 
            'Group':'1,0,0,0,0', 'Materialize':'1,0,0,0,1', 'Gather Merge':'1,0,0,1,0', 'Gather':'1,0,0,1,1', 'Limit':'1,0,1,0,0', 
            'Parallel Hash':'1,0,1,0,1', 'Hash':'1,0,1,1,0', 'Incremental Sort':'1,0,1,1,1', 'Sort':'1,1,0,0,0', 'Memoize':'1,1,0,0,1', 'Finalize Aggregate':'1,1,0,1,0'
            , 'Partial Aggregate':'1,1,0,1,1', 'Aggregate':'1,1,1,0,0', 'BitmapOr':'1,1,1,0,1'}
        self.str_out = "["
        #plan lines
        maxrows = 0
        minrows = 9999999
        for line in plan:
            if "cost=" in line:
                str_in = line.split(" (")
                # str_in=[' Hash Join ', 'cost=11543.49..41153.16 rows=16149235 width=158)', 'actual time=73.046..1547.523 rows=16143826 loops=1)\n']
                row = str_in[1].split(" ")
                row = row[1].replace("rows=", "")
                if maxrows < int(row):
                    maxrows = int(row)
                if minrows > int(row):
                    minrows = int(row)
        for line in plan:
            str_in = ""
            if "cost=" in line:
                str_in = line.split(" (")
                # cost
                kind = str_in[0]
                width = str_in[1].split(" ")
                width = width[2].replace("width=", "")
                width = width.replace(")", "")
                width = width.replace("\n", "")
                cost_row_str = str_in[1].split(" ")
                cost = cost_row_str[0].replace("cost=", "")
                row = cost_row_str[1].replace("rows=", "")
                row = str((math.log(int(row)+1)-math.log(int(minrows)+1))/(math.log(int(maxrows)+1)-math.log(int(minrows)+1)))
                # actual
                # kind = str_in[0]
                # width = str_in[1].split(" ")
                # width = width[2].replace("width=", "")
                # width = width.replace(")", "")
                # cost_row_str = str_in[2].split(" ")
                # cost = cost_row_str[1].replace("time=", "")
                # row = cost_row_str[2].replace("rows=", "")
                location = ""
                #kind & location
                tag = 0
                for item in self.dic:
                    if item in kind:
                        self.str_out = self.str_out + self.dic[item] + ","
                        tag = 1
                        break
                if tag == 0:
                    self.str_out = self.str_out + "1,1,1,1,1,"

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
                cost_f = str((cost_2))
                # self.str_out = self.str_out + location + "," + row + "," + width + "," + cost_f + "]\n"
                self.str_out = self.str_out +  row + "," + width + "," + location + "]\n" #cost->width
                    

    def vector_change(self):
        # print(self.str_out)
        self.str_out = self.str_out.replace("[", "")
        self.str_out = self.str_out.replace("]", "")
        self.str_out = self.str_out.replace("\n", ",")
        self.str_out = self.str_out[:-1]
        str_ = self.str_out.split(",")
        # print(str_)
        data = []
        # print(self.str_out)
        for item in str_:
            # print(item)
            data.append(float(item))
        data = np.array(data)
        i = data.size
        for j in range(i, 512):
            data = np.append(data, 0)

        return data
        
        
def get_x_y_input_tree():
    result = []
    for x in range(1, 2):
        path1 = "/opt/yy/Experiments-Final/PR/RankJOtest/sql/josql/tp" + str(x) + "/test"
        for y in range(1, 2):
            # print(x,y)
            path2 = path1 + str(y) + ".txt"
            file = open(path2, 'r')
            plan_all = file.readlines()
            file.close()
            x_1temp = []
            x_2temp = []
            x_1 = []
            x_2 = []
            plan = []
            ac_time = []
            for item in plan_all:
                if "psql" in item:
                    plan = []
                    continue
                if item == "\n" and plan != []:
                    # raw_data = dataprocess_mysql(plan)
                    x1 = np.empty((64, 8), dtype=float)
                    x2 = np.empty((192, 1), dtype=int)
                    raw_data = dataprocess(plan)
                    data = raw_data.vector_change()
                    i = 0
                    j = 0
                    for k in range(0, 512):
                        x1[i][j] = float(data[k])
                        j += 1
                        if j == 8:
                            i += 1
                            j = 0
                    i = 0
                    for k in range(0, 64):
                        if x1[k][7] < x1[k+1][7]:
                            x2[i][0] = k
                            x2[i+1][0] = k + 1
                            flag = 0
                            for m in range(k+2, 64):
                                if x1[k+1][7] == x1[m][7]:
                                    flag = 1
                                    x2[i+2][0] = m
                                    break
                            if flag == 0:
                                x2[i+2][0] = 63
                            i = i + 3
                        if x1[k][7] == 0:
                            break
                    for m in range(i, 192):
                        x2[m][0] = 63
                    x_1temp.append(x1)
                    x_2temp.append(x2)
                    plan = []

                else:
                    plan.append(item)
                    if "Execution Time: " in item:
                        time = item.split("Execution Time: ")
                        time = time [1]
                        time = time.replace(" ms\n", "")
                        ac_time.append(float(time))
            #actual_rank
            idx = np.argsort(ac_time)
            # print(idx)
            rk_list = [0 for yy in range(0, len(idx))]
            rk = 0
            for yy in idx:
                rk_list[int(yy)] = rk
                rk += 1
            print(rk_list)
            # print(ac_time)

            x_1.append(x_1temp)
            x_2.append(x_2temp)
            result.append([x_1,x_2])
            x_1 = []
            x_2 = []
            x_1temp = []
            x_2temp = []

    return result


def get_x_y_input_tree_():
    result = []
    file = open("/opt/yy/Experiments-Final/PR/RankJOtest/data2/f_plan_explain_analyse_test.txt", 'r')
    planlist = []
    tempplan = []
    for line in file:
        if "cost=" in line:
            tempplan.append(line)
        if "Execution Time" or "Planning Time:" in line:
            tempplan.append(line)
            if "Execution Time" in line:
                planlist.append(tempplan)
                tempplan = []

    _result = []
    for y in range(1, 5):
        x_1temp = []
        x_2temp = []
        x_1 = []
        x_2 = []
        ac_time = []
        for count in range(0, 10):
            while True:
                random.seed(time.time())
                plan = planlist[random.randint(0, len(planlist)-1)]
                if len(plan) < 62:
                    break
            x1 = np.empty((64, 8), dtype=float)
            x2 = np.empty((192, 1), dtype=int)
            raw_data = dataprocess(plan)
            data = raw_data.vector_change()
            i = 0
            j = 0
            for k in range(0, 512):
                x1[i][j] = float(data[k])
                j += 1
                if j == 8:
                    i += 1
                    j = 0
            i = 0
            for k in range(0, 64):
                if x1[k][7] < x1[k+1][7]:
                    x2[i][0] = k
                    x2[i+1][0] = k + 1
                    flag = 0
                    for m in range(k+2, 64):
                        if x1[k+1][7] == x1[m][7]:
                            flag = 1
                            x2[i+2][0] = m
                            break
                    if flag == 0:
                        x2[i+2][0] = 63
                    i = i + 3
                if x1[k][7] == 0:
                    break
            for m in range(i, 192):
                x2[m][0] = 63
            x_1temp.append(x1)
            x_2temp.append(x2)
            for item in plan:
                if "Execution Time: " in item:
                    temptime = item.split("Execution Time: ")
                    temptime = temptime[1]
                    temptime = temptime.replace(" ms\n", "")
                    ac_time.append(float(temptime))
                    break
        #actual_rank
        idx = np.argsort(ac_time)
        # print(idx)
        rk_list = [0 for yy in range(0, len(idx))]
        rk = 0
        for yy in idx:
            rk_list[int(yy)] = rk
            rk += 1
        # print(rk_list)
        print(ac_time)
        _result.append(ac_time)

        x_1.append(x_1temp)
        x_2.append(x_2temp)
        result.append([x_1,x_2])
        x_1 = []
        x_2 = []
        x_1temp = []
        x_2temp = []

    return result, _result

def plantovector(planlist):
    result = []
    x_1temp = []
    x_2temp = []
    x_1 = []
    x_2 = []
    for plan in planlist:
        x1 = np.empty((64, 8), dtype=float)
        x2 = np.empty((192, 1), dtype=int)
        raw_data = dataprocess(plan)
        data = raw_data.vector_change()
        i = 0
        j = 0
        for k in range(0, 512):
            x1[i][j] = float(data[k])
            j += 1
            if j == 8:
                i += 1
                j = 0
        i = 0
        for k in range(0, 64):
            if x1[k][7] < x1[k+1][7]:
                x2[i][0] = k
                x2[i+1][0] = k + 1
                flag = 0
                for m in range(k+2, 64):
                    if x1[k+1][7] == x1[m][7]:
                        flag = 1
                        x2[i+2][0] = m
                        break
                if flag == 0:
                    x2[i+2][0] = 63
                i = i + 3
            if x1[k][7] == 0:
                break
        for m in range(i, 192):
            x2[m][0] = 63
        x_1temp.append(x1)
        x_2temp.append(x2)

    x_1.append(x_1temp)
    x_2.append(x_2temp)
    result.append([x_1,x_2])

    return result

def makeplangroup():
    result = []
    file = open("/opt/yy/Experiments-Final/PCG/IMDB/m5/test/test0_sql1/test0_testsqlrunresult.txt", 'r')
    planlist = []
    tempplan = []
    for line in file:
        if "cost=" in line:
            tempplan.append(line)
        if "Planning Time:" in line:
            planlist.append(tempplan)
            tempplan = []
    _result = []
    count = 0
    x_1temp = []
    x_2temp = []
    for plan in planlist:
        x_1 = []
        x_2 = []
        ac_time = []
        x1 = np.empty((64, 8), dtype=float)
        x2 = np.empty((192, 1), dtype=int)
        raw_data = dataprocess(plan)
        data = raw_data.vector_change()
        i = 0
        j = 0
        for k in range(0, 512):
            x1[i][j] = float(data[k])
            j += 1
            if j == 8:
                i += 1
                j = 0
        i = 0
        for k in range(0, 64):
            if x1[k][7] < x1[k+1][7]:
                x2[i][0] = k
                x2[i+1][0] = k + 1
                flag = 0
                for m in range(k+2, 64):
                    if x1[k+1][7] == x1[m][7]:
                        flag = 1
                        x2[i+2][0] = m
                        break
                if flag == 0:
                    x2[i+2][0] = 63
                i = i + 3
            if x1[k][7] == 0:
                break
        for m in range(i, 192):
            x2[m][0] = 63
        x_1temp.append(x1)
        x_2temp.append(x2)
        count += 1
        #     for item in plan:
        #         if "Execution Time: " in item:
        #             temptime = item.split("Execution Time: ")
        #             temptime = temptime[1]
        #             temptime = temptime.replace(" ms\n", "")
        #             ac_time.append(float(temptime))
        #             break
        # #actual_rank
        # idx = np.argsort(ac_time)
        # # print(idx)
        # rk_list = [0 for yy in range(0, len(idx))]
        # rk = 0
        # for yy in idx:
        #     rk_list[int(yy)] = rk
        #     rk += 1
        # # print(rk_list)
        # print(ac_time)
        # _result.append(ac_time)
        if count == 10:
            x_1.append(x_1temp)
            x_2.append(x_2temp)
            result.append([x_1,x_2])
            count = 0
            x_1 = []
            x_2 = []
            x_1temp = []
            x_2temp = []

    return result

def Plans2Vectors(plans):
    plan_list = []
    for plan in plans:
        planstr = ""
        for line in plan:
            line = line[0]
            planstr = planstr + line + "\n"
        nodelist = planstr.split("\n")
        plan_list.append(nodelist)
    
    result = []
    x_plans = []
    x_idxs = []
    for plan in plan_list:
        x_plan = np.empty((64, 8), dtype=float)
        x_idx = np.empty((192, 1), dtype=int)
        raw_data = dataprocess(plan)
        data = raw_data.vector_change()
        i = 0
        j = 0
        for k in range(0, 512):
            x_plan[i][j] = float(data[k])
            j += 1
            if j == 8:
                i += 1
                j = 0
        i = 0
        for k in range(0, 64):
            if x_plan[k][7] < x_plan[k+1][7]:
                x_idx[i][0] = k
                x_idx[i+1][0] = k + 1
                flag = 0
                for m in range(k+2, 64):
                    if x_plan[k+1][7] == x_plan[m][7]:
                        flag = 1
                        x_idx[i+2][0] = m
                        break
                if flag == 0:
                    x_idx[i+2][0] = 63
                i = i + 3
            if x_plan[k][7] == 0:
                break
        for m in range(i, 192):
            x_idx[m][0] = 63
        x_plans.append(x_plan)
        x_idxs.append(x_idx)
    # x_plans.append(x_1temp)
    # x_idxs.append(x_2temp)
    
    return [x_plans], [x_idxs]

def main():
    print(get_x_y_input_tree())

if __name__ == "__main__":
    main()

