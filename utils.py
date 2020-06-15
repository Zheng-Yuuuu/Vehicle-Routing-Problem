import numpy as np 
from datetime import datetime


def read_file(filename):
    name = []
    route = []
    node = []
    with open(filename) as fp: 
        Lines = fp.readlines() 
        line_num = (int(Lines[0]))
        task_list = [[] for i in range(line_num)]
        count = 0
        for line in Lines[1:line_num+1]:
            info = line.split()
            i_x = int(info[4])
            i_y = int(info[5])
            o_x = int(info[6])
            o_y = int(info[7])
            if [i_x,i_y] not in node:
                node.append([i_x,i_y])
            if [o_x,o_y] not in node:
                node.append([o_x,o_y])
            if info[0] in name:
                name_pos = name.index(info[0])
                route[name_pos].append([i_x,i_y,o_x,o_y])
            else:
                name.append(info[0])
                route.append([])
                route[-1].append([i_x,i_y,o_x,o_y])
            
            task_list[count].append(name.index(info[0]))
            task_list[count].append(node.index([i_x,i_y]))
            task_list[count].append(node.index([o_x,o_y]))
            FMT = '%H:%M'
            time_FMT_begin = datetime.strptime(info[2], FMT)-datetime.strptime('00:00', FMT)
            hours_begin = (time_FMT_begin.total_seconds()/3600)
            time_FMT_end = datetime.strptime(info[3], FMT)-datetime.strptime('00:00', FMT)
            hours_end = (time_FMT_end.total_seconds()/3600)
            task_list[count].append((hours_begin))
            task_list[count].append((hours_end))
            count = count + 1

    graph_dis = np.zeros((len(node),len(node)))
    graph_time = np.zeros((len(node),len(node)))
    for i in range(len(node)):
        for j in range(len(node)):
            graph_dis[i,j] = np.sqrt((node[i][0]-node[j][0])**2+(node[i][1]-node[j][1])**2)/5
            graph_time[i,j] = graph_dis[i,j]/60

    return line_num, name, route, graph_dis, graph_time, task_list


def output_solution(path, time_seq, task_list, filename, name):

    task = []
    passenger = []
    
    task_num = len(task_list)

    i = 0
    count_route = 0
    for i in range(len(time_seq)):
        if time_seq[i]==-1:
            count_route += 1
    
    route = [[] for i in range(count_route)]
    passenger_log = [[] for i in range(count_route)]
    time_list = [[] for i in range(count_route)]
    route_index = -1
    for i in range(len(time_seq)):
        if time_seq[i]==-1:
            route_index += 1
        
        else:
            node = path[i]
            route[route_index].append(node)
            time_stamp = time_seq[i]

            for n in range(task_num):
                if task_list[n][1] == node:
                    if time_stamp>=task_list[n][3] and time_stamp<=task_list[n][4]:
                        passenger.append(task_list[n][0])
                        passenger_log[route_index].append([task_list[n][0],'pickup'])
                        time_list[route_index].append(time_stamp)

            for n in range(task_num):
                if task_list[n][2] == node and task_list[n][0] in passenger:
                    if time_stamp>task_list[n][3] and time_stamp<=task_list[n][4]:
                        passenger.remove(task_list[n][0])
                        passenger_log[route_index].append([task_list[n][0],'take off'])
                        time_list[route_index].append(time_stamp)
    print(route)
    print(passenger)
    print(passenger_log)

    with open(filename,'a') as fp:
        fp.write('\n')
        for i in range(len(route)):
            fp.write('vehicle '+str(i)+ ': ')
            for j in range(len(route[i])):
                t = time_list[i][j]*60
                h = str(int(t//60))
                m = str(int(t%60)) 
                if len(h) == 1:
                    h = '0'+h
                if len(m) == 1:
                    m = '0'+m
                d = h + ':' + m
                fp.write('-> '+'node: '+ str(route[i][j]) +' ' + name[passenger_log[i][j][0]] + ' ' + passenger_log[i][j][1] + ' ' + d+ ' ')
            fp.write('\n')

            




