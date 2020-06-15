import numpy as np 
from utils import read_file, output_solution
from aco import ACO, Graph

# plot(route)

def main():

    file_path = 'C:\\Users\\zy3\\work\\workspace\\rideco\\Simpsons.txt'
    line_num, name, route, graph_dis, graph_time, task_list = read_file(file_path)

    aco = ACO(40, 50, 1.0, 5.0, 0.5, 10, 1, len(name))
    graph = Graph(np.shape(graph_time)[0],line_num,graph_time,task_list)
    path, cost, time_sequence = aco.solve(graph)
    print(path)
    print(cost)
    print(time_sequence)
    print(len(path))
    print(len(time_sequence))

    output_solution(path, time_sequence, task_list, file_path, name)
    # for i in range(line_num):
    #     print(task_list[i])
if __name__ == '__main__':
    main()

'''
[[5, 14, 6, 15, 15, 6, 16, 7, 10, 1, 11, 2, 11, 7, 8, 12, 20, 2, 16, 21, 17, 3, 21, 17, 3, 18, 22, 4], [9, 0, 19, 20, 1, 10, 8, 12, 13, 0]]
[]
[[[1, 'pickup'], [3, 'pickup'], [1, 'take off'], [3, 'take off'], [3, 'pickup'], [1, 'pickup'], [3, 'take off'], [1, 'take off'], [2, 'pickup'], [0, 'pickup'], [2, 'take off'], [0, 'take off'], [2, 'pickup'], [1, 'pickup'], [1, 'take off'], [2, 'take off'], [4, 'pickup'], [0, 'pickup'], [3, 'pickup'], [4, 'take off'], [3, 'take off'], [0, 'take off'], [4, 'pickup'], [3, 'pickup'], [0, 'pickup'], [3, 'take off'], [4, 'take off'], [0, 'take off']], [[2, 'pickup'], [0, 'pickup'], [4, 'pickup'], [4, 'take off'], [0, 'take off'], [2, 'take off'], 
[1, 'pickup'], [2, 'pickup'], [2, 'take off'], [1, 'take off']]]
'''