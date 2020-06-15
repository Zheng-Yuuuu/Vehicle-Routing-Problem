import random
import numpy as np
import numpy.random as npr
from sympy.utilities.iterables import multiset_permutations



npr.seed(10)

class Graph(object):
    def __init__(self, num_node, num_task, graph_time,task_list ):
        """
        :param graph_time: time between nodes
        :param num_node: number of nodes
        """
        self.graph_time = graph_time
        self.num_node = num_node
        self.num_task = num_task
        self.task_list = task_list

        self.pheromone_off_diagnal = np.ones((num_node,num_node))/num_node
        self.pheromone_on_diagnal = np.ones((num_task,num_node))/num_node # at most num_task iterations



class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int, num_person: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy
        self.num_person = num_person

    def _update_pheromone(self, graph: Graph, ants: list):

        graph.pheromone_off_diagnal *= self.rho 
        graph.pheromone_on_diagnal *= self.rho

        for ant in ants:
            graph.pheromone_off_diagnal += ant.pheromone_off_diagnal_delta
            graph.pheromone_on_diagnal += ant.pheromone_on_diagnal_delta

    # noinspection PyProtectedMember
    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_cost = self.num_person+1
        best_solution = []
        time_sequence = []
        for gen in range(self.generations):
            print(gen,'gen')
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            a_num = 0
            for ant in ants:
                print(a_num,'ant')
                a_num += 1
                a_select = 0
                while np.prod(ant.task_flag)!=1:

                    ant._select_next()
                    # print(a_select,'ant select')
                    a_select += 1
                    if ant.tabu[-1] == -1:
                        ant.total_cost += 1
                if ant.total_cost<best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                    time_sequence = [] + ant.time_sequence

                 
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost, time_sequence


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):

        self.init_time = -1 # initial time
        self.time_ = -1 # current time
        self.colony = aco
        self.graph = graph
        self.total_cost = 1
        self.task_flag = np.zeros(graph.num_task)
        self.passenger_list = []
        self.tabu = [-1]  # tabu list
        self.pheromone_off_diagnal_delta = np.zeros((self.graph.num_node,self.graph.num_node))
        self.pheromone_on_diagnal_delta = np.zeros((self.graph.num_task,self.graph.num_node))
        self.current = -1
        self.time_sequence = []
        self.time_sequence.append(self.time_)


    def check_task_todo(self, task_id):


        if self.time_>self.graph.task_list[task_id][4]:
            return False
        # print(self.time_,' self.time in check task')
        dest_time_list = []
        for i in range(len(self.passenger_list)):
            t = self.passenger_list[i][2]
            dest_time_list.append(t)
        
        max_dest_time = max(dest_time_list)

        if self.graph.task_list[task_id][3]>max_dest_time: # can chooes this task later; the task is not urgent
            return False

        if self.task_flag[task_id]==2:
            print('check_task_todo wrong 1')
            exit()
            return False

        else:
            passenger_list_ = self.passenger_list+[[self.graph.task_list[task_id][0],self.graph.task_list[task_id][2] , 
            self.graph.task_list[task_id][4] , task_id]]   
        
        depart_time = self.graph.task_list[task_id][3]
        depart_node = self.graph.task_list[task_id][1]

        time_ = max(depart_time,self.time_+ self.graph.graph_time[self.current,depart_node])
        
        len_l = len(passenger_list_)
        a = [ i for i in range(len_l)]
        for p in multiset_permutations(a):
            flag = 1
            time__ = time_
            node_ = depart_node
            for i in range(len_l):
                next_node = passenger_list_[p[i]][1]
                time__ = self.graph.graph_time[node_,next_node]+time__
                if time__>passenger_list_[p[i]][2]:
                    flag = 0
                    break 
                node_ = next_node
            if flag == 1:
                return True 
        return False
                
    
    def _select_next(self):

        # print(self.passenger_list,'passenger list', self.time_, 'self.time in select next')

        task_todo = []
        for i in range(self.graph.num_task):
            if self.task_flag[i]!=1:
                task_todo.append(i)

        task_todo = np.asarray(task_todo, dtype=int)

        task_index = []
        
        if self.time_ == self.init_time:
            # new start 
            task_index = []
            for i in range(self.graph.num_task):
                if self.task_flag[i]!=1:
                    if self.task_flag[i]==2:
                        print('new start task-flag = 2 wrong 1')
                        exit()
                    task_index.append(i)
            
            task_id = self.choose_new_task(task_index)

            self.task_flag[task_id]=2
            self.passenger_list.append([self.graph.task_list[task_id][0],self.graph.task_list[task_id][2],self.graph.task_list[task_id][4] , task_id])
            
            self.time_ = self.graph.task_list[task_id][3]
            self.current = self.graph.task_list[task_id][1]
            self.tabu.append(self.current)
            self.time_sequence.append(self.time_)
        
        elif self.passenger_list==[]:
            
            # chooes new task or restart
            task_index = []
            for i in range(self.graph.num_task):
                if self.task_flag[i]!=1:
                    if self.task_flag[i]==2:
                        print('task-flag = 2 wrong 1')
                        exit()
                    
                    dest_node = self.graph.task_list[i][2]
                    depart_node = self.graph.task_list[i][1]
                    dest_time = self.graph.task_list[i][4]
                    transition_time = self.graph.graph_time[self.current,depart_node]+self.graph.graph_time[depart_node,dest_node]
                    if dest_time-transition_time>=self.time_:
                        task_index.append(i)
            if task_index==[]:
                self.restart()
                
            else:
                task_id = self.choose_new_task(task_index)
                self.new_progress_task(task_id)
        
        elif len(self.passenger_list)==3:
            # choose the task in progress
            task_id = self.choose_best_task()
            self.in_progress_task(task_id)

        else:
            # choose the potential new task
            
            task_index = []
            for i in range(self.graph.num_task):
                
                if self.task_flag[i]==0:
                    if self.check_task_todo(i):
                        task_index.append(i)
            
            if task_index == []:
                # choose the task in progress

                task_id = self.choose_best_task()
                self.in_progress_task(task_id)
            
            else:
                task_id = self.choose_new_task(task_index)
                # choose the best task in progress
                self.new_progress_task(task_id)
                 
            

    def in_progress_task(self,task_id):

        node = self.graph.task_list[task_id][2]
        del_list = []
        for i in range(len(self.passenger_list)):
            if self.passenger_list[i][1]==node:
                del_list.append(i)
        
        for i in range(len(del_list)-1,-1,-1):
            l = self.passenger_list.pop(del_list[i])
            self.task_flag[l[3]]=1
                
        self.time_ += self.graph.graph_time[self.current,self.graph.task_list[task_id][2]]
        self.current = self.graph.task_list[task_id][2]
        self.tabu.append(self.current)
        self.time_sequence.append(self.time_)
    
    def new_progress_task(self,task_id):

        self.task_flag[task_id]=2
        self.passenger_list.append([self.graph.task_list[task_id][0],self.graph.task_list[task_id][2],self.graph.task_list[task_id][4] , task_id])

        depart_time = self.graph.task_list[task_id][3]
        
        self.time_ = max(depart_time,self.time_+ self.graph.graph_time[self.current,self.graph.task_list[task_id][1]])
        self.current = self.graph.task_list[task_id][1]
        self.tabu.append(self.current)
        self.time_sequence.append(self.time_)

    
    def restart(self):
        self.time_ = self.init_time
        self.time_sequence.append(self.time_)
        self.tabu.append(-1)
        self.current = -1

    def check_order(self):
        pass

    
    def choose_best_task(self,task_id = -1):
        
        passenger_list = []

        if task_id == -1:
            passenger_list = []+self.passenger_list
            time_ = self.time_
        else: 
            passenger_list = self.passenger_list+[[self.graph.task_list[task_id][0],self.graph.task_list[task_id][2], 
            self.graph.task_list[task_id][4] , task_id]]
            depart_time = self.graph.task_list[task_id][3]
            depart_node = self.graph.task_list[task_id][1]
            time_ = max(depart_time,self.time_+ self.graph.graph_time[self.current,depart_node])
            print('choose_best_task wrong')
            exit()

        solution = []
        path = []
        len_l = len(passenger_list)
        a = [ i for i in range(len_l)]
        for p in multiset_permutations(a):
            time__ = time_
            node_ = self.current
            for i in range(len_l):
                next_node = passenger_list[p[i]][1]
                time__ = self.graph.graph_time[node_,next_node]+time__
                if time__>passenger_list[p[i]][2]:
                    time__  = 10000 # overtime
                    break 
                node_ = next_node
            solution.append(time__)
            path.append(p)
        solution = np.asarray(solution)
        best_index = np.argmin(solution)

        if solution[best_index] == 10000:
            print(self.time_,'self time')
            print(self.passenger_list)
            print('10000 solution')
            exit()
        passenger_index = path[best_index][0]
        best_task_id = self.passenger_list[passenger_index][-1]

        return best_task_id


    def choose_new_task(self, task_index):

        if len(task_index)==1:
            return task_index[0]

        task_score_hist = []
        task_score_huristic = []

        for task_id in task_index:

            task_node = self.graph.task_list[task_id][1]
            dest_time = self.graph.task_list[task_id][4]

            urgent_time = dest_time - self.graph.graph_time[self.current,self.graph.task_list[task_id][1]]-\
                self.graph.graph_time[self.graph.task_list[task_id][2],self.graph.task_list[task_id][1]]

            if self.tabu==[] or self.tabu[-1]==-1:
                task_score_hist.append(self.graph.pheromone_on_diagnal[self.total_cost-1][task_node])
                # task_score_huristic.append(dest_time-self.time_)
                task_score_huristic.append(urgent_time-self.time_)

            else:
                task_score_hist.append(self.graph.pheromone_off_diagnal[self.current][task_node])
                # task_score_huristic.append(dest_time-self.time_)
                
                if urgent_time-self.time_<0:
                    print(self.graph.task_list[task_id])
                    print(self.current)
                    print(urgent_time,self.time_)
                    print(self.task_flag[task_id])
                    print('urgent time')
                    exit()
                task_score_huristic.append(urgent_time-self.time_)

        task_score_hist = (np.asarray(task_score_hist))** self.colony.alpha
        task_score_huristic = (1/np.asarray(task_score_huristic))**self.colony.beta
        probability = task_score_hist*task_score_huristic/np.sum(task_score_hist*task_score_huristic)
        index = np.argmax(npr.multinomial(1, probability))
        task_id = task_index[index]

        return task_id
    

    def _update_pheromone_delta(self):
        
        count = 1
        j = self.tabu[0]

        if self.colony.update_strategy == 1:  # ant-quality system                
                self.pheromone_on_diagnal_delta[0,j] = self.colony.Q
        elif self.colony.update_strategy == 2:  # ant-density system
            # noinspection PyTypeChecker
            self.pheromone_on_diagnal_delta[0,j] = self.colony.Q / count
        else:  # ant-cycle system
            self.pheromone_on_diagnal_delta[0,j] = self.colony.Q / self.total_cost

        
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if j!=-1:
                if i == -1:
                    
                    if self.colony.update_strategy == 1:  # ant-quality system
                        self.pheromone_on_diagnal_delta[count,j] = self.colony.Q
                    elif self.colony.update_strategy == 2:  # ant-density system
                        # noinspection PyTypeChecker
                        self.pheromone_on_diagnal_delta[count,j] = self.colony.Q / count
                    else:  # ant-cycle system
                        self.pheromone_on_diagnal_delta[count,j] = self.colony.Q / self.total_cost
                    count = count + 1
                    
                else:

                    if self.colony.update_strategy == 1:  # ant-quality system
                        self.pheromone_off_diagnal_delta[i,j] = self.colony.Q
                    elif self.colony.update_strategy == 2:  # ant-density system
                        # noinspection PyTypeChecker
                        self.pheromone_off_diagnal_delta[i,j] = self.colony.Q / count
                    else:  # ant-cycle system
                        self.pheromone_off_diagnal_delta[i,j] = self.colony.Q / self.total_cost

