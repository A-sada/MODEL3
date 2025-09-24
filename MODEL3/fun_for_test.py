from Strategy_Vehicle_ver1 import Vehicle
from classes import Task
from VRPTW_functions import euclidean_distance
def route_check(route : list[Vehicle],dep_x,dep_y):
    depot_task = Task(0, dep_x, dep_y, 0, 0, 0, 0)  # 仮の開始位置
    current_time = 0
    for i in range(len(route)):
        current_task = route[i]
        if i == 0:
            current_time = max(euclidean_distance(depot_task, current_task),current_task.ready_time)
            if current_time > current_task.due_date:
                return False
            current_time += current_task.service_time
            # 現在のタスクの終了時間を計算
        else:
            current_time = max(current_time + euclidean_distance(pre_task,current_task),current_task.ready_time)
            if current_time > current_task.due_date:
                return False
            #current_timeをカレントタスクの開始時間に更新
            current_time += current_task.service_time
        pre_task = current_task
    return True

def check_arriva_list(arrival_list):
    print("2A")
    for task in arrival_list:
        if (task.late_start_time - task.earliest_start_time) <= 0:
            return False
    return True