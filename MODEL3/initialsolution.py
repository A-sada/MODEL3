import random
import os
import copy
from Strategy_Vehicle_ver1 import Vehicle
from classes import Task
from VRPTW_functions import euclidean_distance
from fun_for_test import route_check   

def check_task(vehicle,new_task,dep_x,dep_y):
    if vehicle.current_weight + new_task.weight > vehicle.max_weight:
        return False
    for i in range(len(vehicle.tasks)):
        route = copy.deepcopy(vehicle.tasks)
        route.insert(i,new_task)
        if route_check(route,dep_x,dep_y) == True:
            return True
    
    return False


def task_add(vehicle, new_task,dep_x, dep_y):
    if vehicle.current_weight + new_task.weight > vehicle.max_weight:
        return False
    for i in range(len(vehicle.tasks)):
        route = copy.deepcopy(vehicle.tasks)
        route.insert(i,new_task)
        if route_check(route,dep_x,dep_y) == True:
            vehicle.tasks.insert(i,new_task)
            vehicle.current_weight += new_task.weight
            return True
    
    return False

def assign_tasks_to_vehicles_with_insert(tasks, vehicles,run_num,dep_x, dep_y):
    # global run_num
    for task in tasks:
        if run_num == 0:
            new_vehicle = Vehicle(run_num, max_weight,dep_x, dep_y)
            new_vehicle.tasks.append(task)
            new_vehicle.current_weight += task.weight
            vehicles.append(new_vehicle)
            run_num += 1
        else:
            apt_cars=[]
            for car in vehicles:
                if check_task(car, task,dep_x, dep_y) == True:
                    apt_cars.append(car)
            if len(apt_cars) == 0:
                new_vehicle = Vehicle(run_num, max_weight,dep_x, dep_y)
                new_vehicle.tasks.append(task)
                new_vehicle.current_weight += task.weight
                vehicles.append(new_vehicle)
                run_num += 1
            elif len(apt_cars)==1:
                task_add(apt_cars[0],task,dep_x,dep_y)
            else:
                rad = random.randint(0, len(apt_cars)-1)
                task_add(apt_cars[rad],task,dep_x,dep_y)
    return run_num

def read_task(filename, tasks):
    global max_weight
    max_x_coordinate = float('-inf')  # 初期値を負の無限大に設定
    max_y_coordinate = float('-inf')  # 初期値を負の無限大に設定
    max_due_date = float('-inf')  # 初期値を負の無限大に設定

    data_path = filename
    if not os.path.isabs(data_path):
        module_dir = os.path.dirname(__file__)
        candidate = os.path.join(module_dir, filename)
        if os.path.exists(candidate):
            data_path = candidate

    with open(data_path, "r") as file:
        current_section = "no"
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line == "VEHICLE":
                current_section = "VEHICLE"
                continue
            elif line == "CUSTOMER":
                current_section = "CUSTOMER"
                continue

            if current_section == "VEHICLE":
                if "NUMBER" in line:
                    continue
                parts = line.split()
                max_num, max_weight = map(int, parts)

            if current_section == "CUSTOMER":
                if "CUST NO." in line:
                    continue
                parts = line.split()
                id, x_coordinate, y_coordinate, weight, ready_time, due_date, service_time = map(int, parts)
                task = Task(id, x_coordinate, y_coordinate, weight, ready_time, due_date, service_time)

                tasks.append(task)

                # 最大値の更新
                max_x_coordinate = max(max_x_coordinate, x_coordinate)
                max_y_coordinate = max(max_y_coordinate, y_coordinate)
                max_due_date = max(max_due_date, due_date)

    return [max(max_x_coordinate, max_y_coordinate), max_due_date]


class exchange_tasks:
    def __init__(self, taskA, taskB, id, vehicleA, vehicleB) :
        self.id = id
        self.vehicleA = vehicleA
        self.vehicleB = vehicleB
        self.taskA = taskA
        self.taskB = taskB
        
