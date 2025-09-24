class Task:
    def __init__(self, id, x_coordinate, y_coordinate, weight, ready_time, due_date, service_time):
        self.id = id  # タスク（顧客）のID
        self.x_coordinate = x_coordinate  # 配送先のx座標
        self.y_coordinate = y_coordinate  # 配送先のy座標
        self.weight = weight  # 荷物の重量またはサイズ
        self.ready_time = ready_time  # 配送可能な最早時間
        self.due_date = due_date  # 配送締切時間
        self.arrival = 0
        self.service_time = service_time  # サービスにかかる時間

class Offer:
    def __init__(self, id, vehicleA,vehicleB,task):
        self.id = id  # タスク（顧客）のID
        self.vehicleA = vehicleA
        self.vehicleB = vehicleB
        self.task = task

class Nego:
    def __init__(self, id, vehicleA, vehicleB):
        self.id = id  # タスク（顧客）のID
        self.vehicleA = vehicleA
        self.vehicleB = vehicleB     

class Agree:
    def __init__(self, vehicleA, vehicleB, TaskA:Task,TaskB:Task) -> None:
        self.vehicleA = vehicleA
        self.vehicleB = vehicleB
        #車両Aが，捨てるタスク
        self.taskA = TaskA
        #車両Bが，捨てるタスク
        self.taskB = TaskB
        pass

class rout_pac:
    def __init__(self,id,arrival_fast,arrival_due) -> None:
        self.id = id
        self.arrival_fast = arrival_fast
        self.arrival_due = arrival_due
        self.slack_time = 0
import pandas as pd

class Balletin:
    def __init__(self,using :bool ,time_board : pd, area_board : pd, X :int ,n  :int,zones) -> None:
        
        self.use = using #掲示板を利用中のエージェントの有無の判断

        self.time_board = time_board

        self.area_board = area_board

        self. n_steps = 0

        self.max_steps = 0

        self.X = X #エリアの最大幅

        self.n = n #エリアの分割数　いっぺんあたり

        self.zones = zones #時間帯情報

        self.dep_x = 0

        self.dep_y = 0

        #何日目かを示す→①日目からスタート（0日目は初期解生成の行う）

class pac_task:
    def __init__(self, task : Task) -> None:
        self.task = task 
        self.earliest_start_time = 0
        self.late_start_time = 0
        self.earliest_arrival_time=0

class cont:
    def __init__(self,agree :Agree,cost) -> None:
        self.agre = agree
        self.cost = cost