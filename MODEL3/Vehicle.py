from classes import Task,Offer,Nego,Balletin
# 車両（エージェント）クラス
import pandas as pd
from negmas import AspirationNegotiator, ResponseType,SAONegotiator
import numpy as np
from negmas import SAOMechanism, AspirationNegotiator, Issue, ResponseType
from typing import Optional, List
from VRPTW_functions import euclidean_distance,slack_time_list
import copy
import math
from balletin_are_search import most_stayed_area_dynamic,update_stay_areas
import random
from Vehicle_Negotiatior import VehicleNegotiator
class Vehicle_BASE:
    def __init__(self, id, max_weight,dep_x, dep_y):
        super().__init__()
        self.id = id  # 車両のID
        self.dep_x = dep_x
        self.dep_y = dep_y
        #self.x_coordinate = x_coordinate  # 現在地のx座標
        #self.y_coordinate = y_coordinate  # 現在地のy座標
        self.max_weight = max_weight  # 最大積載量
        self.current_weight = 0  # 現在の積載量
        self.propose_task = None
        self.tasks = []  # 割り当てられたタスクのリスト
        self.offer_nego_list=[] #自分の交渉リスト・自分が提案側ならここにスタート時の交渉内容を保存する
        self.next_nego={} #交渉IDと自分の交渉リストインデックスと対応
        self.bulletin_board : Balletin
        self.offer_flag = 0
        self.rout_pacs = []
        self.taskA = 0
        self.slack_time = []
        self.Neg = 0
        self.over_task =[]
        self.arrival_time_list =[]
        self.exchange_flag = 0
        self.route_planner = None
        self.negotiation_log_path: Optional[str] = None
    #掲示板を取得
    def set_balletin(self,balletin : Balletin):
        self.bulletin_board = balletin

    def set_route_planner(self, planner):
        self.route_planner = planner

    def evaluate_route_with_planner(self, tasks):
        """Run the RL planner on a task list and return routing metrics."""
        if self.route_planner is None:
            return None
        if tasks is None:
            return None
        try:
            # The planner reorders tasks but does not mutate the provided list.
            _, info = self.route_planner.plan_route(list(tasks), self.max_weight, self.dep_x, self.dep_y)
            return info
        except Exception:
            return None

    #1日の最初に呼び出される
    def first_step(self):
        return
    def step(self):
        self.offer_flag=0
        self.offer_nego_list=[]
        self.exchange_flag = 0
        return
    
    def accept_or_reject(self,offer):
        if len(self.tasks) < 1:
            return False    
        #交渉の受け入れの是非を実装
        if self.check_offer(offer.get("taskB")) == True:
            return True
        else:
            return True
    
    def make_propose(self):
        return {"taskA": self.propose_task, "taskB": self.tasks[random.randint(0,len(self.tasks - 1))]}
  
    #タスクの挿入が可能かチェック->true or false
    def check_task(self,new_task):
        # 車両の開始位置から新しいタスクまでの距離を計算
        start_task = Task(0, 0, 0, 0, 0, 0, 0)  # 仮の開始位置
        travel_time_from_start = euclidean_distance(start_task, new_task)
        if travel_time_from_start <= new_task.due_date :
            if travel_time_from_start + new_task.service_time + euclidean_distance(new_task, self.tasks[0]) <= self.tasks[0].due_date: 
                return True

        # 各タスク間での新しいタスクの挿入を試みる
        for i in range(len(self.tasks) - 1):
            current_task = self.tasks[i]
            next_task = self.tasks[i + 1]

            # 現在のタスクの終了時間を計算
            current_task_end_time = current_task.ready_time + current_task.service_time

            # 新しいタスクへの移動に必要な時間を計算
            travel_time_to_new_task = euclidean_distance(current_task, new_task)

            # 新しいタスクのサービス終了時間を計算
            new_task_end_time = current_task_end_time + travel_time_to_new_task + new_task.service_time

            # 次のタスクへの移動に必要な時間を計算
            travel_time_to_next_task = euclidean_distance(new_task, next_task)

            # 次のタスクの開始時間を計算
            next_task_start_time = new_task_end_time + travel_time_to_next_task

            # 新しいタスクがdue_date前に終了し、次のタスクが時間内に開始できるかどうかを確認
            if current_task_end_time + travel_time_to_new_task <= new_task.due_date and next_task_start_time <= next_task.due_date:
                return True

        # すべてのタスクの後に新しいタスクを追加する場合の判定
        last_task = self.tasks[-1]
        last_task_end_time = last_task.ready_time + last_task.service_time
        travel_time_to_new_task = euclidean_distance(last_task, new_task)
        if last_task_end_time + travel_time_to_new_task <= new_task.due_date:
            return True
        return False
    
    #交渉の提案を行う-> 希望する交渉のリストを送信
    def offer_on_negotiation(self,run_cars,offer_id):
        #list_return = copy.deepcopy(self.offer_nego_list)
        #list_return =[]
        ofe=Offer(offer_id,self.id,run_cars[0].id,self.tasks[0])
        offer_id += 1
        self.offer_nego_list.append(ofe)
        #list_return = copy.deepcopy(self.offer_nego_list)
        return self.offer_nego_list
    
    #提案された交渉について応じるかどうかを判断　->応じるならTrue,応じないならFalse
    def check_offer(self,task):
        return self.check_task(task)
    
    #リストからIDの一致する要素のインデックスを返す
    def find_index(self,obj_list, target_id):
        index = 0
        for obj in obj_list:
            if obj.id == target_id:
                return index
            index += 1
        return None
    
    #提案した交渉が相手に受け入れられたら呼び出される．
    def accept_offer(self,offer,neg_id):
        if offer not in self.offer_nego_list:
            print("error- vehicleA have not task")
            return False
        #実施する交渉のリスト（自分が提案したタスクのみ）
        self.next_nego[neg_id]=offer.task
        return

    #行われる交渉IDを受け取る，この値から自分が提案者側かどうかを判断する，
    def start_negotiation(self,neg_id):
        neg_task = self.next_nego.get(neg_id)
        if neg_task != None:
            self.propose_task = neg_task
            #print("propose_task",self.propose_task)
            #propse_taskはタスクリストのインデックス
            self.offer_flag = True
            #自分が提案した交渉ならフラグがたつ
        return
    def before_negotiation(self):
        return
    def set_negotiation_log_path(self, log_path: Optional[str]):
        self.negotiation_log_path = log_path

    def make_neg_agent(self, negotiation_id: Optional[int] = None, counterparty_id: Optional[int] = None):
        initial_can_propose = bool(self.offer_flag)
        # print(f"initial_can_propose: {initial_can_propose}")
        self.Neg = VehicleNegotiator(
            self.id,
            self.tasks,
            is_vehicle_a=initial_can_propose,
            task_a=self.propose_task,
            negotiation_id=negotiation_id,
            log_path=self.negotiation_log_path,
            counterparty_id=counterparty_id,
            name=self.id,
            can_propose=True,
        )
        self.Neg.bulletin_board = self.bulletin_board
        self.before_negotiation()
        self.Neg.remove_list = self.over_task
        self.Neg.arrival_time_list = self.arrival_time_list
        return self.Neg
    #交渉終了時に呼び出される
    #交渉時に自分が提案者側かどうかを示すフラグの初期化
    def end_negotiation(self):
        self.offer_flag = 0
        return
    #合意した契約のリストが渡される，署名する契約のリストを返す．
    def sign_contracts(self,list: List):
        signed=[]
        for contract in list:
            if contract.vehicleA == self:
                partner = contract.vehicleB
            else:
                partner = contract.vehicleA
            if self.sign_contract(partner,contract.taskA,contract.taskB):
                signed.append(contract)

        return signed
    #署名戦略
    #タスクAとタスクBのどちらが自分持つタスクか確認する必要あり
    def sign_contract(self,partner,taskA,taskB):
        
        return True
    
    #ルートから該当するタスクを削除
    def pop(self,task):
        if task in self.tasks:
            self.tasks.remove(task)
            self.current_weight -= task.weight
            return True
        else:
            return False
        if task in self.tasks:
            return False
        else:
            return True
    def pop_index(self,index):
        if index < len(self.tasks):
            self.tasks.pop(index)
            return True
        else:
            return False
    #とにかく挿入可能な場所に挿入する
    def add_old(self,new_task):
        # 車両の開始位置から新しいタスクまでの距離を計算
        start_task = Task(0, 0, 0, 0, 0, 0, 0)  # 仮の開始位置
        travel_time_from_start = euclidean_distance(start_task, new_task)
        if travel_time_from_start <= new_task.due_date :
            if travel_time_from_start + new_task.service_time + euclidean_distance(new_task, self.tasks[0]) <= self.tasks[0].due_date: 
                self.tasks.insert(0,new_task)
                self.current_weight += new_task.weight
                return True

        # 各タスク間での新しいタスクの挿入を試みる
        for i in range(len(self.tasks) - 1):
            current_task = self.tasks[i]
            next_task = self.tasks[i + 1]

            # 現在のタスクの終了時間を計算
            current_task_end_time = current_task.ready_time + current_task.service_time

            # 新しいタスクへの移動に必要な時間を計算
            travel_time_to_new_task = euclidean_distance(current_task, new_task)

            # 新しいタスクのサービス終了時間を計算
            new_task_end_time = current_task_end_time + travel_time_to_new_task + new_task.service_time

            # 次のタスクへの移動に必要な時間を計算
            travel_time_to_next_task = euclidean_distance(new_task, next_task)

            # 次のタスクの開始時間を計算
            next_task_start_time = new_task_end_time + travel_time_to_next_task

            # 新しいタスクがdue_date前に終了し、次のタスクが時間内に開始できるかどうかを確認
            if current_task_end_time + travel_time_to_new_task <= new_task.due_date and next_task_start_time <= next_task.due_date:
                self.tasks.insert(i+1,new_task)
                self.current_weight += new_task.weight
                return True

        # すべてのタスクの後に新しいタスクを追加する場合の判定
        last_task = self.tasks[-1]
        last_task_end_time = last_task.ready_time + last_task.service_time
        travel_time_to_new_task = euclidean_distance(last_task, new_task)
        if last_task_end_time + travel_time_to_new_task <= new_task.due_date:
            self.tasks.append(new_task)
            self.current_weight += new_task.weight
            return True
        return False
    
    #コストが最小となる場所に挿入する
    def add(self, new_task):
        if self.current_weight + new_task.weight > self.max_weight + 100000:
            return False
        if new_task in self.tasks:
            return False
        self.least_cost_time_sensitive_insertion(new_task)
        if new_task in self.tasks:
            return True
        else:
            return False
    
    #掲示板の更新
    def bulletin_update(self,X,T,zones,n):
        area = self.bulletin_board.area_board
        if not (self.bulletin_board.area_board['id'] == self.id).any():
            new_row = pd.DataFrame({'id': self.id, 'slack_time': [0], 'departure_time': [0], 'return_time': [0]})
            self.bulletin_board.time_board = pd.concat([self.bulletin_board.time_board, new_row], ignore_index=True)
        
        #self.slack_time = slack_time_list(self.tasks,empty_list = list(0 for _ in range(len(self.tasks))))
        self.bulletin_board.time_board.loc[self.bulletin_board.time_board['id']== self.id, 'slack_time'] = self.calculate_slack_time(self.tasks,100000,None)
        self.bulletin_board.time_board.loc[self.bulletin_board.time_board['id'] == self.id, ['departure_time', 'return_time']] = [self.tasks[0].ready_time - euclidean_distance(Task(0,self.dep_x,self.dep_y,0,0,0,0),self.tasks[0]),self.tasks[-1].due_date + self.tasks[-1].service_time + euclidean_distance(self.tasks[-1],Task(0,self.dep_x,self.dep_y,0,0,0,0))]
        new_data = most_stayed_area_dynamic(self.tasks, X, T, zones, n,self.dep_x,self.dep_y)
        if not ( self.bulletin_board.area_board['id'] == self.id).any():
            # 新しい行のインデックスを決定
            new_index = len(self.bulletin_board.area_board)
            # 新しい行を追加
            self.bulletin_board.area_board.loc[new_index] = [self.id] + list(new_data.values())
        else:
            update_stay_areas(self.bulletin_board.area_board, self.id , new_data)
        return 
    

    #スラックタイムの計算　挿入なしの
    def calculate_slack_time(self, route, position_to_insert, task_to_insert):
        total_time = 0  # total time spent so far in the route
        slack_time = 0
        
        for i in range(len(route)):
            task = route[i]
            
            if i == position_to_insert:
                # ここで新しいタスクを挿入すると仮定
                total_time += euclidean_distance(route[i-1], task_to_insert)
                
                # Wait for the task's service to start if necessary
                slack_time += max(0, task_to_insert.due_date - total_time)
                #タスク到着時間から，サービス開始期限までを計算　→スラックタイム
                if total_time < task_to_insert.ready_time:
                    total_time = task_to_insert.ready_time
                
                total_time += task_to_insert.service_time
                total_time += euclidean_distance(task_to_insert, task)
                
                # タスクのサービスが開始するまで必要な場合は待つ
                if total_time < task.ready_time:
                    total_time = task.ready_time
                
                total_time += task.service_time
                
            else:
                if i != 0:
                    total_time += euclidean_distance(route[i-1], task)
                    slack_time += max(0, task.due_date - total_time)
                    # Wait for the task's service to start if necessary
                    if total_time < task.ready_time:
                        total_time = task.ready_time
                    
                    total_time += task.service_time
                else:
                    distance = euclidean_distance(Task(0,self.dep_x,self.dep_y,0,0,0,0), task)
                    if distance < task.ready_time:
                        distance = task.ready_time
                    total_time += distance
                    
                    total_time += task.service_time
            
        return slack_time


    def least_cost_time_sensitive_insertion(self , new_task):
        if not isinstance(new_task, Task):  # 仮定として Task というクラスが存在するとします。
            print("エラー: 'new_task' が Task オブジェクトではありません。")
            return False
        min_cost = float('inf')
        best_position = None
        cost = 0
        if len(self.tasks) == 0:
            return False
        # 車両の開始位置から新しいタスクまでの距離を計算
        start_task = Task(0, self.dep_x, self.dep_y, 0, 0, 0, 0)  # 仮の開始位置
        travel_time_from_start = euclidean_distance(start_task, new_task)
        if travel_time_from_start <= new_task.due_date :
            if travel_time_from_start + new_task.service_time + euclidean_distance(new_task, self.tasks[0]) <= self.tasks[0].due_date: 
                cost = self.calculate_slack_time(self.tasks, 0, new_task)
                if min_cost > cost:
                    min_cost = cost
                    best_position = 0


        # 各タスク間での新しいタスクの挿入を試みる
        for i in range(len(self.tasks) - 1):
            current_task = self.tasks[i]
            next_task = self.tasks[i + 1]

            # 現在のタスクの終了時間を計算
            current_task_end_time = current_task.ready_time + current_task.service_time

            # 新しいタスクへの移動に必要な時間を計算
            travel_time_to_new_task = euclidean_distance(current_task, new_task)

            # 新しいタスクのサービス終了時間を計算
            new_task_end_time = current_task_end_time + travel_time_to_new_task + new_task.service_time

            # 次のタスクへの移動に必要な時間を計算
            travel_time_to_next_task = euclidean_distance(new_task, next_task)

            # 次のタスクの開始時間を計算
            next_task_start_time = new_task_end_time + travel_time_to_next_task

            # 新しいタスクがdue_date前に終了し、次のタスクが時間内に開始できるかどうかを確認
            if current_task_end_time + travel_time_to_new_task <= new_task.due_date and next_task_start_time <= next_task.due_date:
                cost = self.calculate_slack_time(self.tasks,i+1,new_task)
                if min_cost > cost:
                    min_cost = cost
                    best_position = i+1

        if best_position != None:
            self.tasks.insert(best_position,new_task)
            self.current_weight += new_task.weight
            return True
        return False
