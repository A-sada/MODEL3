from negmas import AspirationNegotiator, ResponseType,SAONegotiator
from negmas.negotiators import Controller
from negmas.outcomes import Outcome
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences
from negmas.sao import SAOState
from negmas.sao.common import ResponseType
from negmas.situated import Agent
from VRPTW_functions import euclidean_distance
from classes import Task
from VRPTW_functions import *
import copy
"""
コスト関数をスラック距離だけにする
  """  
import random

# VehicleNegotiatorクラスの定義（NegMASのSAONegotiatorの代わりに基本的なPythonクラスを使用）
class VehicleNegotiator(SAONegotiator):
    def __init__(self, vehicle_id, tasks, is_vehicle_a, task_a, preferences: Preferences | None = None, ufun: BaseUtilityFunction | None = None, name: str | None = None, parent: Controller | None = None, owner: Agent | None = None, id: str | None = None, type_name: str | None = None, can_propose: bool = True):
        self.vehicle_id = vehicle_id  # 車両ID
        self.tasks = tasks            # タスクのリスト（ルート）
        self.is_vehicle_a = is_vehicle_a  # 車両Aかどうかを示すフラグ
        self.task_a = task_a          # taskA（車両Aの場合のみ）
        self.initial_offer_received = None  # 初回に受け取った提案
        self.n_steps=0
        self.bulletin_board = None
        self.remove_list=[]
        self.arrival_time_list=[]
        self.flag = 0
        self.current_weight = 0
        self.max_weight = 100
        super().__init__( preferences, ufun, name, parent, owner, id, type_name, can_propose)
        #self.add_capabilities(dict(propose_for_self=True))

    def propose(self,state):
        # 提案のロジックを実装
        self.n_steps += 1
        task_a = None
        if self.is_vehicle_a:
            # 車両Aの場合、taskAの提案のみ行う
            offer = {"taskA": self.task_a, "taskB": None}
        else:
            if self.n_steps == 1:
                # 車両Bの場合、初回に受け取った提案からtaskAを取得
                task_a = self.initial_offer_received["taskA"]
                task_b = None
                self.make_remove_list(task_a)
                self.flag = 1
            # 車両Bの場合、初回に受け取った提案からtaskAを取得
            task_a = self.initial_offer_received["taskA"] 
            #task_b = None  # 一方的に受け取る場合
            if self.tasks:
                if len(self.remove_list)    == 0:
                    task_b = None
                task_b = self.remove_list.pop(0)  
                if task_b == "0":
                    task_b = None
            offer ={"taskA": task_a, "taskB": task_b}
        #return super().propose(offer)
        #print(offer)
        return offer
    
    def respond(self, state: SAOState, offer: Outcome, source: str):
        cost_border = 1 * (self.bulletin_board.max_steps - self.bulletin_board.n_steps) / self.bulletin_board.max_steps +0
        # 応答のロジックを実装
        if not self.initial_offer_received:
            self.initial_offer_received = offer  # 初回の提案を保存
        # その後の応答ロジックをここに実装
        if self.is_vehicle_a:
            task_b = offer["taskB"]
            if len(self.tasks) < 3:
                cost=calculate_cost_saving(self.arrival_time_list,offer["taskA"],offer["taskB"],self.tasks,self.bulletin_board)
                if task_b == None:
                    return ResponseType.ACCEPT_OFFER
                elif self.current_weight + task_b.weight > self.max_weight:
                    return ResponseType.REJECT_OFFER
                elif calculate_cost_saving(self.arrival_time_list,offer["taskA"],offer["taskB"],self.tasks,self.bulletin_board) < 0:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            else:
                cost = calculate_cost_saving(self.arrival_time_list,offer["taskA"],offer["taskB"],self.tasks,self.bulletin_board)
                if task_b == None:
                    return ResponseType.ACCEPT_OFFER 
                if task_b != None:
                    if self.current_weight + task_b.weight > self.max_weight:
                        return ResponseType.REJECT_OFFER
                if cost < cost_border:
                    return ResponseType.ACCEPT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
                
        else:
            #print("B")
            return ResponseType.REJECT_OFFER
    
    def make_remove_list(self,taskA):
        remove_list = []
        for task_pac in self.arrival_time_list:
            if task_pac.earliest_start_time > task_pac.task.due_date:
                remove_list.append(task_pac.task)
        pac_list = copy.deepcopy(self.arrival_time_list)
        #pac_listから要素を1つずつ削除して，コストの減少が大きい順に並べる
        #remove_listにあるタスクは前に来るようにする
        #pac_listからi番目のタスクを削除したときのコストの増減を計算する
        cost={}
        rt_list = []
        for pac in self.arrival_time_list:
            cost[pac.task] =calculate_cost_saving(self.arrival_time_list,pac.task,taskA,self.tasks,self.bulletin_board)
        
        cost["0"] = calculate_cost_saving(self.arrival_time_list,None,taskA,self.tasks,self.bulletin_board)
        sorted_cost = sorted(cost.items(), key=lambda x:x[1])
        for i in sorted_cost:
            if i[1] < 50:
            
                if i[0] in remove_list:
                    remove_list.remove(i[0])
                    rt_list.append(i[0])
            for i in sorted_cost:
                if i[0] not in remove_list:
                    rt_list.append(i[0])
        self.remove_list = rt_list

# この関数は特定の合意結果のコスト削減を計算します。
def calculate_cost_saving(task_list,taskA,taskB,route,bulletin_board):
#この関数を利用する車両のリスト（パックリスト）と，交換するタスクたち，bulletin_boardを引数にとる
#bulletin_boardはbulletin_board.pyのbulletin_boardクラスのインスタンス
    #print("taskA",taskA)
    #print("taskB",taskB)
    #print(route)
    remove_task = taskA if taskA in route else None
    remove_task = taskB if taskB in route else None

    give_task = taskA if taskA not in route else None
    give_task = taskB if taskB not in route else None


    # 交換でのタスクの交換によるスラックタイムの差分・コストの変化を計算
    slack_cost = calculate_differ_slack(task_list,remove_task,give_task,route,bulletin_board)
    # 交換でのタスクの交換によるover_windowの差分・コストの変化を計算
    # over_cost = caluculate_differ_over_window(task_list,remove_task,give_task,route,bulletin_board)
    # 交換でのタスクの交換による距離の差分・コストの変化を計算
    distans_cost = calculate_differ_distance(route,remove_task,give_task,bulletin_board,task_list)

    slack_late = 0.5


    #over_lateは，時間が進むにつれて値を大きくする
    #現在の時間はbulletin_board.n_stepで取得できる
    #最大時間はbulletin_board.max_stepで取得できる
    #over_costは前半ではほぼ無視をして，後半では大きくする
    #最後の25％の時間ではover_costをかなり大きくする
    # over_late =0* (bulletin_board.n_steps / bulletin_board.max_steps) ** 2
    distance_late = 0.5
    # cost_saving = slack_late * slack_cost + over_late * over_cost + distance_late * distans_cost
    cost_saving = distance_late * distans_cost + slack_late * slack_cost
    return cost_saving

    
def calculate_slacktime(route):
    slack_time = 0
    for task in route:
        slack_time += max(task.late_start_time - task.earliest_start_time ,0)
    return slack_time

def calculate_differ_slack(pac_list,remove,add,route,bulletin_board):
    #元のrouteのスラックタイムと，タスクの交換後のスラックタイムの差分を示す
    #routeはパッケージのリストに限る
    #remove_taskはrouteから削除するタスク
    #add_taskはrouteに追加するタスク
    before_slack_time = calculate_slacktime(pac_list)
    changed_list = copy.deepcopy(pac_list)
    changed_list = remove_task(remove,changed_list)
    changed_list = add_task(add,changed_list,route,bulletin_board)
    earliest_start_time_list(changed_list,bulletin_board.dep_x,bulletin_board.dep_y)
    latest_start_time_list(changed_list)
    after_slack_time = calculate_slacktime(changed_list)
    #スラックタイムが増えれば負の値を返す   
    return after_slack_time - before_slack_time
    #スラックタイムが増えれば負の値を返す


def calculate_over_window(route):
    over_window = 0
    for task in route:
        over_window += max(task.earliest_start_time - task.task.due_date, 0)
    #due_dateトの引き算デいいのか議論の余地あり，最遅サービス開始時間でも
    return over_window  
        
def caluculate_differ_over_window(pac_list,remove,add,route,bulletin_board):
    #元のrouteのover_windowと，タスクの交換後のover_windowの差分を示す
    #routeはパッケージのリストに限る
    #remove_taskはrouteから削除するタスク
    #add_taskはrouteに追加するタスク
    before_over_window = calculate_over_window(pac_list)
    #print("before_over_window",before_over_window)
    changed_list = copy.deepcopy(pac_list)
    changed_list = remove_task(remove,changed_list)
    changed_list = add_task(add,changed_list,route,bulletin_board)
    earliest_start_time_list(changed_list,bulletin_board.dep_x,bulletin_board.dep_y)
    latest_start_time_list(changed_list)
    after_over_window = calculate_over_window(changed_list)
    #print("after_over_window",after_over_window)
    return after_over_window - before_over_window
    #over_windowが減れば負の値を返す

def calculate_differ_distance(route,taskA,taskB,bulletin_board,pac_list):
    #route = copy.deepcopy(route)
    distance = 0
    
    if taskA != None:
        #削除するタスクの前後の移動時間
        index = route.index(taskA)
        if index > 0:
            distance -= euclidean_distance(route[index-1],route[index])
        if index < len(route)-1:
            distance -= euclidean_distance(route[index],route[index+1])
    #タスク追加の前後の移動時間
    if taskB != None:
        index = least_cost_time_insertion_index(route,taskB,pac_list,bulletin_board)
        if index == None:
            return distance
        if index > 0:
            distance += euclidean_distance(route[index-1],taskB)
        if index < len(route):
            distance += euclidean_distance(route[index],taskB)
    return distance
    #距離が短くなれば負の値を返す
