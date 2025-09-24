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
from VRPTW_functions import calculate_cost_saving
import copy
from datetime import datetime
"""
class negotiator(SAONegotiator):
    def __init__(self, owner : Vehicle_Base  ,List : list,neg_flag):
        self.owner = owner
        self.offer_flag = 0 #自分がタスク交換を希望した側かを判断
        self.Negotiate_list = List # 交渉に投げるようのリスト，このリストの先頭からタスクを提案していく（自分が交渉時車両B側でのみつかう）
        self.neg_flag = neg_flag #自分が交渉においてどちらが和なのかを示す　ー＞
    
    def propose(self, state: SAOState) -> Outcome | None:
        return super().propose(state)
    
    def respond(self, state: SAOState, offer: Outcome, source: str) -> ResponseType:
        return super().respond(state, offer, source)
  """  
import random
from typing import Optional

# VehicleNegotiatorクラスの定義（NegMASのSAONegotiatorの代わりに基本的なPythonクラスを使用）
class VehicleNegotiator(SAONegotiator):
    def __init__(
        self,
        vehicle_id,
        tasks,
        is_vehicle_a,
        task_a,
        negotiation_id=None,
        counterparty_id: int | str | None = None,
        log_path: str | None = None,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        type_name: str | None = None,
        can_propose: bool = True,
    ):
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
        self.negotiation_id = negotiation_id
        self.counterparty_id = counterparty_id
        self.log_path = log_path
        super().__init__(preferences, ufun, name, parent, owner, id, type_name, can_propose)
        #self.add_capabilities(dict(propose_for_self=True))

    def propose(self, state: SAOState, dest: Optional[str] = None):
        # 提案のロジックを実装
        self.n_steps += 1
        task_a = None
        # print(self.is_vehicle_a)
        if self.is_vehicle_a:
            # 車両Aの場合、taskAの提案のみ行う
            offer = {"taskA": self.task_a, "taskB": None}
        else:
            none_oofer = {"taskA": None, "taskB": None}
            # print(f"n_steps:{self.n_steps}")
            if self.n_steps == 1:
                # 車両Bの場合、初回に受け取った提案からtaskAを取得
                if not self.initial_offer_received:
                    return none_oofer
                task_a = self.initial_offer_received["taskA"]
                self.make_remove_list(task_a)
                self.flag = 1
            # 車両Bの場合、初回に受け取った提案からtaskAを取得
            if not self.initial_offer_received:
                return none_oofer
            task_a = self.initial_offer_received["taskA"] 
            #task_b = None  # 一方的に受け取る場合
            task_b = None
            if self.tasks:
                if self.remove_list:
                    task_b = self.remove_list.pop(0)  
                    if task_b == "0":
                        task_b = None
            offer ={"taskA": task_a, "taskB": task_b}
        #return super().propose(offer)
        # print(offer)
        self._log_offer(state, offer)
        return offer
    
    def respond(
        self,
        state: SAOState,
        offer: Outcome | None = None,
        source: str | None = None,
    ):
        # if self.is_vehicle_a:
        #     print("A")
        # else:
        #     print("B")
        if offer is None and state is not None:
            offer = state.current_offer
        if offer is None:
            print("offer is None")
            return ResponseType.REJECT_OFFER
        cost_border = 1 * (self.bulletin_board.max_steps - self.bulletin_board.n_steps) / self.bulletin_board.max_steps + 100
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
                
            if task_b != None:
                cost = calculate_cost_saving(self.arrival_time_list,offer["taskA"],offer["taskB"],self.tasks,self.bulletin_board) 
                #print(cost)
                if self.check_task(task_b) == True:
                    if cost < 0:
                        return ResponseType.ACCEPT_OFFER
                    else:
                        return ResponseType.REJECT_OFFER
                else:
                    return ResponseType.REJECT_OFFER
            else:
                return ResponseType.ACCEPT_OFFER
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
            if i[0] in remove_list:
                remove_list.remove(i[0])
                rt_list.append(i[0])
        for i in sorted_cost:
            if i[0] not in remove_list:
                rt_list.append(i[0])
        self.remove_list = rt_list

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



    @property
    def capabilities(self):
        # ここで _capabilities を返すロジックを実装します
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value):
        # ここで _capabilities を設定するロジックを実装します
        self._capabilities = value

    def _log_offer(self, state: SAOState, offer: dict[str, Task | None]):
        if not self.log_path or offer is None:
            return
        try:
            board_step = getattr(self.bulletin_board, "n_steps", "")
            mechanism_step = getattr(state, "step", "") if state is not None else ""
            role = "A" if self.is_vehicle_a else "B"
            agent_offer_number = self.n_steps
            task_a = offer.get("taskA") if offer else None
            task_b = offer.get("taskB") if offer else None

            def task_fields(task: Task | None):
                if task is None:
                    return ("", "", "", "")
                return (
                    getattr(task, "id", ""),
                    getattr(task, "ready_time", ""),
                    getattr(task, "due_date", ""),
                    getattr(task, "weight", ""),
                )

            task_a_fields = task_fields(task_a)
            task_b_fields = task_fields(task_b)
            timestamp = datetime.now().isoformat()
            negotiation_id = self.negotiation_id if self.negotiation_id is not None else ""

            recipient_id = self.counterparty_id if self.counterparty_id is not None else ""

            with open(self.log_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"{timestamp},{board_step},{negotiation_id},{self.vehicle_id},{recipient_id},{role},{mechanism_step},{agent_offer_number},{task_a_fields[0]},{task_a_fields[1]},{task_a_fields[2]},{task_a_fields[3]},{task_b_fields[0]},{task_b_fields[1]},{task_b_fields[2]},{task_b_fields[3]}\n"
                )
        except Exception:
            # ログ記録に失敗しても交渉自体は継続させる
            pass
