from classes import *
import random
from datetime import datetime
from pathlib import Path
from Negotiator import Nego1
import math
import pandas as pd
import copy
import time
from balletin_are_search import create_time_zones
from fun_for_test import route_check,check_arriva_list
from VRPTW_functions import *
from initialsolution import *
from rl_route_planner import load_pretrained_planner

ENABLE_RL_ROUTING = True
RL_PRETRAINED_CHECKPOINT = Path(__file__).resolve().parent / "pretrained" / "planner_checkpoint.pt"
# from initial_ver2 import *
# from initial_ver2 import *
run_num = 0
tasks = []  # タスクを保存するためのリスト
vehicles = []
no_runs=[]
N=200

#時間に関する掲示板
b_board = pd.DataFrame({
    'id': pd.Series(dtype='int'),
    'slack_time': pd.Series(dtype='int'),
    'departure_time': pd.Series(dtype='int'),
    'return_time': pd.Series(dtype='int')
})
#滞在エリアに関する掲示板
# 初期データフレームの作成
stay_areas_bb = pd.DataFrame({
    'id': pd.Series(dtype='int'),
    'A': pd.Series(dtype='str'),
    'B': pd.Series(dtype='str'),
    'C': pd.Series(dtype='str'),
    'D': pd.Series(dtype='str'),
    'E': pd.Series(dtype='str'),
    'F': pd.Series(dtype='str'),
    'G': pd.Series(dtype='str')
})

# taskのリスト化
import os
# ディレクトリの名前を指定
base_directory_name = "output_files"

# 現在の日時を取得して、文字列形式に変換
current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#if not os.path.exists(base_directory_name):
#    os.makedirs(base_directory_name)
# 実行日時を名前とするフォルダのパスを生成
directory_name = os.path.join(base_directory_name, current_time_str)


# ディレクトリが存在しない場合、作成（親ディレクトリも含めて）
os.makedirs(directory_name, exist_ok=True)
negotiation_log_path = os.path.join(directory_name, "negotiation_offers.csv")
with open(negotiation_log_path, 'w') as negotiation_log_file:
    negotiation_log_file.write(
        "timestamp,global_step,negotiation_id,sender_vehicle_id,recipient_vehicle_id,role,mechanism_step,agent_offer_number,"
        "taskA_id,taskA_ready_time,taskA_due_date,taskA_weight,"
        "taskB_id,taskB_ready_time,taskB_due_date,taskB_weight\n"
    )
ll=[]
ll=read_task("c104.txt",tasks)
# ll=read_task("R2.TXT",tasks)
max_xy = ll[0]
max_time = ll[1]
#n = (int)(max_xy /25)
n =(int)(math.sqrt((len(tasks) / 10)))
n_zones = 7
zones = create_time_zones(max_time,n_zones)
#print(max_xy)
dep_x = tasks[0].x_coordinate
dep_y = tasks[0].y_coordinate
tasks.pop(0)
random.shuffle(tasks)
bulletin_board = Balletin(False,b_board,stay_areas_bb, max_xy,n,zones)
# タスクを車両に割り当て（時間制約を含む）
bulletin_board.dep_x = dep_x
bulletin_board.dep_y = dep_y
run_num = assign_tasks_to_vehicles_with_insert(tasks, vehicles,run_num,dep_x, dep_y)
for car in vehicles:
    car.set_balletin(bulletin_board)
    car.set_negotiation_log_path(negotiation_log_path)

route_planner = None
if ENABLE_RL_ROUTING and vehicles:
    try:
        checkpoint_override = os.getenv("RL_PLANNER_CHECKPOINT")
        if checkpoint_override:
            checkpoint_path = Path(checkpoint_override).expanduser().resolve()
        else:
            checkpoint_path = RL_PRETRAINED_CHECKPOINT

        if not checkpoint_path.exists():
            print(
                f"RLプランナーのチェックポイントが見つかりません: {checkpoint_path}. "
                "RLによる経路評価をスキップします。"
            )
        else:
            _, route_planner = load_pretrained_planner(checkpoint_path)
            max_route_tasks = max(len(vehicle.tasks) for vehicle in vehicles)
            planner_max_tasks = getattr(route_planner.config, "max_tasks", None)
            if planner_max_tasks is not None and max_route_tasks > planner_max_tasks:
                print(
                    f"RL警告: 必要なタスク数 {max_route_tasks} が事前学習モデルの対応上限 "
                    f"{planner_max_tasks} を超えています。結果が不安定になる可能性があります。"
                )
            for vehicle in vehicles:
                vehicle.set_route_planner(route_planner)
            print(f"事前学習済みRLプランナーを読み込みました: {checkpoint_path}")
    except ImportError as exc:
        print(f"RLルートプランナーを利用できません: {exc}")
    except Exception as exc:
        print(f"事前学習済みRLプランナーの初期化に失敗しました: {exc}")


#車両とIDの紐付け＿辞書
cars_id = {}
for vehicle in vehicles:
    cars_id[vehicle.id] = vehicle

# 結果を表示（テスト用）
i=0
filename = os.path.join(directory_name, f"Step-0.txt") 
with open(filename, 'w') as f:
    for vehicle in vehicles:
        vehicle.bulletin_update(max_xy,max_time,zones,n)
        task_ids = [task.id for task in vehicle.tasks]
        # ファイル名を生成
        f.write(f"Vehicle {vehicle.id} has tasks {task_ids} with total weight {vehicle.current_weight}.\n")
    f.write(f"CVN {len(vehicles)} CRT {sum_travel_time(vehicles)}\n")
# plot_vehicle_routes(vehicles)
#車両routeの適正比較
from collections import deque
log_CVN=[]
log_CRT=[]
log_CVN.append(len(vehicles))
log_CRT.append(sum_travel_time(vehicles))
log_nego=[0]
Go_flag = False
bulletin_board.max_steps = N
filename = os.path.join(directory_name, "1log_main.txt")
with open(filename, 'w') as f:
    # for i in range(len(log_nego)):
        # ファイル名を生成
    f.write(f"steps {0} CVN {log_CVN[0]} CRT {log_CRT[i]} n_neg {log_nego[0]}.\n")
MMM = int(N/2 + 1)
for negotiate_steps in range(MMM):
    start = time.time()
    for car in vehicles:
        car.first_step()
#        print(f"車両{car.id}のルート：{car.tasks}")
#        for task in car.tasks:
#            print(task.ready_time)
#            print(task.due_date)
    end = time.time()
    time_diff = end - start
    #print(f"first_step関数の実行時間: {time_diff} 秒")
    bulletin_board.n_steps += 1
#全車両から交渉の提案を受け付ける
    Offer_list = deque()
    offer_id=0
    start = time.time()

    for vehicle in vehicles:
        lst=vehicle.offer_on_negotiation(vehicles,offer_id,vehicles)
        Offer_list.append(lst)
        offer_id += len(lst)
    end = time.time()
    time_diff = end - start
    #print(f"offer_on_negotiation関数の実行時間: {time_diff} 秒")
    # 交渉リストを初期化
    negotiation_list = []
    negotiation_id = 0
    # Offer_list内の各リストをループで処理
    for lst in Offer_list:
        
        # 各リスト内のオファーをループで処理
        for offer in lst:
            
            # オファーに含まれる車両IDをキーとして、対応する車両オブジェクトを取得
            cars_A = cars_id[offer.vehicleA]
            cars_B = cars_id[offer.vehicleB]
            
            # 車両Bがオファーのタスクを受け入れられるかどうかをチェック
            if cars_B.check_offer(offer.task) == True:
                
                # 車両Bがオファーを受け入れることができる場合、オファーを交渉リストに追加
                negotiation_list.append(Nego(negotiation_id,cars_A,cars_B))
                
                # 車両Aにオファーの受け入れを通知
                cars_A.accept_offer(offer,negotiation_id)
                negotiation_id += 1
    # print(f"交渉リストの長さ：{len(negotiation_list)}")
    agreements=[]
    start = time.time()
    for neg in negotiation_list:
        neg.vehicleA.start_negotiation(neg.id)
        negA=neg.vehicleA.make_neg_agent(neg.id, counterparty_id=neg.vehicleB.id)
        negA.current_weight = neg.vehicleA.current_weight
        negB=neg.vehicleB.make_neg_agent(neg.id, counterparty_id=neg.vehicleA.id)
        negA.max_weight = neg.vehicleA.max_weight
        negB.max_weight = neg.vehicleB.max_weight
        negB.current_weight = neg.vehicleB.current_weight
        result=Nego1(neg.vehicleA,neg.vehicleB,negA,negB)
        #print(result)
        # 交渉が成功した場合には合意内容をリストに追加
        if result.agreement != None:
            agreement = result.agreement
            taskA = agreement['taskA']if 'taskA' in agreement else None
            taskB = agreement['taskB']if 'taskB' in agreement else None
            agreements.append(Agree(neg.vehicleA,neg.vehicleB,taskA,taskB))
        neg.vehicleA.end_negotiation()
    # print(f"合意リストの長さ：{len(agreements)}")
    end = time.time()
    time_diff = end - start
    #print(f"Nego1関数の実行時間: {time_diff} 秒")
    signed = []
    # 各車両ごとに関連する契約を格納するための辞書
    vehicle_agreements = {vehicle: [] for vehicle in vehicles}
    # 各契約をループし、関連する車両に契約を追加
    for agreement in agreements:
        if agreement.vehicleA in vehicle_agreements:
            vehicle_agreements[agreement.vehicleA].append(agreement)
        if agreement.vehicleB in vehicle_agreements:
            vehicle_agreements[agreement.vehicleB].append(agreement)
    

    
    contracts_signed = {}
    start = time.time()
    #署名の実施
    for vehicle in vehicle_agreements:
        contracts_signed[vehicle] = vehicle.sign_contracts(vehicle_agreements[vehicle])

    end = time.time()
    time_diff = end - start
    #print(f"sign_contracts関数の実行時間: {time_diff} 秒")
    for contract in agreements:
        if contract in contracts_signed.get(contract.vehicleA, [0]) or \
           contract in contracts_signed.get(contract.vehicleB, [0]):
            A_list= contracts_signed.get(contract.vehicleA, [0])
            B_list= contracts_signed.get(contract.vehicleB, [0])
            if contract in A_list:
                A_cost = A_list.index(contract)/len(A_list)
            else:
                A_cost = 0
            if contract in B_list:
                B_cost = B_list.index(contract)/len(B_list)
            else:
                B_cost = 0
            cost = A_cost + B_cost
            signed.append(cont(contract,cost))
    sorted_signed = sorted(signed, key=lambda x: x.cost)
    signed = []
    signed = [x.agre for x in sorted_signed]    
    exchange_count = 0
    #print(f"署名リストの長さ：{len(signed)}")
    start = time.time()
    for sig in signed:
        AgentA = sig.vehicleA
        AgentB = sig.vehicleB
        taskA = sig.taskA
        taskB = sig.taskB
        routeA =[]
        routeB =[]
        if AgentA.exchange_flag == 0 and AgentB.exchange_flag == 0:
            for task in AgentA.tasks:
                routeA.append(task)
            for task in AgentB.tasks:
                routeB.append(task)
            A_weiht = AgentA.current_weight
            B_weiht = AgentB.current_weight

            exchange_successful = False
            if taskB is None:
                if AgentA.pop(taskA) :
                    if AgentB.add(taskA) :
                        exchange_successful = True
            # 交換が成功したかどうかを追跡するためのフラグ
            elif AgentA.pop(taskA) and AgentB.pop(taskB) :
                if AgentA.add(taskB)  and AgentB.add(taskA) :
                    exchange_successful = True

            if not exchange_successful:
                AgentA.tasks = []
                AgentB.tasks = []
                AgentA.tasks = routeA
                AgentB.tasks = routeB
                AgentA.current_weight = A_weiht
                AgentB.current_weight = B_weiht
                # print('交換失敗')
                # print(taskA.id)
                # print(taskB.id)
                # print(AgentA.tasks)
                # print(AgentB.tasks)
            else:
                AgentA.exchange_flag = 1
                AgentB.exchange_flag = 1
                exchange_count += 1

    # end = time.time()
    # time_diff = end - start
    #print(f"交換の実行時間: {time_diff} 秒")


   # for cnt in signed:
   #     #print(cnt)
        
    #未稼働車両の削除
    zzz = 0
    for car in vehicles:
        
        if len(car.tasks)== 0:
            no_runs.append(car)
            vehicles.remove(car)
            if car in vehicles:
                print("error")
                            #vehicles.pop(car)
            #del vehicles[zzz]
            #zzz += 1
            # id = 5 の行のインデックスを見つける
            indices_to_drop = bulletin_board.time_board[bulletin_board.time_board.id == car.id].index
            # これらの行を削除する
            bulletin_board.time_board = bulletin_board.time_board.drop(indices_to_drop)
        
            # 'stay_areas_bb' DataFrame についても同様に行う
            indices_to_drop =bulletin_board.area_board[bulletin_board.area_board.id == car.id].index
            bulletin_board.area_board = bulletin_board.area_board.drop(indices_to_drop)
        else:
            car.bulletin_update(max_xy,max_time,zones,n)
        zzz += 1
        flag =0
        # for pac in car.arrival_time_list:
        #     if pac.late_start_time - pac.earliest_start_time <= 0:
        #         flag ==1
        # # if flag == 1:
        #     for task in car.arrival_time_list:
        #         if task.late_start_time < 0:
        #             print(task.late_start_time)
        #             print(car.tasks)
        #             print("car id {seld.id}")
        #             for task_pac in car.arrival_time_list:
        #                 print(task_pac.task.ready_time)
        #                 print(task_pac.task.due_date)
        #                 print(task_pac.task.service_time)
        #                 print(task_pac.task.x_coordinate)
        #                 print(task_pac.task.y_coordinate)
        #                 print("PPPPPP")
    

#掲示板の更新
   # for car in vehicles:
    #    car.bulletin_update(max_xy,max_time,zones,n)

    for car in vehicles:
        car.step()
    filename = os.path.join(directory_name, f"step-{negotiate_steps+1}.txt")
    with open(filename, 'w') as f:
        for vehicle in vehicles:
            task_ids = [task.id for task in vehicle.tasks]
            # ファイル名を生成
            f.write(f"Vehicle {vehicle.id} has tasks {task_ids} with total weight {vehicle.current_weight}.\n")
        f.write(f"CVN {len(vehicles)} CRT {sum_travel_time(vehicles)}\n")
    filename = os.path.join(directory_name, f"TimeBoard-{negotiate_steps}.txt")
    bulletin_board.time_board.to_csv(filename,sep='\t',index = False)
    filename = os.path.join(directory_name, f"AreaBoard-{negotiate_steps}.txt")
    bulletin_board.area_board.to_csv(filename,sep='\t',index = False)
    plot_vehicle_routes(vehicles,directory_name,negotiate_steps)
    log_CVN.append(len(vehicles))
    log_CRT.append(sum_travel_time(vehicles))
    log_nego.append(exchange_count)
    filename = os.path.join(directory_name, "1log_main.txt")
    with open(filename, 'a') as f:
        # for i in range(len(log_nego)):
            # ファイル名を生成
        f.write(f"steps {negotiate_steps+1} CVN {log_CVN[negotiate_steps+1]} CRT {log_CRT[negotiate_steps+1]} n_neg {log_nego[negotiate_steps+1]}.\n")
