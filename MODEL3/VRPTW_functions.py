import math
from classes import Task,pac_task,Agree,pac_task

ENABLE_ROUTE_PLOTS = False  # disable route plotting during runs

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# タスク間または車両とタスク間のユークリッド距離を計算する関数
def euclidean_distance(task1, task2):
    return (int)(math.sqrt((task1.x_coordinate - task2.x_coordinate)**2 + (task1.y_coordinate - task2.y_coordinate)**2))

def slack_time_list( route,slist):
        total_time = 0  # total time spent so far in the route
        slack_time = 0
        
        for i in range(len(route)):
            task = route[i]           
            
            if i != 0:
                total_time += euclidean_distance(route[i-1], task)
                slack_time[i] = max(0, task.due_date - total_time)
                # Wait for the task's service to start if necessary
                if total_time < task.ready_time:
                    total_time = task.ready_time
                
                total_time += task.service_time
            else:
                distance = euclidean_distance(Task(0,0,0,0,0,0,0), task)
                if distance < task.ready_time:
                    distance = task.ready_time
                total_time += distance
                
                total_time += task.service_time
            
        return slack_time

def find_time_zone(t, zones):
    for zone_name, (start, end) in zones.items():
        if start <= t <= end:
            return zone_name
    return False  # t がどの時間帯にも属さない場合

import pandas as pd

def find_neighboring_areas(area):
    alphabet_part = area[0]  # 最初の文字（アルファベット）
    number_part = int(area[1:])  # 2文字目以降（数字）
    # アルファベットの前後の値を決定
    prev_alphabet = chr(ord(alphabet_part) - 1) if ord(alphabet_part) > ord('A') else None
    next_alphabet = chr(ord(alphabet_part) + 1) 

    # 隣接するエリアのリストを作成
    neighboring_areas = []
    for alpha in [prev_alphabet, alphabet_part, next_alphabet]:
        if alpha:
            for num in [number_part - 1, number_part, number_part + 1]:
                if num >= 0:  # エリア番号は正の数でなければならない
                    neighboring_areas.append(f"{alpha}{num}")

    return neighboring_areas

def find_vehicles_in_neighboring_areas(time_zone, area, df):
    neighbor_areas = find_neighboring_areas(area)
    vehicles = []
    for neighbor_area in neighbor_areas:
        # DataFrameの該当する時間帯の列で隣接エリアを検索し、該当する車両IDをリストに追加
        vehicles_in_area = df[df[time_zone] == neighbor_area]['id'].tolist()
        vehicles.extend(vehicles_in_area)
    return vehicles

def find_vehicle_by_id(vehicle_id, vehicles):
    for vehicle in vehicles:
        if vehicle.id == vehicle_id:
            return vehicle
    return None  # IDと一致するvehicleが見つからなかった場合

def earliest_start_time_list(tasks : list[pac_task],dep_x,dep_y):
    #リストで計算する
    dep_task = Task(0,dep_x,dep_y,0,0,0,0)
    if len(tasks) == 0:
        return
    current_time = max(euclidean_distance(dep_task,tasks[0].task),tasks[0].task.ready_time)
    i = 0 
    for task in tasks:
        if i == 0:
            task.earliest_start_time = task.task.ready_time
            current_time = task.task.ready_time
            current_time += task.task.service_time
            i += 1
            pre_task =task
        else:
            current_time += euclidean_distance(pre_task.task,task.task)
            task.earliest_arrival_time = current_time
            task.earliest_start_time = max(current_time,task.task.ready_time)
            current_time = task.earliest_start_time

            current_time += task.task.service_time
            pre_task =task
    return
            


def calculate_earliest_start_time(previous_task, current_task,current_time):
    # 前のタスクのサービス終了時刻がcurrent_time
    current_time += euclidean_distance(previous_task.task,current_task.task)
    if current_task.task.ready_time > current_time:
        current_task.earliest_arrival_time = current_time
        current_time = current_task.task.ready_time
    else:
        current_task.earliest_arrival_time = current_time
    return current_time

def latest_start_time_list(tasks):
    # 関数内でユークリッド距離を計算するヘルパー関数

    # 最後のタスク（最も遅い締切時間を持つタスク）の最遅開始時間を設定（サービス開始の締切時間とする）
    last_task = tasks[len(tasks)-1]
    last_task.late_start_time = last_task.task.due_date

    # 残りのタスクに対して逆順に最遅開始時間を計算
    for i in range(len(tasks) - 2, -1, -1):
        current_task = tasks[i]
        next_task = tasks[i + 1]

        # 次のタスクまでの移動時間を計算
        time_to_next_task = euclidean_distance(current_task.task, next_task.task)

        # 最遅開始時間を計算
        # 次のタスクの最遅開始時間から、移動時間と現在のタスクのサービス時間を引く
        current_task.late_start_time = min(next_task.late_start_time - time_to_next_task - current_task.task.service_time, 
                                           current_task.task.due_date)

    # 更新されたタスクリストを返却
    return tasks

# 注: この関数は単純化のためにユークリッド距離を使用しています。
# 実際のアプリケーションでは、より複雑な距離計算や時間制約の考慮が必要です。

import copy


# この関数は特定の合意結果のコスト削減を計算します。
def calculate_cost_saving(task_list,taskA,taskB,route,bulletin_board):
#この関数を利用する車両のリスト（パックリスト）と，交換するタスクたち，bulletin_boardを引数にとる
#bulletin_boardはbulletin_board.pyのbulletin_boardクラスのインスタンス
    #print("taskA",taskA)
    #print("taskB",taskB)
    #print(route)
    remove_task = taskA if taskA in route else None
    if remove_task == None:
        remove_task = taskB if taskB in route else None


    give_task = taskA if taskA not in route else None
    if give_task == None:
        give_task = taskB if taskB not in route else None


    # 交換でのタスクの交換によるスラックタイムの差分・コストの変化を計算
    slack_cost = calculate_differ_slack(task_list,remove_task,give_task,route,bulletin_board)
    # 交換でのタスクの交換によるover_windowの差分・コストの変化を計算
    over_cost = caluculate_differ_over_window(task_list,remove_task,give_task,route,bulletin_board)
    # 交換でのタスクの交換による距離の差分・コストの変化を計算
    distans_cost = calculate_differ_distance(route,remove_task,give_task,bulletin_board,task_list)

    slack_late = 0.5


    #over_lateは，時間が進むにつれて値を大きくする
    #現在の時間はbulletin_board.n_stepで取得できる
    #最大時間はbulletin_board.max_stepで取得できる
    #over_costは前半ではほぼ無視をして，後半では大きくする
    #最後の25％の時間ではover_costをかなり大きくする
    over_late = 0.5 * (bulletin_board.n_steps / bulletin_board.max_steps) ** 2
    distance_late = 0.5
    cost_saving = slack_late * slack_cost + over_late * over_cost + distance_late * distans_cost
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
    #スラックタイムが減れば負の値を返す


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

def least_cost_time_insertion_index(route,new_task,pac_list,bulletin_board):
        if not isinstance(new_task, Task):  # 仮定として Task というクラスが存在するとします。
            print("エラー: 'new_task' が Task オブジェクトではありません。")
            return False
        min_cost = float('inf')
        best_position = None
        cost = 0
        if len(route) == 0:
            return False
        # 車両の開始位置から新しいタスクまでの距離を計算
        start_task = Task(0, bulletin_board.dep_x, bulletin_board.dep_y, 0, 0, 0, 0)  # 仮の開始位置
        travel_time_from_start = euclidean_distance(start_task, new_task)
        if travel_time_from_start <= new_task.due_date :
            if travel_time_from_start + new_task.service_time + euclidean_distance(new_task, route[0]) <= route[0].due_date: 
                list = copy.deepcopy(pac_list)
                list.insert(0,pac_task(new_task))
                earliest_start_time_list(list,bulletin_board.dep_x,bulletin_board.dep_y)
                latest_start_time_list(list)
                cost = calculate_slacktime(list)
                if min_cost > cost:
                    min_cost = cost
                    best_position = 0


        # 各タスク間での新しいタスクの挿入を試みる
        for i in range(len(route) - 1):
            current_task = route[i]
            next_task = route[i + 1]

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
                list = copy.deepcopy(pac_list)
                list.insert(i+1,pac_task(new_task))
                earliest_start_time_list(list,bulletin_board.dep_x,bulletin_board.dep_y)  
                latest_start_time_list(list)
                cost = calculate_slacktime(list)
                if min_cost > cost:
                    min_cost = cost
                    best_position = i+1

        if best_position != None:
            return best_position
        return None

def remove_task(task,route):
    #ルートはパッケージのリストに限る
    #routeからtaskを一致するものをもつクラスを削除する
    if task == None:
        return route
    for i in range(len(route)):
        if route[i].task == task:
            del route[i]
            break
    return route



def add_task(task,pac_list,route,bulletin_board):
    #routeはパッケージのリストに限る
    #routeにtaskを追加する
    if task == None:
        return pac_list
    index = least_cost_time_insertion_index(route,task,pac_list,bulletin_board)
    if index == None:
        for i in range(len(route)):
            if pac_list[i].task.due_date > task.due_date:
                index = i
                break
    if index == None:
        return pac_list
    pac_list.insert(index,pac_task(task))
    return pac_list

def cal_travel_time(route,dep_x,dep_y):
    time = 0
    current_time = 0
    if len(route) == 0:
        return 0
    for i in range(len(route)):
        if i == 0:
            time += euclidean_distance(Task(0,dep_x,dep_y,0,0,0,0),route[i])
        else:
            time += euclidean_distance(route[i-1],route[i])
    time += euclidean_distance(route[len(route)-1],Task(0,dep_x,dep_y,0,0,0,0))
    return time

def sum_travel_time(car_list):
    time = 0
    for car in car_list:
        time += cal_travel_time(car.tasks,car.dep_x,car.dep_y)
    return time
import os
def plot_vehicle_routes(vehicles,directory_name,negotiate_steps):
    if not ENABLE_ROUTE_PLOTS or plt is None:
        return
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', '^', 's', 'p', '*', 'x', '+', 'D', 'h', 'v']
    plt.figure()
    for i, vehicle in enumerate(vehicles):
        # 各車両のタスクから座標を抽出
        x_coords = [task.x_coordinate for task in vehicle.tasks]
        y_coords = [task.y_coordinate for task in vehicle.tasks]

        # ルートをプロット
        plt.plot(x_coords, y_coords, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f'Vehicle {vehicle.id}')


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Routing')
    plt.legend()
    save_path = os.path.join(directory_name, f'route_{negotiate_steps}.png')
    plt.savefig(save_path)
    plt.close()
