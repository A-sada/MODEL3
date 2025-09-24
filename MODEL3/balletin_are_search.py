from collections import Counter
import math
import pandas as pd
from classes import Task
from VRPTW_functions import euclidean_distance
# 全てのコードをまとめる（最終版）
# サンプルのタスクデータ
sample_tasks_car1 = [
    Task("T1", 10, 20, 5, 50, 200, 10),
    Task("T2", 40, 50, 6, 100, 300, 15),
    Task("T3", 70, 20, 4, 200, 400, 20)
]

sample_tasks_car2 = [
    Task("T4", 80, 60, 5, 50, 300, 10),
    Task("T5", 30, 40, 4, 120, 350, 15),
    Task("T6", 10, 10, 3, 200, 400, 10)
]
class Task1:
    def __init__(self, id, x_coordinate, y_coordinate, weight, ready_time, due_date, service_time):
        self.id = id  # タスク（顧客）のID
        self.x_coordinate = x_coordinate  # 配送先のx座標
        self.y_coordinate = y_coordinate  # 配送先のy座標
        self.weight = weight  # 荷物の重量またはサイズ
        self.ready_time = ready_time  # 配送可能な最早時間
        self.due_date = due_date  # 配送締切時間
        self.arrival = 0
        self.service_time = service_time  # サービスにかかる時間
# 時間帯の範囲を動的に設定する関数
def create_time_zones(T, num_zones):
    step = T // num_zones
    zones = {}
    for i in range(num_zones):
        start = i * step
        end = start + step - 1
        zones[chr(65 + i)] = (start, end)
    return zones

# タスク間の移動で通過するエリアを計算する関数（修正版）
# このバージョンでは、座標（数値）を返すようにします。
def passing_areas_coordinates(task1, task2):
    distance = euclidean_distance(task1, task2)
    steps = int(distance)
    coordinates = []
    if steps < 1:
        steps = 1
    for i in range(steps + 1):
        ratio = i / steps
        x = task1.x_coordinate + ratio * (task2.x_coordinate - task1.x_coordinate)
        y = task1.y_coordinate + ratio * (task2.y_coordinate - task1.y_coordinate)
        x = int(x)
        y = int(y)
        coordinates.append((x, y))
    return coordinates

# エリアを動的に計算する関数
def calculate_dynamic_area(x, y, X, n):
    cell_size = X / n
    row = chr(65 + int(y // cell_size))
    col = int(x // cell_size) + 1
    return f"{row}{col}"
#アルファベットが，y座標
#数字が，x座標

# 時間帯ごとに最も長く滞在するエリアを動的に計算する関数（修正版）
def most_stayed_area_dynamic(tasks, X, T, zones, n, dep_x, dep_y):
    tasks = [Task(0, dep_x, dep_y, 0, 0, 0, 0)] + tasks[:] + [Task(0, dep_x, dep_y, 0, 0, 0, 0)]
    time_zones = zones
    df_row = {}
    yet_dep = tasks[1].ready_time - euclidean_distance(tasks[0],tasks[1])
    # for zone, (start_time, end_time) in time_zones.items():
    #     area_counter = Counter()
    #     for i in range(len(tasks) - 1):
    #         task1 = tasks[i]
    #         task2 = tasks[i + 1]
    #         if task1.due_date >= start_time and task2.ready_time <= end_time:
    #             # タスク間移動のエリアをカウント
    #             coordinates = passing_areas_coordinates(task1, task2, X)
    #             areas = [calculate_dynamic_area(x, y, X, n) for x, y in coordinates]
    #             area_counter.update(areas)
                
    #             # タスク1のサービス時間をカウント
    #             service_area = calculate_dynamic_area(task1.x_coordinate, task1.y_coordinate, X, n)
    #             area_counter[service_area] += task1.service_time
                
    #             # タスク2までの待ち時間をカウント
    #             wait_time = max(task2.ready_time - task1.due_date, 0)
    #             wait_area = calculate_dynamic_area(task1.x_coordinate, task1.y_coordinate, X, n)
    #             area_counter[wait_area] += wait_time
        
    #     most_common_area, _ = area_counter.most_common(1)[0] if area_counter else (None, None)
    #     if end_time <= yet_dep:
    #         most_common_area = None
    #         #most_common_area, _ = (None, None)
    #     if start_time >= back_dep:
    #         #most_common_area, _ = (None, None)
    #         most_common_area = None
    #     df_row[zone] = most_common_area
    # return df_row
    current_time = 0
    cordinates =[]
    for i in range(len(tasks) - 1):
        task1 = tasks[i]
        task2 = tasks[i + 1]
        current_time = current_time + euclidean_distance(task1,task2)
        cordinates.extend(passing_areas_coordinates(task1,task2))
        if current_time<= task2.ready_time:
            for i in range(task2.ready_time-current_time):
                cordinates.append((task2.x_coordinate,task2.y_coordinate))
            # cordinates+=[(task2.x_coordinate,task2.y_coordinate)]*(task2.ready_time-current_time)
            current_time = task2.ready_time
        for i in range(task2.service_time):
            cordinates.append((task2.x_coordinate,task2.y_coordinate))
        # cordinates+=[(task2.x_coordinate,task2.y_coordinate)]*task2.service_time
        current_time += task2.service_time
        back_dep = current_time 
    # print(cordinates)
    for zone, (start_time, end_time) in time_zones.items():
        area_counter = Counter()
        if end_time < yet_dep:
            df_row[zone] = None
        elif start_time > back_dep:
            df_row[zone] = None
        elif start_time <= yet_dep and end_time >= yet_dep and start_time <= back_dep and end_time >= back_dep:
            #車両が出発．帰還する時間帯:
            areas = [calculate_dynamic_area(x, y, X, n) for x, y in cordinates]
            area_counter.update(areas)
            most_common_area, _ = area_counter.most_common(1)[0] if area_counter else (None, None)
            df_row[zone] = most_common_area
        elif start_time < yet_dep and yet_dep <= end_time:#車両が出発する時間帯
            if yet_dep == end_time:
                x,y = cordinates[0]
                areas =calculate_dynamic_area(x, y, X, n)
            else:
                areas = [calculate_dynamic_area(x, y, X, n) for x, y in cordinates[0 : end_time-yet_dep+1]]
            area_counter.update(areas)
            most_common_area, _ = area_counter.most_common(1)[0] if area_counter else (None, None)
            df_row[zone] = most_common_area
        elif start_time <= back_dep and end_time > back_dep:#車両が帰ってくる時間帯

            areas = [calculate_dynamic_area(x, y, X, n) for x, y in cordinates[start_time - yet_dep  :]]
            area_counter.update(areas)
            most_common_area, _ = area_counter.most_common(1)[0] if area_counter else (None, None)
            df_row[zone] = most_common_area
        else:

            areas = [calculate_dynamic_area(x, y, X, n) for x, y in cordinates[start_time-yet_dep : end_time-yet_dep+1]]
            area_counter.update(areas)
            most_common_area, _ = area_counter.most_common(1)[0] if area_counter else (None, None)
            df_row[zone] = most_common_area
    return df_row



# DataFrameを逐次更新する関数を追加
def update_stay_areas(df, car_id, new_data):
    mask = df['id'] == car_id
    if not mask.any():
        return df
    for zone, area in new_data.items():
        df.loc[mask, zone] = area
    return df


"""
# サンプルデータでテスト
T = 1000  # 時間帯の終端
num_zones = 6  # 分割数
n = 4  # エリアの分割数（4x4）
X = 100  # マップのサイズ

df_row_car1 = most_stayed_area_dynamic(sample_tasks_car1, X, T, num_zones, n)
df_row_car1['id'] = 'Car1'
df_row_car2 = most_stayed_area_dynamic(sample_tasks_car2, X, T, num_zones, n)
df_row_car2['id'] = 'Car2'

# DataFrameに追加
stay_areas_df_dynamic = pd.DataFrame(columns=['id'] + list(create_time_zones(T, num_zones).keys()))
stay_areas_df_dynamic = stay_areas_df_dynamic.append(df_row_car1, ignore_index=True)
stay_areas_df_dynamic = stay_areas_df_dynamic.append(df_row_car, ignore_index=True)
2
# DataFrameを逐次更新（テスト）
new_data_car1 = {'A': 'A1', 'B': 'B2', 'C': 'C1'}
new_data_car2 = {'A': 'A2', 'B': 'B3', 'C': 'C2'}

# 新しく計算された滞在エリアでDataFrameを更新
new_data_car1 = most_stayed_area_dynamic(sample_tasks_car1, X, T, num_zones, n)
new_data_car2 = most_stayed_area_dynamic(sample_tasks_car2, X, T, num_zones, n)

update_stay_areas(stay_areas_df_dynamic, 'Car1', new_data_car1)
update_stay_areas(stay_areas_df_dynamic, 'Car2', new_data_car2)


# コード全体の表示
stay_areas_df_dynamic
"""
