# 車両エージェントの交渉戦略（VRPTW / タスク交換）

この文書は、`MODEL3` 配下の VRPTW（時間窓付き車両経路問題）実験における「車両（Vehicle）エージェントの交渉戦略」を、実装に即して整理した解説です。  
対象は主に `Vehicle_BASE` と、その戦略サブクラス（`Strategy_Vehicle_ver1.Vehicle`, `Strategy_Vehicle_verZ.Vehicle`）、および NegMAS を用いた交渉エージェント `VehicleNegotiator` です。

---

## 1. 全体像：何を交渉しているか

この実験の交渉は「タスク交換（swap）」です。1 回の交渉で合意される結果は次の形を取ります。

- `taskA`: 車両A（提案側）が手放す（相手に渡す）候補タスク（または `None`）
- `taskB`: 車両B（応答側）が手放す（相手に渡す）候補タスク（または `None`）

合意が `{"taskA": x, "taskB": None}` の場合は「B が x を受け取る一方向の移譲」に相当します。  
`{"taskA": x, "taskB": y}` の場合は「交換（A は y を受け取り、B は x を受け取る）」です。

関連クラス:
- `MODEL3/classes.py`: `Task`, `Offer`, `Nego`, `Agree`, `Balletin` などのデータクラス

---

## 2. 交渉ループ（メインループ）の流れ

交渉は `MODEL3/VRPTW-main.py` が「離散ステップ」で回します。概略は以下です。

1. 初期解生成でタスクを車両へ割当てる（`assign_tasks_to_vehicles_with_insert(...)`）
2. 各ステップで掲示板（`Balletin`）の時刻 `n_steps` を進める
3. 各車両が「交渉したい相手」と「放出したいタスク」を列挙して `Offer` を生成する
4. 受け手候補（車両B）が `check_offer()` を通過したものだけが「実際に交渉を走らせるペア」になる
5. 交渉ペアごとに NegMAS の SAO 交渉を実行し、合意が得られたら `Agree(vehicleA, vehicleB, taskA, taskB)` として記録する
6. 合意候補を「署名（実行）するか」を各車両が選別（`sign_contracts()`）
7. グローバルに実行順序を決め、実際にタスクを `pop()` / `add()` で交換する
8. 交換後に `bulletin_update()` で掲示板を更新し、次ステップへ

ポイント:
- **「交渉（合意生成）」と「署名（実行選別）」が分離**されています。
- 交渉はペア単位で並列に走る想定ですが、実装上は `for neg in negotiation_list:` で逐次に実行されています。
- 実行フェーズでは `exchange_flag` により「同一ステップで 1 車両が複数回交換しない」制約がかかります（`MODEL3/VRPTW-main.py`）。

---

## 3. 役割分担：Vehicle と Negotiator

### 3.1 Vehicle（車両本体）が持つもの
`MODEL3/Vehicle.py` の `Vehicle_BASE` は、交渉の土台となる状態を保持します。

- ルート: `self.tasks: list[Task]`
- 容量: `self.max_weight`, `self.current_weight`
- 交渉の事務情報:
  - `offer_nego_list`: このステップで自車が生成した `Offer` の一覧
  - `next_nego`: 「交渉ID → 提案タスク（taskA候補）」の対応表
  - `offer_flag`: その交渉で自分が提案側（vehicleA）かどうか
- 掲示板: `self.bulletin_board: Balletin`
- 交渉用エージェント生成: `make_neg_agent(...)`

### 3.2 VehicleNegotiator（交渉エージェント）がやること
`MODEL3/Vehicle_Negotiatior.py` の `VehicleNegotiator` は NegMAS の `SAONegotiator` として、`propose()` と `respond()` を実装します。

重要な設計として、この実装では **提案側（is_vehicle_a=True）の受諾のみが機能しやすい**形になっています。

- 提案側（A）:
  - `propose()` は常に `{"taskA": self.task_a, "taskB": None}` を投げる
  - `respond()` で相手提案（B の `taskB`）を評価して ACCEPT/REJECT する
- 被提案側（B）:
  - `propose()` で「A の taskA を受け取る代わりに、どの taskB を差し出すか」を順番に提案する
  - `respond()` は基本的に `REJECT_OFFER`（＝A の初回提案を受諾しない）

結果として実質的に、
- A が `taskA` を固定で出す（初回は無償で渡す形）
- B が `taskB` を変えながら譲歩案を作る
- A が採択（ACCEPT）した時点で合意が成立

という「B が譲歩列を提示し、A が承認する」片側承認型に近い動きになります。

---

## 4. 掲示板（Balletin）と近傍探索

### 4.1 掲示板に載っている情報
`MODEL3/classes.py` の `Balletin` は共有状態として、少なくとも次を保持します。

- `time_board`（DataFrame）: 車両ごとの `slack_time`, `departure_time`, `return_time`
- `area_board`（DataFrame）: 車両ごとの「滞在エリア」履歴（時間帯ごと）
- `n_steps` / `max_steps`: 交渉の進行度（時間による閾値制御に利用）
- `zones`: 時間帯（ゾーン）定義
- `X`, `n`: 空間を分割するためのパラメータ

更新は `MODEL3/Vehicle.py` の `bulletin_update(...)` が担います。

- `time_board.slack_time` は `calculate_slack_time(...)`（ルート全体の余裕の合計）で更新
- `departure_time`, `return_time` はルート先頭・末尾タスクとデポ距離から近似更新
- `area_board` は `most_stayed_area_dynamic(...)` の結果で更新

### 4.2 交渉相手の候補を絞るロジック
戦略車両（ver1 / verZ）は、提案生成 `offer_on_negotiation(...)` で候補車両を絞ります。

主なフィルタ:
- 空間近傍: `calculate_dynamic_area(...)` と `find_vehicles_in_neighboring_areas(...)`（`MODEL3/VRPTW_functions.py`）
- 時間近傍: タスクの `ready_time` が属する時間帯 `find_time_zone(...)`
- 稼働時間フィルタ: `find_available_vehicles(...)` が `time_board` を見て「このタスクの時間窓とズレた車両」を抽出

注意:
- `find_available_vehicles(...)` は「時間窓と合わない車両」を返しているため、名称（available）と実際の条件が逆に見えます（実装の意図に依存）。

---

## 5. 提案戦略（offer_on_negotiation）

### 5.1 共通の骨格（ver1 / verZ）
`MODEL3/Strategy_Vehicle_ver1.py` および `MODEL3/Strategy_Vehicle_verZ.py` の `offer_on_negotiation(...)` は概ね同型です。

- タスク数が少ない（`len(self.tasks) < 3`）とき:
  - ルート上の各タスクを「交換候補（taskA）」として列挙し、近傍車両すべてに `Offer(vehicleA=self, vehicleB=neighbor, task=task)` を投げる
- タスク数が多いとき:
  - タスク座標に K-means を適用し、**最もタスクが集中しているクラスタ**（過密領域）を推定
  - 「過密クラスタ以外」のタスク（=外れ値・散在タスク）を `other_cluster_tasks` として交換候補にする
  - さらに積載が限界に近い場合、重いタスクを `other_cluster_tasks` に追加して放出候補を増やす
  - 各候補タスクについて近傍車両へ `Offer` を生成する

### 5.2 K-means による「放出タスク候補」の意図
K-means を使う理由は「ルートが空間的に分散しているタスク」を見つけやすくするためです。

- ルートが一つの塊（近いエリア）なら配送効率が良い
- そこから外れたタスクは移動距離・時間窓違反リスクを増やしやすい

そのため「最大クラスタ以外」を候補にするのは、
> “ルートのまとまりを壊しているタスクを、交渉で他車へ移して局所的に改善する”
というヒューリスティックです。

---

## 6. 交渉（SAO）の中身：提案と応答

交渉の実行は `MODEL3/Negotiator.py` の `Nego1(...)` が担当します。

- outcome 空間（交渉結果候補）:
  - `{"taskA": taskA, "taskB": taskB}` の全組合せ
  - `taskA` は `vehicleA.tasks + [None]`
  - `taskB` は `vehicleB.tasks + [None]`
- SAO の最大ステップ数: `n_steps=10`

### 6.1 A の提案（VehicleNegotiator.propose）
提案側（A）は、交渉開始時に `Vehicle_BASE.start_negotiation(...)` でセットされた `self.propose_task` を `taskA` として固定し、

- `{"taskA": self.task_a, "taskB": None}`

を提示します（初回は無償移譲に近い形）。

### 6.2 B の提案（VehicleNegotiator.propose）
被提案側（B）は初回に A の提示 `taskA` を受け取り、それに対して「どの `taskB` を差し出すか」を列挙します。

- `make_remove_list(taskA)` が `remove_list` を生成
  - 期限違反の可能性があるタスク（`earliest_start_time > due_date`）を優先的に並べる
  - さらに `calculate_cost_saving(...)` を用いて「taskB を差し出したときのコスト改善が大きい順」にソート
  - `"0"` は `taskB=None`（差し出しなし）を表すダミー要素
- 各提案で `remove_list.pop(0)` を `taskB` として提示していく

### 6.3 A の応答（VehicleNegotiator.respond）
A は B の提案 `taskB` を受け取り、コストと容量制約で受諾判断します（`MODEL3/Vehicle_Negotiatior.py`）。

- 容量制約: `current_weight + taskB.weight <= max_weight`（`taskB is not None` の場合）
- コスト指標: `calculate_cost_saving(arrival_time_list, taskA, taskB, tasks, bulletin_board)`
- 進行度による閾値:
  - `cost_border = 1 * (max_steps - n_steps) / max_steps + 100`
  - タスク数が少ない場合（`len(self.tasks) < 3`）は「改善（cost<0）なら受諾」に近い
  - タスク数が多い場合は `cost < cost_border` を満たせば受諾

### 6.4 B の応答（VehicleNegotiator.respond）
現状の実装では B は原則 `REJECT_OFFER` を返します。  
したがって合意成立は「B が提案し、A が受諾する」形に寄ります（A の提案が受諾されて合意になる経路は取りにくい）。

---

## 7. 署名戦略（sign_contracts）：合意を実行候補に絞る

交渉で得た合意 `Agree` は、そのまま即実行されません。各車両が `sign_contracts(...)` で「このステップで実行したい合意」を返します。

対象実装:
- `MODEL3/Strategy_Vehicle_ver1.py: sign_contracts`
- `MODEL3/Strategy_Vehicle_verZ.py: sign_contracts`

### 7.1 署名の基本ロジック（両者共通）

1. 合意ごとに「自分が手放す側のタスク（自車のルートに入っている方）」を `task` として取り出す  
2. `task` ごとに `calculate_cost_saving(agreements)` を計算し、同じ `task` に対する合意候補をコストの良い順に蓄積する  
3. 各 `task` について、進行度で変化する閾値 `cost_border` を満たす合意候補を署名リストへ入れる  
4. 署名リスト全体を `cost` 昇順に並べ替えて返す

### 7.2 進行度で変化する閾値（ver1 / verZ）
両戦略とも、`bulletin_board.n_steps / bulletin_board.max_steps` に応じて閾値が変わります。

（例：ver1 の `sign_contracts`）
- 前半（<0.5）: `cost_border` が大きくなりやすい（多くを受け入れる）
- 後半（>=0.5）: `cost_border` が相対的に厳しくなる（選別が強くなる）

実際の式は実装を参照してください（`MODEL3/Strategy_Vehicle_ver1.py`, `MODEL3/Strategy_Vehicle_verZ.py`）。

---

## 8. コスト関数（calculate_cost_saving）の考え方

ここが「交渉戦略の目的関数」です。車両は合意によって自分のルートがどう変わるかを、複数の観点でスカラーに落として比較します。

### 8.1 verZ のコスト分解（典型）
`MODEL3/Strategy_Vehicle_verZ.py: calculate_cost_saving`

- `slack_cost`: スラックタイム（余裕）の変化
- `over_cost`: 時間窓違反（`earliest_start_time > due_date`）の量の変化
- `distance_cost`: 移動距離の変化（近似）

加重和（概念的）:

```
cost_saving
  = (- slack_late) * slack_cost
  + over_late      * over_cost
  + distance_late  * distance_cost
  + （任意）RL/Q による補正
```

特徴:
- `over_late` が `n_steps` とともに増える（後半ほど時間窓違反を強く嫌う）
- `distance_late` は固定に近い
- `slack_cost` は「余裕が増えるほど有利」になるよう符号設計されています（実装上は差分の取り方に注意）

### 8.2 共通関数版（VRPTW_functions.calculate_cost_saving）
`MODEL3/VRPTW_functions.py: calculate_cost_saving` は、Negotiator 側（`VehicleNegotiator.respond` や `make_remove_list`）でも参照されます。

- コスト分解（slack / over_window / distance）は同様
- 係数は verZ の実装と一致しない場合があります（それぞれ別にチューニングされているため）

---

## 9. RL ルートプランナー統合（ver1 に加わる要素）

この実験では、事前学習済み DQN による「ルート評価」をコストに混ぜられるように作られています。

### 9.1 有効化と読み込み
`MODEL3/VRPTW-main.py`:

- 既定チェックポイント: `MODEL3/pretrained/planner_checkpoint.pt`
- 環境変数 `RL_PLANNER_CHECKPOINT` で上書き可能
- 見つからない場合は RL 評価をスキップ（従来のヒューリスティックのみ）

### 9.2 ver1 の RL 補正（ルート比較ベース）
`MODEL3/Strategy_Vehicle_ver1.py` には `_route_planner_cost_adjustment(...)` があり、

- 現在ルート `self.tasks` の RL 指標（例: `total_reward`, `unassigned_tasks`）と
- 交換後の候補ルートの RL 指標

の差分から、追加の `rl_cost` を作ります。

概念:
- `reward_delta` が増える（良い）なら `rl_cost` は減る（受諾しやすい）
- `unassigned_tasks` が増える（悪い）なら大きなペナルティ

### 9.3 Q 重要度（タスク単位）の補正
`Vehicle_BASE` と `VehicleNegotiator` は、プランナーが提供する `discounted_q_task_scores(...)` を利用して、

- 取り除くタスク（remove_task）の Q を高く見積もる（重要）なら「手放しにくい」
- 受け取るタスク（give_task）の Q が高いなら「受け取りやすい」

という方向でコストを補正します（`q_importance_weight` で重み付け）。

---

## 10. 実行フェーズ（交換の適用）と失敗時ロールバック

`MODEL3/VRPTW-main.py` の実行部は以下の流れです。

1. 合意候補（`signed`）を一定の尺度で並べ替える
   - 各車両の `sign_contracts()` 返り値の「順位」を合算して `cost` として扱う
2. 上位から順に、両者が未交換（`exchange_flag == 0`）なら適用を試す
3. `pop()` / `add()` に失敗したら **ルートと重量を丸ごと復元**してロールバックする

ここで `Vehicle_BASE.add(...)` は `least_cost_time_sensitive_insertion(...)` を使って「挿入位置」を探索し、挿入できなければ失敗します。

---

## 11. ログ：交渉オファーの記録

`MODEL3/VRPTW-main.py` は `output_files/<timestamp>/negotiation_offers.csv` を作り、交渉の各提案を追記します。  
書き込みは `MODEL3/Vehicle_Negotiatior.py: _log_offer(...)` が担当します。

CSV の主な列:
- `global_step`（掲示板の `n_steps`）
- `negotiation_id`
- `sender_vehicle_id`, `recipient_vehicle_id`
- `role`（A/B）
- `mechanism_step`
- `taskA_id`, `taskB_id` と各タスクの `ready_time`, `due_date`, `weight`

このログは「なぜ合意に至った/至らなかったか」を後から分析する際の基礎データになります。

---

## 12. 拡張の指針（新しい戦略を足す場合）

交渉戦略は大きく分けて 3 つの差し替えポイントがあります。

1. **提案生成**（誰に・何を渡すか）: `offer_on_negotiation(...)`
2. **交渉中の譲歩列**（B が何を返すか）: `before_negotiation()` と `over_task` / `remove_list`
3. **署名（実行選別）**: `sign_contracts(...)` と `calculate_cost_saving(...)`

推奨:
- 距離・時間窓・掲示板参照などの共通ロジックは `MODEL3/VRPTW_functions.py` に寄せる
- RL を使う場合は `Vehicle_BASE.set_route_planner(...)` / `evaluate_route_with_planner(...)` 契約に乗せる

