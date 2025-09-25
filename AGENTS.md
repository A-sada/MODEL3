# Agent Architecture

## System Overview
- The VRPTW experiment uses multiple vehicle agents that cooperate through task exchanges managed by negotiations.
- `MODEL3/VRPTW-main.py` drives the discrete negotiation loop, keeping shared state in a `Balletin` bulletin board.
- Agents rely on helper data classes from `MODEL3/classes.py` (`Task`, `Offer`, `Agree`, `Balletin`, etc.) to exchange structured information.

## Core Agent Types

### Vehicle_BASE (`MODEL3/Vehicle.py`)
- Maintains the per-vehicle route (`tasks`), capacity bookkeeping, and the negotiation bookkeeping lists (`offer_nego_list`, `next_nego`, `arrival_time_list`).
- Produces negotiation offers via `offer_on_negotiation`, mixing spatial filters with availability checks from the shared bulletin boards.
- Evaluates feasibility with scheduling helpers (`check_task`, `least_cost_time_sensitive_insertion`, `calculate_slack_time`) and records bulletin updates with `bulletin_update`.
- Participates in contract execution through `sign_contracts` and `accept_offer`, and delegates the actual bargaining to a `VehicleNegotiator`.
- Optionally integrates the RL route planner through `set_route_planner` / `evaluate_route_with_planner`, allowing external scoring of candidate routes.

### Strategy Vehicles
Specific negotiation and signing policies live in strategy subclasses that inherit from `Vehicle_BASE`.

#### `Strategy_Vehicle_ver1.Vehicle` (`MODEL3/Strategy_Vehicle_ver1.py`)
- Uses K-means clustering over customer coordinates to discover overloaded route areas and generate offer candidates even when the vehicle carries many tasks.
- Dynamically combines spatial neighbor searches (`calculate_dynamic_area`, `find_vehicles_in_neighboring_areas`) with bulletin timing filters to pick partners.
- During signing it scores `Agree` objects via `calculate_cost_saving` and applies time-dependent acceptance thresholds that become stricter as `bulletin_board.n_steps` grows.

#### `Strategy_Vehicle_verZ.Vehicle` (`MODEL3/Strategy_Vehicle_verZ.py`)
- Extends the base signing logic with a cost decomposition (`calculate_differ_slack`, `caluculate_differ_over_window`, `calculate_differ_distance`) to favour agreements that lower lateness and travel.
- Builds prioritized task removal lists (`make_remove_list`, `over_task`) that guide both proposal generation and counter-offers.
- Provides additional helpers such as `least_cost_time_insertion_index` to compute insertion positions consistent with time windows.

## Negotiation Agents

### VehicleNegotiator (`MODEL3/Vehicle_Negotiatior.py`)
- Wraps each vehicle for use with `negmas` SAO mechanisms, exposing `propose` and `respond` policies tailored to whether the vehicle initiated the negotiation.
- Stores the first received offer to anchor bilateral exchanges and produces a sequence of counter-proposals based on `remove_list` prioritisation.
- Evaluates offers using `calculate_cost_saving` from the owning vehicle plus dynamic thresholds derived from the bulletin board, and records offers to `negotiation_log_path`.

### Negotiation Mechanism (`MODEL3/Negotiator.py`)
- `Nego1` constructs an `SAOMechanism` over all combinations of `taskA`/`taskB` outcomes for the two negotiators and runs it for up to 10 steps, returning the final `negmas` result.

## Shared Infrastructure
- `Balletin` (`MODEL3/classes.py`) is the shared bulletin board that keeps each vehicle's slack time, operating window, and area occupancy. Vehicles call `bulletin_update` each step to refresh it.
- Time/space utilities in `MODEL3/VRPTW_functions.py` (e.g. `find_time_zone`, `find_vehicles_in_neighboring_areas`, `calculate_cost_saving`) supply the common calculations needed by both vehicles and negotiators.
- `MODEL3/rl_route_planner.py` defines an optional DQN-based planner (`build_default_planner`) that vehicles can consult to rank task sequences before or after negotiations.

## Interaction Cycle (`MODEL3/VRPTW-main.py`)
1. Load tasks, instantiate vehicles, and attach the shared `Balletin` and optional RL planner.
2. At each negotiation step vehicles reset step-local state and push proposals via `offer_on_negotiation`.
3. Offers that pass `check_offer` create `Nego` entries; initiators register them through `accept_offer`.
4. Paired vehicles spin up two `VehicleNegotiator` instances and run `Negotiator.Nego1`, logging every offer to `output_files/.../negotiation_offers.csv`.
5. Successful agreements are filtered in `sign_contracts`; accepted task swaps update vehicle routes and the bulletin board before the next round.
6. Metrics (`CVN`, `CRT`, etc.) and full routes are written under `output_files/<timestamp>` for later analysis.

## Extending the Agent Set
- To add a new vehicle strategy, subclass `Vehicle_BASE`, override offer generation (`offer_on_negotiation`), signing (`sign_contracts`), or cost evaluation helpers, and register the class in the initialisation script that creates vehicles.
- Custom negotiation behaviours can be implemented by deriving from `VehicleNegotiator` or composing a new `negmas` negotiator; ensure it still writes offer logs compatible with the CSV schema in `VRPTW-main.py`.
- Shared heuristics (distance, slack, clustering) should be placed in `VRPTW_functions.py` so all agents can reuse them without duplicating logic.
- When integrating alternative route planners, follow the `set_route_planner` / `evaluate_route_with_planner` contract so vehicles gain new scoring capabilities without altering negotiation interfaces.

## 作業指示
- 「英語でthinkして，日本語でoutputしてください」という指示に従うこと。
