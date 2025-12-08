import copy
import math
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------- Graph + initial data -----------------------------

building_graph = {
    'R301_F3': [('H_F3', 0.5, 5)],
    'H_F3': [('R301_F3', 0.5, 5), ('Stairs_F2_F3', 1.0, 5)],
    'Stairs_F2_F3': [('H_F3', 1.0, 5), ('H_F2', 1.0, 5)],

    'R201_F2': [('H_F2', 0.5, 10)],
    'H_F2': [('R201_F2', 0.5, 10), ('Stairs_F2_F3', 1.0, 5), ('Stairs_F1_F2', 1.0, 15)],
    'Stairs_F1_F2': [('H_F2', 1.0, 15), ('H_F1', 1.0, 15)],

    'R101_F1': [('H_F1', 0.5, 10)],
    'H_F1': [('R101_F1', 0.5, 10), ('Stairs_F1_F2', 1.0, 15),
             ('Exit_A', 1.0, 10), ('Exit_B', 1.0, 10), ('Exit_C', 1.0, 10)],

    'Exit_A': [],
    'Exit_B': [],
    'Exit_C': []
}

starting_groups = {
    'R301_F3': {'pop': 100, 'delay': 1.0},
    'R201_F2': {'pop': 100, 'delay': 4.0},
    'R101_F1': {'pop': 100, 'delay': 1.0}
}

exit_points = ['Exit_A', 'Exit_B', 'Exit_C']


def get_edge_details(graph, u, v):
    """Return (travel_time, capacity) for edge (u, v), or (None, None)."""
    for neighbor, time_weight, capacity in graph.get(u, []):
        if neighbor == v:
            return time_weight, capacity
    return None, None


# ----------------- Floyd–Warshall (all-pairs shortest paths) ------------------

def floyd_warshall_all_pairs(graph):
    """
    Compute all-pairs shortest path times using Floyd–Warshall.
    Returns:
        dist[u][v] = shortest travel time from u to v
        nxt[u][v]  = next node on a shortest path from u to v
    """
    # Collect all nodes mentioned
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            nodes.add(v)
    nodes = list(nodes)

    INF = float('inf')
    dist = {u: {v: INF for v in nodes} for u in nodes}
    nxt = {u: {v: None for v in nodes} for u in nodes}

    # Distance to self = 0
    for u in nodes:
        dist[u][u] = 0.0
        nxt[u][u] = u

    # Direct edges
    for u, nbrs in graph.items():
        for v, t, c in nbrs:
            if t < dist[u][v]:
                dist[u][v] = t
                nxt[u][v] = v

    # Dynamic programming core of Floyd–Warshall
    for k in nodes:
        for i in nodes:
            dik = dist[i][k]
            if dik == INF:
                continue
            for j in nodes:
                if dist[k][j] == INF:
                    continue
                cand = dik + dist[k][j]
                if cand < dist[i][j]:
                    dist[i][j] = cand
                    nxt[i][j] = nxt[i][k]

    return dist, nxt


def get_fw_route(start, exits, dist, nxt):
    """
    Given FW tables dist, nxt, reconstruct the shortest path from 'start'
    to the *best* exit in 'exits'. Returns (distance, [path nodes]).
    """
    best_exit = None
    best_d = math.inf

    if start not in dist:
        return math.inf, []

    for ex in exits:
        if ex in dist[start] and dist[start][ex] < best_d:
            best_d = dist[start][ex]
            best_exit = ex

    if best_exit is None or best_d == math.inf:
        return math.inf, []

    route = [start]
    u = start
    while u != best_exit:
        u = nxt[u][best_exit]
        if u is None:
            return math.inf, []
        route.append(u)

    return best_d, route


# --------------------- NetworkX graph + visualization -------------------------

def build_nx_graph(graph):
    G = nx.DiGraph()
    for u, nbrs in graph.items():
        G.add_node(u)
        for v, t, c in nbrs:
            G.add_edge(u, v, time=t, cap=c)

    # Fixed layout so floors are visually separated
    pos = {
        'R301_F3': (0, 3), 'H_F3': (2, 3),
        'Stairs_F2_F3': (4, 2.3),

        'R201_F2': (0, 2), 'H_F2': (2, 2),
        'Stairs_F1_F2': (4, 1.3),

        'R101_F1': (0, 1), 'H_F1': (2, 1),

        'Exit_A': (4, 0.3), 'Exit_B': (5, 0.6), 'Exit_C': (6, 0.9)
    }

    for n in G.nodes():
        if n not in pos:
            pos[n] = (0, 0)

    return G, pos


def draw_state(G, pos, pop_state, edge_flow, t, step,
               scenario_name="",
               edges_highlight=None,
               step_by_step=False):
    """
    Draw one frame of the simulation:
    - Node size encodes # of people
    - Edge width encodes flow currently on that edge
    - edges_highlight are drawn in red to show where people moved this step
    """
    plt.clf()
    ax = plt.gca()
    ax.set_title(f"{scenario_name}\nt = {t:.1f}   step = {step}")

    node_sizes = []
    node_colors = []
    labels = {}

    for node in G.nodes():
        pop = 0
        if node in pop_state and isinstance(pop_state[node], dict):
            pop = pop_state[node].get('pop', 0)

        size = 300 + pop * 2
        node_sizes.append(size)

        if node.startswith("Exit"):
            color = "green"
        elif node.startswith("R"):
            color = "orange"
        elif "Stairs" in node:
            color = "red"
        else:
            color = "skyblue"
        node_colors.append(color)

        label = node
        if pop > 0:
            label = f"{node}\n({pop})"
        labels[node] = label

    widths = []
    edge_colors = []
    edges_highlight = edges_highlight or set()

    for u, v in G.edges():
        flow = edge_flow.get((u, v), 0)
        if flow < 0:
            flow = 0
        w = 1.0 + flow / 5.0
        if w <= 0:
            w = 0.2
        widths.append(w)

        if (u, v) in edges_highlight:
            edge_colors.append("red")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths,
                           edge_color=edge_colors,
                           arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=8, ax=ax)

    ax.set_axis_off()
    plt.tight_layout()
    plt.draw()

    if step_by_step:
        print("Press Enter for next step (or Ctrl+C to stop)...")
        try:
            input()
        except EOFError:
            pass
    else:
        plt.pause(0.3)


# ------------------------ Evacuation simulation core --------------------------

def EvacuationOptimization(graph, groups, exits,
                           time_limit=None,
                           target_evacuees=None,
                           stop_on_target=False,
                           verbose=False,
                           visualize=False,
                           nx_graph=None,
                           pos=None,
                           scenario_name="",
                           step_by_step=False):
    """
    Evacuation simulation that uses ONLY Floyd–Warshall for routing.
    - FW is computed once up front (dynamic programming).
    - At each step, each room chooses the FW-shortest path to *some* exit.
    - Movement is constrained by per-edge capacities and local delays.
    """
    t = 0.0
    saved = 0
    target_time = None

    total_pop = sum(group['pop'] for group in groups.values())
    pop_state = copy.deepcopy(groups)

    in_transit = []   # list of tuples: (eta, dest, group_id, count, edge_used, speed_mod)
    edge_flow = {}    # current # of people on each edge

    process_order = [
        'R301_F3', 'H_F3',
        'Stairs_F2_F3', 'R201_F2',
        'H_F2', 'Stairs_F1_F2',
        'R101_F1', 'H_F1'
    ]

    # Precompute FW for the static graph
    fw_dist, fw_next = floyd_warshall_all_pairs(graph)

    if visualize and (nx_graph is None or pos is None):
        nx_graph, pos = build_nx_graph(graph)

    if verbose:
        print(f"\n=== {scenario_name or 'Evacuation Simulation (Floyd–Warshall)'} ===")
        if time_limit:
            print(f"Time limit: {time_limit} minutes")
        if target_evacuees:
            print(f"Target evacuees: {target_evacuees}")
        print(f"Total evacuees: {total_pop}")
        print("Routing: Floyd–Warshall all-pairs shortest paths\n")

    step = 0

    if visualize:
        draw_state(nx_graph, pos, pop_state, edge_flow,
                   t, step, scenario_name, set(), step_by_step)

    for _ in range(2000):
        if time_limit and t > time_limit:
            if verbose:
                print(f"[Time {t:.1f}] Time limit reached")
            break

        edges_used_step = set()

        # -------------------- 1. ARRIVALS --------------------
        for arrival in list(in_transit):
            eta, dest, group_id, count, edge_used, speed_mod = arrival

            if t >= eta:
                in_transit.remove(arrival)

                prev_flow = edge_flow.get(edge_used, 0) - count
                if prev_flow <= 0:
                    edge_flow.pop(edge_used, None)
                else:
                    edge_flow[edge_used] = prev_flow

                if dest in exits:
                    saved += count
                    if verbose:
                        print(f"[Time {t:.1f}] {count} from {group_id} arrived at {dest}!")

                    if target_evacuees and saved >= target_evacuees:
                        if target_time is None:
                            target_time = t
                            if verbose:
                                print(f"[Time {t:.1f}] Target reached with {saved} evacuees")
                        if stop_on_target:
                            if verbose:
                                print(f"[Time {t:.1f}] Stopping after reaching target")
                            if visualize:
                                step += 1
                                draw_state(nx_graph, pos, pop_state,
                                           edge_flow, t, step,
                                           scenario_name, edges_used_step,
                                           step_by_step)
                            remaining = sum(
                                g['pop'] for g in pop_state.values()
                                if isinstance(g, dict)
                            )
                            return {
                                "Total_Evacuees_Saved": saved,
                                "Evacuation_Time_T": t,
                                "Remaining_At_Start": remaining,
                                "Target_Reached_Time": target_time
                            }
                else:
                    # join the destination node's population
                    if dest not in pop_state:
                        pop_state[dest] = {'pop': 0, 'delay': speed_mod}
                    pop_state[dest]['pop'] += count
                    if verbose:
                        print(f"[Time {t:.1f}] {count} from {group_id} arrived at {dest}")

        if saved == total_pop:
            if verbose:
                print(f"[Time {t:.1f}] All evacuees are safe")
            if visualize:
                step += 1
                draw_state(nx_graph, pos, pop_state,
                           edge_flow, t, step,
                           scenario_name, edges_used_step,
                           step_by_step)
            break

        # -------------------- 2. DEPARTURES --------------------
        has_movement = False

        for loc in process_order:
            loc_data = pop_state.get(loc)
            if not loc_data or loc_data['pop'] <= 0:
                continue

            people = loc_data['pop']
            speed_mod = loc_data.get('delay', 1.0)

            # Use Floyd–Warshall route (global shortest path)
            _, route = get_fw_route(loc, exits, fw_dist, fw_next)
            if not route or len(route) < 2:
                continue

            src, dst = route[0], route[1]
            base_time, cap = get_edge_details(graph, src, dst)
            if base_time is None or cap is None:
                continue

            edge = (src, dst)
            avail = cap - edge_flow.get(edge, 0)
            if avail <= 0:
                continue

            flow = min(people, avail)

            if flow > 0:
                travel = base_time * speed_mod
                eta = t + travel
                orig_id = loc if loc in groups else f"{loc}"

                in_transit.append((eta, dst, orig_id, flow, edge, speed_mod))
                pop_state[loc]['pop'] -= flow
                edge_flow[edge] = edge_flow.get(edge, 0) + flow
                edges_used_step.add(edge)
                has_movement = True

                if verbose:
                    print(f"[Time {t:.1f}] {flow} from {loc} moving to {dst} "
                          f"(Speed Factor: {speed_mod:.1f}x, ETA: {eta:.1f})")

        step += 1
        if visualize:
            draw_state(nx_graph, pos, pop_state,
                       edge_flow, t, step,
                       scenario_name, edges_used_step,
                       step_by_step)

        if not has_movement and not in_transit:
            if verbose:
                print(f"[Time {t:.1f}] No further movement possible")
            break

        # -------------------- 3. ADVANCE TIME --------------------
        t += 0.5

    remaining = sum(
        g['pop'] for g in pop_state.values()
        if isinstance(g, dict)
    )

    return {
        "Total_Evacuees_Saved": saved,
        "Evacuation_Time_T": t,
        "Remaining_At_Start": remaining,
        "Target_Reached_Time": target_time
    }


def print_scenario_report(name, results, total_pop):
    print("\n" + "-" * 50)
    print(f"SCENARIO REPORT: {name}")
    print("-" * 50)

    saved = results['Total_Evacuees_Saved']
    time_taken = results['Evacuation_Time_T']
    target_time = results.get('Target_Reached_Time')

    print(f"> Total Evacuees: {total_pop}")
    print(f"> Evacuees Saved: {saved} ({saved / total_pop * 100:.1f}%)")
    print(f"> Evacuees Left:  {results['Remaining_At_Start']}")
    print(f"> Time Elapsed:   {time_taken:.1f} minutes")

    if target_time is not None:
        print(f"> Target Reached: {target_time:.1f} minutes")

    if time_taken > 0:
        rate = saved / time_taken
        print(f"> Avg Evac Rate:  {rate:.1f} people/min")

    print("-" * 50 + "\n")


# ---------------------------------- main --------------------------------------

if __name__ == "__main__":
    total_pop = sum(group['pop'] for group in starting_groups.values())
    G_nx, pos = build_nx_graph(building_graph)

    plt.ion()

    # Scenario A – Quick response, step-by-step with graph
    res_a = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        target_evacuees=50,
        stop_on_target=True,
        verbose=True,
        visualize=True,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario A – Floyd–Warshall (Quick Response)",
        step_by_step=True
    )
    print_scenario_report("Scenario A – Floyd–Warshall (Quick Response)", res_a, total_pop)

    # Scenario B – Fire emergency (15-min limit)
    res_b = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        time_limit=15.0,
        verbose=True,
        visualize=True,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario B – Floyd–Warshall (15-min Fire Emergency)",
        step_by_step=True
    )
    print_scenario_report("Scenario B – Floyd–Warshall (Fire Emergency)", res_b, total_pop)

    # Scenario C – Complete evacuation benchmark
    res_c = EvacuationOptimization(
        building_graph,
        starting_groups,
        exit_points,
        verbose=True,
        visualize=True,
        nx_graph=G_nx,
        pos=pos,
        scenario_name="Scenario C – Floyd–Warshall (Complete Evacuation)",
        step_by_step=True
    )
    print_scenario_report("Scenario C – Floyd–Warshall (Complete Evacuation)", res_c, total_pop)

    plt.ioff()
    plt.show()
