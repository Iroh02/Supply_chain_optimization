import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import nashpy as nash

# st.set_option('deprecation.showPyplotGlobalUse', False)

# =============================================================================
# Synthetic Data Generation
# =============================================================================
def generate_synthetic_data(num_entries=1000):
    np.random.seed(42)
    suppliers = ["Supplier_A", "Supplier_B", "Supplier_C"]
    warehouses = ["Warehouse_1", "Warehouse_2", "Warehouse_3"]
    retailers = ["Retailer_X", "Retailer_Y", "Retailer_Z"]
    customers = ["Customer_1", "Customer_2", "Customer_3"]
    data = []
    for _ in range(num_entries):
        entry = {
            "Supplier": random.choice(suppliers),
            "Warehouse": random.choice(warehouses),
            "Retailer": random.choice(retailers),
            "Customer": random.choice(customers),
            "Lead_Time": np.random.randint(2, 10),
            "Demand": np.random.randint(50, 500),
            "Inventory_Level": np.random.randint(0, 1000),
            "Shipping_Cost": round(np.random.uniform(5, 50), 2),
            "Order_Quantity": np.random.randint(10, 200),
        }
        data.append(entry)
    return pd.DataFrame(data)

# =============================================================================
# MDP Model (Value Iteration) for Supply Chain Optimization
# =============================================================================
mdp_states = ['Supplier', 'Warehouse', 'Retailer', 'Customer']
mdp_actions = ['Order', 'Ship', 'Hold']
mdp_transition_probs = {
    ('Supplier', 'Order'): {'Warehouse': 0.8, 'Supplier': 0.2},
    ('Warehouse', 'Ship'): {'Retailer': 0.7, 'Warehouse': 0.3},
    ('Retailer', 'Ship'): {'Customer': 0.9, 'Retailer': 0.1},
    ('Warehouse', 'Hold'): {'Warehouse': 1.0},
    ('Retailer', 'Hold'): {'Retailer': 1.0},
}
mdp_rewards = {
    ('Supplier', 'Order'): -2,
    ('Warehouse', 'Ship'): -1,
    ('Retailer', 'Ship'): -1,
    ('Warehouse', 'Hold'): -0.5,
    ('Retailer', 'Hold'): -0.2,
    ('Retailer', 'Ship'): 5,
}

# ---- DYNAMIC Value Iteration to show intermediate steps ----
# ---- DYNAMIC Value Iteration to show intermediate steps ----
def value_iteration(states, actions, transition_probs, rewards, gamma=0.9, theta=0.0001, max_iters=50):
    """
    Performs value iteration and returns a list of tuples for each iteration.
    Each tuple is (iteration_index, policy, value_function).
    """
    # Initialize value function
    V = {s: 0.0 for s in states}
    # Initialize a random policy for non-terminal states
    policy = {s: random.choice(actions) for s in states if s != 'Customer'}

    iteration_logs = []
    for i in range(max_iters):
        delta = 0
        new_V = V.copy()
        new_policy = policy.copy()

        for s in states:
            if s == 'Customer':  # Terminal state
                continue
            
            max_value = float('-inf')
            best_action = None

            for a in actions:
                if (s, a) in transition_probs:
                    expected_value = 0.0
                    for s_next, p in transition_probs[(s, a)].items():
                        reward = rewards.get((s, a), 0)
                        expected_value += p * (reward + gamma * V[s_next])
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a

            new_V[s] = max_value
            new_policy[s] = best_action
            delta = max(delta, abs(V[s] - max_value))

        V = new_V
        policy = new_policy
        iteration_logs.append((i+1, policy.copy(), V.copy()))
        if delta < theta:
            break

    return iteration_logs

# ---- MDP Optimization Section for Streamlit App ----
def mdp_optimization_section():
    st.header("MDP Optimization - Value Iteration (Dynamic)")
    gamma = st.slider("Discount Factor (gamma)", 0.0, 1.0, 0.9, 0.01)
    theta = st.slider("Convergence Threshold (theta)", 1e-6, 1e-1, 1e-4, format="%.1e")
    max_iters = st.slider("Max Iterations", 1, 100, 50)

    if st.button("Run Value Iteration"):
        logs = value_iteration(mdp_states, mdp_actions, mdp_transition_probs, mdp_rewards,
                               gamma=gamma, theta=theta, max_iters=max_iters)
        st.write(f"**Total Iterations:** {len(logs)}")
        
        # Extract final iteration's policy and value function
        final_iter, final_policy, final_values = logs[-1]
        st.subheader(f"Converged at Iteration {final_iter}")
        st.write("**Final Optimal Policy:**")
        st.write(final_policy)
        st.write("**Final State Values:**")
        st.write(final_values)
        
        # Optionally, display the step-by-step logs
        st.markdown("### Step-by-Step Iteration Logs")
        for (iter_num, p, v) in logs:
            st.markdown(f"**Iteration {iter_num}:**")
            st.write("Policy:", p)
            st.write("State Values:", v)

# =============================================================================
# POMDP Components for Supply Chain
# =============================================================================
pomdp_states = ['Supplier_Good', 'Supplier_Delayed', 'Warehouse_Low', 'Warehouse_Normal', 'Retailer_High', 'Retailer_Low']
pomdp_actions = ['Order', 'Ship', 'Hold']
pomdp_observations = ['On_Time', 'Delayed', 'High_Demand', 'Low_Demand']
pomdp_transition_probs = {
    ('Supplier_Good', 'Order'): {'Warehouse_Normal': 0.8, 'Supplier_Delayed': 0.2},
    ('Supplier_Delayed', 'Order'): {'Warehouse_Low': 0.6, 'Supplier_Delayed': 0.4},
    ('Warehouse_Normal', 'Ship'): {'Retailer_High': 0.7, 'Warehouse_Low': 0.3},
    ('Warehouse_Low', 'Ship'): {'Retailer_Low': 0.5, 'Warehouse_Low': 0.5},
    ('Retailer_High', 'Ship'): {'Customer': 0.9, 'Retailer_High': 0.1},
    ('Retailer_Low', 'Ship'): {'Customer': 0.8, 'Retailer_Low': 0.2},
    ('Retailer_High', 'Hold'): {'Retailer_High': 1.0},
    ('Retailer_Low', 'Hold'): {'Retailer_Low': 1.0},
}
observation_probs = {
    ('Supplier_Good', 'Order'): {'On_Time': 0.9, 'Delayed': 0.1},
    ('Supplier_Delayed', 'Order'): {'On_Time': 0.3, 'Delayed': 0.7},
    ('Retailer_High', 'Ship'): {'High_Demand': 0.8, 'Low_Demand': 0.2},
    ('Retailer_Low', 'Ship'): {'High_Demand': 0.2, 'Low_Demand': 0.8},
}
pomdp_rewards = {
    ('Supplier_Good', 'Order'): -2,
    ('Supplier_Delayed', 'Order'): -3,
    ('Warehouse_Normal', 'Ship'): -1,
    ('Warehouse_Low', 'Ship'): -2,
    ('Retailer_High', 'Ship'): 5,
    ('Retailer_Low', 'Ship'): 3,
    ('Retailer_High', 'Hold'): -1,
    ('Retailer_Low', 'Hold'): -0.5,
}
initial_belief_state = {
    'Supplier_Good': 0.7,
    'Supplier_Delayed': 0.3,
}

def update_belief(belief, action, observation):
    new_belief = {}
    total_prob = 0
    for state in belief:
        if (state, action) in observation_probs and observation in observation_probs[(state, action)]:
            prob_obs_given_state = observation_probs[(state, action)][observation]
            new_belief[state] = belief[state] * prob_obs_given_state
            total_prob += new_belief[state]
    for state in new_belief:
        new_belief[state] /= total_prob if total_prob > 0 else 1
    return new_belief

def pomdp_value_iteration(states, actions, transition_probs, observation_probs, rewards, gamma=0.9, theta=0.01, max_iterations=100):
    V = {s: 0 for s in states}
    policy = {s: None for s in states}
    for iteration in range(max_iterations):
        delta = 0
        for s in states:
            max_value = float('-inf')
            best_action = None
            for a in actions:
                if (s, a) in transition_probs:
                    expected_value = 0
                    for s_next, p in transition_probs[(s, a)].items():
                        r = rewards.get((s, a), -10)
                        expected_value += p * (r + gamma * V.get(s_next, 0))
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
            if best_action is None:
                st.write(f"WARNING: No valid action found for {s}. Assigning default action.")
                best_action = np.random.choice(actions)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
            policy[s] = best_action
        st.write(f"Iteration {iteration+1}: Max Value Change = {delta:.6f}")
        if delta < theta:
            st.write("Converged!")
            break
    return policy, V

# =============================================================================
# Nash Equilibrium & Policy Adjustment Components
# =============================================================================
def nash_equilibrium_analysis():
    supplier_payoff = np.array([[3, -1],
                                [2, 1]])
    distributor_payoff = np.array([[2, 1],
                                   [-1, 3]])
    game = nash.Game(supplier_payoff, distributor_payoff)
    equilibria = list(game.support_enumeration())
    return equilibria

def adjust_supply_chain_policy(supplier_strategy, distributor_strategy):
    supplier_decision = "High Supply" if supplier_strategy[0] > 0.5 else "Low Supply"
    distributor_decision = "High Demand" if distributor_strategy[0] > 0.5 else "Low Demand"
    if supplier_decision == "High Supply" and distributor_decision == "High Demand":
        order_quantity = "Increase Orders (150% of normal)"
        warehouse_policy = "Increase Safety Stock"
        shipping_policy = "Prioritize Express Shipping"
    elif supplier_decision == "Low Supply" and distributor_decision == "Low Demand":
        order_quantity = "Reduce Orders (75% of normal)"
        warehouse_policy = "Reduce Storage Costs"
        shipping_policy = "Use Standard Shipping"
    elif supplier_decision == "High Supply" and distributor_decision == "Low Demand":
        order_quantity = "Maintain Normal Order Levels"
        warehouse_policy = "Optimize Inventory Management"
        shipping_policy = "Use Flexible Shipping"
    else:
        order_quantity = "Adjust Based on Demand Forecast"
        warehouse_policy = "Use Demand-Driven Inventory"
        shipping_policy = "Optimize Logistics Cost"
    return order_quantity, warehouse_policy, shipping_policy

def update_mdp_with_nash(states, actions, transition_probs, rewards, supplier_strategy, distributor_strategy):
    policy, _ = pomdp_value_iteration(states, actions, transition_probs, observation_probs, rewards)
    if supplier_strategy[0] > 0.5:
        policy['Supplier_Good'] = "Order More"
    else:
        policy['Supplier_Good'] = "Reduce Orders"
    if distributor_strategy[0] > 0.5:
        policy['Retailer_High'] = "Ship Faster"
    else:
        policy['Retailer_Low'] = "Hold Inventory"
    return policy

# =============================================================================
# Supply Chain Network Route Visualization
# =============================================================================
def plot_optimal_route(source, destination):
    G = nx.DiGraph()
    nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
    G.add_nodes_from(nodes)
    edges = [
        ("Supplier", "Warehouse_A", {"cost": 3}),
        ("Supplier", "Warehouse_B", {"cost": 4}),
        ("Warehouse_A", "Retailer_A", {"cost": 2}),
        ("Warehouse_B", "Retailer_B", {"cost": 3}),
        ("Retailer_A", "Customer", {"cost": 1}),
        ("Retailer_B", "Customer", {"cost": 2}),
    ]
    G.add_edges_from([(u, v, d) for u, v, d in edges])
    try:
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight="cost")
    except nx.NetworkXNoPath:
        st.error("No path exists between the selected nodes.")
        return None, None, None
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", ax=ax)
    edge_labels = {(u, v): f"{d['cost']} cost" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2, ax=ax)
    ax.set_title(f"Optimal Route from {source} to {destination}")
    return fig, shortest_path, G

# =============================================================================
# Real-Time Simulation
# =============================================================================
def real_time_simulation(duration=100):
    time = np.arange(duration)
    # Simulate demand fluctuations and inventory depletion
    demand = np.random.normal(loc=200, scale=20, size=duration)
    inventory = np.maximum(1000 - np.cumsum(np.random.randint(10, 30, size=duration)), 0)
    df = pd.DataFrame({'Time': time, 'Demand': demand, 'Inventory': inventory})
    df.set_index('Time', inplace=True)
    return df

# =============================================================================
# Streamlit App Sections
# =============================================================================
def synthetic_data_section():
    st.header("Synthetic Data Generation")
    num_entries = st.slider("Select number of entries", 100, 5000, 1000, step=100)
    df = generate_synthetic_data(num_entries)
    st.write("Preview of Synthetic Data:")
    st.dataframe(df.head())
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "supply_chain_synthetic_data.csv", "text/csv")

def mdp_optimization_section():
    st.header("MDP Optimization - Value Iteration")
    policy, values = value_iteration(mdp_states, mdp_actions, mdp_transition_probs, mdp_rewards)
    st.write("Optimal Policy:")
    st.write(policy)
    st.write("State Values:")
    st.write(values)

def transition_visualization_section():
    st.header("Supply Chain Transition Model Visualization")
    G = nx.DiGraph()
    for (s, a), transitions in mdp_transition_probs.items():
        for s_next, prob in transitions.items():
            G.add_edge(s, s_next, label=f"{a} ({prob:.2f})")
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', ax=ax)
    edge_labels = {(s, s_next): d['label'] for s, s_next, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("Supply Chain Transition Model (MDP)")
    st.pyplot(fig)

def pomdp_simulation_section():
    st.header("POMDP Simulation")
    st.write("Initial Belief State:")
    st.write(initial_belief_state)
    action = st.selectbox("Select Action", pomdp_actions)
    observation = st.selectbox("Select Observation", pomdp_observations)
    updated_belief = update_belief(initial_belief_state, action, observation)
    st.write("Updated Belief State:")
    st.write(updated_belief)
    if st.button("Run POMDP Value Iteration"):
        policy, values = pomdp_value_iteration(pomdp_states, pomdp_actions, pomdp_transition_probs, observation_probs, pomdp_rewards)
        st.write("Optimal POMDP Policy:")
        st.write(policy)
        st.write("State Values:")
        st.write(values)

def nash_equilibrium_section():
    st.header("Nash Equilibrium Analysis")
    equilibria = nash_equilibrium_analysis()
    for i, eq in enumerate(equilibria):
        st.write(f"Equilibrium {i+1}:")
        st.write("Supplier Strategy:", eq[0])
        st.write("Distributor Strategy:", eq[1])
    if equilibria:
        chosen_eq = equilibria[0]  # Use the first equilibrium for policy adjustment
        supplier_strategy = chosen_eq[0]
        distributor_strategy = chosen_eq[1]
        order_policy, warehouse_policy, shipping_policy = adjust_supply_chain_policy(supplier_strategy, distributor_strategy)
        st.subheader("Supply Chain Policy Adjustments Based on Nash Equilibrium")
        st.write("Supplier Order Policy:", order_policy)
        st.write("Warehouse Inventory Policy:", warehouse_policy)
        st.write("Shipping & Distribution Policy:", shipping_policy)

def policy_update_section():
    st.header("Update MDP Policy with Nash Equilibrium")
    equilibria = nash_equilibrium_analysis()
    if equilibria:
        chosen_eq = equilibria[0]
        supplier_strategy = chosen_eq[0]
        distributor_strategy = chosen_eq[1]
        updated_policy = update_mdp_with_nash(pomdp_states, pomdp_actions, pomdp_transition_probs, pomdp_rewards, supplier_strategy, distributor_strategy)
        st.write("Updated Supply Chain Policy:")
        st.write(updated_policy)

def route_visualization_section():
    st.header("Optimal Delivery Route Visualization")
    nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
    source = st.selectbox("Select Source", nodes, index=0)
    destination = st.selectbox("Select Destination", nodes, index=len(nodes)-1)
    fig, path, _ = plot_optimal_route(source, destination)
    if fig:
        st.pyplot(fig)
        st.write("Optimal Route:", " -> ".join(path))

def simulation_section():
    st.header("Real-Time Decision-Making Simulation")
    duration = st.slider("Simulation Duration (time steps)", 50, 200, 100, step=10)
    if st.button("Run Simulation"):
        sim_df = real_time_simulation(duration)
        st.line_chart(sim_df)

# =============================================================================
# Main App Layout
# =============================================================================
st.title("Supply Chain Optimization Dashboard")
section = st.sidebar("Select Section", 
                           ["Synthetic Data Generation", 
                            "MDP Optimization", 
                            "Transition Model Visualization", 
                            "POMDP Simulation", 
                            "Nash Equilibrium Analysis", 
                            "Policy Update with Nash", 
                            "Optimal Delivery Route Visualization",
                            "Real-Time Simulation"])

if section == "Synthetic Data Generation":
    synthetic_data_section()
elif section == "MDP Optimization":
    mdp_optimization_section()
elif section == "Transition Model Visualization":
    transition_visualization_section()
elif section == "POMDP Simulation":
    pomdp_simulation_section()
elif section == "Nash Equilibrium Analysis":
    nash_equilibrium_section()
elif section == "Policy Update with Nash":
    policy_update_section()
elif section == "Optimal Delivery Route Visualization":
    route_visualization_section()
elif section == "Real-Time Simulation":
    simulation_section()
