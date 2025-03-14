import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import nashpy as nash
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

def value_iteration(states, actions, transition_probs, rewards, gamma=0.9, theta=0.0001, max_iters=50):
    """
    Performs value iteration and returns a list of tuples for each iteration.
    Each tuple is (iteration_number, policy, value_function).
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
def pomdp_value_iteration_silent(states, actions, transition_probs, observation_probs, rewards,
                                 gamma=0.9, theta=0.01, max_iterations=100):
    """
    Same logic as pomdp_value_iteration, but without any Streamlit outputs.
    Returns (policy, value_function).
    """
    V = {s: 0 for s in states}
    policy = {s: None for s in states}
    for _ in range(max_iterations):
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
                best_action = np.random.choice(actions)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
            policy[s] = best_action
        if delta < theta:
            break
    return policy, V

# def pomdp_value_iteration(states, actions, transition_probs, observation_probs, rewards, gamma=0.9, theta=0.01, max_iterations=100):
#     V = {s: 0 for s in states}
#     policy = {s: None for s in states}
#     for iteration in range(max_iterations):
#         delta = 0
#         for s in states:
#             max_value = float('-inf')
#             best_action = None
#             for a in actions:
#                 if (s, a) in transition_probs:
#                     expected_value = 0
#                     for s_next, p in transition_probs[(s, a)].items():
#                         r = rewards.get((s, a), -10)
#                         expected_value += p * (r + gamma * V.get(s_next, 0))
#                     if expected_value > max_value:
#                         max_value = expected_value
#                         best_action = a
#             if best_action is None:
#                 st.write(f"WARNING: No valid action found for {s}. Assigning default action.")
#                 best_action = np.random.choice(actions)
#             delta = max(delta, abs(V[s] - max_value))
#             V[s] = max_value
#             policy[s] = best_action
#         st.write(f"Iteration {iteration+1}: Max Value Change = {delta:.6f}")
#         if delta < theta:
#             st.write("Converged!")
#             break
#     return policy, V

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

# def update_mdp_with_nash(states, actions, transition_probs, rewards, supplier_strategy, distributor_strategy):
#     policy, _ = pomdp_value_iteration(states, actions, transition_probs, observation_probs, rewards)
#     if supplier_strategy[0] > 0.5:
#         policy['Supplier_Good'] = "Order More"
#     else:
#         policy['Supplier_Good'] = "Reduce Orders"
#     if distributor_strategy[0] > 0.5:
#         policy['Retailer_High'] = "Ship Faster"
#     else:
#         policy['Retailer_Low'] = "Hold Inventory"
#     return policy
def update_mdp_with_nash(states, actions, transition_probs, observation_probs, rewards,
                         supplier_strategy, distributor_strategy):
    """
    1. Runs a silent POMDP value iteration to get the base policy.
    2. Then modifies the policy based on the chosen Nash equilibrium (supplier & distributor strategies).
    """
    # Use the silent version to avoid printing iteration logs
    policy, _ = pomdp_value_iteration_silent(
        states,
        actions,
        transition_probs,
        observation_probs,
        rewards
    )
    # Modify the policy according to the Nash strategies
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
    demand = np.random.normal(loc=200, scale=20, size=duration)
    inventory = np.maximum(1000 - np.cumsum(np.random.randint(10, 30, size=duration)), 0)
    df = pd.DataFrame({'Time': time, 'Demand': demand, 'Inventory': inventory})
    df.set_index('Time', inplace=True)
    return df

# =============================================================================
# Streamlit App Sections
# =============================================================================
# def synthetic_data_section():
#     st.header("Synthetic Data Generation")
#     num_entries = st.slider("Select number of entries", 100, 5000, 1000, step=100)
#     df = generate_synthetic_data(num_entries)
#     st.write("Preview of Synthetic Data:")
#     st.dataframe(df.head())
#     csv = df.to_csv(index=False).encode('utf-8')
#     st.download_button("Download CSV", csv, "supply_chain_synthetic_data.csv", "text/csv")
def synthetic_data_section():
    # A nice big title for the landing page
    st.title("Welcome to ACME Supply Chain Management System")
    
    # Introductory text explaining the purpose of the app
    st.markdown("""
    **ACME Supply Chain Management System** is a comprehensive dashboard designed to help you:
    
    - **Generate** synthetic supply chain data (suppliers, warehouses, retailers, customers, etc.)
    - **Optimize** your supply chain using Markov Decision Processes (MDP)
    - **Visualize** transitions and routes across your network
    - **Handle uncertainties** via Partially Observable MDP (POMDP)
    - **Evaluate strategic interactions** using Nash Equilibria
    - **Update policies** dynamically based on equilibrium strategies
    - **Simulate** real-time demand and inventory fluctuations
    
    Use the slider below to choose how many synthetic data entries to create, then preview, visualize, or download the dataset.
    """)

    # --- Synthetic Data Generation ---
    st.subheader("1) Generate Synthetic Data")
    num_entries = st.slider("Select number of entries", 100, 5000, 1000, step=100)

    df = generate_synthetic_data(num_entries)
    st.write("**Preview of Synthetic Data:**")
    st.dataframe(df.head())

    # Provide download button for CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "supply_chain_synthetic_data.csv", "text/csv")

    # --- Data Visualization Dashboard ---
    st.subheader("2) Data Visualizations")

    # 2a) Bar Chart: Distribution of Suppliers
    st.markdown("**Distribution of Suppliers**")
    supplier_count = df["Supplier"].value_counts().reset_index()
    supplier_count.columns = ["Supplier", "Count"]
    fig_bar = px.bar(
        supplier_count, 
        x="Supplier", 
        y="Count", 
        color="Supplier", 
        title="Supplier Distribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2b) Pie Chart: Distribution of Warehouses
    st.markdown("**Distribution of Warehouses**")
    warehouse_count = df["Warehouse"].value_counts().reset_index()
    warehouse_count.columns = ["Warehouse", "Count"]
    fig_pie = px.pie(
        warehouse_count, 
        names="Warehouse", 
        values="Count", 
        title="Warehouse Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 2c) Correlation Heatmap of Numeric Features
    st.markdown("**Correlation Heatmap** (Lead Time, Demand, Inventory, Shipping Cost, Order Quantity)")
    numeric_cols = ["Lead_Time", "Demand", "Inventory_Level", "Shipping_Cost", "Order_Quantity"]
    corr = df[numeric_cols].corr()
    fig_heatmap = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto", 
        title="Correlation Heatmap (Numeric Features)"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

def mdp_optimization_section():
    st.header("MDP Optimization - Value Iteration")
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

        # --- Display Final Results ---
        st.markdown("**Final Optimal Policy:**")
        # Convert policy dict -> DataFrame for a nicer table
        final_policy_df = pd.DataFrame(list(final_policy.items()), columns=["State", "Action"])
        st.table(final_policy_df)

        st.markdown("**Final State Values:**")
        # Convert values dict -> DataFrame for a nicer table
        final_values_df = pd.DataFrame(list(final_values.items()), columns=["State", "Value"])
        st.table(final_values_df)

        # --- Step-by-Step Iteration Logs ---
        st.markdown("### Step-by-Step Iteration Logs")
        for (iter_num, p, v) in logs:
            st.markdown(f"#### Iteration {iter_num}")
            # Show policy in table form
            df_policy = pd.DataFrame(list(p.items()), columns=["State", "Action"])
            st.markdown("**Policy**:")
            st.table(df_policy)
            # Show value function in table form
            df_values = pd.DataFrame(list(v.items()), columns=["State", "Value"])
            st.markdown("**Value Function**:")
            st.table(df_values)
            st.markdown("---")  # A divider line for clarity

# def mdp_optimization_section():
#     st.header("MDP Optimization - Value Iteration (Dynamic)")
#     gamma = st.slider("Discount Factor (gamma)", 0.0, 1.0, 0.9, 0.01)
#     theta = st.slider("Convergence Threshold (theta)", 1e-6, 1e-1, 1e-4, format="%.1e")
#     max_iters = st.slider("Max Iterations", 1, 100, 50)
#     if st.button("Run Value Iteration"):
#         logs = value_iteration(mdp_states, mdp_actions, mdp_transition_probs, mdp_rewards,
#                                gamma=gamma, theta=theta, max_iters=max_iters)
#         st.write(f"**Total Iterations:** {len(logs)}")
#         final_iter, final_policy, final_values = logs[-1]
#         st.subheader(f"Converged at Iteration {final_iter}")
#         st.write("**Final Optimal Policy:**")
#         st.write(final_policy)
#         st.write("**Final State Values:**")
#         st.write(final_values)
#         st.markdown("### Step-by-Step Iteration Logs")
#         for (iter_num, p, v) in logs:
#             st.markdown(f"**Iteration {iter_num}:**")
#             st.write("Policy:", p)
#             st.write("State Values:", v)
def transition_visualization_section():
    st.header("Supply Chain Transition Model Visualization")

    # Build the directed graph (DiGraph) from your MDP transitions
    G = nx.DiGraph()
    for (s, a), transitions in mdp_transition_probs.items():
        for s_next, prob in transitions.items():
            label = f"{a} ({prob:.2f})"
            G.add_edge(s, s_next, label=label)

    # Define a custom layout for a left-to-right flow
    # Adjust x-coordinates or y-coordinates as desired
    pos = {
        "Supplier":  (0, 0),
        "Warehouse": (2, 0),
        "Retailer":  (4, 0),
        "Customer":  (6, 0)
    }

    # Define unique colors for each state
    node_colors = {
        "Supplier":  "#AED6F1",  # Light blue
        "Warehouse": "#A9DFBF",  # Light green
        "Retailer":  "#F9E79F",  # Light yellow
        "Customer":  "#F5B7B1",  # Light pink
    }

    # Create a figure
    fig, ax = plt.subplots(figsize=(9, 3))

    # Draw nodes with custom colors
    colors = [node_colors.get(n, "lightgray") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color=colors)

    # Draw labels on nodes
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10, font_color="black", font_weight="bold"
    )

    # Draw edges with arrows, slight curvature, and custom styling
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrowstyle='-|>', arrowsize=20,
        width=2, edge_color='gray',
        connectionstyle='arc3,rad=0.1'  # Rad > 0 for slight curved edges
    )

    # Add edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels, font_size=8, label_pos=0.5
    )

    # Title and remove axis
    ax.set_title("Supply Chain Transition Model (MDP)", fontsize=12)
    ax.set_axis_off()

    # Display the figure in Streamlit
    st.pyplot(fig)

# def transition_visualization_section():
#     st.header("Supply Chain Transition Model Visualization")
#     G = nx.DiGraph()
#     for (s, a), transitions in mdp_transition_probs.items():
#         for s_next, prob in transitions.items():
#             G.add_edge(s, s_next, label=f"{a} ({prob:.2f})")
#     pos = nx.spring_layout(G, seed=42)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', ax=ax)
#     edge_labels = {(s, s_next): d['label'] for s, s_next, d in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
#     ax.set_title("Supply Chain Transition Model (MDP)")
#     st.pyplot(fig)
def pomdp_simulation_section():
    st.header("POMDP Simulation")

    # --- Display the Initial Belief State as a Table ---
    st.markdown("**Initial Belief State:**")
    init_belief_df = pd.DataFrame(
        list(initial_belief_state.items()),
        columns=["State", "Belief Probability"]
    )
    st.table(init_belief_df)

    # --- Let User Pick Action & Observation ---
    action = st.selectbox("Select Action", pomdp_actions)
    observation = st.selectbox("Select Observation", pomdp_observations)

    # --- Update Belief on Button Click ---
    if st.button("Update Belief"):
        updated_belief = update_belief(initial_belief_state, action, observation)
        
        st.markdown("**Updated Belief State:**")
        if updated_belief:
            # Convert dict -> DataFrame for a neat table
            updated_belief_df = pd.DataFrame(
                list(updated_belief.items()),
                columns=["State", "Belief Probability"]
            )
            st.table(updated_belief_df)
        else:
            st.warning("No matching (state, action) found in observation_probs for the chosen action/observation. "
                       "The updated belief is empty. Please pick a valid combination or expand observation_probs.")

    # --- Run POMDP Value Iteration on Button Click ---
    if st.button("Run POMDP Value Iteration"):
        policy, values = pomdp_value_iteration(
            pomdp_states,
            pomdp_actions,
            pomdp_transition_probs,
            observation_probs,
            pomdp_rewards
        )
        # Display the policy in table format
        st.markdown("**Optimal POMDP Policy:**")
        policy_df = pd.DataFrame(list(policy.items()), columns=["State", "Action"])
        st.table(policy_df)

        # Display the value function in table format
        st.markdown("**State Values:**")
        values_df = pd.DataFrame(list(values.items()), columns=["State", "Value"])
        st.table(values_df)


# def nash_equilibrium_section():
#     st.header("Nash Equilibrium Analysis")

#     # Define more descriptive labels for each player's actions
#     supplier_labels = ["High Supply", "Low Supply"]
#     distributor_labels = ["High Demand", "Low Demand"]

#     # Calculate all equilibria
#     equilibria = nash_equilibrium_analysis()

#     if not equilibria:
#         st.warning("No Nash equilibria found with the current payoff matrices.")
#         return

#     st.markdown("### All Equilibria Found")

#     # Display each equilibrium with labeled tables
#     for i, eq in enumerate(equilibria):
#         st.subheader(f"Equilibrium {i+1}")
#         # eq is typically a tuple: (supplier_strategy, distributor_strategy)
#         supplier_strat = eq[0]
#         distributor_strat = eq[1]

#         # Convert each strategy to a DataFrame for a neat table
#         supplier_df = pd.DataFrame({
#             "Supplier Action": supplier_labels,
#             "Probability": supplier_strat
#         })
#         distributor_df = pd.DataFrame({
#             "Distributor Action": distributor_labels,
#             "Probability": distributor_strat
#         })

#         # Display tables
#         st.markdown("**Supplier Strategy**")
#         st.table(supplier_df)

#         st.markdown("**Distributor Strategy**")
#         st.table(distributor_df)

#     # Let user pick which equilibrium to apply
#     st.markdown("### Apply an Equilibrium to Update Policy")
#     eq_options = [f"Equilibrium {i+1}" for i in range(len(equilibria))]
#     selected_eq_index = st.selectbox("Select which Equilibrium to Apply", range(len(equilibria)),
#                                      format_func=lambda x: eq_options[x])

#     if st.button("Apply Selected Equilibrium"):
#         chosen_eq = equilibria[selected_eq_index]
#         supplier_strategy = chosen_eq[0]
#         distributor_strategy = chosen_eq[1]

#         # Adjust the supply chain policy based on the chosen equilibrium
#         order_policy, warehouse_policy, shipping_policy = adjust_supply_chain_policy(
#             supplier_strategy, 
#             distributor_strategy
#         )

#         st.subheader("Supply Chain Policy Adjustments Based on Selected Nash Equilibrium")
#         # Show these in a table for clarity
#         adjustments_df = pd.DataFrame({
#             "Policy Aspect": ["Supplier Order Policy", "Warehouse Inventory Policy", "Shipping & Distribution Policy"],
#             "Decision": [order_policy, warehouse_policy, shipping_policy]
#         })
#         st.table(adjustments_df)

#         # ─────────────────────────────────────────────────────────────────
#         # VISUALIZING IMPROVEMENTS FROM THE UPDATED POLICY (using Seaborn)
#         # ─────────────────────────────────────────────────────────────────
#         st.markdown("### Visualization of Policy Improvement")

#         # Example: Compare a performance metric (e.g., cost, profit, or utility)
#         # between the old policy and the new (Nash-based) policy.
#         # In a real app, you'd compute or simulate these metrics. 
#         # Here, we just mock up some random data.

#         old_performance_supplier = np.random.uniform(1, 5)  # e.g. old payoff for supplier
#         old_performance_distributor = np.random.uniform(1, 5)  # old payoff for distributor
#         new_performance_supplier = old_performance_supplier + np.random.uniform(0.5, 2.0)  # improved payoff
#         new_performance_distributor = old_performance_distributor + np.random.uniform(0.5, 2.0)

#         improvement_df = pd.DataFrame({
#             "Agent": ["Supplier", "Distributor"],
#             "Old Policy": [old_performance_supplier, old_performance_distributor],
#             "New (Nash) Policy": [new_performance_supplier, new_performance_distributor]
#         })

#         st.markdown("**Performance Comparison (Example)**")
#         st.dataframe(improvement_df.style.format(precision=2))

#         # Melt the DataFrame so we can plot both old/new as separate bars with Seaborn
#         improvement_melted = improvement_df.melt(id_vars="Agent", 
#                                                  var_name="Policy", 
#                                                  value_name="Performance")

#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.barplot(data=improvement_melted, x="Agent", y="Performance", hue="Policy", ax=ax)
#         ax.set_title("Comparison of Old vs. New Policy Performance")
#         ax.set_ylabel("Performance (Example Units)")
#         st.pyplot(fig)
def nash_equilibrium_section():
    st.header("Nash Equilibrium Analysis")

    # Define descriptive labels for actions
    supplier_labels = ["High Supply", "Low Supply"]
    distributor_labels = ["High Demand", "Low Demand"]

    # Calculate all equilibria
    equilibria = nash_equilibrium_analysis()

    if not equilibria:
        st.warning("No Nash equilibria found with the current payoff matrices.")
        return

    st.markdown("### All Equilibria Found")

    # Display each equilibrium with labeled tables
    for i, eq in enumerate(equilibria):
        st.subheader(f"Equilibrium {i+1}")
        supplier_strat = eq[0]
        distributor_strat = eq[1]

        # Convert strategies to DataFrames for display
        supplier_df = pd.DataFrame({
            "Supplier Action": supplier_labels,
            "Probability": supplier_strat
        })
        distributor_df = pd.DataFrame({
            "Distributor Action": distributor_labels,
            "Probability": distributor_strat
        })

        st.markdown("**Supplier Strategy**")
        st.table(supplier_df)
        st.markdown("**Distributor Strategy**")
        st.table(distributor_df)

    # Let user select which equilibrium to apply
    st.markdown("### Apply an Equilibrium to Update Policy")
    eq_options = [f"Equilibrium {i+1}" for i in range(len(equilibria))]
    selected_eq_index = st.selectbox("Select which Equilibrium to Apply", range(len(equilibria)),
                                     format_func=lambda x: eq_options[x])

    if st.button("Apply Selected Equilibrium"):
        chosen_eq = equilibria[selected_eq_index]
        supplier_strategy = chosen_eq[0]
        distributor_strategy = chosen_eq[1]

        # Adjust policy based on the chosen equilibrium
        order_policy, warehouse_policy, shipping_policy = adjust_supply_chain_policy(
            supplier_strategy, distributor_strategy
        )

        st.subheader("Supply Chain Policy Adjustments Based on Selected Nash Equilibrium")
        adjustments_df = pd.DataFrame({
            "Policy Aspect": ["Supplier Order Policy", "Warehouse Inventory Policy", "Shipping & Distribution Policy"],
            "Decision": [order_policy, warehouse_policy, shipping_policy]
        })
        st.table(adjustments_df)

        # ──────────────────────────────────────────────
        # VISUALIZING POLICY IMPROVEMENTS WITH PLOTLY
        # ──────────────────────────────────────────────
        st.markdown("### Visualization of Policy Improvement")
        
        # For demonstration, we generate example performance metrics
        old_performance_supplier = np.random.uniform(1, 5)
        old_performance_distributor = np.random.uniform(1, 5)
        new_performance_supplier = old_performance_supplier + np.random.uniform(0.5, 2.0)
        new_performance_distributor = old_performance_distributor + np.random.uniform(0.5, 2.0)

        improvement_df = pd.DataFrame({
            "Agent": ["Supplier", "Distributor"],
            "Old Policy": [old_performance_supplier, old_performance_distributor],
            "New (Nash) Policy": [new_performance_supplier, new_performance_distributor]
        })

        st.markdown("**Performance Comparison (Example):**")
        st.dataframe(improvement_df.style.format(precision=2))

        # Melt DataFrame for Plotly
        improvement_melted = improvement_df.melt(id_vars="Agent", 
                                                 var_name="Policy", 
                                                 value_name="Performance")

        # Create interactive Plotly bar plot
        fig = px.bar(improvement_melted, 
                     x="Agent", 
                     y="Performance", 
                     color="Policy",
                     barmode="group",
                     title="Comparison of Old vs. New Policy Performance",
                     labels={"Performance": "Performance (Example Units)"})
        st.plotly_chart(fig)


# def policy_update_section():
#     st.header("Update MDP Policy with Nash Equilibrium")
#     equilibria = nash_equilibrium_analysis()
#     if equilibria:
#         chosen_eq = equilibria[0]
#         supplier_strategy = chosen_eq[0]
#         distributor_strategy = chosen_eq[1]
#         updated_policy = update_mdp_with_nash(pomdp_states, pomdp_actions, pomdp_transition_probs, pomdp_rewards, supplier_strategy, distributor_strategy)
#         st.write("Updated Supply Chain Policy:")
#         st.write(updated_policy)
def policy_update_section():
    st.header("Update MDP Policy with Nash Equilibrium (Dynamic & Silent)")
    equilibria = nash_equilibrium_analysis()

    if not equilibria:
        st.warning("No Nash equilibria found with the current payoff matrices.")
        return

    # Let user pick which equilibrium to apply
    eq_options = [f"Equilibrium {i+1}" for i in range(len(equilibria))]
    chosen_eq_idx = st.selectbox("Select an Equilibrium", range(len(equilibria)),
                                 format_func=lambda i: eq_options[i])

    if st.button("Apply Selected Equilibrium"):
        chosen_eq = equilibria[chosen_eq_idx]
        supplier_strategy = chosen_eq[0]
        distributor_strategy = chosen_eq[1]

        # Run the silent POMDP iteration + Nash policy update
        updated_policy = update_mdp_with_nash(
            pomdp_states,
            pomdp_actions,
            pomdp_transition_probs,
            observation_probs,
            pomdp_rewards,  # or pomdp_rewards if that is your actual dictionary
            supplier_strategy,
            distributor_strategy
        )

        st.subheader("Updated Supply Chain Policy (Table Format)")

        # Convert the policy dict to a DataFrame for a cleaner table
        updated_policy_df = pd.DataFrame(list(updated_policy.items()),
                                         columns=["State", "Action"])
        st.table(updated_policy_df)

# def route_visualization_section():
#     st.header("Optimal Delivery Route Visualization")
#     nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
#     source = st.selectbox("Select Source", nodes, index=0)
#     destination = st.selectbox("Select Destination", nodes, index=len(nodes)-1)
#     fig, path, _ = plot_optimal_route(source, destination)
#     if fig:
#         st.pyplot(fig)
#         st.write("Optimal Route:", " -> ".join(path))
def route_visualization_section():
    st.header("Optimal Delivery Route Visualization (Interactive)")

    # Define the nodes and a fixed layout for them
    nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
    # You can tweak these (x, y) coordinates for a more pleasing layout
    pos = {
        "Supplier":      (0, 0),
        "Warehouse_A":   (2, 1),
        "Warehouse_B":   (2, -1),
        "Retailer_A":    (4, 1),
        "Retailer_B":    (4, -1),
        "Customer":      (6, 0)
    }

    # Define edges (routes) with associated cost
    edges = [
        ("Supplier", "Warehouse_A", {"cost": 3}),
        ("Supplier", "Warehouse_B", {"cost": 4}),
        ("Warehouse_A", "Retailer_A", {"cost": 2}),
        ("Warehouse_B", "Retailer_B", {"cost": 3}),
        ("Retailer_A", "Customer",   {"cost": 1}),
        ("Retailer_B", "Customer",   {"cost": 2}),
    ]

    # Build a simple graph (we can still use NetworkX for pathfinding)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for u, v, d in edges:
        G.add_edge(u, v, cost=d["cost"])

    # User picks source & destination
    source = st.selectbox("Select Source", nodes, index=0)
    destination = st.selectbox("Select Destination", nodes, index=len(nodes)-1)

    # Try to find the shortest path
    try:
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight="cost")
    except nx.NetworkXNoPath:
        st.error("No path exists between the selected source and destination.")
        return

    # Convert the path into a set of edges for easy highlighting
    path_edges = set(zip(shortest_path, shortest_path[1:]))

    # Prepare Plotly figure
    fig = go.Figure()

    # 1) Draw all edges as line segments
    for u, v, d in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Check if this edge is part of the shortest path
        if (u, v) in path_edges or (v, u) in path_edges:
            color = "red"
            width = 4
        else:
            color = "gray"
            width = 2

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="none"  # We'll rely on node hover
            )
        )

    # 2) Draw the nodes
    node_x = []
    node_y = []
    node_names = []
    for node in nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_names,
            textposition="top center",
            marker=dict(size=20, color="lightblue", line=dict(width=2, color="darkblue")),
            hovertext=node_names,
            hoverinfo="text"
        )
    )

    # Update layout for a cleaner look
    fig.update_layout(
        title=f"Optimal Route: {source} → {destination}",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=40, r=40, t=50, b=40),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display the actual route text
    st.write("**Optimal Route:**", " → ".join(shortest_path))
# def simulation_section():
#     st.header("Real-Time Decision-Making Simulation")
#     duration = st.slider("Simulation Duration (time steps)", 50, 200, 100, step=10)
#     if st.button("Run Simulation"):
#         sim_df = real_time_simulation(duration)
#         st.line_chart(sim_df)
def simulation_section():
    st.header("Real-Time Decision-Making Simulation")

    # 1) Let user pick simulation duration
    duration = st.slider("Simulation Duration (time steps)", 50, 200, 100, step=10)

    # 2) Run simulation on button click
    if st.button("Run Simulation"):
        sim_df = real_time_simulation(duration)

        # 3) Display line chart
        st.line_chart(sim_df)

        # 4) Provide some summary statistics/insights
        st.subheader("Simulation Insights")

        # a) Basic Stats
        avg_demand = sim_df["Demand"].mean()
        max_inventory = sim_df["Inventory"].max()
        min_inventory = sim_df["Inventory"].min()

        st.write(f"- **Average Demand**: {avg_demand:.2f} units")
        st.write(f"- **Maximum Inventory**: {max_inventory} units")
        st.write(f"- **Minimum Inventory**: {min_inventory} units")

        # b) Example Additional Insight: Inventory Runout
        # Check if inventory ever hits zero (stockout)
        stockouts = (sim_df["Inventory"] == 0).sum()
        if stockouts > 0:
            st.warning(f"Stockouts occurred {stockouts} time(s). Consider increasing safety stock or improving replenishment.")
        else:
            st.success("No stockouts occurred during this simulation.")

        # c) Another Example: Demand Surges
        # We'll define a 'surge' if demand > average + 2 * std dev
        demand_std = sim_df["Demand"].std()
        surge_threshold = avg_demand + 2 * demand_std
        surge_count = (sim_df["Demand"] > surge_threshold).sum()

        if surge_count > 0:
            st.info(f"High-demand surges occurred {surge_count} time(s). Peak demand can stress the supply chain.")
        else:
            st.write("No significant demand surges occurred during this simulation.")

        # d) You can add more business-specific insights here
        # e.g., fill rate, average backlog, cost implications, etc.

# =============================================================================
# Main App Layout
# =============================================================================
st.title("Supply Chain Optimization ")
section = st.sidebar.selectbox(
    "Select Section", 
    [
        "Dashboard", 
        "Transition Model Visualization", 
        "POMDP Simulation", 
        "Nash Equilibrium Analysis", 
        "MDP Optimization", 
        "MDP Policy Update using Nash", 
        "Optimal Delivery Route Visualization",
        "Real-Time Simulation"
    ]
)

if section == "Dashboard":
    synthetic_data_section()
elif section == "MDP Optimization":
    mdp_optimization_section()
elif section == "Transition Model Visualization":
    transition_visualization_section()
elif section == "POMDP Simulation":
    pomdp_simulation_section()
elif section == "Nash Equilibrium Analysis":
    nash_equilibrium_section()
elif section == "MDP Policy Update using Nash":
    policy_update_section()
elif section == "Optimal Delivery Route Visualization":
    route_visualization_section()
elif section == "Real-Time Simulation":
    simulation_section()

