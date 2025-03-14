import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random
import nashpy as nash

# ---------------------- Streamlit Page Config ---------------------- #
st.set_page_config(page_title="📦 Supply Chain Optimization", layout="wide")

# ---------------------- Define Global Supply Chain Elements ---------------------- #
states = ['Supplier_Good', 'Supplier_Delayed', 'Warehouse_Low', 'Warehouse_Normal', 'Retailer_High', 'Retailer_Low']
actions = ['Order', 'Ship', 'Hold']

# Transition Probabilities (MDP & POMDP)
transition_probs = {
    ('Supplier_Good', 'Order'): {'Warehouse_Normal': 0.8, 'Supplier_Delayed': 0.2},
    ('Supplier_Delayed', 'Order'): {'Warehouse_Low': 0.6, 'Supplier_Delayed': 0.4},
    ('Warehouse_Normal', 'Ship'): {'Retailer_High': 0.7, 'Warehouse_Low': 0.3},
    ('Warehouse_Low', 'Ship'): {'Retailer_Low': 0.5, 'Warehouse_Low': 0.5},
    ('Retailer_High', 'Ship'): {'Customer': 0.9, 'Retailer_High': 0.1},
    ('Retailer_Low', 'Ship'): {'Customer': 0.8, 'Retailer_Low': 0.2},
    ('Retailer_High', 'Hold'): {'Retailer_High': 1.0},
    ('Retailer_Low', 'Hold'): {'Retailer_Low': 1.0},
}

# Rewards for MDP
rewards = {
    ('Supplier_Good', 'Order'): -2,
    ('Supplier_Delayed', 'Order'): -3,
    ('Warehouse_Normal', 'Ship'): -1,
    ('Warehouse_Low', 'Ship'): -2,
    ('Retailer_High', 'Ship'): 5,
    ('Retailer_Low', 'Ship'): 3,
    ('Retailer_High', 'Hold'): -1,
    ('Retailer_Low', 'Hold'): -0.5,
}

# ---------------------- Sidebar Navigation ---------------------- #
menu = st.sidebar.radio("📌 Navigation", 
                        ["Home", "Synthetic Data", "MDP Optimization", "POMDP Optimization", "Nash Equilibrium", "Delivery Routes"])

# ---------------------- Home Page ---------------------- #
if menu == "Home":
    st.title("📦 Supply Chain Optimization System")
    st.markdown("""
    Welcome to the **Supply Chain Optimization System**!  
    🚀 **Key Features:**  
    - **MDP & POMDP:** Optimize inventory & handle uncertainty.  
    - **Nash Equilibrium:** Multi-agent supply chain decisions.  
    - **Optimized Delivery Routes:** Shortest-path visualization.  
    """)

# ---------------------- Synthetic Data Generation ---------------------- #
elif menu == "Synthetic Data":
    st.header("📊 Synthetic Supply Chain Data")
    
    @st.cache_data
    def generate_synthetic_data(num_entries=500):
        np.random.seed(42)
        data = []
        for _ in range(num_entries):
            entry = {
                "Lead_Time": np.random.randint(2, 10),
                "Demand": np.random.randint(50, 500),
                "Inventory_Level": np.random.randint(0, 1000),
                "Shipping_Cost": round(np.random.uniform(5, 50), 2),
                "Order_Quantity": np.random.randint(10, 200),
            }
            data.append(entry)
        return pd.DataFrame(data)

    df = generate_synthetic_data()
    st.dataframe(df, use_container_width=True)

# ---------------------- MDP Optimization ---------------------- #
elif menu == "MDP Optimization":
    st.header("⚙️ MDP Optimization for Supply Chain")
    
    def value_iteration(states, actions, transition_probs, rewards, gamma=0.9, theta=0.0001):
        V = {s: 0 for s in states}
        policy = {s: random.choice(actions) for s in states[:-1]}
        while True:
            delta = 0
            for s in states[:-1]:
                max_value = float('-inf')
                best_action = None
                for a in actions:
                    if (s, a) in transition_probs:
                        expected_value = sum(p * (rewards.get((s, a), 0) + gamma * V.get(s_next, 0))
                                             for s_next, p in transition_probs[(s, a)].items())
                        if expected_value > max_value:
                            max_value = expected_value
                            best_action = a
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
                policy[s] = best_action
            if delta < theta:
                break
        return policy

    optimal_policy = value_iteration(states, actions, transition_probs, rewards)
    st.json(optimal_policy)

# ---------------------- POMDP Optimization ---------------------- #
elif menu == "POMDP Optimization":
    st.header("🔀 Handling Uncertainty Using POMDP")
    
    def update_belief(belief, action, observation):
        new_belief = {}
        total_prob = 0
        for state in belief:
            if (state, action) in transition_probs and observation in transition_probs[(state, action)]:
                prob_obs_given_state = transition_probs[(state, action)][observation]
                new_belief[state] = belief[state] * prob_obs_given_state
                total_prob += new_belief[state]
        for state in new_belief:
            new_belief[state] /= total_prob if total_prob > 0 else 1
        return new_belief
    
    belief_state = {'Supplier_Good': 0.7, 'Supplier_Delayed': 0.3}
    belief_state = update_belief(belief_state, 'Order', 'Delayed')
    st.json(belief_state)

# ---------------------- Nash Equilibrium ---------------------- #
elif menu == "Nash Equilibrium":
    st.header("⚖️ Multi-Agent Decision Making Using Nash Equilibrium")
    
    supplier_payoff = np.array([[3, -1], [2, 1]])
    distributor_payoff = np.array([[2, 1], [-1, 3]])
    game = nash.Game(supplier_payoff, distributor_payoff)
    equilibria = list(game.support_enumeration())
    formatted_equilibria = [{"Supplier": np.round(eq[0], 2), "Distributor": np.round(eq[1], 2)} for eq in equilibria]
    st.table(formatted_equilibria)

# ---------------------- Optimized Delivery Routes ---------------------- #
elif menu == "Delivery Routes":
    st.header("🚚 Optimized Delivery Routes")
    
    G = nx.DiGraph()
    G.add_edges_from([('Supplier', 'Warehouse_A'), ('Warehouse_A', 'Retailer_A'), ('Retailer_A', 'Customer')])
    shortest_path = nx.shortest_path(G, source='Supplier', target='Customer', weight=None, method='dijkstra')
    st.write(f"**Optimal Route:** {' → '.join(shortest_path)}")
