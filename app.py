# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
import random
import nashpy as nash
import matplotlib.pyplot as plt

# ---------------------- Streamlit Page Config ---------------------- #
st.set_page_config(page_title="📦 Supply Chain Optimization", layout="wide")

# ---------------------- Sidebar Navigation ---------------------- #
menu = st.sidebar.radio("Navigation", 
                        ["🏠 Home", "📊 Synthetic Data", "⚙️ MDP Optimization", "⚖️ Nash Equilibrium", "🚚 Delivery Routes", "📉 Supply Chain Model"])

# ---------------------- Home Page ---------------------- #
if menu == "🏠 Home":
    st.title("📦 Supply Chain Optimization System")
    st.markdown("""
    Welcome to the **Supply Chain Optimization System**!  
    🚀 **Key Features:**  
    - **Markov Decision Process (MDP):** Optimize inventory and shipping.  
    - **Partially Observable MDP (POMDP):** Handle uncertainty in supply & demand.  
    - **Nash Equilibrium:** Manage interactions between suppliers & distributors.  
    - **Dynamic Route Visualization:** Explore optimal delivery paths.  
    """)

# ---------------------- Synthetic Data Generation ---------------------- #
elif menu == "📊 Synthetic Data":
    st.header("📊 Synthetic Supply Chain Data")

    @st.cache_data
    def generate_synthetic_data(num_entries=500):
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

    df = generate_synthetic_data()
    st.dataframe(df, use_container_width=True)

# ---------------------- MDP Optimization ---------------------- #
elif menu == "⚙️ MDP Optimization":
    st.header("⚙️ Real-Time Supply Chain Policy Optimization")

    # Define Supply Chain Elements
    states = ['Supplier', 'Warehouse', 'Retailer', 'Customer']
    actions = ['Order', 'Ship', 'Hold']

    # Transition Probabilities
    transition_probs = {
        ('Supplier', 'Order'): {'Warehouse': 0.8, 'Supplier': 0.2},
        ('Warehouse', 'Ship'): {'Retailer': 0.7, 'Warehouse': 0.3},
        ('Retailer', 'Ship'): {'Customer': 0.9, 'Retailer': 0.1},
        ('Warehouse', 'Hold'): {'Warehouse': 1.0},
        ('Retailer', 'Hold'): {'Retailer': 1.0},
    }

    # Rewards (Cost Optimization)
    rewards = {
        ('Supplier', 'Order'): -2,
        ('Warehouse', 'Ship'): -1,
        ('Retailer', 'Ship'): 5,
        ('Warehouse', 'Hold'): -0.5,
        ('Retailer', 'Hold'): -0.2,
    }

    # Value Iteration Function
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
                        expected_value = sum(p * (rewards.get((s, a), 0) + gamma * V[s_next])
                                             for s_next, p in transition_probs[(s, a)].items())

                        if expected_value > max_value:
                            max_value = expected_value
                            best_action = a

                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
                policy[s] = best_action

            if delta < theta:
                break

        return policy, V

    # Run Value Iteration
    optimal_policy, optimal_values = value_iteration(states, actions, transition_probs, rewards)

    st.write("### 🔄 Optimized Supply Chain Policy")
    st.json(optimal_policy)

# ---------------------- Nash Equilibrium ---------------------- #
elif menu == "⚖️ Nash Equilibrium":
    st.header("⚖️ Multi-Agent Strategy Using Nash Equilibrium")

    supplier_payoff = np.array([[3, -1], [2, 1]])
    distributor_payoff = np.array([[2, 1], [-1, 3]])
    game = nash.Game(supplier_payoff, distributor_payoff)
    equilibria = list(game.support_enumeration())

    formatted_equilibria = []
    for eq in equilibria:
        supplier_strategy = f"Supplier: {np.round(eq[0], 2)}"
        distributor_strategy = f"Distributor: {np.round(eq[1], 2)}"
        formatted_equilibria.append({"Supplier Strategy": supplier_strategy, "Distributor Strategy": distributor_strategy})

    st.table(formatted_equilibria)

# ---------------------- Delivery Routes ---------------------- #
elif menu == "🚚 Delivery Routes":
    st.header("🚚 Interactive Optimal Delivery Route")
 
    # Create the PyVis graph
    def create_pyvis_graph(G):
        net = Network(height="500px", width="100%", directed=True, notebook=False)
        
        for node in G.nodes:
            net.add_node(node, label=node, color="lightblue")
    
        for edge in G.edges:
            net.add_edge(edge[0], edge[1])
    
        return net
    
    # Example: Create the network
    G = nx.DiGraph()
    G.add_edges_from([
        ("Supplier", "Warehouse_A"),
        ("Supplier", "Warehouse_B"),
        ("Warehouse_A", "Retailer_A"),
        ("Warehouse_B", "Retailer_B"),
        ("Retailer_A", "Customer"),
        ("Retailer_B", "Customer"),
    ])
    
    # Generate PyVis graph
    net = create_pyvis_graph(G)
    
    # Create a temporary file to save the graph
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        html_path = tmp_file.name
        net.write_html(html_path)
    
    # Display the graph in Streamlit
    st.components.v1.html(open(html_path, "r", encoding="utf-8").read(), height=550, scrolling=True)
    
    # Remove the temporary file after rendering
    os.remove(html_path)

    # # Interactive PyVis Graph
    # net = Network(height="500px", width="100%", directed=True, bgcolor="#222222", font_color="white")

    # # Define Supply Chain Nodes
    # nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
    # for node in nodes:
    #     net.add_node(node, label=node, color='lightblue')

    # # Define Edges with Costs
    # edges = [
    #     ("Supplier", "Warehouse_A", 3),
    #     ("Supplier", "Warehouse_B", 4),
    #     ("Warehouse_A", "Retailer_A", 2),
    #     ("Warehouse_B", "Retailer_B", 3),
    #     ("Retailer_A", "Customer", 1),
    #     ("Retailer_B", "Customer", 2),
    # ]

    # for u, v, cost in edges:
    #     net.add_edge(u, v, label=f"{cost} days")

    # # Save and display
    # net.show("routes.html")
    # st.components.v1.html(open("routes.html", "r").read(), height=600)

# ---------------------- Supply Chain Model ---------------------- #
elif menu == "📉 Supply Chain Model":
    st.header("📉 Supply Chain Transition Model")
    st.image("supply_chain_diagram.png", caption="Supply Chain Model")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import nashpy as nash
# import random

# st.set_page_config(page_title="📦 Supply Chain Management System", layout="wide")

# # --- TITLE ---
# st.title("📦 Supply Chain Management System")
# st.subheader("📊 Optimizing Logistics & Distribution Using AI")

# st.markdown("🔹 **Using MDP, POMDP & Nash Equilibrium to optimize supply chain decisions**")

# # --- SECTION 1: SYNTHETIC DATA GENERATION ---
# st.header("📊 Synthetic Supply Chain Data")

# @st.cache_data
# def generate_synthetic_data(num_entries=500):
#     np.random.seed(42)
#     suppliers = ["Supplier_A", "Supplier_B", "Supplier_C"]
#     warehouses = ["Warehouse_1", "Warehouse_2", "Warehouse_3"]
#     retailers = ["Retailer_X", "Retailer_Y", "Retailer_Z"]
#     customers = ["Customer_1", "Customer_2", "Customer_3"]
    
#     data = []
#     for _ in range(num_entries):
#         entry = {
#             "Supplier": random.choice(suppliers),
#             "Warehouse": random.choice(warehouses),
#             "Retailer": random.choice(retailers),
#             "Customer": random.choice(customers),
#             "Lead_Time": np.random.randint(2, 10),
#             "Demand": np.random.randint(50, 500),
#             "Inventory_Level": np.random.randint(0, 1000),
#             "Shipping_Cost": round(np.random.uniform(5, 50), 2),
#             "Order_Quantity": np.random.randint(10, 200),
#         }
#         data.append(entry)
    
#     return pd.DataFrame(data)

# df = generate_synthetic_data()
# st.dataframe(df.head(), use_container_width=True)

# # --- SECTION 2: SUPPLY CHAIN OPTIMIZATION USING MDP ---
# st.header("⚙️ Real-Time Supply Chain Policy Optimization")

# # Define states and actions
# states = ['Supplier', 'Warehouse', 'Retailer', 'Customer']
# actions = ['Order', 'Ship', 'Hold']

# # Define transition probabilities
# transition_probs = {
#     ('Supplier', 'Order'): {'Warehouse': 0.8, 'Supplier': 0.2},
#     ('Warehouse', 'Ship'): {'Retailer': 0.7, 'Warehouse': 0.3},
#     ('Retailer', 'Ship'): {'Customer': 0.9, 'Retailer': 0.1},
#     ('Warehouse', 'Hold'): {'Warehouse': 1.0},
#     ('Retailer', 'Hold'): {'Retailer': 1.0},
# }

# # User Inputs for Real-Time Adjustments
# lead_time_factor = st.slider("⏳ Adjust Lead Time Impact", 0.5, 2.0, 1.0)
# demand_fluctuation = st.slider("📊 Adjust Demand Fluctuation", 0.5, 2.0, 1.0)
# inventory_sensitivity = st.slider("📦 Adjust Inventory Impact", 0.5, 2.0, 1.0)

# # Value Iteration Function
# def value_iteration(states, actions, transition_probs, rewards, gamma=0.9, theta=0.0001):
#     V = {s: 0 for s in states}
#     policy = {s: random.choice(actions) for s in states[:-1]}
    
#     while True:
#         delta = 0
#         for s in states[:-1]:
#             max_value = float('-inf')
#             best_action = None
            
#             for a in actions:
#                 if (s, a) in transition_probs:
#                     expected_value = sum(p * (rewards.get((s, a), 0) + gamma * V[s_next])
#                                          for s_next, p in transition_probs[(s, a)].items())
                    
#                     if expected_value > max_value:
#                         max_value = expected_value
#                         best_action = a
            
#             delta = max(delta, abs(V[s] - max_value))
#             V[s] = max_value
#             policy[s] = best_action
        
#         if delta < theta:
#             break
    
#     return policy, V

# # Run Value Iteration with User Inputs
# def dynamic_value_iteration():
#     modified_rewards = {
#         ('Supplier', 'Order'): -2 * lead_time_factor,
#         ('Warehouse', 'Ship'): -1 * demand_fluctuation,
#         ('Retailer', 'Ship'): 5 * inventory_sensitivity,
#         ('Warehouse', 'Hold'): -0.5,
#         ('Retailer', 'Hold'): -0.2,
#     }
    
#     return value_iteration(states, actions, transition_probs, modified_rewards)

# dynamic_policy, _ = dynamic_value_iteration()

# # Display Updated Policy
# st.write("### 🔄 Optimized Policy Based on Adjustments")
# st.json(dynamic_policy)

# # --- SECTION 3: NASH EQUILIBRIUM FOR MULTI-AGENT DECISIONS ---
# st.header("⚖️ Multi-Agent Strategy Using Nash Equilibrium")

# supplier_payoff = np.array([[3, -1], [2, 1]])
# distributor_payoff = np.array([[2, 1], [-1, 3]])
# game = nash.Game(supplier_payoff, distributor_payoff)
# equilibria = list(game.support_enumeration())

# # Format Nash Equilibrium Output
# formatted_equilibria = []
# for eq in equilibria:
#     formatted_equilibria.append({"Supplier Strategy": np.round(eq[0], 2).tolist(), 
#                                  "Distributor Strategy": np.round(eq[1], 2).tolist()})

# # Display as a Table
# st.write("### Nash Equilibrium Strategies")
# st.table(formatted_equilibria)

# # --- SECTION 4: INTERACTIVE DELIVERY ROUTE VISUALIZATION ---
# st.header("🚚 Interactive Optimal Delivery Route")

# # Define Graph
# G = nx.DiGraph()
# nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
# G.add_nodes_from(nodes)

# # Add Edges (Routes & Costs)
# edges = [
#     ("Supplier", "Warehouse_A", {"cost": 3}),
#     ("Supplier", "Warehouse_B", {"cost": 4}),
#     ("Warehouse_A", "Retailer_A", {"cost": 2}),
#     ("Warehouse_B", "Retailer_B", {"cost": 3}),
#     ("Retailer_A", "Customer", {"cost": 1}),
#     ("Retailer_B", "Customer", {"cost": 2}),
# ]
# G.add_edges_from([(u, v, d) for u, v, d in edges])

# # User Selection
# selected_source = st.selectbox("📦 Select Source", nodes)
# selected_destination = st.selectbox("🎯 Select Destination", nodes)

# # Compute Shortest Path Based on Selection
# if selected_source and selected_destination:
#     try:
#         shortest_path = nx.shortest_path(G, source=selected_source, target=selected_destination, weight="cost")

#         # Display Route
#         st.write(f"**Optimal Route:** {' → '.join(shortest_path)}")

#         # Plot Updated Graph
#         fig, ax = plt.subplots(figsize=(8, 5))
#         pos = nx.spring_layout(G)
#         nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
#         nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['cost']} days" for u, v, d in edges})
#         path_edges = list(zip(shortest_path, shortest_path[1:]))
#         nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)
#         st.pyplot(fig)

#     except nx.NetworkXNoPath:
#         st.error("⚠️ No available route between the selected locations!")

# st.write("---")
# st.write("💡 **Developed for Hackathon: Reasoning & Decision Making Under Uncertainty**")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import nashpy as nash
# import random

# st.title("🚀 Supply Chain Optimization System")

# # --- SECTION 1: SYNTHETIC DATA GENERATION ---
# st.header("📊 Synthetic Supply Chain Data")

# # Generate Synthetic Data
# def generate_synthetic_data(num_entries=1000):
#     np.random.seed(42)
#     suppliers = ["Supplier_A", "Supplier_B", "Supplier_C"]
#     warehouses = ["Warehouse_1", "Warehouse_2", "Warehouse_3"]
#     retailers = ["Retailer_X", "Retailer_Y", "Retailer_Z"]
#     customers = ["Customer_1", "Customer_2", "Customer_3"]
    
#     data = []
#     for _ in range(num_entries):
#         entry = {
#             "Supplier": random.choice(suppliers),
#             "Warehouse": random.choice(warehouses),
#             "Retailer": random.choice(retailers),
#             "Customer": random.choice(customers),
#             "Lead_Time": np.random.randint(2, 10),  # Days
#             "Demand": np.random.randint(50, 500),  # Units
#             "Inventory_Level": np.random.randint(0, 1000),  # Stock count
#             "Shipping_Cost": round(np.random.uniform(5, 50), 2),  # Cost per unit
#             "Order_Quantity": np.random.randint(10, 200),
#         }
#         data.append(entry)
    
#     return pd.DataFrame(data)

# df = generate_synthetic_data()
# st.dataframe(df.head())

# # --- SECTION 2: NASH EQUILIBRIUM BASED DECISIONS ---
# st.header("⚖️ Nash Equilibrium for Supplier & Distributor")

# supplier_reliability = st.slider("Supplier Reliability (%)", 50, 100, 80)
# demand_variability = st.slider("Demand Variability (%)", 10, 50, 20)

# # Compute Nash Strategies
# if supplier_reliability > 75:
#     supplier_strategy = [0.8, 0.2]
# else:
#     supplier_strategy = [0.4, 0.6]

# if demand_variability > 30:
#     distributor_strategy = [0.6, 0.4]
# else:
#     distributor_strategy = [0.8, 0.2]

# # Compute Nash Equilibrium
# supplier_payoff = np.array([[3, -1], [2, 1]])
# distributor_payoff = np.array([[2, 1], [-1, 3]])
# game = nash.Game(supplier_payoff, distributor_payoff)
# equilibria = list(game.support_enumeration())

# st.subheader("🔄 Nash Equilibrium Strategy")
# st.write(f"📦 **Supplier Strategy:** {supplier_strategy}")
# st.write(f"🛒 **Distributor Strategy:** {distributor_strategy}")
# st.write(f"⚖️ **Computed Nash Equilibria:** {equilibria}")

# # --- SECTION 3: OPTIMAL DELIVERY ROUTES VISUALIZATION ---
# st.header("🚚 Optimal Delivery Route")

# # Define Graph
# G = nx.DiGraph()
# nodes = ["Supplier", "Warehouse_A", "Warehouse_B", "Retailer_A", "Retailer_B", "Customer"]
# G.add_nodes_from(nodes)

# # Add Edges (Routes & Costs)
# edges = [
#     ("Supplier", "Warehouse_A", {"cost": 3}),
#     ("Supplier", "Warehouse_B", {"cost": 4}),
#     ("Warehouse_A", "Retailer_A", {"cost": 2}),
#     ("Warehouse_B", "Retailer_B", {"cost": 3}),
#     ("Retailer_A", "Customer", {"cost": 1}),
#     ("Retailer_B", "Customer", {"cost": 2}),
# ]
# G.add_edges_from([(u, v, d) for u, v, d in edges])

# # User Selection for Warehouses & Retailers
# warehouse_choice = st.radio("Select Warehouse:", ["Warehouse_A", "Warehouse_B"])
# retailer_choice = st.radio("Select Retailer:", ["Retailer_A", "Retailer_B"])

# # Compute Optimal Path
# shortest_path = nx.shortest_path(G, source="Supplier", target="Customer", weight="cost")

# # Plot the Graph
# fig, ax = plt.subplots(figsize=(8, 5))
# pos = nx.spring_layout(G)  # Layout
# nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
# nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['cost']} days" for u, v, d in edges})

# # Highlight Optimal Route
# path_edges = list(zip(shortest_path, shortest_path[1:]))
# nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

# st.pyplot(fig)
# st.subheader("🚀 Recommended Route")
# st.write(f"**Optimal Route:** {' → '.join(shortest_path)}")

# st.write("---")
# st.write("💡 **Developed for Hackathon: Reasoning & Decision Making Under Uncertainty**")

