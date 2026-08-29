import streamlit as st
import numpy as np
import plotly.graph_objects as go

def lorenz_system(state, t, sigma, rho, beta):
    """
    Returns the derivatives [dx/dt, dy/dt, dz/dt] for the Lorenz system.
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def rk4_step(state, t, dt, sigma, rho, beta):
    """
    Performs a single step of the Runge-Kutta 4th order method.
    """
    k1 = lorenz_system(state, t, sigma, rho, beta)
    k2 = lorenz_system(state + 0.5 * dt * k1, t + 0.5 * dt, sigma, rho, beta)
    k3 = lorenz_system(state + 0.5 * dt * k2, t + 0.5 * dt, sigma, rho, beta)
    k4 = lorenz_system(state + dt * k3, t + dt, sigma, rho, beta)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_lorenz(initial_state, num_steps, dt, sigma, rho, beta):
    """
    Simulates the Lorenz system using RK4 and returns the time series of states.
    """
    states = np.zeros((num_steps, 3))
    states[0] = initial_state

    current_state = initial_state
    t = 0.0

    for i in range(1, num_steps):
        next_state = rk4_step(current_state, t, dt, sigma, rho, beta)
        states[i] = next_state
        current_state = next_state
        t += dt

    return states

def create_plot(states1, states2, skip_frames=10):
    """
    Creates a 3D animated plot using Plotly Graph Objects.
    """
    # Downsample for animation performance
    s1 = states1[::skip_frames]
    s2 = states2[::skip_frames]

    # Create the figure
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[s1[0, 0]], y=[s1[0, 1]], z=[s1[0, 2]],
        mode='lines',
        line=dict(color='#00ffff', width=3),
        name='Trajectory 1 (Neon Blue)'
    ))

    fig.add_trace(go.Scatter3d(
        x=[s2[0, 0]], y=[s2[0, 1]], z=[s2[0, 2]],
        mode='lines',
        line=dict(color='#ff00ff', width=3),
        name='Trajectory 2 (Neon Magenta)'
    ))

    # Create frames for animation
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(x=s1[:k+1, 0], y=s1[:k+1, 1], z=s1[:k+1, 2]),
                go.Scatter3d(x=s2[:k+1, 0], y=s2[:k+1, 1], z=s2[:k+1, 2])
            ],
            name=str(k)
        )
        for k in range(1, len(s1))
    ]

    fig.frames = frames

    # Update layout with Play/Pause buttons and dark theme
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='Y', showbackground=False),
            zaxis=dict(title='Z', showbackground=False),
            # Set fixed ranges to prevent axis jumping during animation
            xaxis_range=[min(np.min(states1[:,0]), np.min(states2[:,0])) - 5, max(np.max(states1[:,0]), np.max(states2[:,0])) + 5],
            yaxis_range=[min(np.min(states1[:,1]), np.min(states2[:,1])) - 5, max(np.max(states1[:,1]), np.max(states2[:,1])) + 5],
            zaxis_range=[min(np.min(states1[:,2]), np.min(states2[:,2])) - 5, max(np.max(states1[:,2]), np.max(states2[:,2])) + 5]
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=5, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))])
            ],
            showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top"
        )],
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.8, y=0.9, xanchor='left', yanchor='top')
    )

    return fig

def main():
    st.set_page_config(page_title="Lorenz Attractor", layout="wide", initial_sidebar_state="expanded")

    # Custom CSS for a darker, sleeker theme
    st.markdown("""
        <style>
        .reportview-container {
            background: #0e1117;
            color: #fafafa;
        }
        .sidebar .sidebar-content {
            background: #262730;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Lorenz Attractor & The Butterfly Effect")
    st.markdown("### A 3D Visualization of Chaotic Systems")

    # --- Sidebar Controls ---
    st.sidebar.header("System Parameters")
    st.sidebar.markdown("Adjust the constants for the Lorenz system equations:")
    sigma = st.sidebar.slider("Sigma (σ) - Prandtl number", 0.0, 50.0, 10.0, 0.1)
    rho = st.sidebar.slider("Rho (ρ) - Rayleigh number", 0.0, 100.0, 28.0, 0.1)
    beta = st.sidebar.slider("Beta (β)", 0.0, 10.0, 2.667, 0.001)

    st.sidebar.header("Initial Conditions")
    st.sidebar.markdown("Set the starting coordinates for the first trajectory:")
    x0 = st.sidebar.number_input("Initial X", value=1.0)
    y0 = st.sidebar.number_input("Initial Y", value=1.0)
    z0 = st.sidebar.number_input("Initial Z", value=1.0)

    st.sidebar.header("The Butterfly Effect")
    st.sidebar.markdown("Offset for the second trajectory to observe sensitive dependence on initial conditions:")
    delta = st.sidebar.slider("Initial Difference (Δ)", 1e-6, 1.0, 1e-4, format="%e")

    st.sidebar.header("Simulation Settings")
    time_steps = st.sidebar.slider("Number of Time Steps", 1000, 10000, 5000, 500)
    dt = st.sidebar.number_input("Time Step (dt)", value=0.01, format="%.4f")
    anim_speed = st.sidebar.slider("Animation Speed (Frame Skip)", 1, 50, 10)

    # --- Split Layout for Plot and Docs ---
    col1, col2 = st.columns([2, 1])

    with col1:
        # --- Simulation ---
        with st.spinner("Simulating trajectories..."):
            initial_state1 = np.array([x0, y0, z0])
            initial_state2 = np.array([x0 + delta, y0 + delta, z0 + delta])

            states1 = simulate_lorenz(initial_state1, time_steps, dt, sigma, rho, beta)
            states2 = simulate_lorenz(initial_state2, time_steps, dt, sigma, rho, beta)

        # --- Plotting ---
        st.write("Click **Play** to animate the trajectories.")
        fig = create_plot(states1, states2, skip_frames=anim_speed)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # --- Documentation ---
        st.markdown("""
        ## Mathematical Foundations

        The **Lorenz system** is a system of ordinary differential equations first studied by Edward Lorenz. It is notable for having chaotic solutions for certain parameter values and initial conditions.

        ### The Equations

        $$
        \\begin{align*}
        \\frac{dx}{dt} &= \\sigma(y - x) \\\\
        \\frac{dy}{dt} &= x(\\rho - z) - y \\\\
        \\frac{dz}{dt} &= xy - \\beta z
        \\end{align*}
        $$

        - $\\sigma$ (Prandtl number): Defines fluid viscosity relative to thermal conductivity.
        - $\\rho$ (Rayleigh number): Relates to the temperature difference between the top and bottom of the fluid system.
        - $\\beta$: Geometric factor.

        ### Numerical Integration: RK4

        To simulate this system over time, we use the **Runge-Kutta 4th Order (RK4)** method. RK4 is a standard technique for numerical integration of ordinary differential equations (ODEs). It calculates a weighted average of four different increments to approximate the next state:

        $$ y_{n+1} = y_n + \\frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4) $$

        where $h$ is the time step (`dt`), and $k_1, k_2, k_3, k_4$ represent the slopes at the beginning, midpoints, and end of the interval. This offers an excellent balance between computational efficiency and accuracy compared to simpler methods like Euler's method.

        ---

        ## The Butterfly Effect

        Chaos theory is beautifully summarized by the "Butterfly Effect" — the concept that small causes can have larger effects. In mathematically chaotic systems, we observe **sensitive dependence on initial conditions**.

        In this visualization, two trajectories are plotted:
        1. **Neon Blue:** Starts exactly at $(X_0, Y_0, Z_0)$.
        2. **Neon Magenta:** Starts at $(X_0 + \\Delta, Y_0 + \\Delta, Z_0 + \\Delta)$.

        When $\\Delta$ is extremely small (e.g., $10^{-4}$), the two trajectories will follow nearly identical paths initially. However, due to the chaotic nature of the Lorenz attractor, they will eventually diverge completely, mapping out the shape of the strange attractor independently.

        *Try adjusting the sliders to see how the system transitions from stable to chaotic states!*
        """)

if __name__ == "__main__":
    main()
