import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def check_diagonal_dominance(A):
    """Check if matrix A is diagonally dominant"""
    n = A.shape[0]
    for i in range(n):
        diagonal = abs(A[i, i])
        off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if i != j)
        if diagonal <= off_diagonal_sum:
            return False
    return True

def gauss_seidel_iteration(A, b, x, iteration_num):
    """Perform one Gauss-Seidel iteration and return detailed steps"""
    n = len(x)
    x_old = x.copy()
    steps = []
    
    for i in range(n):
        # Calculate sum of known terms
        sum_known = b[i]
        calculation_str = f"x{i+1}⁽{iteration_num+1}⁾ = ({b[i]}"
        
        for j in range(n):
            if i != j:
                sum_known -= A[i, j] * x[j]
                calculation_str += f" - ({A[i, j]:.3f})×({x[j]:.6f})"
        
        # Update x[i]
        x[i] = sum_known / A[i, i]
        calculation_str += f") ÷ {A[i, i]:.3f} = {x[i]:.6f}"
        
        steps.append({
            'variable': f'x{i+1}',
            'calculation': calculation_str,
            'new_value': x[i],
            'old_value': x_old[i]
        })
    
    return x, steps, x_old

def solve_gauss_seidel(A, b, x0, tolerance, max_iterations):
    """Solve system using Gauss-Seidel method with detailed tracking"""
    n = len(x0)
    x = x0.copy()
    iterations_data = []
    convergence_data = []
    
    # Check diagonal dominance
    is_diag_dominant = check_diagonal_dominance(A)
    
    for iteration in range(max_iterations):
        x, steps, x_old = gauss_seidel_iteration(A, b, x, iteration)
        
        # Calculate errors
        errors = [abs(x[i] - x_old[i]) for i in range(n)]
        max_error = max(errors)
        
        # Store iteration data
        iteration_data = {
            'iteration': iteration + 1,
            'steps': steps,
            'solution': x.copy(),
            'errors': errors,
            'max_error': max_error,
            'converged': max_error < tolerance
        }
        
        iterations_data.append(iteration_data)
        convergence_data.append({
            'iteration': iteration + 1,
            'max_error': max_error,
            **{f'x{i+1}': x[i] for i in range(n)}
        })
        
        if max_error < tolerance:
            return iterations_data, convergence_data, True, is_diag_dominant
    
    return iterations_data, convergence_data, False, is_diag_dominant

def main():
    # Page configuration
    st.set_page_config(
        page_title="Gauss-Seidel Calculator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .method-explanation {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .iteration-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #e8f4f8;
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔢 Gauss-Seidel Method Calculator</h1>
        <p>Iterative method for solving systems of linear equations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🧮 Calculator", "📚 Method Explanation", "📊 Convergence Analysis"])
    
    with tab1:
        # Input section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📝 Input System")
            
            # System size
            n = st.selectbox("System Size (n×n):", [2, 3, 4], index=1)
            
            # Initialize session state for matrix values
            if f'matrix_{n}' not in st.session_state:
                if n == 3:
                    # Default 3x3 system
                    st.session_state[f'matrix_{n}'] = np.array([
                        [4.0, 1.0, 2.0],
                        [3.0, 5.0, 1.0],
                        [1.0, 1.0, 3.0]
                    ])
                    st.session_state[f'constants_{n}'] = np.array([4.0, 7.0, 3.0])
                else:
                    st.session_state[f'matrix_{n}'] = np.eye(n) * (n + 1) + np.ones((n, n))
                    st.session_state[f'constants_{n}'] = np.ones(n)
            
            # Matrix input
            st.subheader("Coefficient Matrix [A]:")
            matrix_data = []
            
            for i in range(n):
                row_data = []
                cols = st.columns(n)
                for j in range(n):
                    with cols[j]:
                        value = st.number_input(
                            f"a{i+1}{j+1}",
                            value=float(st.session_state[f'matrix_{n}'][i, j]),
                            key=f"a_{i}_{j}_{n}",
                            format="%.3f"
                        )
                        row_data.append(value)
                matrix_data.append(row_data)
            
            A = np.array(matrix_data)
            
            # Constants vector
            st.subheader("Constants Vector [b]:")
            constants_data = []
            b_cols = st.columns(n)
            
            for i in range(n):
                with b_cols[i]:
                    value = st.number_input(
                        f"b{i+1}",
                        value=float(st.session_state[f'constants_{n}'][i]),
                        key=f"b_{i}_{n}",
                        format="%.3f"
                    )
                    constants_data.append(value)
            
            b = np.array(constants_data)
            
            # Parameters
            st.subheader("Solution Parameters:")
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                tolerance = st.number_input("Tolerance (ε):", value=0.001, format="%.6f", min_value=1e-10)
            
            with col1_2:
                max_iterations = st.number_input("Max Iterations:", value=50, min_value=1)
            
            # Initial guess
            st.subheader("Initial Guess [x₀]:")
            initial_guess = []
            x0_cols = st.columns(n)
            
            for i in range(n):
                with x0_cols[i]:
                    value = st.number_input(
                        f"x{i+1}⁽⁰⁾",
                        value=0.0,
                        key=f"x0_{i}_{n}",
                        format="%.3f"
                    )
                    initial_guess.append(value)
            
            x0 = np.array(initial_guess)
            
        with col2:
            st.header("🔍 System Overview")
            
            # Display system in matrix form
            st.subheader("System: Ax = b")
            
            # Create a formatted display of the system
            system_df = pd.DataFrame(A, columns=[f'x{i+1}' for i in range(n)])
            system_df['='] = ['='] * n
            system_df['b'] = b
            
            st.dataframe(system_df, use_container_width=True)
            
            # Display equations
            st.subheader("Equations:")
            for i in range(n):
                equation = ""
                for j in range(n):
                    if j == 0:
                        equation += f"{A[i,j]:.3f}x{j+1}"
                    else:
                        sign = "+" if A[i,j] >= 0 else ""
                        equation += f" {sign}{A[i,j]:.3f}x{j+1}"
                equation += f" = {b[i]:.3f}"
                st.write(f"**Equation {i+1}:** {equation}")
            
            # Gauss-Seidel rearranged form
            st.subheader("Gauss-Seidel Form:")
            for i in range(n):
                equation = f"x{i+1} = ({b[i]:.3f}"
                for j in range(n):
                    if i != j:
                        equation += f" - ({A[i,j]:.3f})x{j+1}"
                equation += f") / {A[i,i]:.3f}"
                st.write(f"**x{i+1}:** {equation}")
        
        # Solve button
        if st.button("🚀 Solve System", type="primary", use_container_width=True):
            try:
                # Check for zero diagonal elements
                if any(abs(A[i, i]) < 1e-10 for i in range(n)):
                    st.error("❌ Error: Zero diagonal elements detected. Gauss-Seidel method cannot proceed.")
                    return
                
                # Solve the system
                iterations_data, convergence_data, converged, is_diag_dominant = solve_gauss_seidel(
                    A, b, x0, tolerance, max_iterations
                )
                
                # Display results
                st.header("📊 Solution Results")
                
                # Diagonal dominance warning
                if not is_diag_dominant:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Warning:</strong> The coefficient matrix is not diagonally dominant. 
                        Convergence is not guaranteed.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Final result
                if converged:
                    final_solution = iterations_data[-1]['solution']
                    st.markdown("""
                    <div class="success-box">
                        <h3>🎉 Solution Found!</h3>
                        <p><strong>Converged after {} iterations</strong></p>
                    </div>
                    """.format(len(iterations_data)), unsafe_allow_html=True)
                    
                    # Display final solution
                    solution_df = pd.DataFrame({
                        'Variable': [f'x{i+1}' for i in range(n)],
                        'Value': [f"{final_solution[i]:.6f}" for i in range(n)]
                    })
                    st.dataframe(solution_df, use_container_width=True)
                else:
                    st.error(f"❌ Solution did not converge within {max_iterations} iterations.")
                
                # Detailed iterations
                st.subheader("🔄 Detailed Iterations")
                
                for iteration_data in iterations_data:
                    with st.expander(f"Iteration {iteration_data['iteration']}", expanded=(iteration_data['iteration'] <= 3)):
                        
                        # Show calculations for each variable
                        for step in iteration_data['steps']:
                            st.write(f"**{step['variable']}:** {step['calculation']}")
                        
                        # Current solution
                        solution_str = ", ".join([f"x{i+1} = {iteration_data['solution'][i]:.6f}" 
                                                for i in range(n)])
                        st.write(f"**Current solution:** [{solution_str}]")
                        
                        # Error analysis
                        error_str = ", ".join([f"|Δx{i+1}| = {iteration_data['errors'][i]:.6f}" 
                                             for i in range(n)])
                        st.write(f"**Errors:** {error_str}")
                        st.write(f"**Max Error:** {iteration_data['max_error']:.6f} " +
                               ("✅ (Converged!)" if iteration_data['converged'] 
                                else f"≥ {tolerance} ❌ (Continue)"))
                        
                        if iteration_data['converged']:
                            break
                
                # Store data for convergence analysis
                st.session_state['convergence_data'] = convergence_data
                st.session_state['iterations_data'] = iterations_data
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.header("📚 Gauss-Seidel Method Explanation")
        
        st.markdown("""
        <div class="method-explanation">
            <h3>What is the Gauss-Seidel Method?</h3>
            <p>The Gauss-Seidel method is an iterative technique used to solve systems of linear equations. 
            It's particularly useful for large systems where direct methods become computationally expensive.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="formula-box">
            <strong>General Formula:</strong><br>
            x<sub>i</sub><sup>(k+1)</sup> = (b<sub>i</sub> - Σ<sub>j=1</sub><sup>i-1</sup> a<sub>ij</sub>x<sub>j</sub><sup>(k+1)</sup> - Σ<sub>j=i+1</sub><sup>n</sup> a<sub>ij</sub>x<sub>j</sub><sup>(k)</sup>) / a<sub>ii</sub>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Algorithm Steps")
            st.write("""
            1. **Rearrange equations**: For each equation i, solve for xᵢ in terms of other variables
            2. **Initialize**: Start with an initial guess x⁽⁰⁾
            3. **Iterate**: Use the most recent values to compute new approximations
            4. **Check convergence**: Stop when the change is smaller than tolerance
            """)
            
            st.subheader("⚡ Key Features")
            st.write("""
            - Uses updated values immediately (unlike Jacobi method)
            - Generally faster convergence than Jacobi
            - Requires diagonal dominance for guaranteed convergence
            - Memory efficient - only stores one set of values
            """)
        
        with col2:
            st.subheader("🎯 Convergence Criteria")
            st.write("""
            **Diagonal Dominance**: For guaranteed convergence, the matrix should satisfy:
            
            |aᵢᵢ| > Σⱼ≠ᵢ |aᵢⱼ| for all i
            
            This means each diagonal element should be larger than the sum of absolute values 
            of other elements in that row.
            """)
            
            st.subheader("🔍 Example Process")
            st.write("""
            For a 3×3 system in iteration k+1:
            - x₁⁽ᵏ⁺¹⁾ uses: x₂⁽ᵏ⁾, x₃⁽ᵏ⁾ (old values)
            - x₂⁽ᵏ⁺¹⁾ uses: x₁⁽ᵏ⁺¹⁾ (new), x₃⁽ᵏ⁾ (old)
            - x₃⁽ᵏ⁺¹⁾ uses: x₁⁽ᵏ⁺¹⁾, x₂⁽ᵏ⁺¹⁾ (both new)
            """)
    
    with tab3:
        st.header("📊 Convergence Analysis")
        
        if 'convergence_data' in st.session_state:
            df = pd.DataFrame(st.session_state['convergence_data'])
            
            # Convergence plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Variable Values vs Iteration", "Error vs Iteration"),
                vertical_spacing=0.1
            )
            
            # Variable convergence
            variables = [col for col in df.columns if col.startswith('x')]
            for var in variables:
                fig.add_trace(
                    go.Scatter(x=df['iteration'], y=df[var], name=var, mode='lines+markers'),
                    row=1, col=1
                )
            
            # Error convergence
            fig.add_trace(
                go.Scatter(x=df['iteration'], y=df['max_error'], name='Max Error', 
                          mode='lines+markers', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text="Variable Value", row=1, col=1)
            fig.update_yaxes(title_text="Error", type="log", row=2, col=1)
            fig.update_xaxes(title_text="Iteration", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Convergence table
            st.subheader("📋 Convergence Data")
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            if st.session_state['iterations_data']:
                final_iteration = len(st.session_state['iterations_data'])
                final_error = st.session_state['iterations_data'][-1]['max_error']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iterations to Convergence", final_iteration)
                with col2:
                    st.metric("Final Max Error", f"{final_error:.2e}")
                with col3:
                    converged = st.session_state['iterations_data'][-1]['converged']
                    st.metric("Status", "✅ Converged" if converged else "❌ Not Converged")
        else:
            st.info("👆 Please solve a system in the Calculator tab to see convergence analysis.")

if __name__ == "__main__":
    main()
