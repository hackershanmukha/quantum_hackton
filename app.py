import io
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# Compatible imports with fallback options
try:
    from qiskit_aer import AerSimulator
    USE_AER = True
except ImportError:
    print("qiskit-aer not available, using basic simulator")
    USE_AER = False

from qiskit.utils import algorithm_globals

# Import with fallbacks for different Qiskit versions
try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import SPSA
except ImportError:
    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import SPSA
    except ImportError:
        raise ImportError("Could not import QAOA and SPSA. Please check qiskit installation.")

try:
    from qiskit.primitives import Sampler
except ImportError:
    try:
        from qiskit_algorithms.primitives import Sampler
    except ImportError:
        print("Sampler not available, will use basic backend")

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

algorithm_globals.random_seed = 123

def compute_returns_and_cov(prices_df):
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    mu = returns.mean().values
    cov = returns.cov().values
    return mu, cov


def build_quadratic_program(mu, cov, budget, risk_aversion, penalty=10.0):
    n = len(mu)
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    linear = {f'x{i}': float(-mu[i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(n):
            quadratic[(f'x{i}', f'x{j}')] = float(risk_aversion * cov[i, j])
    qp.minimize(linear=linear, quadratic=quadratic)
    for i in range(n):
        qp.objective.linear[f'x{i}'] += penalty * (1 - 2 * budget)
    for i in range(n):
        for j in range(n):
            if i != j:
                qp.objective.quadratic[(f'x{i}', f'x{j}')] = qp.objective.quadratic.get((f'x{i}', f'x{j}'), 0.0) + 2 * penalty
    return qp


def run_qaoa(qp, p=1, shots=1024, maxiter=100):
    # Flexible backend initialization
    backend = None
    
    if USE_AER:
        try:
            backend = AerSimulator()
        except Exception as e:
            print(f"Error creating AerSimulator: {e}")
    
    # Fallback to basic simulators if AerSimulator fails
    if backend is None:
        try:
            from qiskit import BasicAer
            backend = BasicAer.get_backend('qasm_simulator')
        except ImportError:
            try:
                from qiskit.providers.basic_provider import BasicProvider
                provider = BasicProvider()
                backend = provider.get_backend('basic_simulator')
            except Exception as e:
                print(f"Could not initialize any backend: {e}")
                return None
    
    optimizer = SPSA(maxiter=maxiter)
    
    # Try to create sampler with fallbacks
    try:
        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
    except Exception:
        # Fallback for older versions
        try:
            qaoa = QAOA(optimizer=optimizer, reps=p, quantum_instance=backend)
        except Exception as e:
            print(f"Could not create QAOA instance: {e}")
            return None
    
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)
    return result


def optimize_portfolio(csv_file, budget, risk_aversion, qaoa_p, shots, maxiter):
    try:
        # Better file handling
        if csv_file is None:
            return 'Please upload a CSV file.', None
        
        if hasattr(csv_file, 'name'):
            raw = pd.read_csv(csv_file.name)
        else:
            raw = pd.read_csv(csv_file)
    except Exception as e:
        return f'Error reading CSV: {e}', None
    
    if raw.shape[1] < 2:
        return 'CSV must have at least 2 columns (date + asset(s) or asset columns).', None
    
    df = raw.copy()
    first_col = df.columns[0]
    
    # Handle date column
    if pd.api.types.is_datetime64_any_dtype(df[first_col]) or any(c in str(first_col).lower() for c in ['date', 'day']):
        try:
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.set_index(first_col)
        except Exception:
            pass
    
    # Select only numeric columns
    df = df.select_dtypes(include=[np.number]).dropna(how='all')
    
    if df.shape[1] > 12:
        df = df.iloc[:, :12]
    
    if df.shape[1] == 0:
        return 'No numeric columns found in CSV.', None
    
    asset_names = df.columns.tolist()
    
    try:
        mu, cov = compute_returns_and_cov(df)
    except Exception as e:
        return f'Error computing returns/covariance: {e}', None
    
    qp = build_quadratic_program(mu, cov, budget, risk_aversion, penalty=10.0)
    
    try:
        res = run_qaoa(qp, p=qaoa_p, shots=shots, maxiter=maxiter)
        if res is None:
            return 'QAOA run failed - check backend initialization.', None
    except Exception as e:
        return f'QAOA run failed: {e}', None
    
    selection = []
    picked = []
    for var, val in res.variables_dict.items():
        selection.append((var, int(round(val))))
        if int(round(val)) == 1:
            picked.append(var)
    
    picked_indices = [int(v[1:]) for v in picked]
    chosen_assets = [asset_names[i] for i in picked_indices]
    
    x = np.zeros(len(mu))
    for i in picked_indices:
        x[i] = 1
    
    exp_return = float(mu @ x)
    risk = float(x @ cov @ x)
    
    out_text = f"Chosen assets (budget target={budget}): {', '.join(chosen_assets) if chosen_assets else 'None'}\n"
    out_text += f"Estimated (daily) expected return: {exp_return:.6f}  |  Risk (variance proxy): {risk:.6f}\n"
    out_text += f"QAOA solver status: {getattr(res, 'status', 'OK')}\n"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(asset_names, x)
    ax.set_ylabel('Selected (1) / Not selected (0)')
    ax.set_title('QAOA Portfolio Selection')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close figure to free memory
    
    return out_text, buf


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# QAOA Portfolio Optimizer\nUpload asset price CSV (columns = assets, rows = dates). Keep <=12 assets for simulation.")
        
        with gr.Row():
            csv_in = gr.File(label='Upload CSV')
            
            with gr.Column():
                budget = gr.Slider(1, 12, value=3, step=1, label='Target number of assets (budget)')
                risk_aversion = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label='Risk aversion (higher -> safer)')
                qaoa_p = gr.Slider(1, 3, value=1, step=1, label='QAOA depth p')
                shots = gr.Slider(256, 4096, value=1024, step=256, label='Shots (simulator)')
                maxiter = gr.Slider(10, 500, value=100, step=10, label='Max optimizer iterations (SPSA)')
                run_btn = gr.Button('Run QAOA')
        
        output_text = gr.Textbox(label='Result summary', lines=6)
        output_plot = gr.Image(label='Selection plot')
        
        run_btn.click(
            fn=optimize_portfolio, 
            inputs=[csv_in, budget, risk_aversion, qaoa_p, shots, maxiter], 
            outputs=[output_text, output_plot]
        )
    
    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))