import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np

class LinearAlgebraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Algebra Solver")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Variables
        self.method_var = tk.StringVar()
        self.digit_var = tk.IntVar(value=4)
        self.epsilon_var = tk.DoubleVar(value=1e-6)
        self.w_var = tk.DoubleVar(value=1.0)
        self.max_iter_var = tk.IntVar(value=100)
        self.n_var = tk.IntVar(value=3)
        
        # Main container with notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input frame
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Input")
        
        # Output frame with tabs for different steps
        self.output_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_frame, text="Output")
        
        # Create main selection page
        self.create_selection_page()
    
    def create_selection_page(self):
        """Create the method selection page"""
        # Clear existing widgets
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        # Title
        title = ttk.Label(self.input_frame, text="Select a Linear Algebra Method", font=("Arial", 16))
        title.pack(pady=20)
        
        # Method selection frame
        method_frame = ttk.LabelFrame(self.input_frame, text="Available Methods")
        method_frame.pack(fill="both", expand=True, padx=50, pady=20)
        
        # Method buttons
        methods = [
            ("Gaussian Elimination", "Gauss"),
            ("Gaussian Elimination (Lower Triangular)", "Gauss_lower"),
            ("LU Decomposition (Doolittle)", "dolittel"),
            ("LU Decomposition (Crout)", "crout"),
            ("Modified Cholesky", "cholesky_modified"),
            ("Jacobi Iterative", "Jokobi"),
            ("Gauss-Seidel Iterative", "Guass_sidel"),
            ("SOR Iterative", "SOR"),
            ("Solution Refinement", "refine"),
            ("Matrix Inverse", "inverse")
        ]
        
        # Use grid layout with 2 columns
        for i, (name, code) in enumerate(methods):
            btn = ttk.Button(
                method_frame, 
                text=name,
                command=lambda c=code: self.create_input_page(c),
                width=35
            )
            btn.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="ew")
        
        # Digits selection at bottom
        digit_frame = ttk.Frame(self.input_frame)
        digit_frame.pack(side="bottom", pady=10)
        ttk.Label(digit_frame, text="Precision Digits:").pack(side="left", padx=5)
        ttk.Spinbox(digit_frame, from_=1, to=15, textvariable=self.digit_var, width=5).pack(side="left", padx=5)
    
    def create_input_page(self, method):
        """Create input page for a specific method"""
        # Clear existing widgets
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        # Create a scrollable container for inputs
        input_canvas = tk.Canvas(self.input_frame)
        scrollbar = ttk.Scrollbar(self.input_frame, orient="vertical", command=input_canvas.yview)
        scrollable_frame = ttk.Frame(input_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: input_canvas.configure(scrollregion=input_canvas.bbox("all"))
        )
        
        input_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        input_canvas.configure(yscrollcommand=scrollbar.set)
        
        input_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Back button
        back_btn = ttk.Button(scrollable_frame, text="← Back", command=self.create_selection_page)
        back_btn.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Method title
        title = ttk.Label(scrollable_frame, text=f"{method} Method", font=("Arial", 14))
        title.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Matrix size input
        size_frame = ttk.Frame(scrollable_frame)
        size_frame.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(size_frame, text="Matrix Size (n):").pack(side="left", padx=5)
        ttk.Spinbox(size_frame, from_=1, to=10, textvariable=self.n_var, width=5,
                    command=lambda: self.update_vector_inputs(method, scrollable_frame)).pack(side="left", padx=5)
        
        # Create input widgets based on method
        if method in ["Gauss", "Gauss_lower", "dolittel", "crout", "cholesky_modified"]:
            self.create_direct_method_inputs(method, scrollable_frame)
        elif method in ["Jokobi", "Guass_sidel", "SOR"]:
            self.create_iterative_method_inputs(method, scrollable_frame)
        elif method == "refine":
            self.create_refinement_inputs(scrollable_frame)
        elif method == "inverse":
            self.create_inverse_inputs(scrollable_frame)
        
        # Create matrix inputs
        self.create_matrix_inputs(scrollable_frame)
        
        # Solve button
        solve_btn = ttk.Button(scrollable_frame, text="Solve", command=lambda: self.solve(method))
        solve_btn.grid(row=100, column=0, padx=10, pady=20, sticky="w")
    
    def create_direct_method_inputs(self, method, parent):
        """Inputs for direct methods"""
        params_frame = ttk.LabelFrame(parent, text="Method Parameters")
        params_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Create vector b inputs
        self.b_frame = ttk.Frame(params_frame)
        self.b_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(self.b_frame, text="Vector b:").pack(side="left", padx=5)
        
        self.b_entries = []
        for i in range(self.n_var.get()):
            entry = ttk.Entry(self.b_frame, width=8)
            entry.pack(side="left", padx=2)
            self.b_entries.append(entry)
        
        # Additional parameters for Gaussian Elimination
        if method == "Gauss":
            pivoting_frame = ttk.Frame(params_frame)
            pivoting_frame.pack(fill="x", padx=5, pady=5)
            self.pivoting_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(pivoting_frame, text="Pivoting", variable=self.pivoting_var).pack(side="left")
            
            scaling_frame = ttk.Frame(params_frame)
            scaling_frame.pack(fill="x", padx=5, pady=5)
            self.scaling_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(scaling_frame, text="Scaling", variable=self.scaling_var).pack(side="left")
    
    def create_iterative_method_inputs(self, method, parent):
        """Inputs for iterative methods"""
        params_frame = ttk.LabelFrame(parent, text="Iterative Parameters")
        params_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Create vector b inputs
        self.b_frame = ttk.Frame(params_frame)
        self.b_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(self.b_frame, text="Vector b:").pack(side="left", padx=5)
        
        self.b_entries = []
        for i in range(self.n_var.get()):
            entry = ttk.Entry(self.b_frame, width=8)
            entry.pack(side="left", padx=2)
            self.b_entries.append(entry)
        
        # Initial guess
        self.x0_frame = ttk.Frame(params_frame)
        self.x0_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(self.x0_frame, text="Initial Guess (x0):").pack(side="left", padx=5)
        
        self.x0_entries = []
        for i in range(self.n_var.get()):
            entry = ttk.Entry(self.x0_frame, width=8)
            entry.pack(side="left", padx=2)
            entry.insert(0, "0")
            self.x0_entries.append(entry)
        
        # Epsilon
        epsilon_frame = ttk.Frame(params_frame)
        epsilon_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(epsilon_frame, text="Epsilon (tolerance):").pack(side="left", padx=5)
        ttk.Entry(epsilon_frame, textvariable=self.epsilon_var, width=10).pack(side="left", padx=5)
        
        # Max iterations
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(iter_frame, text="Max Iterations:").pack(side="left", padx=5)
        ttk.Label(iter_frame, text="100", width=10).pack(side="left", padx=5)
        
        # w parameter for SOR
        if method == "SOR":
            w_frame = ttk.Frame(params_frame)
            w_frame.pack(fill="x", padx=5, pady=5)
            ttk.Label(w_frame, text="w (relaxation factor):").pack(side="left", padx=5)
            ttk.Entry(w_frame, textvariable=self.w_var, width=10).pack(side="left", padx=5)
    
    def create_refinement_inputs(self, parent):
        """Inputs for solution refinement"""
        params_frame = ttk.LabelFrame(parent, text="Refinement Parameters")
        params_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Create vector b inputs
        self.b_frame = ttk.Frame(params_frame)
        self.b_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(self.b_frame, text="Vector b:").pack(side="left", padx=5)
        
        self.b_entries = []
        for i in range(self.n_var.get()):
            entry = ttk.Entry(self.b_frame, width=8)
            entry.pack(side="left", padx=2)
            self.b_entries.append(entry)
        
        # Initial solution
        self.x0_frame = ttk.Frame(params_frame)
        self.x0_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(self.x0_frame, text="Initial Solution:").pack(side="left", padx=5)
        
        self.x0_entries = []
        for i in range(self.n_var.get()):
            entry = ttk.Entry(self.x0_frame, width=8)
            entry.pack(side="left", padx=2)
            self.x0_entries.append(entry)
        
        # Epsilon
        epsilon_frame = ttk.Frame(params_frame)
        epsilon_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(epsilon_frame, text="Epsilon (tolerance):").pack(side="left", padx=5)
        ttk.Entry(epsilon_frame, textvariable=self.epsilon_var, width=10).pack(side="left", padx=5)
    
    def create_inverse_inputs(self, parent):
        """Inputs for matrix inverse"""
        pass  # Already handled by matrix inputs
    
    def create_matrix_inputs(self, parent):
        """Create matrix input widgets"""
        # Clear existing matrix widgets if any
        if hasattr(self, 'matrix_frame'):
            self.matrix_frame.destroy()
        
        n = self.n_var.get()
        
        # Matrix frame
        self.matrix_frame = ttk.LabelFrame(parent, text="Matrix A")
        self.matrix_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        # Create matrix inputs
        self.matrix_entries = []
        for i in range(n):
            row_frame = ttk.Frame(self.matrix_frame)
            row_frame.pack(fill="x")
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(row_frame, width=8)
                entry.pack(side="left", padx=2, pady=2)
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
    
    def update_vector_inputs(self, method, parent):
        """Update vector inputs when matrix size changes"""
        # Update matrix inputs first
        self.create_matrix_inputs(parent)
        
        # Update vector b inputs
        if hasattr(self, 'b_frame'):
            # Destroy existing b inputs
            for widget in self.b_frame.winfo_children():
                widget.destroy()
            
            # Recreate b inputs with current size
            ttk.Label(self.b_frame, text="Vector b:").pack(side="left", padx=5)
            self.b_entries = []
            for i in range(self.n_var.get()):
                entry = ttk.Entry(self.b_frame, width=8)
                entry.pack(side="left", padx=2)
                self.b_entries.append(entry)
        
        # Update x0 inputs if applicable
        if hasattr(self, 'x0_frame'):
            # Destroy existing x0 inputs
            for widget in self.x0_frame.winfo_children():
                widget.destroy()
            
            # Recreate x0 inputs with current size
            if method in ["Jokobi", "Guass_sidel", "SOR"]:
                label_text = "Initial Guess (x0):"
                default_value = "0"
            else:
                label_text = "Initial Solution:"
                default_value = ""
            
            ttk.Label(self.x0_frame, text=label_text).pack(side="left", padx=5)
            self.x0_entries = []
            for i in range(self.n_var.get()):
                entry = ttk.Entry(self.x0_frame, width=8)
                entry.pack(side="left", padx=2)
                entry.insert(0, default_value)
                self.x0_entries.append(entry)
    
    def get_matrix_inputs(self):
        """Get matrix values from input fields"""
        n = self.n_var.get()
        A = []
        for i in range(n):
            row = []
            for j in range(n):
                val = self.matrix_entries[i][j].get()
                if not val:
                    raise ValueError(f"Missing value at A[{i}][{j}]")
                row.append(float(val))
            A.append(row)
        return A
    
    def get_vector_inputs(self, entries):
        """Get vector values from input fields"""
        n = self.n_var.get()
        vector = []
        for i in range(n):
            val = entries[i].get()
            if not val:
                raise ValueError(f"Missing value at position {i}")
            vector.append(float(val))
        return vector
    
    def format_matrix(self, matrix, digits):
        """Format matrix or vector for nice display"""
        if not matrix:
            return ""
        
        if isinstance(matrix[0], list):
            # Matrix
            col_widths = [0] * len(matrix[0])
            str_matrix = []
            
            # Convert numbers to strings and find max column width
            for row in matrix:
                str_row = []
                for j, val in enumerate(row):
                    s = f"{val:.{digits}f}"
                    str_row.append(s)
                    if len(s) > col_widths[j]:
                        col_widths[j] = len(s)
                str_matrix.append(str_row)
            
            # Build formatted string
            s = ""
            for row in str_matrix:
                for j, val_str in enumerate(row):
                    s += val_str.rjust(col_widths[j] + 2)
                s += "\n"
            return s
        else:
            # Vector
            max_width = 0
            str_vector = []
            for val in matrix:
                s = f"{val:.{digits}f}"
                str_vector.append(s)
                if len(s) > max_width:
                    max_width = len(s)
            
            return "  ".join([s.rjust(max_width + 2) for s in str_vector])
        
    
    def display_matrix_step(self, parent, title, matrix):
        """Display a matrix in a separate frame"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame, text=title, font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Create frame for matrix display
        matrix_frame = ttk.Frame(frame)
        matrix_frame.pack(fill="x", padx=20, pady=5)
        
        # Display matrix row by row
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                val_str = f"{val:.{self.digit_var.get()}f}"
                ttk.Label(matrix_frame, text=val_str, width=8, anchor="e").grid(row=i, column=j, padx=2, pady=2)
    
    def display_vector_step(self, parent, title, vector):
        """Display a vector in a separate frame"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame, text=title, font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Create frame for vector display
        vector_frame = ttk.Frame(frame)
        vector_frame.pack(fill="x", padx=20, pady=5)
        
        # Display vector row
        for i, val in enumerate(vector):
            val_str = f"{val:.{self.digit_var.get()}f}"
            ttk.Label(vector_frame, text=val_str, width=8, anchor="e").grid(row=0, column=i, padx=2, pady=2)
    
    def create_step_tab(self, title, content):
        """Create a new tab for solution steps"""
        tab_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(tab_frame, text=title)
        
        # Create scroll area for steps
        canvas = tk.Canvas(tab_frame)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
    
    
    def solve(self, method):
        try:
            # Clear previous outputs
            for child in self.output_frame.winfo_children():
                child.destroy()
            
            # Create new notebook for output
            self.output_notebook = ttk.Notebook(self.output_frame)
            self.output_notebook.pack(fill="both", expand=True)
            
            # Create main output tab
            self.main_output = ttk.Frame(self.output_notebook)
            self.output_notebook.add(self.main_output, text="Genral info")
            
            # Create text area for solution steps
            self.output_text = scrolledtext.ScrolledText(
                self.main_output, 
                wrap=tk.WORD,
                font=("Consolas", 10),
                padx=10,
                pady=10
            )
            self.output_text.pack(fill="both", expand=True)
            self.output_text.tag_configure("header", font=("Arial", 12, "bold"))
            self.output_text.tag_configure("result", font=("Arial", 12, "bold"), foreground="blue")
            self.output_text.tag_configure("error", foreground="red")
            self.output_text.tag_configure("matrix", font=("Consolas", 10))
            
            # Get input values
            digit = self.digit_var.get()
            
            # Get matrix A
            A = self.get_matrix_inputs()
            
            # Initialize solver
            from main import linear_algebra
            solver = linear_algebra(A, [0]*len(A), digit)  # Temporary b for inverse
            
            # Display initial information
            self.output_text.insert(tk.END, f"Method: {method.upper()}\n", "header")
            self.output_text.insert(tk.END, f"Matrix size: {self.n_var.get()}\n")
            self.output_text.insert(tk.END, f"Precision: {digit} decimal digits\n\n")
            self.output_text.insert(tk.END, "Initial Matrix A:\n", "header")
            self.output_text.insert(tk.END, self.format_matrix(A, digit) + "\n\n", "matrix")
            
            # Process based on method
            if method in ["Gauss", "Gauss_lower", "dolittel", "crout", "cholesky_modified"]:
                # Get vector b
                b = self.get_vector_inputs(self.b_entries)
                solver.b = b  # Update b in solver
                self.output_text.insert(tk.END, "Vector b:\n", "header")
                self.output_text.insert(tk.END, self.format_matrix(b, digit) + "\n\n", "matrix")
                
                if method == "Gauss":
                    pivoting = getattr(self, 'pivoting_var', tk.BooleanVar(value=False)).get()
                    scaling = getattr(self, 'scaling_var', tk.BooleanVar(value=False)).get()
                    gen = solver.Gauss(scaling=scaling, pivoting=pivoting)
                elif method == "Gauss_lower":
                    gen = solver.Gauss_lower()
                elif method == "dolittel":
                    gen = solver.dolittel()
                elif method == "crout":
                    gen = solver.crout()
                elif method == "cholesky_modified":
                    gen = solver.cholesky_modified()
            
            elif method in ["Jokobi", "Guass_sidel", "SOR"]:
                # Get inputs for iterative methods
                b = self.get_vector_inputs(self.b_entries)
                x0 = self.get_vector_inputs(self.x0_entries)
                epsilon = self.epsilon_var.get()
                max_iter = self.max_iter_var.get()
                
                solver.b = b  # Update b in solver
                self.output_text.insert(tk.END, "Vector b:\n", "header")
                self.output_text.insert(tk.END, self.format_matrix(b, digit) + "\n\n", "matrix")
                self.output_text.insert(tk.END, "Initial Guess (x0):\n", "header")
                self.output_text.insert(tk.END, self.format_matrix(x0, digit) + "\n\n", "matrix")
                self.output_text.insert(tk.END, f"Tolerance: {epsilon}\n")
                self.output_text.insert(tk.END, f"Max Iterations: {max_iter}\n\n")
                
                if method == "Jokobi":
                    gen = solver.Jokobi(x0, epsilon)
                elif method == "Guass_sidel":
                    gen = solver.Guass_sidel(x0, epsilon)
                elif method == "SOR":
                    w = self.w_var.get()
                    self.output_text.insert(tk.END, f"Relaxation factor (w): {w}\n\n")
                    gen = solver.SOR(x0, w, epsilon)
            
            elif method == "refine":
                # Get inputs for solution refinement
                b = self.get_vector_inputs(self.b_entries)
                x0 = self.get_vector_inputs(self.x0_entries)
                epsilon = self.epsilon_var.get()
                
                solver.b = b  # Update b in solver
                self.output_text.insert(tk.END, "Vector b:\n", "header")
                self.output_text.insert(tk.END, self.format_matrix(b, digit) + "\n\n", "matrix")
                self.output_text.insert(tk.END, "Initial Solution:\n", "header")
                self.output_text.insert(tk.END, self.format_matrix(x0, digit) + "\n\n", "matrix")
                self.output_text.insert(tk.END, f"Tolerance: {epsilon}\n\n")
                gen = solver.refine(x0, epsilon)
            
            elif method == "inverse":
                # Only matrix A is needed for inverse
                gen = solver.inverse()
            
            iteration_count = 0
            step_number = 1
            final_solution = None
            current_tab = self.main_output
            step_tab_created = False
            
            for step in gen:
                iteration_count += 1
                if iteration_count > self.max_iter_var.get() and method in ["Jokobi", "Guass_sidel", "SOR", "refine"]:
                    self.output_text.insert(tk.END, f"\n\n⚠️ Reached maximum iterations ({self.max_iter_var.get()}) without convergence ⚠️\n", "error")
                    break
                
                # Display matrix steps
                if isinstance(step, tuple) and len(step) == 2:
                    label, data = step
                    
                    # Format and display in main output
                    formatted_data = self.format_matrix(data, digit)
                    self.output_text.insert(tk.END, f"{label}:\n", "header")
                    self.output_text.insert(tk.END, formatted_data + "\n\n", "matrix")
                    
                    # Create new tab for steps if doesn't exist
                    if not step_tab_created:
                        step_tab = self.create_step_tab("Detailed Steps", step)
                        step_tab_created = True
                    
                    if isinstance(data[0], list):
                        # Matrix
                        self.display_matrix_step(step_tab, f"Step {step_number}: {label}", data)
                    else:
                        # Vector
                        self.display_vector_step(step_tab, f"Step {step_number}: {label}", data)
                    
                    step_number += 1
                
                elif isinstance(step, list):
                    # Display result vector
                    if not step_tab_created:
                        step_tab = self.create_step_tab("Detailed Steps", step)
                        step_tab_created = True
                    
                    self.display_vector_step(step_tab, f"Step {step_number}", step)
                    step_number += 1
                    
                    if method in ["Gauss", "Gauss_lower", "dolittel", "crout", "cholesky_modified", "inverse"]:
                        final_solution = step
                        # Format and display in main output
                        self.output_text.insert(tk.END, "Current Solution:\n", "header")
                        self.output_text.insert(tk.END, self.format_matrix(step, digit) + "\n\n", "matrix")
                
                elif isinstance(step, str):
                    # Display text message with formatting
                    if "error" in step.lower() or "warning" in step.lower():
                        self.output_text.insert(tk.END, f"\n⚠️ {step} ⚠️\n", "error")
                    elif "solution" in step.lower() or "result" in step.lower():
                        self.output_text.insert(tk.END, f"\n✅ {step} ✅\n", "result")
                    else:
                        self.output_text.insert(tk.END, f"\n➡️ {step}\n")
                
                else:
                    # Display other data types
                    self.output_text.insert(tk.END, f"\n➡️ {str(step)}\n")
                
                self.output_text.see(tk.END)
                self.root.update()
            
            # Display final solution prominently
            if final_solution:
                # Create frame for final solution
                result_frame = ttk.LabelFrame(self.main_output, text="🌟 FINAL SOLUTION 🌟")
                result_frame.pack(fill="x", padx=10, pady=10, ipady=10)
                
                # Display title
                ttk.Label(result_frame, text="Solution Vector:", 
                         font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=5)
                
                # Display solution values
                values_frame = ttk.Frame(result_frame)
                values_frame.pack(fill="x", padx=20, pady=5)
                
                for i, val in enumerate(final_solution):
                    row_frame = ttk.Frame(values_frame)
                    row_frame.pack(fill="x", pady=2)
                    ttk.Label(row_frame, text=f"x_{i} =", width=5, 
                             anchor="e", font=("Arial", 10)).pack(side="left")
                    ttk.Label(row_frame, text=f"{val:.{digit}f}", 
                             font=("Arial", 10, "bold")).pack(side="left", padx=5)
            
            # Add save results button
            save_btn = ttk.Button(self.main_output, text="💾 Save Results", 
                                 command=self.save_results, style="Accent.TButton")
            save_btn.pack(pady=15)
            self.root.style = ttk.Style()
            self.root.style.configure("Accent.TButton", font=("Arial", 10, "bold"))
        
        except Exception as e:
            messagebox.showerror("❌ Error", f"Operation failed:\n\n{str(e)}")
    
    def save_results(self):
        """Save the results to a text file"""
        content = self.output_text.get("1.0", tk.END)
        if not content.strip():
            messagebox.showwarning("Empty Content", "No results to save")
            return
        
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")