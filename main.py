# main.py
import numpy as np
from gui import LinearAlgebraApp
import tkinter as tk


class linear_algebra:
    def __init__(self, A, b, digit):
        if not isinstance(A, list) or not all(isinstance(row, list) for row in A):
            raise TypeError("A must be a list of lists.")
        # Check that all elements of A are numbers
        if not all(all(isinstance(x, (int, float)) for x in row) for row in A):
            raise ValueError("All elements of A must be numbers.")
        n = len(A)
        # Check if matrix is square
        if any(len(row) != n for row in A):
            raise ValueError("A must be a square matrix.")
        # Check that b is a list of numbers
        if not isinstance(b, list) or not all(isinstance(x, (int, float)) for x in b):
            raise ValueError("b must be a list of numbers.")
        # Check that b has the same size as A
        if len(b) != n:
            raise ValueError("Size of b must match number of rows in A.")
        # Check determinant
        array = np.array(A)
        if np.linalg.det(array) == 0:
            raise ValueError("Matrix is singular (det = 0); not supported.")
        # Save values
        self.A = A
        self.b = b
        self.n = n
        self.np_A=array
        self.digit = digit

    
    def Gauss(self, scaling=False, pivoting=False):
        """Gaussian Elimination with optional scaling and pivoting"""
        digit = self.digit
        A = [row[:] for row in self.A]  # Create a copy of A
        b = self.b[:]  # Create a copy of b
        
        n = self.n
        yield "🚀 Starting Gaussian Elimination"
        yield "--------------------------------"
        
        # Apply scaling if requested
        if scaling:
            yield "🔧 Applying row scaling..."
            for i in range(n):
                max_val = max(abs(x) for x in A[i])
                if max_val == 0:
                    yield "⚠️ Warning: Row contains all zeros, skipping scaling"
                    continue
                for j in range(n):
                    A[i][j] = round(A[i][j] / max_val, digit)
                b[i] = round(b[i] / max_val, digit)
            yield ("Scaled Matrix A", A)
            yield ("Scaled Vector b", b)
        
        # Perform elimination
        for i in range(n-1):
            if pivoting:
                # Find pivot row
                pivot_row = i
                for j in range(i+1, n):
                    if abs(A[j][i]) > abs(A[pivot_row][i]):
                        pivot_row = j
                
                if pivot_row != i:
                    # Swap rows
                    A[i], A[pivot_row] = A[pivot_row], A[i]
                    b[i], b[pivot_row] = b[pivot_row], b[i]
                    yield f"🔄 Pivoting: Swapped row {i} and row {pivot_row}"
                    yield ("Matrix after pivoting", A)
                    yield ("Vector after pivoting", b)
            
            # Check for zero pivot
            if abs(A[i][i]) < 1e-10:
                for j in range(i+1,len(A)):
                    if(A[j][i]!=0):
                        A[i], A[j]= A[j],A[i]
                        b[i],b[j]=b[j],b[i]
                        break
            
            # Elimination
            for j in range(i+1, n):
                factor = round(A[j][i] / A[i][i], digit)
                for k in range(i, n):
                    A[j][k] = round(A[j][k] - factor * A[i][k], digit)
                b[j] = round(b[j] - factor * b[i], digit)
                yield f"➖ Elimination: Row {j} = Row {j} - ({factor}) * Row {i}"
                yield (f"Matrix after row {j} elimination", A)
                yield (f"Vector after row {j} elimination", b)
        
        yield "✅ Gaussian Elimination completed successfully"
        yield ("Upper Triangular Matrix", A)
        yield ("Modified Vector b", b)
        
        # Back substitution
        x = [0] * n
        for i in range(n-1, -1, -1):
            s = b[i]
            for j in range(i+1, n):
                s -= A[i][j] * x[j]
            x[i] = round(s / A[i][i], digit)
            yield f"🔙 Back Substitution: x[{i}] = {x[i]}"
        
        yield "🎉 Final Solution"
        yield x

    def Gauss_lower(self):
        """Gaussian Elimination for Lower Triangular System"""
        digit = self.digit
        A = [row[:] for row in self.A]  # Create a copy of A
        b = self.b[:]  # Create a copy of b
        
        n = self.n
        yield "🚀 Starting Gaussian Elimination (Lower Triangular)"
        yield "-----------------------------------------------"
        
        # Forward elimination
        for i in range(n):
            # Normalize the current row
            diag = A[i][i]
            if abs(diag) < 1e-10:
                raise ValueError(f"Zero diagonal at position ({i},{i})")
            
            for j in range(i):
                factor = round(A[i][j] / diag, digit)
                for k in range(j, n):
                    A[i][k] = round(A[i][k] - factor * A[j][k], digit)
                b[i] = round(b[i] - factor * b[j], digit)
                yield f"➖ Elimination: Row {i} = Row {i} - ({factor}) * Row {j}"
                yield (f"Matrix after row {i} elimination", A)
                yield (f"Vector after row {i} elimination", b)
        
        yield "✅ Forward Elimination completed"
        yield ("Lower Triangular Matrix", A)
        yield ("Modified Vector b", b)
        
        # Forward substitution
        x = [0] * n
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i][j] * x[j]
            x[i] = round(s / A[i][i], digit)
            yield f"🔜 Forward Substitution: x[{i}] = {x[i]}"
        
        yield "🎉 Final Solution"
        yield x

    def dolittel(self):
        """Doolittle LU Decomposition"""
        digit = self.digit
        n = self.n
        L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        U = [[0.0 for _ in range(n)] for _ in range(n)]
        
        yield "🚀 Starting Doolittle LU Decomposition"
        yield "------------------------------------"
        
        for i in range(n):
            # Calculate U row
            for j in range(i, n):
                s = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = round(self.A[i][j] - s, digit)
                yield f"🔧 Computing U[{i}][{j}] = A[{i}][{j}] - Σ(L[{i}][k]*U[k][{j}])"
            
            # Calculate L column
            for j in range(i+1, n):
                s = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = round((self.A[j][i] - s) / U[i][i], digit)
                yield f"🔧 Computing L[{j}][{i}] = (A[{j}][{i}] - Σ(L[{j}][k]*U[k][{i}])) / U[{i}][{i}]"
            
            yield (f"Step {i+1}: L matrix", L)
            yield (f"Step {i+1}: U matrix", U)
        
        yield "✅ LU Decomposition completed"
        yield ("L Matrix", L)
        yield ("U Matrix", U)
        
        # Solve Ly = b
        y = [0.0] * n
        for i in range(n):
            s = self.b[i]
            for j in range(i):
                s -= L[i][j] * y[j]
            y[i] = round(s / L[i][i], digit)
            yield f"🔜 Forward Substitution: y[{i}] = {y[i]}"
        
        # Solve Ux = y
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            s = y[i]
            for j in range(i+1, n):
                s -= U[i][j] * x[j]
            x[i] = round(s / U[i][i], digit)
            yield f"🔙 Back Substitution: x[{i}] = {x[i]}"
        
        yield "🎉 Final Solution"
        yield x

    def crout(self):
        """Crout LU Decomposition"""
        digit = self.digit
        n = self.n
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        yield "🚀 Starting Crout LU Decomposition"
        yield "---------------------------------"
        
        for i in range(n):
            # Calculate L column
            for j in range(i, n):
                s = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = round(self.A[j][i] - s, digit)
                yield f"🔧 Computing L[{j}][{i}] = A[{j}][{i}] - Σ(L[{j}][k]*U[k][{i}])"
            
            # Calculate U row
            for j in range(i+1, n):
                s = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = round((self.A[i][j] - s) / L[i][i], digit)
                yield f"🔧 Computing U[{i}][{j}] = (A[{i}][{j}] - Σ(L[{i}][k]*U[k][{j}])) / L[{i}][{i}]"
            
            yield (f"Step {i+1}: L matrix", L)
            yield (f"Step {i+1}: U matrix", U)
        
        yield "✅ LU Decomposition completed"
        yield ("L Matrix", L)
        yield ("U Matrix", U)
        
        # Solve Ly = b
        y = [0.0] * n
        for i in range(n):
            s = self.b[i]
            for j in range(i):
                s -= L[i][j] * y[j]
            y[i] = round(s / L[i][i], digit)
            yield f"🔜 Forward Substitution: y[{i}] = {y[i]}"
        
        # Solve Ux = y
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            s = y[i]
            for j in range(i+1, n):
                s -= U[i][j] * x[j]
            x[i] = round(s / U[i][i], digit)
            yield f"🔙 Back Substitution: x[{i}] = {x[i]}"
        
        yield "🎉 Final Solution"
        yield x

    def cholesky_modified(self):
        """Modified Cholesky Decomposition"""
        digit = self.digit
        n = self.n
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        U = [[0.0 for _ in range(n)] for _ in range(n)]
        
        yield "🚀 Starting Modified Cholesky Decomposition"
        yield "-----------------------------------------"
        
        for i in range(n):
            # Diagonal elements
            s = sum(L[i][k] * U[k][i] for k in range(i))
            diag_val = self.A[i][i] - s
            if diag_val <= 0:
                raise ValueError("Matrix is not positive definite")
            
            diag = round(diag_val ** 0.5, digit)
            L[i][i] = U[i][i] = diag
            yield f"🔧 Diagonal: L[{i}][{i}] = U[{i}][{i}] = √{self.A[i][i]} - Σ = {diag}"
            
            # Calculate row i of U
            for j in range(i+1, n):
                s = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = round((self.A[i][j] - s) / diag, digit)
                yield f"🔧 Computing U[{i}][{j}] = (A[{i}][{j}] - Σ) / L[{i}][{i}]"
            
            # Calculate column i of L
            for j in range(i+1, n):
                s = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = round((self.A[j][i] - s) / diag, digit)
                yield f"🔧 Computing L[{j}][{i}] = (A[{j}][{i}] - Σ) / U[{i}][{i}]"
            
            yield (f"Step {i+1}: L matrix", L)
            yield (f"Step {i+1}: U matrix", U)
        
        yield "✅ Decomposition completed"
        yield ("L Matrix", L)
        yield ("U Matrix", U)
        
        # Solve Ly = b
        y = [0.0] * n
        for i in range(n):
            s = self.b[i]
            for j in range(i):
                s -= L[i][j] * y[j]
            y[i] = round(s / L[i][i], digit)
            yield f"🔜 Forward Substitution: y[{i}] = {y[i]}"
        
        # Solve Ux = y
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            s = y[i]
            for j in range(i+1, n):
                s -= U[i][j] * x[j]
            x[i] = round(s / U[i][i], digit)
            yield f"🔙 Back Substitution: x[{i}] = {x[i]}"
        
        yield "🎉 Final Solution"
        yield x

    def refine(self, x, epsilon):
        """Solution Refinement (Iterative Improvement)"""
        digit = self.digit
        high_precision = max(16, self.digit * 2)
        
        yield "🚀 Starting Solution Refinement"
        yield f"Initial tolerance: {epsilon}"
        yield f"Using high precision: {high_precision} digits"
        yield "Initial solution:"
        yield x
        
        iteration = 0
        max_iter = 100
        residual_norm = float('inf')
        
        while residual_norm > epsilon and iteration < max_iter:
            iteration += 1
            # Calculate residual r = b - Ax
            Ax = np.dot(self.np_A, np.array(x))
            r = np.subtract(np.array(self.b), Ax).tolist()
            residual_norm = round(np.linalg.norm(r, np.inf), digit)
            
            yield f"🔁 Iteration {iteration}:"
            yield "Residual vector r = b - Ax:"
            yield r
            yield f"Residual norm: {residual_norm}"
            
            # Solve Aδ = r
            solver = linear_algebra(self.A, r, high_precision)
            gen = solver.Gauss()
            for step in gen:
                if isinstance(step, list):
                    delta = step
            
            # Update solution x = x + δ
            x = (np.array(x) + np.array(delta)).tolist()
            yield "Correction vector δ:"
            yield delta
            yield "Updated solution:"
            yield x
            yield "--------------------------------"
        
        if residual_norm <= epsilon:
            yield f"✅ Convergence achieved after {iteration} iterations"
            yield f"Final residual norm: {residual_norm}"
        else:
            yield f"⚠️ Maximum iterations reached ({max_iter})"
            yield f"Current residual norm: {residual_norm}"
        
        yield "🎉 Final Refined Solution"
        yield x

    def inverse(self):
        """Matrix Inversion using LU Decomposition"""
        digit = self.digit
        n = self.n
        
        yield "🚀 Starting Matrix Inversion"
        yield "---------------------------"
        
        # Perform LU decomposition
        solver = linear_algebra(self.A, [0]*n, digit)
        gen = solver.dolittel()
        for step in gen:
            if isinstance(step, tuple) and len(step) == 2:
                L, U = step
                yield "✅ LU Decomposition obtained"
                yield ("L Matrix", L)
                yield ("U Matrix", U)
        
        # Create identity matrix
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        inv = [[0.0]*n for _ in range(n)]
        
        yield "Computing inverse by solving AX = I:"
        
        for col in range(n):
            yield f"🔧 Solving for column {col+1}/{n}"
            b = [I[i][col] for i in range(n)]
            
            # Solve Ly = b
            y = [0.0] * n
            for i in range(n):
                s = b[i]
                for j in range(i):
                    s -= L[i][j] * y[j]
                y[i] = s / L[i][i]
            
            # Solve Ux = y
            x = [0.0] * n
            for i in range(n-1, -1, -1):
                s = y[i]
                for j in range(i+1, n):
                    s -= U[i][j] * x[j]
                x[i] = s / U[i][i]
                inv[i][col] = round(x[i], digit)
            
            yield f"Column {col+1} solution:"
            yield x
        
        yield "✅ Matrix Inverse computed"
        yield ("Inverse Matrix", inv)
        
        # Verify inverse
        product = np.dot(self.np_A, np.array(inv))
        identity_approx = product.round(digit)
        error = np.linalg.norm(identity_approx - np.eye(n), np.inf)
        
        yield f"Verification: ||A·A⁻¹ - I||∞ = {error:.{digit}e}"
        if error < 1e-6:
            yield "✅ Verification successful"
        else:
            yield "⚠️ Verification warning: Product not exactly identity"
        
        yield "🎉 Final Inverse Matrix"
        yield inv

    def Jokobi(self, x, epsilon):
        """Jacobi Iterative Method"""
        digit = self.digit
        n = self.n
        x_old = x[:]
        iteration = 0
        
        yield "🚀 Starting Jacobi Iterative Method"
        yield f"Initial tolerance: {epsilon}"
        yield "Initial guess:"
        yield x_old
        
        while True:
            iteration += 1
            x_new = [0.0] * n
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += self.A[i][j] * x_old[j]
                x_new[i] = round((self.b[i] - sigma) / self.A[i][i], digit)
            
            yield f"🔁 Iteration {iteration}:"
            yield x_new
            
            # Calculate error
            error = max(abs(x_new[i] - x_old[i]) for i in range(n))
            yield f"Maximum component change: {error:.{digit}e}"
            
            if error < epsilon:
                yield f"✅ Convergence achieved after {iteration} iterations"
                yield f"Final error: {error:.{digit}e}"
                break
                
            if iteration >= 100:
                yield f"⚠️ Maximum iterations reached 100"
                yield f"Current error: {error:.{digit}e}"
                break
                
            x_old = x_new[:]
        
        yield "🎉 Final Solution"
        yield x_new

    def Guass_sidel(self, x, epsilon):
        """Gauss-Seidel Iterative Method"""
        digit = self.digit
        n = self.n
        x_old = x[:]
        iteration = 0
        
        yield "🚀 Starting Gauss-Seidel Iterative Method"
        yield f"Initial tolerance: {epsilon}"
        yield "Initial guess:"
        yield x_old
        
        while True:
            iteration += 1
            x_new = x_old[:]  # Start with previous values
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += self.A[i][j] * x_new[j]  # Use updated values
                x_new[i] = round((self.b[i] - sigma) / self.A[i][i], digit)
            
            yield f"🔁 Iteration {iteration}:"
            yield x_new
            
            # Calculate error
            error = max(abs(x_new[i] - x_old[i]) for i in range(n))
            yield f"Maximum component change: {error:.{digit}e}"
            
            if error < epsilon:
                yield f"✅ Convergence achieved after {iteration} iterations"
                yield f"Final error: {error:.{digit}e}"
                break
                
            if iteration >= 100:
                yield f"⚠️ Maximum iterations reached 100"
                yield f"Current error: {error:.{digit}e}"
                break
                
            x_old = x_new[:]
        
        yield "🎉 Final Solution"
        yield x_new

    def SOR(self, x, w, epsilon):
        """Successive Over-Relaxation (SOR) Method"""
        digit = self.digit
        n = self.n
        x_old = x[:]
        iteration = 0
        
        yield "🚀 Starting SOR Method"
        yield f"Relaxation factor ω: {w}"
        yield f"Initial tolerance: {epsilon}"
        yield "Initial guess:"
        yield x_old
        
        while True:
            iteration += 1
            x_new = x_old[:]  # Start with previous values
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += self.A[i][j] * x_new[j]  # Use updated values
                
                # SOR update formula
                new_val = (1 - w) * x_old[i] + w * (self.b[i] - sigma) / self.A[i][i]
                x_new[i] = round(new_val, digit)
            
            yield f"🔁 Iteration {iteration}:"
            yield x_new
            
            # Calculate error
            error = max(abs(x_new[i] - x_old[i]) for i in range(n))
            yield f"Maximum component change: {error:.{digit}e}"
            
            if error < epsilon:
                yield f"✅ Convergence achieved after {iteration} iterations"
                yield f"Final error: {error:.{digit}e}"
                break
                
            if iteration >= 100:
                yield f"⚠️ Maximum iterations reached 100"
                yield f"Current error: {error:.{digit}e}"
                break
                
            x_old = x_new[:]
        
        yield "🎉 Final Solution"
        yield x_new
    
    
    def solve_multiple_b(self,*args):
        x=self.dolittel()
        for i in x:
            if isinstance(i, tuple) and len(i) == 2:
                last_lu = i
        result=last_lu
        yield "L and U matrices:"
        yield result
        l=result[0]
        u=result[1]
        del result
        j=1
        for i in args:
            yield(f"Solving system {j}: Ly = b")
            result=(l,i)
            X=self.forwardSubstitution(result)
            yield X
            result=(u,X)
            yield(f"Solution for system {j}:")
            yield(self.backSubstitution(result))
            j+=1

def k(matrix, num):
    inverse = np.linalg.inv(np.array(matrix))
    return np.linalg.norm(inverse, ord=num) * np.linalg.norm(np.array(matrix), ord=num)

def norm(matrix, num):
    return np.linalg.norm(np.array(matrix), ord=num)

def do_pivoting(A, b, i):
    maximum = i
    for j in range(i+1, len(A)):
        if abs(A[maximum][i]) < abs(A[j][i]):
            maximum = j
    A[maximum], A[i] = A[i], A[maximum]
    b[maximum], b[i] = b[i], b[maximum]

def Gauss_Helper(A, b, pivoting, digit):
    for i in range(len(A)-1): 
        if pivoting:
            do_pivoting(A, b, i)
            yield (A, b)
        for j in range(i+1, len(A)):
            mij = (-1) * (A[j][i] / A[i][i])
            A[j][i] = 0
            val = b[i] * mij
            b[j] = round(val + b[j], digit)
            for k in range(i+1, len(A)):
                val = A[i][k] * mij
                A[j][k] = round(val + A[j][k], digit)
        yield (A, b)

if __name__ == "__main__":
    from gui import LinearAlgebraApp
    import tkinter as tk
    
    root = tk.Tk()
    app = LinearAlgebraApp(root)
    root.mainloop()