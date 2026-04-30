
### **Lesson 9: Machine Learning with Linear Algebra**

This final lesson of the "Machine Learning Foundations" series synthesizes prior concepts to perform advanced operations and solve real-world problems such as data compression and linear regression.

---

#### **9.1 Singular Value Decomposition (SVD)**

**Singular Value Decomposition (SVD)** is a method for decomposing a matrix into its fundamental components. While eigendecomposition is restricted to square matrices, **SVD is applicable to any real-valued matrix**, including rectangular ones.

- **The SVD Formula:** Any matrix $A$ can be represented as **$A = UDV^T$**.
    - **$U$**: An $m \times m$ **orthogonal matrix** whose columns are the **left-singular vectors** of $A$.
    - **$V$**: An $n \times n$ **orthogonal matrix** whose columns are the **right-singular vectors** of $A$.
    - **$D$**: An $m \times n$ **diagonal matrix** containing the **singular values** of $A$ along its main diagonal, typically arranged in descending order.
- **Relationship to Eigendecomposition:**
    - The **left-singular vectors** of $A$ are the **eigenvectors of $AA^T$**.
    - The **right-singular vectors** of $A$ are the **eigenvectors of $A^TA$**.
    - The non-zero **singular values** are the **square roots of the eigenvalues** of both $AA^T$ and $A^TA$.


<<div align="center">
  <img src="images/Pasted%20image%2020260430072517.png" width="600">
</div>




#### **9.2 Media File Compression**

SVD is a ubiquitous technique for **lossy data compression**, allowing models to retain the most informative components of a dataset while dramatically reducing its size.

- **Prominence of Singular Vectors:** Because singular values are arranged in descending order, the first singular vector represents the **most prominent feature** of the data.
- **Image Compression Example:** An image can be represented as a matrix of pixels. By keeping only the first few singular vectors (e.g., $n=64$), one can reconstruct the image with high fidelity.
- **Efficiency:** In a practical demonstration using an image of a dog, using only 64 singular vectors reduced the data footprint to **3.7% of the original size**.


<div align="center">
  <img src="images/Pasted%20image%2020260430072728.png" width="600">
</div>



---

#### **9.3 The Moore-Penrose Pseudoinverse**

The **Moore-Penrose Pseudoinverse ($A^+$)** is a generalization of the matrix inverse for **non-square matrices**. Regular matrix inversion requires a matrix to be square and non-singular; the pseudoinverse resolves these limitations.

- **The Formula:** It is computed using the SVD components: **$A^+ = VD^+U^T$**.
- **$D^+$ Calculation:** $D^+$ is derived by taking the **reciprocal of all non-zero elements** in the diagonal matrix $D$ and then transposing it.
- **Implementation:** In code, this can be calculated automatically using **`np.linalg.pinv()`** in NumPy or **`torch.pinverse()`** in PyTorch.

<div align="center">
  <img src="images/Pasted%20image%2020260430073436.png" width="500">
  <br>


---

#### **9.4 Regression via Pseudoinversion**

In machine learning, systems of equations are frequently **overdetermined**, meaning there are many more data points (rows) than features (columns). Matrix inversion cannot solve these, but the pseudoinverse can.

- **Solving for Weights:** Linear regression can be represented as **$y = Xw$**. We can solve for the unknown weights $w$ using the formula **$w = X^+y$**.
- **Optimization Intuition:** For overdetermined systems, the pseudoinverse finds the solution $w$ that minimizes the **Euclidean distance ($L^2$ norm)** between the predicted and actual values ($Xw - y$).
- **Practical Use:** This method allows a model to fit a regression line (calculating the slope $m$ and y-intercept $b$) using only linear algebra, bypassing iterative optimization like gradient descent for small datasets.

  <br>
  <img src="images/Pasted%20image%2020260430073644.png" width="500">
  <br>

---

#### **9.5 Principal Component Analysis (PCA)**

**Principal Component Analysis (PCA)** is a simple yet powerful **unsupervised machine learning algorithm** used to identify structure in unlabeled data and perform dimensionality reduction.

- **Core Objective:** PCA distills a high-dimensional space into a lower-dimensional one while retaining the most informative parts of the data.
- **Mechanics:** It identifies **Principal Components**, which are vectors defining a new coordinate system where the first axis aligns with the **direction of maximum variance** in the data.
- **Linear Algebra Foundations:** PCA relies on several previously learned concepts, including norms, orthogonal matrices, identity matrices, and the **trace operator**.
- **Application:** In the provided Iris dataset example, PCA was used to reduce 4 features down to 2 principal components, which allowed for the visual segregation of three different flower species on a 2D scatter plot.
 <br>
  <img src="images/Pasted%20image%2020260430073806.png" width="500">
  <br>

---

#### **9.6 Resources for Further Study of Linear Algebra**

To deepen expertise beyond the essential foundations, the following resources are recommended:

- **Video Tutorials:** **"3Blue1Brown"** on YouTube for high-level geometric intuition and **Khan Academy** for brushing up on basic algebra.
- **Textbooks:**
    - **"Deep Learning"** (2016) by Goodfellow, Bengio, and Courville (Chapter 2).
    - **"Mathematics for Machine Learning"** by Deisenroth, Faisal, and Ong.
    - **"Linear Algebra Done Right"** by Sheldon Axler.
- **Next Steps in the Journey:** The completion of these linear algebra modules prepares students for the next subjects in the _Machine Learning Foundations_ series: **Calculus I & II**, **Probability & Information Theory**, **Statistics**, and **Optimization**.
 <br>
  <img src="images/Pasted%20image%2020260430074023.png" width="500">
</div>
