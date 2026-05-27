from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.add_font("CJK", "", "/System/Library/Fonts/STHeiti Medium.ttc")
pdf.add_font("CJK", "B", "/System/Library/Fonts/STHeiti Medium.ttc")

def title(text):
    pdf.set_font("CJK", "B", 16)
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

def heading(text):
    pdf.set_font("CJK", "B", 13)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

def subheading(text):
    pdf.set_font("CJK", "B", 11)
    pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def body(text):
    pdf.set_font("CJK", "", 10)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)

def formula(text):
    pdf.set_font("CJK", "", 10)
    pdf.set_x(20)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)

title("analytic_gradient 数学推导")

heading("1. 模型结构（单个位置 i）")
body("给定 token 对 (t_i, t_{i+1})，令 y = t_{i+1} 为 target：")
formula("x = wte[t_i]  in  R^d")
formula("h_pre = W1 * x  in  R^(4d)")
formula("h = ReLU(h_pre)")
formula("logits = W2 * h  in  R^V")
formula("probs = softmax(logits)")
formula("L_i = -log(probs[y])")
body("其中 d = n_embd，V = vocab_size，W1 = mlp_fc1，W2 = mlp_fc2。")

heading("2. Loss")
formula("Loss(t, theta) = (1/n) * SUM_{i=0}^{n-1} L_i")

heading("3. 反向传播（链式法则）")

subheading("3.1  dL_i / d(logits)（softmax + 交叉熵）")
formula("dL_i / d(logits[k]) = probs[k] - 1[k == y]")
formula("dLoss / d(logits[k]) = (1/n) * (probs[k] - 1[k == y])")

subheading("3.2  dLoss / dW2（logits = W2 * h）")
formula("dLoss / dW2[k, j] = (dLoss / d(logits[k])) * h[j]")

subheading("3.3  dLoss / dh（W2^T * dlogits）")
formula("dLoss / dh[j] = SUM_{k=0}^{V-1} W2[k, j] * (dLoss / d(logits[k]))")

subheading("3.4  dLoss / d(h_pre)（ReLU 反向）")
formula("dLoss / d(h_pre[j]) = (dLoss / dh[j]) * 1[h_pre[j] > 0]")

subheading("3.5  dLoss / dW1（h_pre = W1 * x）")
formula("dLoss / dW1[j, k] = (dLoss / d(h_pre[j])) * x[k]")

subheading("3.6  dLoss / dx（W1^T * d(h_pre)）")
formula("dLoss / dx[k] = SUM_{j=0}^{4d-1} W1[j, k] * (dLoss / d(h_pre[j]))")

subheading("3.7  dLoss / d(wte[t_i])（x = wte[t_i]）")
formula("dLoss / d(wte[t_i][k]) = dLoss / dx[k]")

heading("4. 最终梯度（累加所有位置）")
body("对每个参数 theta_p：")
formula("dLoss / d(theta_p) = SUM_{i=0}^{n-1} (dL_i / d(theta_p)) * (1/n)")
body("代码中通过 += 累加每个位置的梯度，最后除以 n 体现在第 1 步的 dlogits 里已经乘了 1/n。")

pdf.output("/Users/bytedance/Programs/microgpt/analytic_gradient.pdf")
print("PDF generated: analytic_gradient.pdf")
