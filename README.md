# 🧠 CNN + Learning Rate Schedulers (FashionMNIST)

This project explores how different **Learning Rate Schedulers** affect the training of a CNN on the **FashionMNIST** dataset using **PyTorch**.

## 📌 Task (from coursework)
The goal is to apply the following LR schedulers to a simplified CNN:
- `ExponentialLR`
- `MultiStepLR`
- `CosineAnnealingLR`

and compare results to a baseline model without scheduling.

## 🧱 Model: Reduced CNN
To simulate a lighter model (as per Assignment 4), one convolution block was removed.

## 📊 Results (Test Accuracy)

| Scheduler         | Accuracy |
|------------------|----------|
| No Scheduler      | 91.17%   |
| ExponentialLR     | 91.96%   |
| MultiStepLR       | 91.88%   |
| CosineAnnealingLR | **91.93%** ✅ |

## 📈 Training Loss Comparison
![Training Loss](imgs/loss_plot_comparison.png)

## 🧠 Conclusion
> All schedulers slightly improved training. **CosineAnnealingLR** achieved the best accuracy, likely due to smoother LR decay that avoids overfitting in early epochs.

## 🧰 Technologies
- Python
- PyTorch
- Matplotlib
- FashionMNIST (torchvision)

## 🚀 Run this notebook
```bash
pip install -r requirements.txt
jupyter notebook
```

## 📷 Sample Outputs

**No Scheduler**
![Baseline](imgs/loss_baseline.png)

**With ExponentialLR**
![ExponentialLR](imgs/loss_exponentiallr.png)

**With MultiStepLR**
![MultiStepLR](imgs/loss_multisteplr.png)

**With CosineAnnealingLR**
![CosineAnnealingLR](imgs/loss_cosineannealinglr.png)
