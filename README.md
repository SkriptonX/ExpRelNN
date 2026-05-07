# ER-Optim: Accurate Second-Order Optimization for PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

**ER-Optim** — это библиотека для оптимизации нейронных сетей на основе метода **Экспоненциальной Релаксации (ER)**. Она разработана специально для решения «жестких» (stiff) задач, таких как Physics-Informed Neural Networks (**PINNs**) и **SIREN**, где классические методы первого порядка (Adam, SGD) демонстрируют медленную сходимость или стагнацию.



---

## 🚀 Ключевые особенности

* **Точность второго порядка:** Использование честных произведений Гессиана на вектор (HVP) через `torch.autograd`.
* **Спектральные аппроксимации:**
    * **Chebyshev ER:** Аппроксимация экспоненты полиномами Чебышева для индефинитных матриц.
    * **Lanczos ER:** Итеративное построение подпространства Крылова для точного поиска экстремальных собственных значений.
* **Гибридный режим:** Бесшовное переключение с Adam/L-BFGS на ER при стагнации Loss.
* **Умный Trust Region:** Встроенный механизм Backtracking Line Search и послойное демпфирование для защиты от взрыва градиентов.
* **ER Studio:** Встроенный GUI на PyQt5 для визуализации ландшафта функции потерь и послойной обусловленности в реальном времени.

---

## 📦 Установка

### Из исходников (режим разработчика)
```bash
git clone https://github.com/your-username/er-optim.git
cd er-optim
pip install -e .
```

### Зависимости
* PyTorch >= 2.0.0
* NumPy, SciPy
* Matplotlib
* PyQt5 (для GUI)

---

## 💡 Быстрый старт

Использовать `EROptimizer` так же просто, как и любой стандартный оптимизатор PyTorch.

```python
from er_optim import EROptimizer

model = MyNeuralNet()
# Инициализация ER с аппроксимацией Чебышева
optimizer = EROptimizer(
    model, 
    er_method='Chebyshev', 
    chebyshev_k=7, 
    h=1.0, 
    init_damping=1e-4
)

def loss_fn(output, target):
    return torch.mean((output - target)**2)

# В методе step необходимо передавать функцию потерь и данные
for epoch in range(100):
    loss_val, cond_numbers = optimizer.step(loss_fn, inputs, targets)
    print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
```

---

## 🔬 Математическое обоснование

Метод ER основывается на вычислении шага через операторную экспоненту матрицы Гессе $H$:

$$\Delta w = -(I - e^{-hH})H^{-1}g$$

Где $g$ — вектор градиента, а $h$ — параметр релаксации. В отличие от метода Ньютона, ER естественным образом ограничивает длину шага в областях с высокой кривизной, что делает его значительно более стабильным при обучении PINN для уравнений типа Аллена-Кана или Навье-Стокса.

---

## 🖥 ER Studio (GUI)

Библиотека поставляется с графической средой для анализа процесса обучения. Она позволяет сравнивать методы «лоб в лоб» и видеть, как веса сети адаптируются к геометрии задачи.

Запуск из терминала:
```bash
er-studio
```

* **Вкладка 1:** Сходимость (Loss vs Epochs/Time).
* **Вкладка 2:** Визуализация решения (Heatmaps для PINN, Image Reconstruction для SIREN).
* **Аналитика:** Мониторинг чисел обусловленности каждого слоя.

---

## 🛠 Архитектура проекта

* `er_optim.optimizer`: Ядро библиотеки, реализация `EROptimizer`.
* `er_optim.trainer`: Высокоуровневые функции для проведения экспериментов.
* `er_optim.gui`: Код графического интерфейса пользователя.

---

## 🤝 Участие в разработке

Ваши предложения и Pull Requests приветствуются! 
1. Форкните репозиторий.
2. Создайте ветку вашей фичи (`git checkout -b feature/AmazingFeature`).
3. Закоммитьте изменения (`git commit -m 'Add AmazingFeature'`).
4. Отправьте ветку (`git push origin feature/AmazingFeature`).
5. Откройте Pull Request.

---

## 📄 Лицензия

Распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).

---
*Разработано в рамках научно-исследовательской работы "Обучение нейронных сетей на основе методов с экспоненциальной релаксацией".*