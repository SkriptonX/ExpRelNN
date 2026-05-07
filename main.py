import sys
import random
import numpy as np
import torch
from scipy.interpolate import griddata
from matplotlib.colors import SymLogNorm
from trainer import train_network
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QSpinBox, QPushButton, QMessageBox,
                             QDialog, QScrollArea, QFrame, QCheckBox, QTableWidgetItem,
                             QTableWidget, QHeaderView, QProgressBar, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LayerConfigDialog(QDialog):
    def __init__(self, current_configs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Архитектура нейронной сети")
        self.setMinimumWidth(450)
        self.setMinimumHeight(300)
        self.layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Количество скрытых слоев:"))
        self.num_layers_sb = QSpinBox()
        self.num_layers_sb.setRange(1, 10)
        self.num_layers_sb.setValue(len(current_configs))
        self.num_layers_sb.valueChanged.connect(self.update_layer_widgets)
        top_layout.addWidget(self.num_layers_sb)
        self.layout.addLayout(top_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)

        self.layer_widgets = []
        self.current_configs = current_configs
        self.build_initial_widgets()

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

    def build_initial_widgets(self):
        for config in self.current_configs:
            self.add_layer_widget(config['units'], config['activation'])

    def update_layer_widgets(self, count):
        current_count = len(self.layer_widgets)
        if count > current_count:
            for _ in range(count - current_count):
                self.add_layer_widget(8, 'Tanh')
        elif count < current_count:
            for _ in range(current_count - count):
                widget_data = self.layer_widgets.pop()
                widget_data['frame'].deleteLater()

    def add_layer_widget(self, units, activation):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        flayout = QHBoxLayout(frame)

        idx = len(self.layer_widgets) + 1
        flayout.addWidget(QLabel(f"<b>Слой {idx}:</b>"))

        flayout.addWidget(QLabel("Нейроны:"))
        u_sb = QSpinBox()
        u_sb.setRange(1, 1024)
        u_sb.setValue(units)
        flayout.addWidget(u_sb)

        flayout.addWidget(QLabel("Активация:"))
        a_cb = QComboBox()
        a_cb.addItems(["Tanh", "ReLU", "Sigmoid"])
        a_cb.setCurrentText(activation)
        flayout.addWidget(a_cb)

        self.scroll_layout.addWidget(frame)
        self.layer_widgets.append({'frame': frame, 'units': u_sb, 'activation': a_cb})

    def get_configs(self):
        return [{'units': w['units'].value(), 'activation': w['activation'].currentText()}
                for w in self.layer_widgets]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ послойной овражности (Метод ЭР)")
        self.resize(1400, 950)

        self.layer_configs = [{'units': 8, 'activation': 'Tanh'}, {'units': 8, 'activation': 'Tanh'}]
        self.run_counter = 1

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        settings_panel = QWidget()
        settings_panel.setFixedWidth(320)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setAlignment(Qt.AlignTop)

        settings_layout.addWidget(QLabel("<b>Настройки эксперимента</b>"))

        settings_layout.addWidget(QLabel("Датасет:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems([
            "Synthetic Ravine", "Moons", "Classification",
            "SIREN (Image Fitting)", "PINN (Allen-Cahn)"
        ])
        self.dataset_cb.currentIndexChanged.connect(self.auto_configure_experiment)
        settings_layout.addWidget(self.dataset_cb)

        settings_layout.addWidget(QLabel("Оптимизатор:"))
        self.optim_cb = QComboBox()
        self.optim_cb.addItems(["Adam", "SGD", "RMSProp", "L-BFGS", "ER", "Hybrid"])
        self.optim_cb.currentTextChanged.connect(self.toggle_optim_settings)
        settings_layout.addWidget(self.optim_cb)

        self.hybrid_base_label = QLabel("База для Гибрида:")
        self.hybrid_base_cb = QComboBox()
        self.hybrid_base_cb.addItems(["Adam", "RMSProp", "SGD", "L-BFGS"])
        settings_layout.addWidget(self.hybrid_base_label)
        settings_layout.addWidget(self.hybrid_base_cb)

        self.er_method_label = QLabel("Метод ER:")
        self.er_method_cb = QComboBox()
        self.er_method_cb.addItems(["Spectral", "Recursive", "Lanczos", "Chebyshev", "Kaczmarz"])
        self.er_method_cb.currentTextChanged.connect(self.toggle_optim_settings)
        settings_layout.addWidget(self.er_method_label)
        settings_layout.addWidget(self.er_method_cb)

        self.k_lanczos_label = QLabel("Размер подпространства (Lanczos k):")
        self.k_lanczos_sb = QSpinBox()
        self.k_lanczos_sb.setRange(2, 50)
        self.k_lanczos_sb.setValue(10)
        settings_layout.addWidget(self.k_lanczos_label)
        settings_layout.addWidget(self.k_lanczos_sb)

        self.chebyshev_k_label = QLabel("Степень полинома (Chebyshev K):")
        self.chebyshev_k_sb = QSpinBox()
        self.chebyshev_k_sb.setRange(2, 100)
        self.chebyshev_k_sb.setValue(15)
        settings_layout.addWidget(self.chebyshev_k_label)
        settings_layout.addWidget(self.chebyshev_k_sb)

        self.switch_label = QLabel("Условие переключения:")
        self.switch_cb = QComboBox()
        self.switch_cb.addItems(["Стагнация (Stagnation)", "Спектральный (Cost-Benefit)", "Фиксированная эпоха"])
        self.switch_cb.currentTextChanged.connect(self.toggle_optim_settings)
        settings_layout.addWidget(self.switch_label)
        settings_layout.addWidget(self.switch_cb)

        self.switch_epoch_label = QLabel("Эпоха переключения:")
        self.switch_epoch_sb = QSpinBox()
        self.switch_epoch_sb.setRange(1, 1000)
        self.switch_epoch_sb.setValue(15)
        settings_layout.addWidget(self.switch_epoch_label)
        settings_layout.addWidget(self.switch_epoch_sb)

        settings_layout.addWidget(QLabel("Целевой функционал (Loss):"))
        self.loss_cb = QComboBox()
        self.loss_cb.addItems(["Cross-Entropy", "MSE", "Log Loss"])
        settings_layout.addWidget(self.loss_cb)

        self.arch_btn = QPushButton("Настроить архитектуру сети")
        self.arch_btn.clicked.connect(self.open_arch_dialog)
        settings_layout.addWidget(self.arch_btn)

        settings_layout.addWidget(QLabel("Эпохи:"))
        self.epochs_sb = QSpinBox()
        self.epochs_sb.setRange(5, 5000)
        self.epochs_sb.setValue(30)
        self.epochs_sb.setSingleStep(5)
        settings_layout.addWidget(self.epochs_sb)

        self.batching_cb = QCheckBox("Использовать мини-батчи")
        self.batching_cb.setChecked(False)
        self.batching_cb.stateChanged.connect(self.toggle_batching)
        settings_layout.addWidget(self.batching_cb)

        self.batch_label = QLabel("Размер батча:")
        self.batch_sb = QSpinBox()
        self.batch_sb.setRange(16, 300)
        self.batch_sb.setValue(64)
        self.batch_sb.setSingleStep(16)
        self.batch_sb.setEnabled(False)
        settings_layout.addWidget(self.batch_label)
        settings_layout.addWidget(self.batch_sb)

        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("<b>Воспроизводимость</b>"))

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))

        self.seed_sb = QSpinBox()
        self.seed_sb.setRange(0, 99999999)
        self.seed_sb.setValue(random.randint(0, 99999999))
        self.seed_sb.setEnabled(False)
        seed_layout.addWidget(self.seed_sb)

        self.seed_auto_cb = QCheckBox("Авто")
        self.seed_auto_cb.setChecked(True)
        self.seed_auto_cb.toggled.connect(lambda checked: self.seed_sb.setEnabled(not checked))
        seed_layout.addWidget(self.seed_auto_cb)
        settings_layout.addLayout(seed_layout)
        settings_layout.addSpacing(10)

        self.ema_cb = QCheckBox("Использовать EMA Гессиана")
        self.ema_cb.setChecked(True)
        settings_layout.addWidget(self.ema_cb)

        self.compress_cb = QCheckBox("Сжатие памяти (CSR Pruning)")
        self.compress_cb.setChecked(False)
        settings_layout.addWidget(self.compress_cb)

        settings_layout.addSpacing(10)

        self.run_btn = QPushButton("Запустить тест")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_training)
        settings_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.progress_bar)

        self.clear_btn = QPushButton("Сбросить графики и таблицу")
        self.clear_btn.clicked.connect(self.clear_data)
        settings_layout.addWidget(self.clear_btn)

        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("<b>Результаты:</b>"))

        self.toggle_optim_settings()

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(["Оптим.", "Режим", "Loss", "Время", "Смена", "Seed", "Память"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        settings_layout.addWidget(self.results_table)

        layout.addWidget(settings_panel)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tab_metrics = QWidget()
        metrics_layout = QVBoxLayout(self.tab_metrics)
        self.figure = Figure(figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        metrics_layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(411)
        self.ax2 = self.figure.add_subplot(412)
        self.ax3 = self.figure.add_subplot(413)
        self.ax4 = self.figure.add_subplot(414)
        self.setup_axes()

        self.tab_visual = QWidget()
        visual_layout = QVBoxLayout(self.tab_visual)
        self.res_figure = Figure(figsize=(10, 8))
        self.res_canvas = FigureCanvas(self.res_figure)
        visual_layout.addWidget(self.res_canvas)

        self.tabs.addTab(self.tab_metrics, "Графики сходимости")
        self.tabs.addTab(self.tab_visual, "Визуализация результата")

    def auto_configure_experiment(self):
        ds = self.dataset_cb.currentText()

        if ds == "SIREN (Image Fitting)":
            self.layer_configs = [
                {'units': 128, 'activation': 'Sine'},
                {'units': 128, 'activation': 'Sine'},
                {'units': 128, 'activation': 'Sine'}
            ]
            self.optim_cb.setCurrentText("Hybrid")
            self.hybrid_base_cb.setCurrentText("Adam")
            self.er_method_cb.setCurrentText("Lanczos")
            self.k_lanczos_sb.setValue(20)
            self.batching_cb.setChecked(False)
            self.epochs_sb.setValue(300)
            self.loss_cb.setCurrentText("MSE")
            self.switch_cb.setCurrentText("Фиксированная эпоха")
            self.switch_epoch_sb.setValue(100)

        elif ds == "PINN (Allen-Cahn)":
            self.layer_configs = [
                {'units': 50, 'activation': 'Tanh'},
                {'units': 50, 'activation': 'Tanh'},
                {'units': 50, 'activation': 'Tanh'}
            ]
            self.optim_cb.setCurrentText("Hybrid")
            self.hybrid_base_cb.setCurrentText("L-BFGS")
            self.er_method_cb.setCurrentText("Chebyshev")
            self.chebyshev_k_sb.setValue(7)
            self.batching_cb.setChecked(False)
            self.epochs_sb.setValue(500)
            self.loss_cb.setCurrentText("MSE")
            self.switch_cb.setCurrentText("Стагнация (Stagnation)")

        else:
            self.layer_configs = [{'units': 8, 'activation': 'Tanh'}, {'units': 8, 'activation': 'Tanh'}]

        QMessageBox.information(self, "Авто-настройка", f"Применены настройки для: {ds}")

    def toggle_batching(self, state):
        self.batch_sb.setEnabled(state == Qt.Checked)

    def toggle_optim_settings(self, text=None):
        optim = self.optim_cb.currentText()
        is_er = (optim == "ER")
        is_hybrid = (optim == "Hybrid")
        is_er_family = is_er or is_hybrid

        current_er = self.er_method_cb.currentText()
        is_lanczos = current_er == "Lanczos"
        is_chebyshev = current_er == "Chebyshev"
        is_switch_fixed = self.switch_cb.currentText() == "Фиксированная эпоха"

        self.hybrid_base_label.setVisible(is_hybrid)
        self.hybrid_base_cb.setVisible(is_hybrid)
        self.er_method_label.setVisible(is_er_family)
        self.er_method_cb.setVisible(is_er_family)
        self.k_lanczos_label.setVisible(is_er_family and is_lanczos)
        self.k_lanczos_sb.setVisible(is_er_family and is_lanczos)
        self.chebyshev_k_label.setVisible(is_er_family and is_chebyshev)
        self.chebyshev_k_sb.setVisible(is_er_family and is_chebyshev)
        self.switch_label.setVisible(is_hybrid)
        self.switch_cb.setVisible(is_hybrid)
        self.switch_epoch_label.setVisible(is_hybrid and is_switch_fixed)
        self.switch_epoch_sb.setVisible(is_hybrid and is_switch_fixed)
        self.ema_cb.setVisible(is_er_family)
        self.compress_cb.setVisible(is_er_family)
        self.compress_cb.setEnabled(not (is_lanczos or is_chebyshev))

    def setup_axes(self):
        self.ax1.set_title("Сходимость: Эпохи / Loss")
        self.ax1.set_xlabel("Эпоха")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_yscale('log')
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        self.ax2.set_title("Сходимость: Время (с) / Loss")
        self.ax2.set_xlabel("Время (с)")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_yscale('log')
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        self.ax3.set_title("Обусловленность Гессиана")
        self.ax3.set_xlabel("Эпоха")
        self.ax3.set_ylabel("Cond")
        self.ax3.set_yscale('log')
        self.ax3.grid(True, linestyle='--', alpha=0.7)

        self.ax4.set_title("Обусловленность весов")
        self.ax4.set_xlabel("Эпоха")
        self.ax4.set_ylabel("Cond")
        self.ax4.set_yscale('log')
        self.ax4.grid(True, linestyle='--', alpha=0.7)
        self.figure.tight_layout()

    def clear_data(self):
        self.figure.clear()
        self.setup_axes()
        self.canvas.draw()

        self.res_figure.clear()
        self.res_canvas.draw()

        self.run_counter = 1
        self.results_table.setRowCount(0)

    def open_arch_dialog(self):
        dialog = LayerConfigDialog(self.layer_configs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.layer_configs = dialog.get_configs()

    def run_training(self):
        dataset = self.dataset_cb.currentText()
        optim = self.optim_cb.currentText()
        loss_name = self.loss_cb.currentText()
        epochs = self.epochs_sb.value()
        batch_size = self.batch_sb.value()
        use_ema = self.ema_cb.isChecked()
        use_batching = self.batching_cb.isChecked()
        er_method = self.er_method_cb.currentText()
        k_lanczos = self.k_lanczos_sb.value()
        chebyshev_k = self.chebyshev_k_sb.value()
        switch_method = self.switch_cb.currentText()
        switch_epoch = self.switch_epoch_sb.value()
        use_compression = self.compress_cb.isChecked()
        hybrid_base = self.hybrid_base_cb.currentText()

        if self.seed_auto_cb.isChecked():
            current_seed = random.randint(0, 99999999)
            self.seed_sb.setValue(current_seed)
        else:
            current_seed = self.seed_sb.value()

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Выполнение...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, epochs)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Эпоха %v из %m")
        QApplication.processEvents()

        def update_progress(current_epoch, total_epochs):
            self.progress_bar.setValue(current_epoch)
            QApplication.processEvents()

        try:
            history = train_network(dataset, optim, self.layer_configs, epochs,
                                    batch_size, use_ema, use_batching, loss_name,
                                    er_method, k_lanczos, switch_method, switch_epoch,
                                    current_seed, use_compression, chebyshev_k, hybrid_base,
                                    progress_callback=update_progress)

            self.plot_results(history, optim, current_seed)
            self.plot_visualization(history, dataset, optim, current_seed)

            display_optim = f"Hybrid ({hybrid_base} -> ER)" if optim == "Hybrid" else optim
            if optim == "ER": display_optim = f"ER ({er_method})"

            self.add_table_row(display_optim, use_batching, use_ema, history['final_loss'],
                               history['total_time'], history.get('switch_epoch', -1),
                               current_seed, history['memory_mb'])
            self.run_counter += 1

            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка выполнения", str(e))
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Запустить тест")
            self.progress_bar.setVisible(False)

    def plot_results(self, history, optim, seed_val):
        label_prefix = f"Т{self.run_counter}: {optim} (s:{seed_val})"

        self.ax1.plot(history['loss'], linewidth=2, label=label_prefix)
        switch_ep = history.get('switch_epoch', -1)
        if switch_ep != -1:
            self.ax1.axvline(x=switch_ep, color='cyan', linestyle=':', linewidth=1.5)
        self.ax1.legend(loc='upper right', fontsize=8)

        self.ax2.plot(history['time'], history['loss'], linewidth=2, label=label_prefix)
        self.ax2.legend(loc='upper right', fontsize=8)

        if (optim == 'ER' or optim == 'Hybrid') and history.get('cond'):
            for layer_name, conds in history['cond'].items():
                if 'weight' in layer_name:
                    layer_label = layer_name.replace('.weight', '')
                    x_axis = range(switch_ep, switch_ep + len(conds)) if switch_ep != -1 else range(len(conds))
                    self.ax3.plot(x_axis, conds, label=f"{label_prefix} - {layer_label}")
            self.ax3.legend(loc='upper right', fontsize=8)

        if history.get('weight_cond'):
            for layer_name, conds in history['weight_cond'].items():
                layer_label = layer_name.replace('.weight', '')
                self.ax4.plot(conds, label=f"{label_prefix} - {layer_label}")
            self.ax4.legend(loc='upper right', fontsize=8)

        self.canvas.draw()

    def plot_visualization(self, history, dataset_name, optim, seed_val):
        model = history.get('model')
        X_tensor = history.get('X')
        y_tensor = history.get('y')

        if model is None or X_tensor is None or y_tensor is None:
            return

        self.res_figure.clear()

        try:
            X_numpy = X_tensor.cpu().numpy()
            y_raw = y_tensor.cpu().numpy()
            model.eval()

            if dataset_name in ["Moons", "Classification"]:
                ax = self.res_figure.add_subplot(111)

                x_min, x_max = X_numpy[:, 0].min() - 0.5, X_numpy[:, 0].max() + 0.5
                y_min, y_max = X_numpy[:, 1].min() - 0.5, X_numpy[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=X_tensor.device)

                with torch.no_grad():
                    Z = model(grid_tensor).cpu().numpy()

                Z = Z.reshape(xx.shape) if Z.shape[1] == 1 else Z.argmax(axis=1).reshape(xx.shape)

                ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.5)
                ax.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_raw.squeeze(), cmap='RdBu', edgecolors='k', s=40)
                ax.set_title(f"Разделяющая плоскость: {optim}")

            elif dataset_name == "SIREN (Image Fitting)":
                size = int(np.sqrt(X_numpy.shape[0]))
                if size * size == X_numpy.shape[0]:
                    ax1 = self.res_figure.add_subplot(121)
                    ax2 = self.res_figure.add_subplot(122)

                    with torch.no_grad():
                        pred = model(X_tensor).cpu().numpy()

                    y_img = y_raw.reshape(size, size) if y_raw.shape[1] == 1 else y_raw.reshape(size, size, 3)
                    pred_img = pred.reshape(size, size) if pred.shape[1] == 1 else pred.reshape(size, size, 3)

                    ax1.imshow(y_img, cmap='gray' if y_raw.shape[1] == 1 else None)
                    ax1.set_title("Original Image")
                    ax1.axis('off')

                    ax2.imshow(np.clip(pred_img, 0, 1), cmap='gray' if pred.shape[1] == 1 else None)
                    ax2.set_title(f"Reconstruction: {optim}")
                    ax2.axis('off')

            elif dataset_name == "PINN (Allen-Cahn)":
                axes = self.res_figure.subplots(1, 2, sharey=True)
                ax_pred, ax_exact = axes
                x_coord = X_numpy[:, 0]
                t_coord = X_numpy[:, 1]
                x_min, x_max = x_coord.min(), x_coord.max()
                t_min, t_max = t_coord.min(), t_coord.max()
                with torch.no_grad():
                    pred_raw = model(X_tensor).cpu().numpy()
                y_exact_vals = y_raw.squeeze()
                pred_final_vals = pred_raw.squeeze()
                N_plot = 200
                xi = np.linspace(x_min, x_max, N_plot)
                ti = np.linspace(t_min, t_max, N_plot)
                grid_x, grid_t = np.meshgrid(xi, ti)
                grid_exact = griddata((x_coord, t_coord), y_exact_vals, (grid_x, grid_t), method='linear')
                grid_pred = griddata((x_coord, t_coord), pred_final_vals, (grid_x, grid_t), method='linear')
                print(np.nanmin(y_exact_vals), np.nanmax(y_exact_vals), np.nanstd(y_exact_vals))
                print(np.nanmin(pred_final_vals), np.nanmax(pred_final_vals), np.nanstd(pred_final_vals))
                linthresh = 1e-2
                abs_max = np.nanmax(np.abs(np.concatenate([grid_exact.ravel(), grid_pred.ravel()])))
                norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-abs_max, vmax=abs_max, base=10)

                extent = [x_min, x_max, t_min, t_max]
                im1 = ax_pred.imshow(grid_pred, extent=extent, origin='lower',
                                     cmap='magma', aspect='auto', norm=norm)
                ax_pred.set_title(f"Предсказано ({optim})")
                ax_pred.set_xlabel("x (Пространство)")
                ax_pred.set_ylabel("t (Время)")

                im2 = ax_exact.imshow(grid_exact, extent=extent, origin='lower',
                                      cmap='magma', aspect='auto', norm=norm)
                ax_exact.set_title("Точное решение")
                ax_exact.set_xlabel("x (Пространство)")

                self.res_figure.subplots_adjust(right=0.85, wspace=0.3)
                cbar_ax = self.res_figure.add_axes([0.88, 0.15, 0.03, 0.7])
                self.res_figure.colorbar(im2, cax=cbar_ax, label='Поле u(t,x)')
            else:
                ax = self.res_figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Отрисовка '{dataset_name}' в разработке.", ha='center', va='center')

        except Exception as e:
            print(f"Ошибка визуализации: {e}")

        self.res_canvas.draw()
        model.train()

    def add_table_row(self, optim, is_batched, use_ema, final_loss, total_time, switch_ep, seed_val, mem_mb):
        row_pos = self.results_table.rowCount()
        self.results_table.insertRow(row_pos)
        settings_str = "Batched" if is_batched else "Full"
        if ("ER" in optim or "Hybrid" in optim) and is_batched and use_ema: settings_str += "+EMA"
        switch_str = f"Эпоха {switch_ep}" if switch_ep != -1 else "-"
        self.results_table.setItem(row_pos, 0, QTableWidgetItem(optim))
        self.results_table.setItem(row_pos, 1, QTableWidgetItem(settings_str))
        self.results_table.setItem(row_pos, 2, QTableWidgetItem(f"{final_loss:.4e}"))
        self.results_table.setItem(row_pos, 3, QTableWidgetItem(f"{total_time:.2f}s"))
        self.results_table.setItem(row_pos, 4, QTableWidgetItem(switch_str))
        self.results_table.setItem(row_pos, 5, QTableWidgetItem(str(seed_val)))
        self.results_table.setItem(row_pos, 6, QTableWidgetItem(f"{mem_mb:.2f} MB"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())