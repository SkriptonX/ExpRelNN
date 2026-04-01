import sys
from trainer import train_network
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QSpinBox, QPushButton, QMessageBox,
                             QDialog, QScrollArea, QFrame)
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

        # Управление количеством слоев
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Количество скрытых слоев:"))
        self.num_layers_sb = QSpinBox()
        self.num_layers_sb.setRange(1, 10)
        self.num_layers_sb.setValue(len(current_configs))
        self.num_layers_sb.valueChanged.connect(self.update_layer_widgets)
        top_layout.addWidget(self.num_layers_sb)
        self.layout.addLayout(top_layout)

        # Прокручиваемая область для индивидуальной настройки каждого слоя
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
                self.add_layer_widget(8, 'Tanh')  # Значения по умолчанию для новых слоев
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
        self.resize(1200, 900)

        # Значения по умолчанию
        self.layer_configs = [{'units': 8, 'activation': 'Tanh'}, {'units': 8, 'activation': 'Tanh'}]
        self.run_counter = 1

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # === ЛЕВАЯ КОЛОНКА: НАСТРОЙКИ ===
        settings_panel = QWidget()
        settings_panel.setFixedWidth(280)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setAlignment(Qt.AlignTop)

        settings_layout.addWidget(QLabel("<b>Настройки эксперимента</b>"))

        settings_layout.addWidget(QLabel("Датасет:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.addItems(["Synthetic Ravine", "Moons", "Classification"])
        settings_layout.addWidget(self.dataset_cb)

        settings_layout.addWidget(QLabel("Оптимизатор:"))
        self.optim_cb = QComboBox()
        self.optim_cb.addItems(["ER", "Adam", "SGD"])
        settings_layout.addWidget(self.optim_cb)

        self.arch_btn = QPushButton("Настроить архитектуру сети")
        self.arch_btn.clicked.connect(self.open_arch_dialog)
        settings_layout.addWidget(self.arch_btn)

        settings_layout.addWidget(QLabel("Эпохи:"))
        self.epochs_sb = QSpinBox()
        self.epochs_sb.setRange(5, 1000)
        self.epochs_sb.setValue(30)
        self.epochs_sb.setSingleStep(5)
        settings_layout.addWidget(self.epochs_sb)

        settings_layout.addSpacing(20)

        self.run_btn = QPushButton("Запустить тест")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_training)
        settings_layout.addWidget(self.run_btn)

        self.clear_btn = QPushButton("Сбросить графики")
        self.clear_btn.clicked.connect(self.clear_plots)
        settings_layout.addWidget(self.clear_btn)

        layout.addWidget(settings_panel)

        # === ПРАВАЯ КОЛОНКА: ГРАФИКИ ===
        self.figure = Figure(figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(313)
        self.setup_axes()

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

        self.ax3.set_title("Обусловленность Гессиана (только метод ER)")
        self.ax3.set_xlabel("Эпоха")
        self.ax3.set_ylabel("Condition Number")
        self.ax3.set_yscale('log')
        self.ax3.grid(True, linestyle='--', alpha=0.7)

        self.figure.tight_layout()

    def clear_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.setup_axes()
        self.canvas.draw()
        self.run_counter = 1

    def open_arch_dialog(self):
        dialog = LayerConfigDialog(self.layer_configs, self)
        if dialog.exec_() == QDialog.Accepted:
            self.layer_configs = dialog.get_configs()

    def run_training(self):
        dataset = self.dataset_cb.currentText()
        optim = self.optim_cb.currentText()
        epochs = self.epochs_sb.value()

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Выполнение...")
        QApplication.processEvents()

        try:
            history = train_network(dataset, optim, self.layer_configs, epochs)
            self.plot_results(history, optim)
            self.run_counter += 1
        except Exception as e:
            QMessageBox.critical(self, "Ошибка выполнения", str(e))
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Запустить тест")

    def plot_results(self, history, optim):
        label_prefix = f"Тест {self.run_counter} ({optim})"

        # Наложение новых графиков поверх старых
        self.ax1.plot(history['loss'], linewidth=2, label=label_prefix)
        self.ax1.legend(loc='upper right', fontsize=8)

        self.ax2.plot(history['time'], history['loss'], linewidth=2, label=label_prefix)
        self.ax2.legend(loc='upper right', fontsize=8)

        if optim == 'ER' and history['cond']:
            for layer_name, conds in history['cond'].items():
                if 'weight' in layer_name:
                    layer_label = layer_name.replace('.weight', '')
                    self.ax3.plot(conds, label=f"{label_prefix} - {layer_label}")
            self.ax3.legend(loc='upper right', fontsize=8)

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())