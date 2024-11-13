import sys
import numpy as np
import winshell
import cv2
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QLabel,
    QMainWindow,
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QFileDialog,
    QCheckBox,
    QLineEdit, QMessageBox
)
from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QPixmap, QImage, Qt
from PySide6.QtMultimedia import QMediaDevices

net = cv2.dnn.readNetFromONNX('best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

last_x = last_y = 0


class NEMO(QMainWindow):

    def __init__(self):
        super(NEMO, self).__init__()
        self.setWindowTitle('Немо')
        self.setStyleSheet("background-color: #191919; color: white")
        self.setFixedSize(830, 620)

        # Кнопка для открытия/закрытия sidebar
        self.sidebar_button = QPushButton("☰")
        self.sidebar_button.setFixedSize(40, 40)
        self.sidebar_button.setStyleSheet('QPushButton {'
                                          'border-radius: 20px;'
                                          'font-size: 12px;'
                                          'color: white;'
                                          'background-color: #303136;}')

        self.sidebar_button.clicked.connect(self.toggle_sidebar)

        self.color = {
            'Зелёный': (0, 255, 0),
            'Красный': (0, 0, 255),
            'Синий': (255, 0, 0)
        }

        self.color_label = QLabel('Цвет рамок:')
        self.color_selector = QComboBox()
        self.color_selector.setFixedSize(140, 25)
        self.color_selector.setStyleSheet('QComboBox { '
                                          'background-color: #252525; '
                                          'border-radius: 7px; '
                                          'padding: 5px }'
                                          'QComboBox:drop-down { '
                                          'background-color: #252525;'
                                          'border-radius: 5px; '
                                          'padding: 5px}'
                                          'QComboBox:disabled { '
                                          'background-color: #141414; '
                                          'color: darkgray }')

        self.color_selector.addItems(list(self.color.keys()))
        self.color_mode = QCheckBox('Цвет по RGB')
        self.color_mode.clicked.connect(self.color_mode_check)
        self.Red = QSpinBox(minimum=0, maximum=255)
        self.Red.setFixedSize(75, 25)
        self.Green = QSpinBox(minimum=0, maximum=255)
        self.Green.setFixedSize(75, 25)
        self.Blue = QSpinBox(minimum=0, maximum=255)
        self.Blue.setFixedSize(75, 25)
        self.Red.setDisabled(True)
        self.Green.setDisabled(True)
        self.Blue.setDisabled(True)
        self.Red.setStyleSheet('QSpinBox { '
                               'background-color: #252525; '
                               'border-radius: 7px; '
                               'padding: 5px }'
                               'QSpinBox:up-button, QSpinBox:down-button { '
                               'border-radius: 1px }'
                               'QSpinBox:disabled { '
                               'background-color: #141414; '
                               'color: darkgray }')

        self.Green.setStyleSheet('QSpinBox { '
                                 'background-color: #252525; '
                                 'border-radius: 7px; '
                                 'padding: 5px }'
                                 'QSpinBox:up-button, QSpinBox:down-button { '
                                 'border-radius: 1px }'
                                 'QSpinBox:disabled { '
                                 'background-color: #141414; '
                                 'color: darkgray }')

        self.Blue.setStyleSheet('QSpinBox { '
                                'background-color: #252525; '
                                'border-radius: 7px; padding: 5px }'
                                'QSpinBox:up-button, QSpinBox:down-button { '
                                'border-radius: 1px }'
                                'QSpinBox:disabled { '
                                'background-color: #141414; '
                                'color: darkgray }')

        self.resize_label = QLabel('Размер изображения:')
        self.origin_size = (1920, 1080)
        sizes = (
            '1920x1080', '1600x1200',
            '1280x720', '1024x768',
            '960x540', '800x600'
        )
        self.size_selector = QComboBox()
        self.size_selector.setFixedSize(140, 25)
        self.size_selector.setStyleSheet('QComboBox { '
                                         'background-color: #252525; '
                                         'border-radius: 7px; '
                                         'padding: 5px }'
                                         'QComboBox:drop-down { '
                                         'background-color: #252525; '
                                         'border-radius: 5px; '
                                         'padding: 5px}'
                                         'QComboBox:disabled { '
                                         'background-color: #141414; '
                                         'color: darkgray }')

        self.size_selector.addItems(sizes)
        self.resize_mode = QCheckBox('Свой размер')
        self.resize_mode.clicked.connect(self.resize_mode_check)
        self.frame_width = QSpinBox(minimum=100, maximum=3840, value=self.origin_size[0])
        self.frame_width.setFixedSize(100, 25)
        self.frame_height = QSpinBox(minimum=100, maximum=2160, value=self.origin_size[1])
        self.frame_height.setFixedSize(100, 25)
        self.frame_width.setDisabled(True)
        self.frame_height.setDisabled(True)
        self.frame_width.setStyleSheet('QSpinBox { '
                                       'background-color: #252525; '
                                       'border-radius: 7px; '
                                       'padding: 5px }'
                                       'QSpinBox:up-button, QSpinBox:down-button { '
                                       'border-radius: 1px }'
                                       'QSpinBox:disabled { '
                                       'background-color: #141414; '
                                       'color: darkgray }')

        self.frame_height.setStyleSheet('QSpinBox { '
                                        'background-color: #252525; '
                                        'border-radius: 7px; '
                                        'padding: 5px }'
                                        'QSpinBox:up-button, QSpinBox:down-button { '
                                        'border-radius: 1px }'
                                        'QSpinBox:disabled { '
                                        'background-color: #141414; '
                                        'color: darkgray }')

        self.conf_label = QLabel('Уровень достоверности:')
        self.conf = QDoubleSpinBox(minimum=0, maximum=1, value=0.7)
        self.conf.setFixedSize(140, 25)
        self.conf.setStyleSheet('QDoubleSpinBox { '
                                'background-color: #252525; '
                                'border-radius: 7px; '
                                'padding: 5px }'
                                'QDoubleSpinBox:up-button, QDoubleSpinBox:down-button { '
                                'border-radius: 1px }')

        self.name_label = QLabel('Имя объекта:')
        self.obj_name = QLineEdit('Fish')
        self.obj_name.setFixedSize(140, 25)
        self.obj_name.setStyleSheet('QLineEdit { '
                                    'background-color: #252525; '
                                    'border-radius: 7px; '
                                    'padding: 5px }')

        self.path_label = QLabel('Путь для сохранения:')
        self.path = QLineEdit(f'{winshell.desktop()}')
        self.path.setFixedSize(140, 25)
        self.path.setStyleSheet('QLineEdit { '
                                'background-color: #252525; '
                                'border-radius: 7px; '
                                'padding: 5px }')

        self.browse_button = QPushButton('Выбрать путь')
        self.browse_button.setFixedSize(QSize(120, 30))
        self.browse_button.setStyleSheet('QPushButton {'
                                         'border-radius: 15px;'
                                         'font-size: 12px;'
                                         'color: white;'
                                         'background-color: #303136;}')

        self.browse_button.clicked.connect(self.browse_path)

        self.start_stream = QPushButton('Прямая трансляция')
        self.start_stream.setFixedSize(QSize(170, 30))
        self.start_stream.setStyleSheet('QPushButton {'
                                        'border-radius: 15px;'
                                        'font-size: 14px;'
                                        'color: white;'
                                        'background-color: #ad1818;}')

        self.start_stream.clicked.connect(lambda: self.setup_video(self.mode_selector.currentIndex()))
        self.start_stream.clicked.connect(lambda: self.pl_bot_widget.setVisible(False))

        self.mode_selector = QComboBox()
        self.mode_selector.setFixedSize(170, 30)
        self.mode_selector.setStyleSheet('QComboBox { '
                                         'background-color: #252525; '
                                         'border-radius: 7px; '
                                         'padding: 5px }'
                                         'QComboBox:drop-down { '
                                         'background-color: #252525; '
                                         'border-radius: 5px; '
                                         'padding: 5px}'
                                         'QComboBox:disabled { '
                                         'background-color: #141414; '
                                         'color: darkgray }')

        self.mode_selector.addItems([cam.description() for cam in QMediaDevices.videoInputs()])

        self.nemo_button = QPushButton('Включить Nemo')
        self.nemo_button.setFixedSize(170, 30)
        self.nemo_button.setStyleSheet('QPushButton { '
                                       'background-color: #1851ad; '
                                       'color: white; '
                                       'border-radius: 15px;'
                                       'font-size: 14px }')

        self.nemo_button.clicked.connect(self.get_data)

        self.image_label = QLabel(self)
        frame = cv2.imread('20190103_171727A (40).jpg')
        frame = cv2.resize(frame, (720, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.file_extension = ''
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.display)

        self.file = ''

        self.start_button = QPushButton('С начала')
        self.start_button.setFixedSize(QSize(150, 30))
        self.start_button.setStyleSheet('QPushButton {'
                                        'border-radius: 15px;'
                                        'font-size: 14px;'
                                        'color: white;'
                                        'background-color: #303136;}')

        self.start_button.setDisabled(True)
        self.start_button.clicked.connect(lambda: self.setup_video(self.file))

        self.stop_button = QPushButton("Остановить")
        self.stop_button.setFixedSize(QSize(150, 30))
        self.stop_button.setStyleSheet('QPushButton {'
                                       'border-radius: 15px;'
                                       'font-size: 14px;'
                                       'color: white;'
                                       'background-color: #303136;}')

        self.stop_button.clicked.connect(self.stop_player)

        self.download_button = QPushButton('Скачать')
        self.download_button.setFixedSize(QSize(120, 30))
        self.download_button.setStyleSheet('QPushButton {'
                                           'border-radius: 15px;'
                                           'font-size: 12px;'
                                           'color: white;'
                                           'background-color: #303136;}')

        self.load_button = QPushButton('Загрузить файл')
        self.load_button.setFixedSize(QSize(700, 30))
        self.load_button.setStyleSheet('QPushButton {'
                                       'border-radius: 15px;'
                                       'font-size: 14px;'
                                       'color: white;'
                                       'background-color: #303136;}')

        self.load_button.clicked.connect(self.load_file)

        self.main_layout = QHBoxLayout()
        self.bar_button_widget = QWidget()
        self.bar_button_layout = QVBoxLayout(self.bar_button_widget)
        self.sidebar = QWidget()
        self.bar_layout = QVBoxLayout(self.sidebar)
        self.color_layout = QVBoxLayout()
        self.RGB_box_layout = QHBoxLayout()
        self.resize_layout = QVBoxLayout()
        self.size_params_layout = QHBoxLayout()
        self.confidence_layout = QVBoxLayout()
        self.name_layout = QVBoxLayout()
        self.path_layout = QVBoxLayout()
        self.player_layout = QVBoxLayout()
        self.pl_top_layout = QHBoxLayout()
        self.pl_bot_widget = QWidget()
        self.pl_bot_layout = QHBoxLayout(self.pl_bot_widget)
        self.player = QVBoxLayout()

        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.addWidget(self.bar_button_widget)
        self.bar_button_widget.setMinimumHeight(600)
        self.bar_button_widget.setFixedWidth(58)
        self.bar_button_widget.setStyleSheet('background-color: #131313; border-radius: 15px')
        self.bar_button_layout.addWidget(self.sidebar_button)
        self.bar_button_layout.setAlignment(self.sidebar_button, Qt.AlignCenter)
        self.bar_button_layout.addStretch()

        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addSpacing(10)
        self.sidebar.setVisible(False)
        self.sidebar.setFixedWidth(270)
        self.sidebar.setMinimumHeight(600)
        self.sidebar.setStyleSheet('background-color: #131313; border-radius: 15px')
        self.bar_layout.setContentsMargins(9, 9, 15, 15)
        self.bar_layout.setSpacing(20)

        self.bar_layout.addLayout(self.color_layout)
        self.color_layout.setSpacing(10)
        self.color_layout.addWidget(self.color_label)
        self.color_layout.addWidget(self.color_selector)
        self.color_layout.addWidget(self.color_mode)
        self.color_layout.addLayout(self.RGB_box_layout)
        self.RGB_box_layout.addWidget(self.Red)
        self.RGB_box_layout.addWidget(self.Green)
        self.RGB_box_layout.addWidget(self.Blue)

        self.bar_layout.addLayout(self.resize_layout)
        self.resize_layout.setSpacing(10)
        self.resize_layout.addWidget(self.resize_label)
        self.resize_layout.addWidget(self.size_selector)
        self.resize_layout.addWidget(self.resize_mode)
        self.resize_layout.addLayout(self.size_params_layout)
        self.size_params_layout.setSpacing(0)
        self.size_params_layout.addWidget(self.frame_width)
        self.size_params_layout.addWidget(self.frame_height)

        self.bar_layout.addLayout(self.confidence_layout)
        self.confidence_layout.setSpacing(0)
        self.confidence_layout.addWidget(self.conf_label)
        self.confidence_layout.addWidget(self.conf)

        self.bar_layout.addLayout(self.name_layout)
        self.name_layout.setSpacing(0)
        self.name_layout.addWidget(self.name_label)
        self.name_layout.addWidget(self.obj_name)

        self.bar_layout.addLayout(self.path_layout)
        self.path_layout.setSpacing(10)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path)
        self.path_layout.addWidget(self.browse_button)

        self.main_layout.addLayout(self.player_layout)
        self.player_layout.setSpacing(10)
        self.player_layout.addLayout(self.pl_top_layout)
        self.player_layout.addLayout(self.player)
        self.player_layout.setAlignment(self.player, Qt.AlignCenter)
        self.player_layout.addWidget(self.pl_bot_widget)
        self.pl_bot_widget.setVisible(False)
        self.pl_top_layout.addWidget(self.start_stream)
        self.pl_top_layout.addWidget(self.mode_selector)
        self.pl_top_layout.addWidget(self.nemo_button)

        self.player.addWidget(self.image_label)
        self.pl_bot_layout.addWidget(self.start_button)
        self.pl_bot_layout.addWidget(self.stop_button)
        self.pl_bot_layout.setContentsMargins(0, 0, 0, 0)

        self.player_layout.addWidget(self.load_button)
        self.player_layout.setAlignment(self.load_button, Qt.AlignCenter)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setMenuWidget(central_widget)

    def toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.bar_layout.removeWidget(self.sidebar_button)
            self.bar_button_layout.insertWidget(0, self.sidebar_button)
            self.bar_button_widget.setVisible(True)
            self.sidebar.setVisible(False)
            return self.setFixedSize(830, 620)

        self.bar_button_layout.removeWidget(self.sidebar_button)
        self.bar_layout.insertWidget(0, self.sidebar_button)
        self.sidebar.setVisible(True)
        self.bar_button_widget.setVisible(False)
        return self.setFixedSize(1030, 620)

    def color_mode_check(self):
        if self.color_mode.isChecked():
            self.Red.setDisabled(False)
            self.Green.setDisabled(False)
            self.Blue.setDisabled(False)
            return self.color_selector.setDisabled(True)

        self.Red.setDisabled(True)
        self.Green.setDisabled(True)
        self.Blue.setDisabled(True)
        return self.color_selector.setDisabled(False)

    def resize_mode_check(self):
        if self.resize_mode.isChecked():
            self.frame_width.setDisabled(False)
            self.frame_height.setDisabled(False)
            return self.size_selector.setDisabled(True)

        self.frame_width.setDisabled(True)
        self.frame_height.setDisabled(True)
        return self.size_selector.setDisabled(False)

    def browse_path(self):
        directory = QFileDialog.getExistingDirectory(self, 'Выберите директорию')
        if directory:
            self.path.setText(directory)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', filter='*jpg *jpeg *mp4')
        extension = file_path.split('.')[-1]

        if not extension:
            return None

        if self.file_extension:
            self.stop_player()

        self.start_button.setDisabled(False)
        self.file = file_path
        self.file_extension = extension

        match self.file_extension:
            case 'jpg':
                print('jpg')
                self.pl_bot_widget.setVisible(False)
                return self.setup_photo(file_path)

            case 'jpeg':
                print('jpeg')
                self.pl_bot_widget.setVisible(False)
                return self.setup_photo(file_path)

            case 'mp4':
                print('mp4')
                self.pl_bot_widget.setVisible(True)
                return self.setup_video(file_path)

    def setup_photo(self, file_dic):
        self.capture = cv2.VideoCapture(file_dic)
        return self.display()

    def setup_video(self, file_dic):
        if isinstance(file_dic, int):
            self.file_extension = 'stream'
        self.capture = cv2.VideoCapture(file_dic)
        self.timer.start(30)

    def display(self):
        ret, frame = self.capture.read()

        if not ret:
            return self.timer.stop()

        frame = cv2.resize(frame, (720, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def stop_player(self):
        self.timer.stop()
        self.capture.release()

    def get_data(self):
        if self.color_mode.isChecked():
            color = (self.Blue.value(), self.Green.value(), self.Red.value())
        else:
            color = self.color.get(self.color_selector.currentText())

        if self.resize_mode.isChecked():
            width, height = self.frame_width.value(), self.frame_height.value()
        else:
            width, height = map(int, self.size_selector.currentText().split('x'))

        conf = self.conf.value()
        name = self.obj_name.text()
        directory = self.path.text()

        if self.file_extension in ['jpg', 'jpeg']:
            return self.process_photo(color, width, height, conf, name, directory)

        elif self.file_extension == 'mp4':
            return self.process_video(color, width, height, conf, name, directory)

        elif self.file_extension == 'stream':
            return self.process_stream(color, conf, name)

        return None

    def process_photo(self, color, width, height, conf, name, directory):
        img = cv2.imread(self.file)
        frame = self.yolo_predictions(img, net, color, conf, name)[0]
        frame = cv2.resize(frame, (720, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        file_name = self.file.split('/')[-1]
        outfile = f'{directory}/NEMO_{file_name}'

        return cv2.imwrite(outfile, frame)

    def process_video(self, color, width, height, conf, name, directory):
        cap = cv2.VideoCapture(self.file)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        file_name = self.file.split('/')[-1]
        outfile = f'{directory}/NEMO_{file_name}'
        out = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
        count = 0
        ind = set()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results, index, last_x, last_y, buf = self.yolo_predictions(frame, net, color, conf, name)
            cv2.rectangle(results, (width - 300, 20), (width - 80, 172), (255, 255, 255), -1)

            if len(index) != 0:
                count += buf
                for val in index:
                    ind.add(val)

            cv2.putText(results, str(len(ind)), (width - 310, 165), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 3)

            out.write(results)

        cap.release()
        out.release()
        info = QMessageBox()
        info.setWindowTitle('NEMO')
        info.setText(f'Видео обработано и сохранено в директории: {directory}!')
        info.setIcon(QMessageBox.Icon.Information)
        info.exec()

    def process_stream(self, color, conf, name):
        cv2.VideoCapture(self.mode_selector.currentIndex())
        while True:
            ret, frame = self.capture.read()

            if not ret:
                return None

            frame = self.yolo_predictions(frame, net, color, conf, name)[0]
            frame = cv2.resize(frame, (720, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))

    @staticmethod
    def get_detections(img, nn):
        image = img.copy()
        row, col, d = image.shape

        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (640, 640), swapRB=True, crop=False)
        nn.setInput(blob)
        preds = nn.forward()
        detections = preds[0]

        return input_image, detections

    @staticmethod
    def non_maximum_supression(input_image, detections, conf):
        boxes = []
        confidences = []

        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / 640
        y_factor = image_h / 640

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > conf:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.8, 0.25)

        return boxes_np, confidences_np, index

    @staticmethod
    def drawings(image, boxes_np, confidences_np, index, color, name):
        x, y = last_x, last_y
        c = 0

        for ind in index:
            x, y, w, h = boxes_np[ind]
            bb_conf = confidences_np[ind]
            conf_text = '{}: {:.0f}%'.format(name, bb_conf * 100)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv2.rectangle(image, (x, y - 45), (x + w, y), color, -1)
            cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

            if abs(last_x - x) > 20 and abs(last_y - y) > 20:
                c += 1

        return image, x, y, c

    def yolo_predictions(self, img, nn, color, conf, name):
        input_image, detections = self.get_detections(img, nn)
        boxes_np, confidences_np, index = self.non_maximum_supression(input_image, detections, conf)
        result_img, x, y, c = self.drawings(img, boxes_np, confidences_np, index, color, name)
        return result_img, index, x, y, c


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NEMO()
    win.show()
    sys.exit(app.exec())
