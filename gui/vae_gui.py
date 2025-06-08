import sys
import os
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QFrame, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

latent_dim = 128
img_size = 128

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, 8, 8)
        return self.deconv(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class VAEApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAE Image Generator")
        # 更合理的初始尺寸，适配常见屏幕
        self.resize(1050, 650)
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)
        self.current_theme = "light"

        self.model = None
        self.model_name = ""
        self.input_image = None
        self.input_img_path = ""
        self.recon_image = None

        # 顶部栏
        self.model_name_bar = QLabel("")
        self.model_name_bar.setAlignment(Qt.AlignCenter)
        self.model_name_bar.setStyleSheet("""
            background: transparent;
            font-size: 15px;
            color: #3B86F7;
            font-weight: bold;
            padding: 6px;
        """)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.model_name_bar)
        top_bar.addStretch(1)
        self.theme_btn = QPushButton("☀ Theme")
        self.theme_btn.setMinimumWidth(80)
        self.theme_btn.setFont(QFont("Segoe UI", 13))
        self.theme_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: #E5E6EA;
                border-radius: 8px;
                font-size: 13px;
                color: #333;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background: #D1D2D6;
            }
        """)
        self.theme_btn.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.theme_btn)

        # 侧边栏
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setAlignment(Qt.AlignTop)
        side_layout.setContentsMargins(18, 24, 18, 20)
        app_title = QLabel("VAE Tool")
        app_title.setObjectName("sideLabel")
        app_title.setFont(QFont("Segoe UI", 17, QFont.Bold))
        side_layout.addWidget(app_title)
        side_layout.addSpacing(10)

        self.model_label = QLabel("No model loaded")
        self.model_label.setObjectName("modelLabel")
        self.model_label.setFont(QFont("Segoe UI", 10))
        self.model_label.setWordWrap(True)
        side_layout.addWidget(self.model_label)
        side_layout.addSpacing(15)

        button_font = QFont("Segoe UI", 12, QFont.Bold)
        button_style = """
            QPushButton {
                background-color: #E5E6EA;
                border-radius: 12px;
                padding: 10px 20px;
                font-weight: 600;
                color: #333;
                border: 1.2px solid #DEE0E7;
                margin: 7px 2px 7px 2px;
                font-size: 12px;
                min-width: 170px;
                max-width: 170px;
            }
            QPushButton:hover {
                background-color: #D1D2D6;
                border: 1.2px solid #BFC1C8;
            }
        """

        self.load_model_btn = QPushButton("Choose VAE Model (.pth)")
        self.load_model_btn.setFont(button_font)
        self.load_model_btn.setMinimumHeight(36)
        self.load_model_btn.setStyleSheet(button_style)
        self.load_model_btn.clicked.connect(self.load_model)
        side_layout.addWidget(self.load_model_btn)

        self.load_img_btn = QPushButton("Choose Image")
        self.load_img_btn.setFont(button_font)
        self.load_img_btn.setMinimumHeight(36)
        self.load_img_btn.setStyleSheet(button_style)
        self.load_img_btn.clicked.connect(self.load_image)
        side_layout.addWidget(self.load_img_btn)

        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setFont(button_font)
        self.gen_btn.setMinimumHeight(36)
        self.gen_btn.setStyleSheet(button_style)
        self.gen_btn.clicked.connect(self.generate_image)
        side_layout.addWidget(self.gen_btn)

        self.random_btn = QPushButton("Random Generate")
        self.random_btn.setFont(button_font)
        self.random_btn.setMinimumHeight(36)
        self.random_btn.setStyleSheet(button_style)
        self.random_btn.clicked.connect(self.random_generate)
        side_layout.addWidget(self.random_btn)

        side_layout.addItem(QSpacerItem(10, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 主区域（图片展示和信息）
        main_area = QVBoxLayout()
        main_area.setContentsMargins(12, 12, 12, 12)
        main_area.setSpacing(9)

        # 右上角 tip
        tip_row = QHBoxLayout()
        tip_row.addStretch(1)
        self.info_label = QLabel("Tip: Load a model and image, then click Generate!")
        self.info_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #3B86F7; margin-right: 10px;")
        tip_row.addWidget(self.info_label)
        main_area.addLayout(tip_row)

        img_row = QHBoxLayout()
        img_row.setSpacing(12)

        # 左图
        self.orig_img_title = QLabel("Original Image")
        self.orig_img_title.setObjectName("imgTitle")
        self.orig_img_title.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.orig_img_label = QLabel()
        self.orig_img_label.setAlignment(Qt.AlignCenter)
        self.orig_img_label.setMinimumSize(220, 220)
        self.orig_img_label.setMaximumSize(260, 260)
        self.orig_img_label.setStyleSheet("background: #FFF; border-radius: 12px; border: 1.2px solid #E2E3E9;")

        orig_panel = QVBoxLayout()
        orig_panel.setSpacing(7)
        orig_panel.addWidget(self.orig_img_title)
        orig_panel.addWidget(self.orig_img_label)

        # 右图
        self.recon_img_title = QLabel("Generated Image")
        self.recon_img_title.setObjectName("imgTitle")
        self.recon_img_title.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.recon_img_label = QLabel()
        self.recon_img_label.setAlignment(Qt.AlignCenter)
        self.recon_img_label.setMinimumSize(220, 220)
        self.recon_img_label.setMaximumSize(260, 260)
        self.recon_img_label.setStyleSheet("background: #FFF; border-radius: 12px; border: 1.2px solid #E2E3E9;")

        recon_panel = QVBoxLayout()
        recon_panel.setSpacing(7)
        recon_panel.addWidget(self.recon_img_title)
        recon_panel.addWidget(self.recon_img_label)

        img_row.addLayout(orig_panel, stretch=1)
        img_row.addSpacing(12)
        img_row.addLayout(recon_panel, stretch=1)

        main_area.addLayout(img_row)

        # 主布局
        all_layout = QHBoxLayout(self)
        all_layout.setContentsMargins(0, 0, 0, 0)
        all_layout.setSpacing(0)
        all_layout.addWidget(sidebar)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(top_bar)
        content_layout.addLayout(main_area)
        all_layout.addLayout(content_layout)

        self.apply_theme(self.current_theme)

    def apply_theme(self, theme):
        if theme == "light":
            self.setStyleSheet("""
                QWidget {
                    background-color: #F5F6FA;
                    font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif';
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #E5E6EA;
                    border-radius: 12px;
                    padding: 10px 20px;
                    font-weight: 600;
                    color: #333;
                    border: 1.2px solid #DEE0E7;
                    margin: 7px 2px 7px 2px;
                    font-size: 12px;
                    min-width: 170px;
                    max-width: 170px;
                }
                QPushButton:hover {
                    background-color: #D1D2D6;
                    border: 1.2px solid #BFC1C8;
                }
                QLabel#sideLabel {
                    font-size: 17px;
                    font-weight: 700;
                    color: #222;
                    padding: 10px 6px 7px 2px;
                }
                QFrame#sidebar {
                    background: #EBECEF;
                    border-radius: 16px;
                    min-width: 180px;
                    max-width: 250px;
                }
                QLabel#imgTitle {
                    font-size: 12px;
                    color: #555;
                    font-weight: 600;
                    margin-bottom: 5px;
                }
                QLabel#modelLabel {
                    font-size: 10px;
                    color: #3B86F7;
                    font-weight: 600;
                    margin-left: 2px;
                }
            """)
            self.theme_btn.setText("☀ Theme")
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #21232B;
                    font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif';
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #3A3B45;
                    border-radius: 12px;
                    padding: 10px 20px;
                    font-weight: 600;
                    color: #EEE;
                    border: 1.2px solid #373849;
                    margin: 7px 2px 7px 2px;
                    font-size: 12px;
                    min-width: 170px;
                    max-width: 170px;
                }
                QPushButton:hover {
                    background-color: #2D2E36;
                    border: 1.2px solid #64657B;
                }
                QLabel#sideLabel {
                    font-size: 17px;
                    font-weight: 700;
                    color: #F2F3FA;
                    padding: 10px 6px 7px 2px;
                }
                QFrame#sidebar {
                    background: #292B37;
                    border-radius: 16px;
                    min-width: 180px;
                    max-width: 250px;
                }
                QLabel#imgTitle {
                    font-size: 12px;
                    color: #B6B9CC;
                    font-weight: 600;
                    margin-bottom: 5px;
                }
                QLabel#modelLabel {
                    font-size: 10px;
                    color: #71A5FE;
                    font-weight: 600;
                    margin-left: 2px;
                }
            """)
            self.theme_btn.setText("🌙 Theme")

        # 图片区主题
        for label in [self.orig_img_label, self.recon_img_label]:
            if theme == "light":
                label.setStyleSheet("background: #FFF; border-radius: 12px; border: 1.2px solid #E2E3E9;")
            else:
                label.setStyleSheet("background: #23242C; border-radius: 12px; border: 1.2px solid #32334A;")

    def toggle_theme(self):
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme(self.current_theme)

    def load_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose model file", "", "PyTorch Model (*.pth)")
        if file:
            self.model = VAE()
            self.model.load_state_dict(torch.load(file, map_location='cpu'))
            self.model.eval()
            self.model_name = file.split("/")[-1]
            self.model_label.setText(f"Model loaded:\n{self.model_name}")
            self.model_name_bar.setText(f"Current Model: {self.model_name}")

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.bmp)")
        if file:
            self.input_img_path = file
            self.input_image = Image.open(file).convert('RGB')
            self.show_image(self.input_image, self.orig_img_label)

    def show_image(self, pil_img, label_widget):
        img = pil_img.resize((240, 240))
        img = np.array(img)
        if img.shape[2] == 4:
            img = img[..., :3]
        qimg = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label_widget.setPixmap(pixmap)

    def generate_image(self):
        if self.model is None or self.input_image is None:
            self.recon_img_label.setText("Please load model and image")
            return

        img = self.input_image.resize((img_size, img_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            recon, mu, logvar = self.model(img)
            recon_img = recon.squeeze().permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img * 255, 0, 255).astype(np.uint8)
            pil_recon = Image.fromarray(recon_img)
            self.recon_image = pil_recon
            self.show_image(pil_recon, self.recon_img_label)

    def random_generate(self):
        if self.model is None:
            self.recon_img_label.setText("Please load model")
            return
        with torch.no_grad():
            z = torch.randn(1, latent_dim)
            gen_img = self.model.decoder(z)
            gen_img = gen_img.squeeze().permute(1, 2, 0).numpy()
            gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
            pil_gen = Image.fromarray(gen_img)
            self.recon_image = pil_gen
            self.show_image(pil_gen, self.recon_img_label)

if __name__ == "__main__":
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 12)
    app.setFont(font)
    window = VAEApp()
    window.show()
    sys.exit(app.exec_())