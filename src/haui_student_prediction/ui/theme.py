from __future__ import annotations

APP_STYLESHEET = """
QMainWindow, QDialog {
    background: #F0F4FA;
}

QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #111827;
    background: transparent;
}

QFrame#Sidebar {
    background: #0F1F3D;
    border: none;
}

QLabel#BrandTitle {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    background: transparent;
}

QLabel#BrandCaption {
    color: #8FA3C8;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    background: transparent;
}

QLabel#SidebarSection {
    color: #6B84B0;
    font-size: 10px;
    font-weight: 700;
    padding: 10px 10px 4px 10px;
    background: transparent;
}

QPushButton#NavButton {
    background: transparent;
    color: #C5D3EB;
    text-align: left;
    border: none;
    border-radius: 8px;
    padding: 11px 14px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton#NavButton:hover {
    background: #1B3262;
    color: #FFFFFF;
}

QPushButton#NavButton:checked {
    background: #2563EB;
    color: #FFFFFF;
    font-weight: 700;
}

QFrame#TopBar {
    background: #FFFFFF;
    border-bottom: 1.5px solid #D9E1EE;
}

QWidget#PageBody {
    background: #F0F4FA;
}

QScrollArea {
    background: #F0F4FA;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background: #F0F4FA;
}

QLabel#PageTitle {
    font-size: 22px;
    font-weight: 700;
    color: #0F1F3D;
    background: transparent;
}

QLabel#PageSubtitle {
    font-size: 12px;
    color: #6B7280;
    background: transparent;
}

QFrame#Surface {
    background: #FFFFFF;
    border: 1.5px solid #D9E1EE;
    border-radius: 12px;
}

QFrame#MetricCard {
    background: #FFFFFF;
    border: 1.5px solid #D9E1EE;
    border-radius: 12px;
}

QLabel#SectionTitle {
    font-size: 14px;
    font-weight: 700;
    color: #0F1F3D;
    background: transparent;
}

QLabel#MetricTitle {
    font-size: 11px;
    color: #6B7280;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: transparent;
}

QLabel#MetricValue {
    font-size: 26px;
    color: #2563EB;
    font-weight: 800;
    background: transparent;
}

QLabel#MetricValue[tone="blue"]   { color: #2563EB; }
QLabel#MetricValue[tone="green"]  { color: #16A34A; }
QLabel#MetricValue[tone="amber"]  { color: #D97706; }
QLabel#MetricValue[tone="red"]    { color: #DC2626; }
QLabel#MetricValue[tone="violet"] { color: #7C3AED; }

QLabel#MetricDetail {
    font-size: 11px;
    color: #6B7280;
    background: transparent;
}

QLabel {
    color: #111827;
    background: transparent;
}

QLabel#Muted {
    color: #6B7280;
    background: transparent;
}

QPushButton {
    background: #FFFFFF;
    color: #374151;
    border: 1.5px solid #CBD5E1;
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background: #F1F5F9;
    border-color: #94A3B8;
    color: #111827;
}

QPushButton:disabled {
    background: #F3F4F6;
    color: #9CA3AF;
    border-color: #E5E7EB;
}

QPushButton#PrimaryButton {
    background: #2563EB;
    color: #FFFFFF;
    border: 1.5px solid #2563EB;
}

QPushButton#PrimaryButton:hover {
    background: #1D4ED8;
    border-color: #1D4ED8;
}

QPushButton#PrimaryButton:disabled {
    background: #93C5FD;
    border-color: #93C5FD;
    color: #FFFFFF;
}

QPushButton#SuccessButton {
    background: #16A34A;
    color: #FFFFFF;
    border: 1.5px solid #16A34A;
}

QPushButton#SuccessButton:hover {
    background: #15803D;
    border-color: #15803D;
}

QPushButton#WarningButton {
    background: #D97706;
    color: #FFFFFF;
    border: 1.5px solid #D97706;
}

QPushButton#WarningButton:hover {
    background: #B45309;
    border-color: #B45309;
}

QPushButton#DangerButton {
    background: #DC2626;
    color: #FFFFFF;
    border: 1.5px solid #DC2626;
}

QPushButton#DangerButton:hover {
    background: #B91C1C;
    border-color: #B91C1C;
}

QLineEdit, QComboBox {
    background: #FFFFFF;
    color: #111827;
    border: 1.5px solid #CBD5E1;
    border-radius: 7px;
    padding: 8px 10px;
    min-height: 20px;
    selection-background-color: #DBEAFE;
    selection-color: #1E3A8A;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #2563EB;
    background: #FFFFFF;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}

QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #111827;
    border: 1.5px solid #CBD5E1;
    border-radius: 6px;
    selection-background-color: #EFF6FF;
    selection-color: #1D4ED8;
    padding: 4px;
}

QTableWidget {
    background: #FFFFFF;
    color: #111827;
    alternate-background-color: #F8FAFC;
    border: 1.5px solid #D9E1EE;
    border-radius: 8px;
    gridline-color: #E9EDF5;
    selection-background-color: #DBEAFE;
    selection-color: #1E3A8A;
    font-size: 12px;
}

QHeaderView::section {
    background: #F1F5FB;
    color: #374151;
    border: none;
    border-bottom: 2px solid #D9E1EE;
    padding: 10px 8px;
    font-weight: 700;
    font-size: 12px;
}

QTableWidget::item {
    padding: 6px 8px;
    color: #111827;
    border: none;
}

QTableWidget::item:selected {
    background: #DBEAFE;
    color: #1E3A8A;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background: #E5E7EB;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: #2563EB;
    border-radius: 4px;
}

QToolTip {
    background: #1E293B;
    color: #FFFFFF;
    border: none;
    padding: 6px 8px;
    border-radius: 6px;
    font-size: 12px;
}

QScrollBar:vertical {
    background: #F0F4FA;
    width: 8px;
    margin: 0;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #F0F4FA;
    height: 8px;
    margin: 0;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #CBD5E1;
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #94A3B8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

QMessageBox {
    background: #FFFFFF;
}

QMessageBox QLabel {
    color: #111827;
    background: transparent;
}
"""


def apply_to_app(app) -> None:
    app.setStyleSheet(APP_STYLESHEET)