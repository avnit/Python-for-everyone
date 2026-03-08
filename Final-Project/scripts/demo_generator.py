"""
Demo Image Generator
====================
Generates sample financial document images for the OCR demo.

Creates two types of images:
1. Financial report / invoice (text-heavy document)
2. Stock performance chart with annotations

Run this script once to create demo images:
    python demo_generator.py

Images saved in the 'data/demo_images/' subfolder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import random


DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_images")


def create_demo_dir():
    os.makedirs(DEMO_DIR, exist_ok=True)
    print(f"[Demo] Output directory: {DEMO_DIR}")


def generate_financial_report_image():
    """
    Generate a fake financial report document as a PNG image.
    This simulates what OCR would read from a real financial document.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Company header
    ax.text(5, 13.3, 'ACME FINANCIAL CORP', fontsize=22, fontweight='bold',
            ha='center', va='center', color='#1a237e',
            fontfamily='DejaVu Sans')
    ax.text(5, 12.9, 'Quarterly Financial Report - Q4 2025', fontsize=14,
            ha='center', va='center', color='#37474f')
    ax.text(5, 12.55, 'Fiscal Year: January 1, 2025 - December 31, 2025',
            fontsize=10, ha='center', va='center', color='#78909c')

    # Divider line
    ax.axhline(y=12.3, xmin=0.05, xmax=0.95, color='#1a237e', linewidth=2)

    # Revenue Section
    ax.text(0.5, 11.9, 'REVENUE SUMMARY', fontsize=13, fontweight='bold',
            va='center', color='#1a237e')
    ax.axhline(y=11.65, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)

    revenue_items = [
        ('Total Revenue', '$12,450,000', '+8.3%'),
        ('Operating Income', '$3,120,000', '+12.1%'),
        ('Net Income', '$2,345,000', '+15.7%'),
        ('Earnings Per Share (EPS)', '$4.82', '+14.2%'),
        ('EBITDA', '$4,670,000', '+9.8%'),
    ]

    y = 11.4
    ax.text(0.6, y + 0.1, 'Item', fontsize=10, fontweight='bold', color='#455a64')
    ax.text(6.5, y + 0.1, 'Amount', fontsize=10, fontweight='bold', color='#455a64')
    ax.text(8.8, y + 0.1, 'YoY Change', fontsize=10, fontweight='bold', color='#455a64')
    y -= 0.2

    for i, (item, amount, change) in enumerate(revenue_items):
        bg_color = '#E8EAF6' if i % 2 == 0 else '#FAFAFA'
        rect = mpatches.FancyBboxPatch((0.4, y - 0.15), 9.2, 0.35,
                                        boxstyle="square,pad=0.02",
                                        facecolor=bg_color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(0.6, y + 0.02, item, fontsize=10, va='center', color='#263238')
        ax.text(6.5, y + 0.02, amount, fontsize=10, va='center',
                color='#1a237e', fontweight='bold')
        change_color = '#2e7d32' if '+' in change else '#c62828'
        ax.text(8.8, y + 0.02, change, fontsize=10, va='center', color=change_color,
                fontweight='bold')
        y -= 0.42

    # Stock Performance Section
    ax.text(0.5, y - 0.1, 'STOCK PERFORMANCE', fontsize=13, fontweight='bold',
            va='center', color='#1a237e')
    y -= 0.5
    ax.axhline(y=y + 0.1, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)

    stock_items = [
        ('Stock Ticker', 'ACME'),
        ('Current Price', '$187.45'),
        ('52-Week High', '$215.30'),
        ('52-Week Low', '$142.80'),
        ('Market Cap', '$45.2 Billion'),
        ('P/E Ratio', '22.4'),
        ('Dividend Yield', '1.8%'),
        ('Beta', '1.12'),
    ]

    y -= 0.15
    for i, (label, value) in enumerate(stock_items):
        col = i % 2
        row = i // 2
        x_label = 0.6 if col == 0 else 5.5
        x_value = 3.0 if col == 0 else 8.5
        y_pos = y - row * 0.42

        ax.text(x_label, y_pos, label + ':', fontsize=10, va='center', color='#546e7a')
        ax.text(x_value, y_pos, value, fontsize=10, va='center',
                color='#1a237e', fontweight='bold', ha='right')

    y -= (len(stock_items) // 2) * 0.42 + 0.5

    # Key Highlights
    ax.text(0.5, y, 'KEY HIGHLIGHTS', fontsize=13, fontweight='bold',
            va='center', color='#1a237e')
    y -= 0.4
    ax.axhline(y=y + 0.1, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)

    highlights = [
        '• Record quarterly revenue of $3.2 billion, up 8.3% year-over-year',
        '• Expanded gross margin to 42.5%, driven by product mix shift',
        '• Cloud division grew 34% to $890 million in quarterly revenue',
        '• Returned $450 million to shareholders via buybacks and dividends',
        '• Raised full-year 2026 guidance: Revenue $13.2B - $13.6B',
    ]

    for highlight in highlights:
        y -= 0.38
        ax.text(0.6, y, highlight, fontsize=9.5, va='center', color='#37474f')

    # Footer
    y -= 0.6
    ax.axhline(y=y + 0.1, xmin=0.05, xmax=0.95, color='#1a237e', linewidth=1)
    ax.text(5, y - 0.15, 'CONFIDENTIAL - FOR INTERNAL USE ONLY', fontsize=8,
            ha='center', va='center', color='#90a4ae', style='italic')
    ax.text(5, y - 0.35, f'Generated: {datetime.now().strftime("%B %d, %Y")}',
            fontsize=8, ha='center', va='center', color='#90a4ae')

    output_path = os.path.join(DEMO_DIR, "financial_report.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"[Demo] Created financial report image: {output_path}")
    return output_path


def generate_stock_chart_image():
    """
    Generate a sample annotated stock chart as a PNG image.
    The chart will have text labels that OCR can extract.
    """
    # Generate fake stock price data
    np.random.seed(42)
    days = 252  # ~1 year of trading days
    dates = [datetime(2025, 1, 2) + timedelta(days=i * 1.4) for i in range(days)]
    prices = [150.0]
    for _ in range(days - 1):
        change = np.random.normal(0.0003, 0.015)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)

    # Simple moving average calculation
    def moving_avg(arr, n):
        result = np.full_like(arr, np.nan)
        for i in range(n - 1, len(arr)):
            result[i] = arr[i - n + 1:i + 1].mean()
        return result

    ma20 = moving_avg(prices, 20)
    ma50 = moving_avg(prices, 50)

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.1)

    # --- Price chart ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(range(days), prices, color='#1565C0', linewidth=1.5, label='ACME Price', zorder=3)
    ax1.plot(range(days), ma20, color='#FF6F00', linewidth=1.2, linestyle='--',
             label='MA(20)', alpha=0.8)
    ax1.plot(range(days), ma50, color='#2E7D32', linewidth=1.2, linestyle='--',
             label='MA(50)', alpha=0.8)

    # Annotations
    peak_idx = np.argmax(prices)
    low_idx = np.argmin(prices)
    ax1.annotate(f'52W HIGH: ${prices[peak_idx]:.2f}',
                xy=(peak_idx, prices[peak_idx]),
                xytext=(peak_idx - 30, prices[peak_idx] + 5),
                fontsize=9, color='#2E7D32', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

    ax1.annotate(f'52W LOW: ${prices[low_idx]:.2f}',
                xy=(low_idx, prices[low_idx]),
                xytext=(low_idx + 10, prices[low_idx] - 8),
                fontsize=9, color='#C62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

    ax1.set_title('ACME Financial Corp (ACME) | 1-Year Price Chart | NYSE',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('Price (USD $)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Price info text box
    info_text = (
        f"Current: ${prices[-1]:.2f}\n"
        f"Change: +${prices[-1]-prices[-2]:.2f} (+{((prices[-1]/prices[-2])-1)*100:.2f}%)\n"
        f"Volume: 4,235,891\n"
        f"Market Cap: $45.2B\n"
        f"P/E: 22.4 | Beta: 1.12"
    )
    ax1.text(0.02, 0.97, info_text, transform=ax1.transAxes,
             fontsize=8.5, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#90a4ae'))

    # --- Volume bars ---
    volumes = np.random.randint(2_000_000, 8_000_000, days)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    colors = ['#2e7d32' if prices[i] >= prices[i - 1] else '#c62828' for i in range(days)]
    ax2.bar(range(days), volumes / 1e6, color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volume (M)')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.grid(True, alpha=0.3)

    # --- RSI ---
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = moving_avg(gain, 14)
    avg_loss = moving_avg(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(range(days), rsi, color='#7B1FA2', linewidth=1.2, label='RSI(14)')
    ax3.axhline(y=70, color='#C62828', linestyle='--', alpha=0.7, linewidth=0.8)
    ax3.axhline(y=30, color='#2E7D32', linestyle='--', alpha=0.7, linewidth=0.8)
    ax3.fill_between(range(days), rsi, 70, where=(rsi > 70),
                     alpha=0.3, color='#C62828', label='Overbought')
    ax3.fill_between(range(days), rsi, 30, where=(rsi < 30),
                     alpha=0.3, color='#2E7D32', label='Oversold')
    ax3.text(days - 1, 72, 'Overbought (70)', fontsize=7.5, color='#C62828', ha='right')
    ax3.text(days - 1, 28, 'Oversold (30)', fontsize=7.5, color='#2E7D32', ha='right',
             va='top')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Trading Days (Jan 2025 - Dec 2025)')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    output_path = os.path.join(DEMO_DIR, "stock_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Demo] Created stock chart image: {output_path}")
    return output_path


def generate_invoice_image():
    """
    Generate a simple financial invoice image for OCR demonstration.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Header
    ax.text(4.25, 10.5, 'INVOICE', fontsize=28, fontweight='bold',
            ha='center', color='#1a237e')
    ax.text(0.5, 10.5, 'TechFinance Solutions LLC', fontsize=11, fontweight='bold',
            color='#37474f')
    ax.text(0.5, 10.2, '1234 Market Street, Suite 500', fontsize=9, color='#78909c')
    ax.text(0.5, 10.0, 'New York, NY 10005', fontsize=9, color='#78909c')
    ax.text(0.5, 9.8, 'Tel: (212) 555-0100 | info@techfinance.com', fontsize=9,
            color='#78909c')

    ax.text(5.5, 10.2, f'Invoice #: INV-2025-0847', fontsize=10, fontweight='bold',
            color='#1a237e')
    ax.text(5.5, 9.95, f'Date: December 15, 2025', fontsize=9, color='#37474f')
    ax.text(5.5, 9.70, f'Due Date: January 15, 2026', fontsize=9, color='#c62828')

    ax.axhline(y=9.5, xmin=0.05, xmax=0.95, color='#1a237e', linewidth=2)

    # Bill To
    ax.text(0.5, 9.2, 'BILL TO:', fontsize=10, fontweight='bold', color='#546e7a')
    ax.text(0.5, 8.95, 'Acme Investment Partners', fontsize=10, color='#263238')
    ax.text(0.5, 8.72, 'Attn: Robert Johnson, CFO', fontsize=9, color='#37474f')
    ax.text(0.5, 8.49, '500 Wall Street, Floor 12', fontsize=9, color='#37474f')
    ax.text(0.5, 8.26, 'New York, NY 10005', fontsize=9, color='#37474f')

    # Table Header
    ax.axhline(y=7.9, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)
    rect = mpatches.Rectangle((0.4, 7.7), 7.7, 0.35, facecolor='#1a237e', edgecolor='none')
    ax.add_patch(rect)
    ax.text(0.6, 7.87, 'DESCRIPTION', fontsize=9, fontweight='bold', color='white', va='center')
    ax.text(4.5, 7.87, 'QTY', fontsize=9, fontweight='bold', color='white', ha='center', va='center')
    ax.text(5.8, 7.87, 'UNIT PRICE', fontsize=9, fontweight='bold', color='white', ha='center', va='center')
    ax.text(7.8, 7.87, 'AMOUNT', fontsize=9, fontweight='bold', color='white', ha='right', va='center')

    items = [
        ('Financial Analytics Dashboard (Annual License)', '1', '$8,500.00', '$8,500.00'),
        ('ML Model Training & Deployment Service', '3', '$2,200.00', '$6,600.00'),
        ('OCR Document Processing - 10K pages/mo', '6 mo', '$450.00', '$2,700.00'),
        ('Private LLM Server Setup & Configuration', '1', '$3,500.00', '$3,500.00'),
        ('Priority Support Package (Annual)', '1', '$1,200.00', '$1,200.00'),
        ('Training Sessions (4 hours)', '2', '$750.00', '$1,500.00'),
    ]

    y = 7.5
    for i, (desc, qty, unit, amount) in enumerate(items):
        bg_color = '#E8EAF6' if i % 2 == 0 else 'white'
        rect = mpatches.Rectangle((0.4, y - 0.22), 7.7, 0.35, facecolor=bg_color, edgecolor='none')
        ax.add_patch(rect)
        ax.text(0.6, y - 0.04, desc, fontsize=8.5, va='center', color='#263238')
        ax.text(4.5, y - 0.04, qty, fontsize=8.5, ha='center', va='center', color='#37474f')
        ax.text(5.8, y - 0.04, unit, fontsize=8.5, ha='center', va='center', color='#37474f')
        ax.text(7.8, y - 0.04, amount, fontsize=8.5, ha='right', va='center',
                color='#1a237e', fontweight='bold')
        y -= 0.38

    ax.axhline(y=y - 0.1, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)

    # Totals
    y -= 0.35
    subtotal = 24000.00
    tax = subtotal * 0.085
    total = subtotal + tax

    for label, amount, bold in [
        ('Subtotal:', f'${subtotal:,.2f}', False),
        ('Tax (8.5%):', f'${tax:,.2f}', False),
        ('TOTAL DUE:', f'${total:,.2f}', True),
    ]:
        ax.text(5.8, y, label, fontsize=10 if bold else 9,
                fontweight='bold' if bold else 'normal',
                color='#1a237e' if bold else '#37474f', va='center')
        ax.text(7.8, y, amount, fontsize=10 if bold else 9,
                fontweight='bold' if bold else 'normal', ha='right',
                color='#1a237e' if bold else '#263238', va='center')
        y -= 0.38

    # Payment info
    y -= 0.3
    ax.axhline(y=y + 0.1, xmin=0.05, xmax=0.95, color='#90a4ae', linewidth=0.5)
    ax.text(0.5, y - 0.2, 'PAYMENT INFORMATION', fontsize=10, fontweight='bold', color='#546e7a')
    ax.text(0.5, y - 0.45, 'Bank: First National Bank | Account: 1234567890 | Routing: 021000021',
            fontsize=8.5, color='#37474f')
    ax.text(0.5, y - 0.65, 'Payment Terms: Net 30 | Late fee: 1.5%/month after due date',
            fontsize=8.5, color='#78909c', style='italic')

    output_path = os.path.join(DEMO_DIR, "invoice.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Demo] Created invoice image: {output_path}")
    return output_path


def generate_all():
    """Generate all demo images."""
    create_demo_dir()
    print("\n[Demo] Generating sample images for OCR demonstration...\n")

    images = []
    images.append(generate_financial_report_image())
    images.append(generate_invoice_image())

    # Stock chart needs special handling (no pandas.rolling_mean_simulated)
    try:
        images.append(generate_stock_chart_image())
    except Exception as e:
        print(f"[Demo] Stock chart skipped: {e}")

    print(f"\n[Demo] {len(images)} demo images created in: {DEMO_DIR}")
    print("[Demo] You can now run the OCR module on these images!")
    return images


if __name__ == "__main__":
    generate_all()
