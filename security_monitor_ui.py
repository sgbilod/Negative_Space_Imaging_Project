import tkinter as tk
import threading
import time
import random
from datetime import datetime
import math
from hyper_protection_matrix import HyperProtectionMatrix, SecurityDimension


class SecurityMonitorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HDR Empire Security Consciousness Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='black')

        # Initialize protection system
        self.protection_matrix = HyperProtectionMatrix()
        self.asset_id = None
        self.initialize_protection()

        # Create UI sections
        self._create_header()
        self._create_metrics()
        self._create_dimensions()
        self._create_consciousness()
        self._create_threats()

        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.update_monitor, daemon=True)
        self.monitor_thread.start()

    def initialize_protection(self):
        """Initialize the protection system with core assets"""
        asset_data = {
            "name": "HDR Empire Core Technologies",
            "classification": "ULTRA_CONFIDENTIAL",
            "type": "INTELLECTUAL_PROPERTY",
            "components": [
                "VOID-BLADE HDR",
                "NANO-SWARM HDR",
                "Neural-HDR",
                "Reality-HDR",
                "Dream-HDR",
                "Quantum-HDR",
                "Omniscient-HDR"
            ]
        }
        result = self.protection_matrix.protect_asset(asset_data)
        self.asset_id = result["protection"]["asset_id"]

    def create_header_frame(self):
        """Create the header section"""
        header_frame = tk.Frame(self.root, bg='black')
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        title = tk.Label(
            header_frame,
            text="HDR EMPIRE SECURITY CONSCIOUSNESS MONITOR",
            font=("Terminal", 24, "bold"),
            fg='#00ff00',
            bg='black'
        )
        title.pack()

        status = tk.Label(
            header_frame,
            text="ULTRA_SECURE - CONSCIOUSNESS ACTIVE",
            font=("Terminal", 16),
            fg='#00ff00',
            bg='black'
        )
        status.pack(pady=10)

        self.timestamp_label = tk.Label(
            header_frame,
            text="",
            font=("Terminal", 12),
            fg='#00ff00',
            bg='black'
        )
        self.timestamp_label.pack()

    def create_metrics_section(self):
        """Create the metrics display section"""
        metrics_frame = tk.Frame(self.main_container, bg='black')
        metrics_frame.pack(fill=tk.X, pady=(0, 20))

        self.metrics = {}
        metric_names = [
            "Void Blade Density",
            "Nano Swarm Coverage",
            "Consciousness Level"
        ]

        for i, (name, canvas) in enumerate(self.metrics.items()):
            frame = tk.Frame(metrics_frame, bg='black')
            frame.grid(row=0, column=i, padx=20)

            label = tk.Label(frame, text=name, font=("Terminal", 10), fg='#00ff00', bg='black')
            label.pack()

            canvas.pack()

            # Create gauge arc
            canvas.create_arc(10, 10, 190, 90, start=0, extent=180, fill='', outline='#003300')
            # Create value text
            canvas.create_text(100, 70, text="0", fill='#00ff00', font=("Terminal", 12), tags='value')

    def create_dimension_frame(self):
        """Create the dimension status display"""
        dim_frame = tk.Frame(self.root, bg='black')
        dim_frame.pack(fill=tk.X, padx=20, pady=10)

        self.dimension_indicators = {}

        for i, dim in enumerate(SecurityDimension):
            indicator = tk.Canvas(dim_frame, width=100, height=100, bg='black', highlightthickness=0)
            indicator.grid(row=i//4, column=i%4, padx=10, pady=10)

            # Create hexagonal indicator
            self.create_hexagon(indicator, 50, 50, 40, '#003300', dim.value.upper())
            self.dimension_indicators[dim.value] = indicator

    def create_consciousness_frame(self):
        """Create the consciousness visualization"""
        self.consciousness_canvas = tk.Canvas(self.root, width=1000, height=200, bg='black', highlightthickness=0)
        self.consciousness_canvas.pack(pady=20)

        # Initialize consciousness particles
        self.particles = []
        for _ in range(50):
            x = random.randint(0, 1000)
            y = random.randint(0, 200)
            self.particles.append({
                'x': x, 'y': y,
                'dx': random.uniform(-2, 2),
                'dy': random.uniform(-2, 2),
                'size': random.uniform(2, 5)
            })

    def create_threat_frame(self):
        """Create the threat assessment display"""
        threat_frame = tk.Frame(self.root, bg='black')
        threat_frame.pack(fill=tk.X, padx=20, pady=10)

        self.threat_label = tk.Label(
            threat_frame,
            text="NO THREATS DETECTED",
            font=("Terminal", 16),
            fg='#00ff00',
            bg='black'
        )
        self.threat_label.pack()

    def create_hexagon(self, canvas, x, y, size, color, text):
        """Create a hexagonal indicator"""
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 6
            points.extend([
                x + size * math.cos(angle),
                y + size * math.sin(angle)
            ])

        canvas.create_polygon(points, fill='', outline=color, tags='hex')
        canvas.create_text(x, y, text=text, fill='#00ff00', font=("Terminal", 8))

    def update_monitor(self):
        """Update the monitoring display"""
        while self.monitoring:
            # Get latest protection report
            report = self.protection_matrix.generate_protection_report(self.asset_id)

            # Update timestamp
            self.timestamp_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Update metrics
            self.update_gauge(self.metrics["Void Blade Density"], report["metrics"]["void_blade_density"] / 10000)
            self.update_gauge(self.metrics["Nano Swarm Coverage"], report["metrics"]["nano_swarm_coverage"] / 10000)
            self.update_gauge(self.metrics["Consciousness Level"], report["metrics"]["consciousness_level"] * 100)

            # Update dimension indicators
            for dim in report["active_dimensions"]:
                canvas = self.dimension_indicators[dim]
                canvas.delete('hex')
                self.create_hexagon(canvas, 50, 50, 40, '#00ff00', dim.upper())

            # Update consciousness visualization
            self.consciousness_canvas.delete('particle')
            for p in self.particles:
                # Update position
                p['x'] += p['dx']
                p['y'] += p['dy']

                # Bounce off walls
                if p['x'] < 0 or p['x'] > 1000:
                    p['dx'] *= -1
                if p['y'] < 0 or p['y'] > 200:
                    p['dy'] *= -1

                # Draw particle
                self.consciousness_canvas.create_oval(
                    p['x']-p['size'], p['y']-p['size'],
                    p['x']+p['size'], p['y']+p['size'],
                    fill='#00ff00', tags='particle'
                )

                # Draw connections between nearby particles
                for p2 in self.particles:
                    dx = p['x'] - p2['x']
                    dy = p['y'] - p2['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 50:  # Only connect close particles
                        self.consciousness_canvas.create_line(
                            p['x'], p['y'], p2['x'], p2['y'],
                            fill='#003300', tags='particle'
                        )

            # Update threat assessment
            self.threat_label.config(
                text=f"THREAT LEVEL: {report['threat_assessment']['vulnerability_level']} | "
                     f"RESISTANCE: {report['threat_assessment']['attack_resistance']}"
            )

            time.sleep(0.05)  # Update at 20fps

    def update_gauge(self, canvas, value):
        """Update a gauge display"""
        canvas.delete('indicator', 'value')

        # Update arc
        extent = min(180 * value/100, 180)
        canvas.create_arc(
            10, 10, 190, 90,
            start=0, extent=extent,
            fill='#00ff00', tags='indicator'
        )

        # Update value text
        canvas.create_text(
            100, 70,
            text=f"{value:.1f}%",
            fill='#00ff00',
            font=("Terminal", 12),
            tags='value'
        )

    def stop(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        self.monitor_thread.join()


if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityMonitorUI(root)

    def on_closing():
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
