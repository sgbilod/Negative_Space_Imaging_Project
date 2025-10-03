import tkinter as tk
import threading
import time
import random
from datetime import datetime
import math
from hyper_protection_matrix import HyperProtectionMatrix, SecurityDimension

class SecurityVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HDR Empire Security Consciousness Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='black')

        # Initialize protection system
        self.matrix = HyperProtectionMatrix()
        self.asset_id = self._initialize_protection()

        # Create main container
        self.container = tk.Frame(self.root, bg='black')
        self.container.pack(expand=True, fill='both', padx=20, pady=20)

        # Initialize UI components
        self._init_header()
        self._init_metrics()
        self._init_dimensions()
        self._init_consciousness()
        self._init_threats()

        # Start monitoring
        self.is_monitoring = True
        self.monitor = threading.Thread(target=self._update_display, daemon=True)
        self.monitor.start()

    def _initialize_protection(self):
        """Initialize protection for HDR Empire assets"""
        result = self.matrix.protect_asset({
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
        })
        return result["protection"]["asset_id"]

    def _init_header(self):
        """Initialize header display"""
        header = tk.Frame(self.container, bg='black')
        header.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            header,
            text="HDR EMPIRE SECURITY CONSCIOUSNESS MONITOR",
            font=("Terminal", 24, "bold"),
            fg='#00ff00',
            bg='black'
        ).pack()

        tk.Label(
            header,
            text="ULTRA_SECURE - CONSCIOUSNESS ACTIVE",
            font=("Terminal", 16),
            fg='#00ff00',
            bg='black'
        ).pack(pady=10)

        self.time_display = tk.Label(
            header,
            text="",
            font=("Terminal", 12),
            fg='#00ff00',
            bg='black'
        )
        self.time_display.pack()

    def _init_metrics(self):
        """Initialize metrics gauges"""
        metrics = tk.Frame(self.container, bg='black')
        metrics.pack(fill=tk.X, pady=(0, 20))

        self.gauges = {}

        for i, name in enumerate([
            "Void Blade Density",
            "Nano Swarm Coverage",
            "Consciousness Level"
        ]):
            frame = tk.Frame(metrics, bg='black')
            frame.grid(row=0, column=i, padx=20)

            tk.Label(
                frame,
                text=name,
                font=("Terminal", 10),
                fg='#00ff00',
                bg='black'
            ).pack(pady=(0, 5))

            canvas = tk.Canvas(
                frame,
                width=200,
                height=100,
                bg='black',
                highlightthickness=0
            )
            canvas.pack()

            canvas.create_arc(
                10, 10, 190, 90,
                start=0,
                extent=180,
                fill='',
                outline='#003300'
            )

            canvas.create_text(
                100, 70,
                text="0",
                fill='#00ff00',
                font=("Terminal", 12),
                tags='value'
            )

            self.gauges[name] = canvas

    def _init_dimensions(self):
        """Initialize dimension indicators"""
        dims = tk.Frame(self.container, bg='black')
        dims.pack(fill=tk.X, pady=(0, 20))

        self.indicators = {}

        for i, dim in enumerate(SecurityDimension):
            frame = tk.Frame(dims, bg='black')
            frame.grid(row=i//4, column=i%4, padx=10, pady=10)

            canvas = tk.Canvas(
                frame,
                width=100,
                height=100,
                bg='black',
                highlightthickness=0
            )
            canvas.pack()

            self._draw_hexagon(
                canvas,
                50, 50, 40,
                '#003300',
                dim.value.upper()
            )

            self.indicators[dim.value] = canvas

    def _init_consciousness(self):
        """Initialize consciousness visualization"""
        conscious = tk.Frame(self.container, bg='black')
        conscious.pack(fill=tk.X, pady=(0, 20))

        self.neural_net = tk.Canvas(
            conscious,
            width=1000,
            height=200,
            bg='black',
            highlightthickness=0
        )
        self.neural_net.pack()

        self.neurons = []
        for _ in range(50):
            self.neurons.append({
                'x': random.randint(0, 1000),
                'y': random.randint(0, 200),
                'dx': random.uniform(-2, 2),
                'dy': random.uniform(-2, 2),
                'size': random.uniform(2, 5)
            })

    def _init_threats(self):
        """Initialize threat display"""
        threats = tk.Frame(self.container, bg='black')
        threats.pack(fill=tk.X)

        self.threat_display = tk.Label(
            threats,
            text="NO THREATS DETECTED",
            font=("Terminal", 16),
            fg='#00ff00',
            bg='black'
        )
        self.threat_display.pack()

    def _draw_hexagon(self, canvas, x, y, size, color, text):
        """Draw a hexagonal indicator"""
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 6
            points.extend([
                x + size * math.cos(angle),
                y + size * math.sin(angle)
            ])

        canvas.create_polygon(
            points,
            fill='',
            outline=color,
            tags='hex'
        )

        canvas.create_text(
            x, y,
            text=text,
            fill='#00ff00',
            font=("Terminal", 8)
        )

    def _update_display(self):
        """Update all display elements"""
        while self.is_monitoring:
            try:
                # Get latest security report
                report = self.matrix.generate_protection_report(
                    self.asset_id
                )

                # Update timestamp
                self.time_display.config(
                    text=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

                # Update metric gauges
                self._update_gauge(
                    self.gauges["Void Blade Density"],
                    report["metrics"]["void_blade_density"] / 10000
                )
                self._update_gauge(
                    self.gauges["Nano Swarm Coverage"],
                    report["metrics"]["nano_swarm_coverage"] / 10000
                )
                self._update_gauge(
                    self.gauges["Consciousness Level"],
                    report["metrics"]["consciousness_level"] * 100
                )

                # Update dimension indicators
                for dim in report["active_dimensions"]:
                    canvas = self.indicators[dim]
                    canvas.delete('hex')
                    self._draw_hexagon(
                        canvas,
                        50, 50, 40,
                        '#00ff00',
                        dim.upper()
                    )

                # Update neural network visualization
                self._update_neural_net()

                # Update threat assessment
                self._update_threat_status(report["threat_assessment"])

            except Exception as e:
                print(f"Display update error: {e}")

            time.sleep(0.05)  # 20 FPS

    def _update_gauge(self, canvas, value):
        """Update a metric gauge"""
        canvas.delete('indicator', 'value')

        extent = min(180 * value/100, 180)
        canvas.create_arc(
            10, 10, 190, 90,
            start=0,
            extent=extent,
            fill='#00ff00',
            tags='indicator'
        )

        canvas.create_text(
            100, 70,
            text=f"{value:.1f}%",
            fill='#00ff00',
            font=("Terminal", 12),
            tags='value'
        )

    def _update_neural_net(self):
        """Update neural network visualization"""
        self.neural_net.delete('neuron')

        for n in self.neurons:
            # Update position
            n['x'] += n['dx']
            n['y'] += n['dy']

            # Bounce at boundaries
            if n['x'] < 0 or n['x'] > 1000:
                n['dx'] *= -1
            if n['y'] < 0 or n['y'] > 200:
                n['dy'] *= -1

            # Draw neuron
            self.neural_net.create_oval(
                n['x']-n['size'],
                n['y']-n['size'],
                n['x']+n['size'],
                n['y']+n['size'],
                fill='#00ff00',
                tags='neuron'
            )

            # Draw connections
            for n2 in self.neurons:
                dx = n['x'] - n2['x']
                dy = n['y'] - n2['y']
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 50:  # Connect nearby neurons
                    self.neural_net.create_line(
                        n['x'], n['y'],
                        n2['x'], n2['y'],
                        fill='#003300',
                        tags='neuron'
                    )

    def _update_threat_status(self, assessment):
        """Update threat status display"""
        self.threat_display.config(
            text=(
                f"THREAT LEVEL: {assessment['vulnerability_level']} | "
                f"RESISTANCE: {assessment['attack_resistance']}"
            )
        )

    def start(self):
        """Start the security monitor"""
        def on_closing():
            self.is_monitoring = False
            self.monitor.join()
            self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    monitor = SecurityVisualizer()
    monitor.start()
