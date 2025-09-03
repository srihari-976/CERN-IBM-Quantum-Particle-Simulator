# main.py

import sys
import os
import logging
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation import ParticleSimulation
from visualization import ParticleVisualizer
from scipy.constants import c
import numpy as np

def setup_logging():
    """Configure logging for the simulation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('particle_simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_sample_simulation():
    """Run a sample electron-positron collision simulation"""
    print("=" * 80)
    print("RESEARCH-GRADE PARTICLE PHYSICS SIMULATOR")
    print("CERN-Inspired Detector with IBM Quantum Integration")
    print("=" * 80)
    
    # Initialize simulation
    print("\n1. Initializing simulation engine...")
    simulation = ParticleSimulation(dt=1e-11)   # bigger step so particles travel farther
    simulation.create_electron_positron_collision_scenario(demo=True)
    
    # Run long enough so they definitely cross nano-layers
    simulation.run_simulation(5e-9)

    
    # Create electron-positron collision scenario
    print("\n2. Setting up electron-positron collision scenario...")
    particles = simulation.create_electron_positron_collision_scenario()
    
    print(f"   - Electron: position={particles['electron'].position.tolist()}, velocity={np.linalg.norm(particles['electron'].velocity)/c:.3f}c")
    print(f"   - Positron: position={particles['positron'].position.tolist()}, velocity={np.linalg.norm(particles['positron'].velocity)/c:.3f}c")
    print(f"   - Muon: position={particles['muon'].position.tolist()}, velocity={np.linalg.norm(particles['muon'].velocity)/c:.3f}c")
    
    # Run simulation
    print("\n3. Running relativistic particle simulation...")
    duration = 1e-12  # 1 picosecond
    event = simulation.run_simulation(duration)
    
    print(f"   - Simulation completed: {simulation.time:.2e} seconds")
    print(f"   - Detector events recorded: {len(simulation.detector.events)}")
    print(f"   - Total detector hits: {event['hit_count']}")
    
    # Quantum analysis
    print("\n4. Performing quantum effects analysis...")
    quantum_analysis = simulation.analyze_quantum_effects()
    
    for particle_name, analysis in quantum_analysis.items():
        print(f"\n   {particle_name.upper()} Quantum Properties:")
        print(f"   - de Broglie wavelength: {analysis['de_broglie_wavelength']:.2e} m")
        print(f"   - Compton wavelength: {analysis['compton_wavelength']:.2e} m")
        
        uncertainty = analysis['uncertainty_analysis']
        print(f"   - Heisenberg uncertainty: ΔxΔp = {uncertainty['uncertainty_product']:.2e}")
        print(f"   - Satisfies uncertainty principle: {uncertainty['satisfies_uncertainty']}")
        
        spin = analysis['spin_measurement']
        print(f"   - Quantum spin measurement: ↑{spin['spin_up_probability']:.3f}, ↓{spin['spin_down_probability']:.3f}")
    
    # Detector analysis
    print("\n5. Detector response analysis...")
    detector_summary = simulation.detector.get_event_summary()
    
    if not detector_summary.empty:
        total_energy_deposit = detector_summary['energy_deposit'].sum()
        print(f"   - Total energy deposited: {total_energy_deposit:.2e} J")
        print(f"   - Hits by material:")
        
        for material, group in detector_summary.groupby('layer_material'):
            energy_sum = group['energy_deposit'].sum()
            hit_count = len(group)
            print(f"     * {material}: {hit_count} hits, {energy_sum:.2e} J")
    else:
        print("   - No detector hits recorded")
    
    # Performance summary
    print(f"\n6. Simulation performance:")
    summary = simulation.get_simulation_summary()
    print(f"   - Total simulation steps: {summary['total_steps']}")
    print(f"   - Average particle speed: {summary['particles']['electron']['average_speed']/c:.3f}c (electron)")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("Launch web interface with: python app.py")
    print("=" * 80)
    
    return simulation

def run_web_interface():
    """Launch the Flask web interface"""
    print("\nLaunching web interface...")
    print("Access the simulator at: http://localhost:5000")
    
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)

def main():
    """Main execution function"""
    setup_logging()
    
    print("Research-Grade Particle Physics Simulator")
    print("Choose execution mode:")
    print("1. Run sample simulation (console)")
    print("2. Launch web interface")
    print("3. Run both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_sample_simulation()
        elif choice == '2':
            run_web_interface()
        elif choice == '3':
            run_sample_simulation()
            input("\nPress Enter to launch web interface...")
            run_web_interface()
        else:
            print("Invalid choice. Running sample simulation...")
            run_sample_simulation()
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())