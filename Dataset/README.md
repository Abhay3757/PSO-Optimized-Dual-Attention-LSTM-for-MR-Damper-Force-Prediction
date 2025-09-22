# MR Damper Dataset Documentation

## Dataset Overview

This directory contains experimental data from **Prototype MR Damper Characterization Tests Using White Noise** conducted by Prof. Shirley Dyke as part of the NEES (Network for Earthquake Engineering Simulation) program.

### 📋 Dataset Information

- **Title**: Prototype MR Damper Characterization Tests Using White Noise
- **Creator**: Shirley Dyke
- **DOI**: [10.4231/D3V11VM6X](https://doi.org/10.4231/D3V11VM6X)
- **Source**: [DesignSafe-CI](https://www.designsafe-ci.org/data/browser/public/nees.public/NEES-2012-1158)
- **Experiment**: NEES-2012-1158, Experiment-3
- **Data Collection Period**: January 1994 - August 1996
- **Type**: Experimental characterization data for MR damper modeling

### 🎯 Citation

```bibtex
@dataset{dyke1996prototype,
  title={Prototype MR Damper Characterization Tests Using White Noise},
  author={Dyke, Shirley},
  year={1996},
  publisher={DesignSafe-CI},
  doi={10.4231/D3V11VM6X},
  url={https://www.designsafe-ci.org/data/browser/public/nees.public/NEES-2012-1158}
}
```

## 🔬 Experimental Design

### Test Parameters
The experiments were conducted with systematic variations across multiple parameters:

- **Amplitudes**: 2 different displacement amplitudes
- **Frequency Ranges**: 2 different frequency bands (0-20 Hz, 0-50 Hz)
- **Voltage Ranges**: 2 different voltage levels (0-1V, 0-2.25V)
- **Trials**: 5 trials per configuration for statistical reliability

### Input-Output System
- **Inputs**: 
  - Displacement (hydraulic actuator command)
  - Voltage (to MR damper circuit)
- **Output**: 
  - Force (measured damper response)
- **System Type**: Two-input, single-output (TISO) dynamic system

## 📊 Data Structure

### File Naming Convention

| Filename | Duration | Frequency Range | Sampling Details |
|----------|----------|-----------------|------------------|
| `ranran_50_u` | 60 sec | 0-50 Hz | ffilter = 1500, fs = 3000 |
| `ransim` | 20 sec | Simulation | Simulation data |
| `ransim_u` | 20 sec | Simulation | fs = 3000, Y filtered at 20000Hz |
| `ranran_50` | 20 sec | 0-50 Hz | fs = 3000, Y filtered at 20000Hz |
| `ranran_20` | 40 sec | 0-20 Hz | fs = 3000, Y filtered at 20000Hz |

### Channel Descriptions

#### Unprocessed Data Files (Voltage Units)
1. **D1** - Main Displacement Measurement (V)
2. **F** - Force (V)
3. **A** - Acceleration of Actuator (V)
4. **U** - Command to Actuator (V)
5. **Y** - Voltage to MR Damper Circuit (V)
6. **D2** - Second Displacement Measurement (V)

#### Converted Data Files (Engineering Units)
1. **D** - Damper Displacement (cm)
2. **F** - Force (N)
3. **A** - Acceleration of Actuator (acceleration units)
4. **Y** - Voltage to MR Damper Circuit (V)
5. **V** - Calculated Velocity of MR Damper (cm/s)

## 🛠️ Data Acquisition System

### Hardware Configuration
- **Filters**: 8x Syminex XFM82 3-decade programmable antialiasing filters
  - Signal-to-noise ratio: 90 dB
  - Simultaneous sample and hold capability
- **A/D Conversion**: Analogic LSDAS-16-AC-mod2 data acquisition board
  - Resolution: 16-bit A/D converters
- **Timing**: Analogic CTRTM-05 counter-timer board
- **Computer**: Gateway 2000 P5-90
- **Software**: HEM Data Corporation Snap-Master

### Sampling Specifications
- **Sampling Frequency**: 3,000 Hz
- **Anti-aliasing Filter**: 100 Hz cutoff (all channels)
- **Filter Type**: Programmable anti-aliasing filters

## 🎲 Signal Generation

### White Noise Input Generation
The experimental inputs were generated using MATLAB with the following process:

1. **Base Signal**: 1000 Hz random signal (`raninp`)
2. **Filtering**: 8-pole elliptic filter (`ranfilt.m`) for frequency shaping
3. **Processing**: 
   - Exponentiation to ensure positive values
   - Scaling to appropriate voltage ranges (0-1V or 0-2.25V)
4. **Output**: Broadband excitation signals for system identification

### Frequency Content
- **Low Frequency**: 0-20 Hz (typical structural dynamics range)
- **High Frequency**: 0-50 Hz (extended dynamics range)
- **Filter Characteristics**: Elliptic design for sharp cutoff

## 📈 Dataset Statistics

### File in Use: `ranran_50_converted.csv`

This project uses the converted data file with the following characteristics:

- **Duration**: 20 seconds
- **Frequency Range**: 0-50 Hz
- **Sampling Rate**: 3,000 Hz
- **Total Samples**: ~60,000 samples
- **Channels**: 5 (D, F, A, Y, V)
- **Units**: Engineering units (cm, N, V, cm/s)

### Data Quality
- **Completeness**: No missing values
- **Signal-to-Noise Ratio**: High (90 dB acquisition system)
- **Calibration**: Factory-calibrated transducers
- **Filtering**: Proper anti-aliasing applied

## 🎯 Intended Use Cases

### Primary Applications
1. **MR Damper Modeling**: Develop mathematical models of MR damper dynamics
2. **System Identification**: Parameter estimation for control-oriented models
3. **Machine Learning**: Training data for neural networks and ML algorithms
4. **Validation Studies**: Benchmark data for model comparison
5. **Control Design**: Data for developing feedback control systems

### Research Areas
- **Structural Control**: Semi-active damping systems
- **Vibration Isolation**: Adaptive isolation systems
- **Automotive**: Suspension system modeling
- **Aerospace**: Landing gear and vibration control
- **Civil Engineering**: Seismic protection systems

## 🔍 Data Preprocessing Notes

### For Machine Learning Applications
The data in this project has been preprocessed as follows:

1. **Time Series Windows**: Created sliding windows for sequence modeling
2. **Normalization**: Standard scaling applied to features
3. **Train/Validation/Test Split**: 70/15/15 time-aware splitting
4. **Feature Selection**: 
   - **Inputs**: D (displacement), A (acceleration), Y (voltage), V (velocity)
   - **Target**: F (force)

### Physical Relationships
- **Force-Displacement**: Hysteretic behavior due to MR fluid properties
- **Force-Velocity**: Damping characteristics
- **Voltage Dependency**: Controllable damping via magnetic field

## ⚠️ Usage Considerations

### Data Limitations
1. **Prototype System**: Laboratory prototype, not production hardware
2. **Limited Operating Range**: Specific amplitude and frequency ranges tested
3. **Environmental Conditions**: Laboratory conditions (temperature, etc.)
4. **Aging Effects**: Data from 1994-1996, MR fluid properties may have evolved

### Recommended Practices
1. **Validation**: Always validate models on independent test data
2. **Scaling**: Consider scaling factors when applying to different damper sizes
3. **Operating Conditions**: Be aware of operating range limitations
4. **Physical Constraints**: Ensure model predictions are physically reasonable

## 📁 File Structure

```
Dataset/
├── ranran_50_converted.csv      # Main dataset used in this project
├── ranran_50_u_converted.csv    # Alternative dataset (60 sec duration)
├── ransim_u_converted.csv       # Simulation-based dataset
└── README.md                    # This documentation
```

## 🔗 Additional Resources

- **Detailed Signal Description**: `white_noise_signal_description.csv` (in original dataset)
- **DesignSafe-CI Portal**: [Full dataset access](https://www.designsafe-ci.org/data/browser/public/nees.public/NEES-2012-1158)
- **NEES Documentation**: Additional experimental details and metadata
- **Related Publications**: Search for "Shirley Dyke MR damper" for associated research papers

## 📞 Contact & Support

For questions about the dataset:
- **Original Data**: Contact DesignSafe-CI support
- **This Project Usage**: See main project README and documentation
- **Technical Issues**: Open an issue in the project repository

---

## 🏆 Acknowledgments

This research was supported by the Network for Earthquake Engineering Simulation (NEES) program. We acknowledge Prof. Shirley Dyke and the experimental team for providing this valuable dataset to the research community.

The data has been made publicly available through DesignSafe-CI, enabling reproducible research in structural control and MR damper modeling.

---

*Dataset README generated for MR Damper Force Prediction project*  
*Last updated: September 22, 2025*