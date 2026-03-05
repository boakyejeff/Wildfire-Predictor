# Wildfire Predictor (Konza Prairie Burn Detection)

Wildfire detection and prevention are critical for environmental sustainability and community safety. This project leverages advanced Machine Learning techniques with stacked multitemporal Landsat TM data to predict and analyze prairie burn events.

## Features
- **Data Preprocessing Pipelines:** Converts raw flat files into structured geographic fields (`Step1`) and builds normalized baseline data arrays (`Step6`).
- **Geospatial Visualizations:** Employs geographic visualization scripts to map stacked remote sensing datasets onto actual prairie topologies (`Step3` & `Step4`).
- **Baseline Machine Learning:** Tests multiple scikit-learn algorithms (e.g., K-Nearest Neighbors, Support Vector Classifiers, Logistic Regression, Decision Trees) to establish a baseline efficacy for burn predictions on raw and normalized data (`Step5` & `Step6`).
- **Production Scripts:** Previously scattered Jupyter Notebooks have been extracted into automated, PEP8-formatted Python scripts to allow for programmatic execution.

## Setup & Installation

It is recommended to run this repository in an isolated virtual environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/boakyejeff/Wildfire-Predictor.git
   cd Wildfire-Predictor
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Execute Pipeline Scripts (Examples):**
   ```bash
   python Step5_Stacked_BaseLine_ML.py
   python Step6_NormalizedData_BaseLine_ML.py
   ```

## Dependencies Highlights
Core analytical and geospatial libraries include:
- `pandas`
- `numpy`
- `geopandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## License
MIT License.
