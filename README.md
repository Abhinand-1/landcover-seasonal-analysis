# Multi-Season Land Cover Classification & Seasonal Vegetation Analysis

## Overview

This project combines **GIS-based remote sensing analysis, seasonal vegetation assessment, feature engineering, and Random Forest machine learning** to analyse land-cover and vegetation dynamics in the Palakkad region of Kerala, India.

The analysis uses **Sentinel-2 Surface Reflectance Harmonized imagery** and Google Earth Engine to derive spectral and vegetation features across different seasons. These features are then used to train and evaluate a **Random Forest classifier** for five land-cover classes.

The project consists of two connected components:

1. **GIS & Seasonal Vegetation Analysis**
2. **Random Forest Land-Cover Classification**

The workflow demonstrates how multi-season satellite data can be used to distinguish persistent vegetation such as forest and agroforestry from seasonally varying cropland.

---

## Objectives

- Analyse seasonal vegetation behaviour using Sentinel-2 imagery.
- Calculate vegetation, water and moisture indices.
- Identify areas with strong seasonal vegetation variability.
- Develop a multi-season feature stack for machine learning.
- Classify land cover using Random Forest.
- Evaluate classification performance using validation data.
- Analyse feature importance and class-wise performance.
- Generate a spatial land-cover classification map.

---

## Study Area

The study focuses on a sub-region of **Palakkad District, Kerala, India**.

The selected study area covers approximately **1,270 km²** and contains diverse landscapes, including:

- Dense forest
- Agroforestry / tree-outside-forest
- Agricultural and paddy areas
- Built-up areas
- Water bodies

The study area was selected to represent all five target land-cover classes.

---

## Data & Technologies

### Satellite Data

**Sentinel-2 Surface Reflectance Harmonized**

Google Earth Engine dataset:

```text
COPERNICUS/S2_SR_HARMONIZED
