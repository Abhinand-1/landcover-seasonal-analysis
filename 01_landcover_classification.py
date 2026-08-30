"""
Palakkad (Kerala) Land-Cover Classification — Python / earthengine-api port
============================================================================
Classes: 1 Dense Forest | 2 Agroforestry / Tree-outside-Forest |
         3 Cropland | 4 Built-up | 5 Water
Sensor : Sentinel-2 SR Harmonized, two seasons (post-monsoon + summer)
Classifier: Random Forest (smileRandomForest)

Setup
-----
    pip install earthengine-api geemap
    earthengine authenticate          # one-time, opens a browser
    # then edit EE_PROJECT below to your GCP/EE project id

AOI note: full Palakkad district is ~4,500 sq km, above the 500-1,500 sq km
brief. This uses a ~1,270 sq km sub-region (Malampuzha reservoir, Palakkad
city, the Palakkad Gap paddy belt, Walayar/Western Ghats foothill forest,
and rural agroforestry) so all five classes appear in one frame.
"""

import ee

EE_PROJECT = "your-gcp-project-id"  # <-- set this
ee.Initialize(project=EE_PROJECT)

# ---------------------------------------------------------------------------
# 0. AOI
# ---------------------------------------------------------------------------
AOI_RECT = ee.Geometry.Rectangle([76.55, 10.65, 76.90, 10.95], None, False)

# OPTIONAL: clip to the true district boundary if you have the geojson.
# import json
# with open("palakkad_district.geojson") as f:
#     district_geojson = json.load(f)
# district = ee.FeatureCollection(district_geojson)
# AOI_RECT = AOI_RECT.intersection(district.geometry(), 30)

aoi = AOI_RECT

# ---------------------------------------------------------------------------
# 1. SENTINEL-2 SR: CLOUD MASK + TWO SEASONAL COMPOSITES
# ---------------------------------------------------------------------------
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def mask_s2_sr(img):
    scl = img.select("SCL")
    good_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return (
        img.updateMask(good_mask)
        .select(S2_BANDS)
        .divide(10000)
        .copyProperties(img, ["system:time_start"])
    )


def seasonal_composite(start_date, end_date):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_s2_sr)
        .median()
        .clip(aoi)
    )


# Post-monsoon: peak greenness, minimal cloud in Kerala.
post_monsoon = seasonal_composite("2023-10-01", "2023-12-31")
# Summer / dry season: exposes seasonal cropland vs evergreen forest.
summer = seasonal_composite("2024-02-01", "2024-04-30")

# ---------------------------------------------------------------------------
# 2. SPECTRAL INDICES + TEXTURE + SEASONAL-VARIABILITY METRIC
# ---------------------------------------------------------------------------


def add_indices(img, tag):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{tag}")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename(f"EVI_{tag}")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename(f"NDWI_{tag}")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename(f"NDBI_{tag}")
    return img.addBands([ndvi, evi, ndwi, ndbi])


post_monsoon = add_indices(post_monsoon, "pm")
summer = add_indices(summer, "sm")

# GLCM texture (contrast) on post-monsoon NIR -- separates the coarse,
# heterogeneous canopy texture of natural/mixed forest from the smoother
# texture of monoculture cropland or plantations.
nir_int = post_monsoon.select("B8").multiply(255).toByte()
glcm = nir_int.glcmTexture(size=3)
texture = glcm.select("B8_contrast").rename("NIR_contrast_pm")

# Seasonal NDVI variability -- evergreen forest stays high & stable across
# seasons; cropland and deciduous agroforestry swing more.
ndvi_seasonal_diff = (
    post_monsoon.select("NDVI_pm")
    .subtract(summer.select("NDVI_sm"))
    .abs()
    .rename("NDVI_seasonal_diff")
)

# ---------------------------------------------------------------------------
# 3. FEATURE STACK
# ---------------------------------------------------------------------------
feature_stack = (
    post_monsoon.select(S2_BANDS)
    .addBands(post_monsoon.select(["NDVI_pm", "EVI_pm", "NDWI_pm", "NDBI_pm"]))
    .addBands(summer.select(["NDVI_sm", "EVI_sm", "NDWI_sm", "NDBI_sm"]))
    .addBands(texture)
    .addBands(ndvi_seasonal_diff)
    .clip(aoi)
)

band_names = feature_stack.bandNames()
print("Feature stack bands:", band_names.getInfo())

# ---------------------------------------------------------------------------
# 4. TRAINING / VALIDATION LABELS
# ---------------------------------------------------------------------------
# Class codes:
#   1 = Dense forest      2 = Agroforestry / TOF      3 = Cropland
#   4 = Built-up          5 = Water
#
# Auto-generates a *stratified starting sample* from ESA WorldCover 2021
# (10 m) + Hansen Global Forest Change tree-cover %. TREAT THESE AS A FIRST
# DRAFT: export `training_pts` (see below), open it in QGIS / geemap next to
# high-res basemap imagery, and manually inspect/relabel/add points by eye
# -- especially to separate class 1 (dense, contiguous canopy, high NDVI,
# low seasonal variability) from class 2 (scattered/homestead trees mixed
# with crops or built structures). This manual QA step cannot be automated.

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
tree_cover_pct = (
    ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    .select("treecover2000")
    .clip(aoi)
)

wc_tree = worldcover.eq(10)
wc_cropland = worldcover.eq(40)
wc_builtup = worldcover.eq(50)
wc_water = worldcover.eq(80).Or(worldcover.eq(90))

dense_forest = wc_tree.And(tree_cover_pct.gte(70))
agroforestry_from_tree = wc_tree.And(tree_cover_pct.lt(70))
# Cropland pixels with scattered tree canopy (>10%) are also agroforestry
# (common Kerala homestead-garden / crop-tree mosaic).
agroforestry_from_crop = wc_cropland.And(tree_cover_pct.gte(10)).And(
    tree_cover_pct.lt(40)
)
pure_cropland = wc_cropland.And(tree_cover_pct.lt(10))

label_image = (
    ee.Image(0)
    .where(dense_forest, 1)
    .where(agroforestry_from_tree.Or(agroforestry_from_crop), 2)
    .where(pure_cropland, 3)
    .where(wc_builtup, 4)
    .where(wc_water, 5)
    .rename("class")
    .updateMask(
        ee.Image(0)
        .where(
            dense_forest.Or(agroforestry_from_tree)
            .Or(agroforestry_from_crop)
            .Or(pure_cropland)
            .Or(wc_builtup)
            .Or(wc_water),
            1,
        )
        .selfMask()
    )
)

# Stratified random sample: fewer points for rarer built-up/water classes.
sample_pts = label_image.addBands(feature_stack).stratifiedSample(
    numPoints=150,
    classBand="class",
    region=aoi,
    scale=10,
    seed=42,
    geometries=True,
    classValues=[1, 2, 3, 4, 5],
    classPoints=[150, 150, 150, 100, 60],
)

print("Auto-generated training/validation points:", sample_pts.size().getInfo())

# ---------------------------------------------------------------------------
# 5. TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------------------------
with_random = sample_pts.randomColumn("rand", 42)
train_set = with_random.filter(ee.Filter.lt("rand", 0.7))
valid_set = with_random.filter(ee.Filter.gte("rand", 0.7))
print("Train size:", train_set.size().getInfo(), "Validation size:", valid_set.size().getInfo())

# ---------------------------------------------------------------------------
# 6. RANDOM FOREST CLASSIFIER
# ---------------------------------------------------------------------------
classifier_bands = band_names
rf_classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=200, minLeafPopulation=3, seed=42
).train(features=train_set, classProperty="class", inputProperties=classifier_bands)

classified = feature_stack.classify(rf_classifier).rename("classification")

CLASS_NAMES = ["Dense Forest", "Agroforestry/TOF", "Cropland", "Built-up", "Water"]
CLASS_PALETTE = ["0b6623", "76b041", "e3c700", "d7191c", "2c7fb8"]

# ---------------------------------------------------------------------------
# 7. ACCURACY ASSESSMENT (held-out validation set)
# ---------------------------------------------------------------------------
validated = valid_set.classify(rf_classifier)
confusion_matrix = validated.errorMatrix("class", "classification")

print("--- VALIDATION CONFUSION MATRIX (rows=reference, cols=predicted) ---")
print(confusion_matrix.getInfo())
print("Overall Accuracy:", confusion_matrix.accuracy().getInfo())
print("Kappa Coefficient:", confusion_matrix.kappa().getInfo())
print("Producers Accuracy (per class, recall):", confusion_matrix.producersAccuracy().getInfo())
print("Consumers Accuracy (per class, precision):", confusion_matrix.consumersAccuracy().getInfo())

# Training (resubstitution) accuracy, for reference/overfitting check only.
train_accuracy = rf_classifier.confusionMatrix()
print("Training (resubstitution) Overall Accuracy:", train_accuracy.accuracy().getInfo())

# ---------------------------------------------------------------------------
# 8. CLASS AREA STATISTICS (sq km)
# ---------------------------------------------------------------------------
pixel_area_km2 = ee.Image.pixelArea().divide(1e6)

area_results = {}
for i, name in enumerate(CLASS_NAMES):
    class_val = i + 1
    area = (
        pixel_area_km2.updateMask(classified.eq(class_val))
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e10
        )
        .get("area")
    )
    area_results[name] = ee.Number(area).getInfo()

print("Class-wise area (sq km):", area_results)

# ---------------------------------------------------------------------------
# 9. EXPORTS
# ---------------------------------------------------------------------------
image_task = ee.batch.Export.image.toDrive(
    image=classified.toByte(),
    description="Palakkad_LandCover_RF",
    folder="GEE_exports",
    fileNamePrefix="palakkad_landcover_rf",
    region=aoi,
    scale=10,
    maxPixels=1e10,
)
image_task.start()

table_task = ee.batch.Export.table.toDrive(
    collection=sample_pts,
    description="Palakkad_TrainingValidationPoints",
    folder="GEE_exports",
    fileFormat="SHP",
)
table_task.start()

print(
    "Export tasks started:",
    image_task.id,
    table_task.id,
    "-- check status in the GEE Code Editor 'Tasks' tab, "
    "or poll with task.status() in Python.",
)

# Optional: export the classified image as an EE asset so the Streamlit app
# (02_landcover_app.py) can load it instantly instead of recomputing.
# asset_task = ee.batch.Export.image.toAsset(
#     image=classified.toByte(),
#     description="Export_classified_asset",
#     assetId=f"projects/{EE_PROJECT}/assets/palakkad_landcover_rf",
#     region=aoi, scale=10, maxPixels=1e10,
# )
# asset_task.start()
