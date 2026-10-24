Street Vendor Survival Prediction — Power BI Starter Kit

Included
- Urban_Analytics_Theme.json: import through View > Themes > Browse for themes.
- DAX_Measures.txt: starter measures for the cleaned dataset.
- Copy of the cleaned dataset, if available in this package.
- This setup guide.

Recommended data source
Use street_vendor_survival_cleaned.csv for the main business dashboard. It retains readable categories such as city, zone_type, food_category, license_status, and season_of_observation.
Use the feature-engineered dataset mainly for model-related analysis; it contains one-hot encoded categories and is less convenient for slicers.

Setup
1. Open Power BI Desktop.
2. Home > Get data > Text/CSV and select street_vendor_survival_cleaned.csv.
3. Select Transform Data and verify types:
   - vendor_id: Text
   - city, zone_type, food_category, license_status, season_of_observation: Text
   - vendor_survived, had_fine_last_year, has_online_presence, num_helpers: Whole number
   - revenue, rent, rainfall, complaint rate, and other measurements: appropriate numeric type
4. Close & Apply.
5. In the Fields pane, rename the imported table to Vendors.
6. View > Themes > Browse for themes and select Urban_Analytics_Theme.json.
7. Create the measures from DAX_Measures.txt using Modeling > New measure.
8. Format Survival Rate and Closure Rate as percentages; revenue and rent measures as Indian Rupees.

Suggested pages
Page 1 — Executive Overview
- Cards: Total Vendors, Survived Vendors, Closed Vendors, Survival Rate
- Donut: survived vs closed
- Bar chart: vendor count by city
- Bar chart: average daily revenue by city
- Slicers: city, zone_type, food_category, license_status

Page 2 — Business Analysis
- Survival rate by license status
- Survival rate by city and zone type
- Revenue vs daily customers
- Competition and complaint rate comparisons
- Inspection score analysis; treat zero as no recorded inspection score based on the cleaning logic

Page 3 — ML Insights
- Add model metrics, confusion matrix, and feature importance only after the final model run.
- Do not invent accuracy or other evaluation metrics.
- A feature-engineered CSV alone does not contain trained model evaluation results.

Design
- Page background: #E9F0F5
- Navy: #172B4D
- Teal: #287D8E
- Muted red: #D65A5A
- Accent amber: #D89B36
- Font: Segoe UI
- Page title: 24–28 pt; chart title: 13–15 pt; KPI value: 28–36 pt
- Use a 16:9 page layout and keep margins and spacing consistent.

Interpretation
The target variable is coded 0 = Closed and 1 = Survived in the provided project notes. Label outcomes as 12-month survival outcomes rather than current status unless the dataset defines current status.
