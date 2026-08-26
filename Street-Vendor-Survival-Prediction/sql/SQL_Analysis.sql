use minor_project_v_sem;

-- Basic Data Validation
SELECT COUNT(*) AS total_vendors FROM street_vendor_survival_cleaned;
SELECT vendor_id, COUNT(*) AS count FROM street_vendor_survival_cleaned GROUP BY vendor_id HAVING COUNT(*) > 1;
SELECT vendor_survived, COUNT(*) AS vendor_count FROM street_vendor_survival_cleaned GROUP BY vendor_survived;
SELECT
    SUM(vendor_id IS NULL) AS missing_vendor_id,
    SUM(city IS NULL) AS missing_city,
    SUM(avg_daily_revenue_inr IS NULL) AS missing_revenue,
    SUM(avg_daily_customers IS NULL) AS missing_customers,
    SUM(license_status IS NULL) AS missing_license,
    SUM(vendor_survived IS NULL) AS missing_target
FROM street_vendor_survival_cleaned;

/*SQL Analysis Part 1:
Basic Vendor Analysis */
-- Vendors by city
SELECT city, COUNT(*) AS vendor_count FROM street_vendor_survival_cleaned GROUP BY city ORDER BY vendor_count DESC;

-- Vendors by zone type
SELECT food_category, COUNT(*) AS vendor_count FROM street_vendor_survival_cleaned GROUP BY food_category ORDER BY vendor_count DESC;

-- Vendors by season
SELECT season_of_observation, COUNT(*) AS vendor_count FROM street_vendor_survival_cleaned GROUP BY season_of_observation ORDER BY vendor_count DESC;

/* Part 2: Revenue Analysis*/
-- Average revenue
SELECT ROUND(AVG(avg_daily_revenue_inr), 2) AS average_daily_revenue FROM street_vendor_survival_cleaned;

-- Highest revenue vendors
SELECT vendor_id, city, food_category, avg_daily_revenue_inr FROM street_vendor_survival_cleaned ORDER BY avg_daily_revenue_inr DESC LIMIT 10;

-- Revenue by city
SELECT city, ROUND(AVG(avg_daily_revenue_inr), 2) AS average_revenue FROM street_vendor_survival_cleaned GROUP BY city ORDER BY average_revenue DESC;

-- Revenue by food category
SELECT food_category, ROUND(AVG(avg_daily_revenue_inr), 2) AS average_revenue FROM street_vendor_survival_cleaned GROUP BY food_category ORDER BY average_revenue DESC;

-- Revenue by zone
SELECT
    zone_type,
    ROUND(AVG(avg_daily_revenue_inr), 2) AS average_revenue
FROM street_vendor_survival_cleaned GROUP BY zone_type ORDER BY average_revenue DESC;

/*Combined Business Analysis*/
-- Highest-performing vendors
SELECT
    vendor_id,
    city,
    avg_daily_revenue_inr,
    avg_daily_customers,
    monthly_health_inspection_score,
    customer_complaint_rate,
    vendor_survived
FROM street_vendor_survival_cleaned
WHERE vendor_survived = 1
ORDER BY
    avg_daily_revenue_inr DESC,
    avg_daily_customers DESC
LIMIT 10;

-- Active vendors with high revenue and low complaints
SELECT
    vendor_id,
    city,
    avg_daily_revenue_inr,
    customer_complaint_rate
FROM street_vendor_survival_cleaned
WHERE vendor_survived = 1
  AND avg_daily_revenue_inr > (
      SELECT AVG(avg_daily_revenue_inr)
      FROM street_vendor_survival_cleaned)
  AND customer_complaint_rate < (
      SELECT AVG(customer_complaint_rate)
      FROM street_vendor_survival_cleaned)
ORDER BY avg_daily_revenue_inr DESC;

-- City-level business performance
SELECT
    city,
    COUNT(*) AS total_vendors,
    ROUND(AVG(avg_daily_revenue_inr), 2) AS avg_revenue,
    ROUND(AVG(avg_daily_customers), 2) AS avg_customers,
    ROUND(AVG(competition_within_100m), 2) AS avg_competition,
    ROUND(AVG(customer_complaint_rate), 3) AS avg_complaints,
    ROUND(AVG(vendor_survived) * 100, 2) AS survival_rate
FROM street_vendor_survival_cleaned
GROUP BY city
ORDER BY survival_rate DESC;

-- Vendor age and survival
SELECT
    CASE
        WHEN vendor_age_years < 25 THEN 'Young'
        WHEN vendor_age_years < 40 THEN 'Middle'
        ELSE 'Older'
    END AS age_group,
    COUNT(*) AS vendor_count,
    ROUND(AVG(vendor_survived) * 100, 2) AS survival_rate
FROM street_vendor_survival_cleaned
GROUP BY age_group
ORDER BY survival_rate DESC;

-- Business experience and survival
SELECT
    CASE
        WHEN years_in_business < 5 THEN 'Less than 5 Years'
        WHEN years_in_business < 10 THEN '5-10 Years'
        ELSE 'More than 10 Years'
    END AS experience_group,
    COUNT(*) AS vendor_count,
    ROUND(AVG(vendor_survived) * 100, 2) AS survival_rate
FROM street_vendor_survival_cleaned
GROUP BY experience_group
ORDER BY survival_rate DESC;

-- Overall vendor risk indicators
SELECT
    vendor_id,
    city,
    avg_daily_revenue_inr,
    competition_within_100m,
    customer_complaint_rate,
    monthly_health_inspection_score,
    license_status,
    had_fine_last_year,
    vendor_survived
FROM street_vendor_survival_cleaned
WHERE vendor_survived = 0
ORDER BY
    customer_complaint_rate DESC,
    competition_within_100m DESC;