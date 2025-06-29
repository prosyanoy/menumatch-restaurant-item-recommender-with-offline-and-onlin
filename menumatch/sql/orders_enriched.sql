-- One row per order with time-of-day context and the time-based split label.
-- daypart_idx mirrors menumatch.context.daypart_of.
CREATE OR REPLACE TABLE orders_enriched AS
SELECT
    o.order_id,
    o.restaurant_id,
    o.ts,
    hour(o.ts) AS hour,
    CASE
        WHEN hour(o.ts) BETWEEN 5 AND 10 THEN 0
        WHEN hour(o.ts) BETWEEN 11 AND 14 THEN 1
        WHEN hour(o.ts) BETWEEN 15 AND 17 THEN 2
        WHEN hour(o.ts) BETWEEN 18 AND 21 THEN 3
        ELSE 4
    END AS daypart_idx,
    CASE WHEN dayofweek(o.ts) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    CASE
        WHEN o.ts < $train_end THEN 'train'
        WHEN o.ts < $val_end THEN 'val'
        ELSE 'test'
    END AS split
FROM orders o;
