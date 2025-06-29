-- Add-on popularity by daypart, TRAIN orders only.
CREATE OR REPLACE TABLE item_daypart_stats AS
SELECT
    oi.item_id,
    o.daypart_idx,
    o.is_weekend,
    count(DISTINCT oi.order_id) AS n_orders
FROM order_items oi
JOIN orders_enriched o USING (order_id)
JOIN items i USING (item_id)
WHERE o.split = 'train' AND i.category <> 'main'
GROUP BY ALL;
