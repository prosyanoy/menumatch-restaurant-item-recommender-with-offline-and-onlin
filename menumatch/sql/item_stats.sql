-- Item popularity computed from TRAIN orders only (no look-ahead).
CREATE OR REPLACE TABLE item_stats AS
SELECT
    oi.item_id,
    i.restaurant_id,
    i.category,
    count(DISTINCT oi.order_id) AS n_orders,
    sum(oi.quantity) AS n_units
FROM order_items oi
JOIN orders_enriched o USING (order_id)
JOIN items i USING (item_id)
WHERE o.split = 'train'
GROUP BY ALL;
