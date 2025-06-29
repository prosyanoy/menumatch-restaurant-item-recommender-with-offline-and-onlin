-- Evaluation examples for val/test: ONE add-on per basket is held out, chosen by a
-- deterministic hash so results are reproducible. The cart is the remaining basket.
CREATE OR REPLACE TABLE examples_eval AS
WITH ranked AS (
    SELECT
        o.order_id, o.split, o.restaurant_id, o.daypart_idx, o.is_weekend, o.hour,
        oi.item_id AS target_item,
        row_number() OVER (PARTITION BY o.order_id ORDER BY hash(o.order_id::VARCHAR || '-' || oi.item_id::VARCHAR)) AS rn
    FROM orders_enriched o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN items i ON i.item_id = oi.item_id AND i.category <> 'main'
    WHERE o.split IN ('val', 'test')
)
SELECT
    r.order_id, r.split, r.restaurant_id, r.daypart_idx, r.is_weekend, r.hour, r.target_item,
    coalesce(list(c.item_id ORDER BY c.item_id) FILTER (WHERE c.item_id IS NOT NULL), []::BIGINT[]) AS cart
FROM ranked r
LEFT JOIN order_items c ON c.order_id = r.order_id AND c.item_id <> r.target_item
WHERE r.rn = 1
GROUP BY r.order_id, r.split, r.restaurant_id, r.daypart_idx, r.is_weekend, r.hour, r.target_item;
