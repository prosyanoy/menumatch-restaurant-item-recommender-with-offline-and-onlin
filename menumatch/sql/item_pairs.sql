-- Directed item co-occurrence counts over TRAIN baskets (graph edges).
CREATE OR REPLACE TABLE item_pairs AS
SELECT
    a.item_id AS item_a,
    b.item_id AS item_b,
    count(*) AS n_orders
FROM order_items a
JOIN order_items b ON a.order_id = b.order_id AND a.item_id <> b.item_id
JOIN orders_enriched o ON o.order_id = a.order_id
WHERE o.split = 'train'
GROUP BY ALL;
