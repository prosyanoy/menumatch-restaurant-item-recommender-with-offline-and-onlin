-- Leave-one-out training examples: every add-on in a TRAIN basket is a target,
-- the cart is the rest of the basket.
CREATE OR REPLACE TABLE examples_train AS
SELECT
    o.order_id,
    'train' AS split,
    o.restaurant_id,
    o.daypart_idx,
    o.is_weekend,
    o.hour,
    t.item_id AS target_item,
    coalesce(list(c.item_id ORDER BY c.item_id) FILTER (WHERE c.item_id IS NOT NULL), []::BIGINT[]) AS cart
FROM orders_enriched o
JOIN order_items t ON t.order_id = o.order_id
JOIN items ti ON ti.item_id = t.item_id AND ti.category <> 'main'
LEFT JOIN order_items c ON c.order_id = o.order_id AND c.item_id <> t.item_id
WHERE o.split = 'train'
GROUP BY o.order_id, o.restaurant_id, o.daypart_idx, o.is_weekend, o.hour, t.item_id;
