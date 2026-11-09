"""Validation of Part A reference checkpoints. Run: python validate.py"""
import sqlite3
import pandas as pd

o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv")
comp = o[o.status == "completed"]
with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    db_rows = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
assert len(o) == 12309 and o.order_id.is_unique
assert (o.status == "completed").sum() == 11110 and (o.status == "cancelled").sum() == 1199
assert (o.status == "pending").sum() == 0
assert o.order_amount.isna().sum() == 0
assert comp.delivery_minutes.isna().sum() == 0
assert o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199
assert round(comp.order_amount.sum(), 2) == 2998668.80
assert c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c.age.isna().sum() == 0 and c.age.between(18, 60).all()
assert tables >= {"customers", "products", "orders"} and db_rows == 12309
print("ALL CHECKPOINTS PASSED")
