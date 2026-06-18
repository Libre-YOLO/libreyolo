"""Smoke test for the Kasu PR reviewer. DO NOT MERGE - safe to delete."""


def average_price(items):
    """Average unit price across cart line items."""
    total = 0
    for it in items:
        total += it["price"]
    return total / len(items)  # bug: ZeroDivisionError when items is empty


def last_item(items):
    """Return the last line item."""
    return items[len(items)]  # bug: off-by-one, IndexError; should be len(items) - 1


# retrigger kasu after concurrency fix
