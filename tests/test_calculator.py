"""
tests/test_calculator.py — Unit tests for the safe AST calculator.

Tests verify:
- Correct arithmetic results
- Rejection of unsafe operations (__import__, attribute access, etc.)
- Edge cases (division by zero, very large exponents)
"""

from __future__ import annotations

import pytest


def calc(expr: str) -> str:
    from rag.graph.tools import calculator  # noqa: PLC0415
    # calculator is a LangChain @tool — call its underlying function
    return calculator.invoke({"expression": expr})


class TestCalculatorCorrectness:
    def test_addition(self):
        assert calc("2 + 3") == "5"

    def test_subtraction(self):
        assert calc("10 - 4") == "6"

    def test_multiplication(self):
        assert calc("6 * 7") == "42"

    def test_division(self):
        result = calc("10 / 4")
        assert float(result) == pytest.approx(2.5)

    def test_floor_division(self):
        assert calc("10 // 3") == "3"

    def test_modulo(self):
        assert calc("10 % 3") == "1"

    def test_power(self):
        assert calc("2 ** 8") == "256"

    def test_nested_expression(self):
        assert calc("(3 + 4) * 2") == "14"

    def test_negative_number(self):
        assert calc("-5 + 3") == "-2"

    def test_float_result(self):
        result = calc("1 / 3")
        assert "0.333" in result


class TestCalculatorSafety:
    def test_rejects_dunder(self):
        result = calc("__import__('os').system('echo hi')")
        assert "Error" in result

    def test_rejects_attribute_access_via_dunder(self):
        result = calc("().__class__.__bases__[0].__subclasses__()")
        assert "Error" in result

    def test_rejects_string_literal(self):
        result = calc("'hello'")
        assert "Error" in result

    def test_rejects_list(self):
        result = calc("[1, 2, 3]")
        assert "Error" in result

    def test_empty_expression(self):
        result = calc("")
        assert "Error" in result

    def test_syntax_error(self):
        result = calc("2 +* 3")
        assert "Error" in result


class TestCalculatorEdgeCases:
    def test_division_by_zero(self):
        result = calc("1 / 0")
        assert "zero" in result.lower() or "Error" in result

    def test_large_exponent_rejected(self):
        result = calc("2 ** 200")
        assert "Error" in result or "too large" in result.lower()

    def test_reasonable_exponent_allowed(self):
        assert calc("2 ** 10") == "1024"
