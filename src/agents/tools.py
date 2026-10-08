import ast
import math
import operator

from langchain.tools import tool

from src.retrieval.vector_store import search


@tool
def financial_search(query: str) -> str:
    """
    Search the financial document knowledge base.

    Use this tool when the user asks about information that may be
    contained in the uploaded financial reports or regulatory documents.

    The search returns relevant document passages along with source,
    page number, document type, and relevance score.
    """
    if not query or not query.strip():
        return "Search query cannot be empty."

    results = search(query.strip(), top_k=5)

    if not results:
        return "No relevant financial documents were found."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        document = result["document"]
        metadata = document.metadata

        formatted_results.append(
            f"""[{index}]
Source: {metadata["source"]}
Page: {metadata["page"]}
Document type: {metadata["document_type"]}
Relevance score: {result["score"]:.4f}

{document.page_content}
"""
        )

    return "\n---\n".join(formatted_results)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_calculate(expression: str) -> int | float:
    """Evaluate a restricted arithmetic expression safely."""

    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    expression = expression.strip()

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expression is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Boolean values are not allowed.")  # noqa: TRY004

            if isinstance(node.value, (int, float)):
                if not math.isfinite(node.value):
                    raise ValueError("Non-finite numbers are not allowed.")
                return node.value

            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.BinOp):
            operator_function = _ALLOWED_OPERATORS.get(type(node.op))

            if operator_function is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent is too large.")

            result = operator_function(left, right)

            if not math.isfinite(result):
                raise ValueError("Calculation produced a non-finite result.")

            return result

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return +evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            raise ValueError("Unary operator is not allowed.")

        raise ValueError(f"Expression element {type(node).__name__} is not allowed.")

    return evaluate(tree)


@tool
def calculator(expression: str) -> str:
    """
    Perform safe arithmetic calculations.

    Use this tool when the user asks for a numerical calculation,
    percentage calculation, comparison, or arithmetic expression.

    Supported operators:
    +, -, *, /, %, **, parentheses, and unary +/-
    """
    try:
        result = _safe_calculate(expression)

        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.12g}"

        return str(result)

    except Exception as exc:  # noqa: BLE001
        return f"Calculator error: {exc}"


if __name__ == "__main__":
    print("Financial Search Test")
    print("=" * 50)

    result = financial_search.invoke({"query": "What was HDFC Bank's profit?"})

    print(result)

    print("\nCalculator Test")
    print("=" * 50)

    calculations = [
        "14 * 250",
        "(1000 - 250) / 5",
        "12.5 * 8",
        "2 ** 10",
    ]

    for expression in calculations:
        result = calculator.invoke({"expression": expression})
        print(f"{expression} = {result}")
