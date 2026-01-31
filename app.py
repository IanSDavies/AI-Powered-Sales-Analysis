from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests

app = Flask(__name__)

# ============================================================
# 1. MANUAL .env LOADER
# ============================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"

GEMINI_API_KEY = None
if ENV_PATH.exists():
    with open(ENV_PATH, "r") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                GEMINI_API_KEY = line.split("=", 1)[1].strip()
                os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
                break

# ============================================================
# 2. LOAD SALES DATA
# ============================================================
DATA_PATH = Path(__file__).resolve().parent / "data" / "TableauSalesData.csv"

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print("CSV load error:", e)
    df = pd.DataFrame()

if not df.empty and "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Year"] = df["Order Date"].dt.year

cols_to_numeric = ["Sales", "Profit", "Discount"]
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 3. HELPERS
# ============================================================
REGION_COLORS = {
    "East": "#10b981",     # green
    "West": "#3b82f6",     # blue
    "Central": "#f59e0b",  # amber
    "South": "#ef4444"     # red
}

SEGMENT_COLORS = {
    "Consumer": "#0ea5e9",
    "Corporate": "#8b5cf6",
    "Home Office": "#ef4444"
}

def get_unique_values(series: pd.Series):
    return sorted(series.dropna().unique())

def format_currency(value: float) -> str:
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"

def fig_to_html(fig):
    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn",
        config={"displaylogo": False, "responsive": True}
    )


# ============================================================
# 4. BUILD CONTEXT-AWARE SUMMARY FOR GEMINI
# ============================================================
def build_llm_summary(filtered: pd.DataFrame,
                      selected_query: str,
                      category, subcategory, region, segment) -> str:

    if filtered.empty:
        return "No data after filtering."

    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    avg_discount = filtered["Discount"].mean() if "Discount" in filtered.columns else None
    row_count = len(filtered)

    filter_parts = []
    if category: filter_parts.append(f"Category={category}")
    if subcategory: filter_parts.append(f"Sub-Category={subcategory}")
    if region: filter_parts.append(f"Region={region}")
    if segment: filter_parts.append(f"Segment={segment}")

    filter_text = ", ".join(filter_parts) if filter_parts else "No filters applied"

    # Category contributions
    top_cat = worst_cat = None
    if "Category" in filtered.columns:
        gp = filtered.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        if not gp.empty:
            top_cat = f"{gp.index[0]} (${gp.iloc[0]:,.2f})"
            worst_cat = f"{gp.index[-1]} (${gp.iloc[-1]:,.2f})"

    # Subcategory
    top_sub = worst_sub = None
    if "Sub-Category" in filtered.columns:
        gp = filtered.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
        if not gp.empty:
            top_sub = f"{gp.index[0]} (${gp.iloc[0]:,.2f})"
            worst_sub = f"{gp.index[-1]} (${gp.iloc[-1]:,.2f})"

    # Segment
    seg_summary = None
    if "Segment" in filtered.columns:
        gp = filtered.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
        seg_summary = "; ".join([f"{i}: ${v:,.0f}" for i, v in gp.items()])

    # Region
    region_summary = None
    if "Region" in filtered.columns:
        gp = filtered.groupby("Region")["Profit"].sum().sort_values(ascending=False)
        region_summary = "; ".join([f"{i}: ${v:,.0f}" for i, v in gp.items()])

    # High discount
    high_disc = None
    if "Product Name" in filtered.columns and "Discount" in filtered.columns:
        topd = filtered.sort_values("Discount", ascending=False).head(3)
        high_disc = ", ".join([
            f"{row['Product Name']} ({row['Discount']:.1%})"
            for _, row in topd.iterrows()
        ])

    # Negative profit
    neg_count = filtered[filtered["Profit"] < 0]["Product ID"].nunique() \
        if "Product ID" in filtered.columns else \
        (filtered["Profit"] < 0).sum()

    summary = [
        f"Query executed: {selected_query}.",
        f"Filters applied: {filter_text}.",
        f"Rows included: {row_count}.",
        f"Total sales: ${total_sales:,.2f}.",
        f"Total profit: ${total_profit:,.2f}.",
    ]
    if avg_discount is not None:
        summary.append(f"Average discount: {avg_discount:.2%}.")

    if top_cat: summary.append(f"Top category: {top_cat}.")
    if worst_cat: summary.append(f"Worst category: {worst_cat}.")
    if top_sub: summary.append(f"Top sub-category: {top_sub}.")
    if worst_sub: summary.append(f"Worst sub-category: {worst_sub}.")
    if seg_summary: summary.append(f"Segment sales distribution: {seg_summary}.")
    if region_summary: summary.append(f"Regional profit distribution: {region_summary}.")
    if high_disc: summary.append(f"Highest discounting products: {high_disc}.")
    summary.append(f"Distinct negative-profit products: {neg_count}.")

    return " ".join(summary)


# ============================================================
# 5. FORMAT GEMINI OUTPUT INTO HTML WITH COLLAPSIBLE SECTIONS
# ============================================================
def format_recommendations(raw: str) -> str:
    """
    Convert Gemini structured output into:
    Title
    Reasoning
    Collapsible Action Steps
    With business-verb bullet headers.
    """

    if not raw:
        return ""

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return ""

    formatted_cards = []
    current_title = ""
    reasoning_lines = []
    bullets = []
    mode = None

    def flush_card():
        if not current_title:
            return ""

        collapse_id = f"collapse_{abs(hash(current_title))}"

        html = []
        html.append('<div class="rec-card">')
        html.append(f'<div class="rec-title">{current_title}</div>')

        if reasoning_lines:
            html.append(
                '<div class="rec-body"><strong>Reasoning:</strong> '
                + " ".join(reasoning_lines)
                + '</div>'
            )

        if bullets:
            html.append(
                f'''
                <button class="btn btn-sm btn-outline-primary mt-2"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target="#{collapse_id}">
                    Show Action Steps
                </button>

                <div class="collapse" id="{collapse_id}">
                    <div class="card card-body bg-light mt-2 ms-2">
                '''
            )

            for b in bullets:
                if ":" in b:
                    head, rest = b.split(":", 1)
                    html.append(
                        f'<div class="rec-bullet"><strong>{head.strip()}:</strong> {rest.strip()}</div>'
                    )
                else:
                    html.append(f'<div class="rec-bullet">{b}</div>')

            html.append("</div></div>")

        html.append("</div>")
        return "\n".join(html)

    for line in lines:
        low = line.lower()

        # Detect title
        if low.startswith("1.") or low.startswith("2.") or low.startswith("3."):
            if current_title:
                formatted_cards.append(flush_card())
                reasoning_lines = []
                bullets = []

            current_title = line[2:].strip().strip("* ")
            mode = None
            continue

        if low.startswith("reasoning:"):
            mode = "reasoning"
            content = line[len("Reasoning:"):].strip()
            if content:
                reasoning_lines.append(content)
            continue

        if low.startswith("action steps:"):
            mode = "actions"
            continue

        if mode == "reasoning":
            reasoning_lines.append(line)
            continue

        if mode == "actions":
            if line.startswith("*"):
                clean = line[1:].strip().strip("* ")
                bullets.append(clean)
            else:
                if bullets:
                    bullets[-1] += " " + line
            continue

    if current_title:
        formatted_cards.append(flush_card())

    return "\n".join(formatted_cards)


# ============================================================
# 6. CALL GEMINI 2.5 FLASH
#
# call_gemini(summary_text)
#
# Sends a structured summary of the filtered dataset to the
# Google Gemini 2.5 Flash API (free tier) using a simple POST
# request via the `requests` library.
#
# The prompt instructs the model to generate exactly 3
# business recommendations tied to Office Solutions’ goals:
#  • Increase overall sales by 20%
#  • Increase profits by 10%
#  • Increase corporate sales by 10%
#  • Increase corporate accounts by 50% over 3 years 
#
# ============================================================
def call_gemini(summary_text: str) -> str:

    if not GEMINI_API_KEY:
        return "Gemini API key missing."

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        "gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    prompt = f"""
You are a senior business analytics consultant.

SUMMARY OF FILTERED DATA:
{summary_text}

Your task:
Provide EXACTLY 3 recommendations.

FORMAT STRICTLY AS:

1. **Recommendation Title**
Reasoning: (1–2 sentences)
Action Steps:
* Identify: …
* Implement: …
* Develop: …
* Optimize: …
* Strengthen: …

RULES:
- Each bullet MUST begin with a business verb + colon.
- Reference filters (Region, Segment, Category, Sub-Category).
- Reference query type.
- Tie recommendations to goals:
  • Increase overall sales by 20%
  • Increase profits by 10%
  • Increase corporate sales by 10%
  • Increase corporate accounts by 50% over 3 years 

OUTPUT ONLY THE RECOMMENDATIONS IN THIS EXACT FORMAT.
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, json=payload, timeout=20)
        data = resp.json()

        if "error" in data:
            return f"Gemini API error: {data['error'].get('message')}"

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"Gemini request failed: {e}"


# ============================================================
# 7. DYNAMIC SUBCATEGORY ENDPOINT
# ============================================================
@app.route("/get_subcategories", methods=["POST"])
def get_subcategories():
    data = request.get_json()
    category = data.get("category")
    if category:
        subs = get_unique_values(df[df["Category"] == category]["Sub-Category"])
    else:
        subs = get_unique_values(df["Sub-Category"])
    return jsonify(subs)


# ============================================================
# 8. MAIN ROUTE
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():

    category = subcategory = region = segment = selected_query = None

    if request.method == "POST":
        category = request.form.get("category") or None
        subcategory = request.form.get("subcategory") or None
        region = request.form.get("region") or None
        segment = request.form.get("segment") or None
        selected_query = request.form.get("query") or None

    # Dropdowns
    if df.empty:
        categories = subcategories = regions = segments = []
    else:
        categories = get_unique_values(df["Category"])
        subcategories = (
            get_unique_values(df[df["Category"] == category]["Sub-Category"])
            if category else get_unique_values(df["Sub-Category"])
        )
        regions = get_unique_values(df["Region"])
        segments = get_unique_values(df["Segment"])

    chart_html = None
    table_html = None
    recommendations_html = None

    # ===========================
    # RUN QUERIES
    # ===========================
    if request.method == "POST" and selected_query:

        filtered = df.copy()
        if category:
            filtered = filtered[filtered["Category"] == category]
        if subcategory:
            filtered = filtered[filtered["Sub-Category"] == subcategory]
        if region:
            filtered = filtered[filtered["Region"] == region]
        if segment:
            filtered = filtered[filtered["Segment"] == segment]

        if filtered.empty:
            table_html = "<p class='text-danger fw-semibold'>No data for selected filters.</p>"
            return render_template(
                "index.html",
                categories=categories,
                subcategories=subcategories,
                regions=regions,
                segments=segments,
                query_name=selected_query,
                chart_html=None,
                table_html=table_html,
                recommendations_html=None,
                selected_category=category,
                selected_subcategory=subcategory,
                selected_region=region,
                selected_segment=segment,
                selected_query=selected_query,
            )

        # ====================================================
        # QUERY: Total Sales & Profit
        # ====================================================
        if selected_query == "Total Sales and Profit":
            total_sales = filtered["Sales"].sum()
            total_profit = filtered["Profit"].sum()

            table_df = pd.DataFrame({
                "Metric": ["Total Sales", "Total Profit"],
                "Value": [format_currency(total_sales), format_currency(total_profit)]
            })

            table_html = table_df.to_html(
                classes="table table-striped table-hover",
                index=False, escape=False
            )

            # Manual Graph Building for absolute precision
            fig = go.Figure()

            # Add Sales Bar
            fig.add_trace(go.Bar(
                x=["Total Sales"],
                y=[total_sales],
                name="Sales",
                text=[format_currency(total_sales)],
                textposition="outside",
                marker_color="#10b981",
                customdata=[format_currency(total_sales)],
                hovertemplate="<b>Total Sales</b><br>Value: %{customdata}<extra></extra>"
            ))

            # Add Profit Bar
            fig.add_trace(go.Bar(
                x=["Total Profit"],
                y=[total_profit],
                name="Profit",
                text=[format_currency(total_profit)],
                textposition="outside",
                marker_color="#3b82f6",
                customdata=[format_currency(total_profit)],
                hovertemplate="<b>Total Profit</b><br>Value: %{customdata}<extra></extra>"
            ))

            fig.update_layout(
                title="Total Sales vs Total Profit",
                xaxis_title="Metric",
                yaxis_title="Amount ($)",
                yaxis=dict(tickformat="-$"),
                height=650,
                margin=dict(t=90, b=80, l=80, r=80),
                showlegend=False
            )

            chart_html = "<div class='scroll-x'>" + fig_to_html(fig) + "</div>"


        # ====================================================
        # QUERY: Average Discount by Product
        # ====================================================
        elif selected_query == "Average Discount by Product":
            gp = filtered.groupby("Product Name")["Discount"].mean().reset_index()
            gp = gp.sort_values("Discount", ascending=False)
            gp["Discount_Display"] = gp["Discount"].apply(format_percentage)

            table_html = gp[["Product Name", "Discount_Display"]].rename(
                columns={"Discount_Display": "Discount"}
            ).to_html(
                classes="table table-striped table-hover",
                index=False,
                escape=False
            )

            top = gp.head(10)

            fig = px.bar(
                top,
                x="Discount",
                y="Product Name",
                text="Discount_Display",
                orientation="h",
                color_discrete_sequence=["#facc15"],
                title="Top 10 Highest-Discounted Products"
            )

            # Map specific display strings to the hover template
            fig.update_traces(
                customdata=top[["Discount_Display"]],
                hovertemplate="<b>%{y}</b><br>Avg Discount: %{customdata[0]}<extra></extra>",
                textposition="outside"
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                xaxis_title="Average Discount (%)",
                yaxis_title="Product Name"
            )
            fig.update_xaxes(tickformat=".0%")
            chart_html = "<div class='scroll-x'>" + fig_to_html(fig) + "</div>"

        # ====================================================
        # QUERY: Total Sales by Year
        # ====================================================
        elif selected_query == "Total Sales by Year":
            yearly = (
                filtered.groupby(["Year", "Segment"])["Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Sales": "Total Sales"})
            )
            yearly["Sales_Display"] = yearly["Total Sales"].apply(format_currency)

            fig = px.line(
                yearly,
                x="Year", y="Total Sales", color="Segment",
                text="Sales_Display", markers=True, title="Total Sales by Year"
            )

            fig.update_traces(
                customdata=yearly[["Sales_Display"]],
                hovertemplate="<b>Segment: %{fullData.name}</b><br>Year: %{x}<br>Total Sales: %{customdata[0]}<extra></extra>",
                textposition="top center"
            )

            fig.update_layout(xaxis_title="Year", yaxis_title="Total Sales ($)")
            fig.update_xaxes(dtick=1, tickformat="d")
            chart_html = "<div class='scroll-x'>" + fig_to_html(fig) + "</div>"

            # Pivot the data so Years are columns and Segments are rows
            pivot = (
                yearly.pivot(index="Segment", columns="Year", values="Total Sales")
                .fillna(0)
                .map(format_currency)
            )
            pivot.index.name = None
            pivot.columns.name = ""
            table_html = pivot.to_html(classes="table table-striped table-hover")

        # ====================================================
        # QUERY: Profit by Region
        # ====================================================
        elif selected_query == "Profit by Region":
            gp = filtered.groupby("Region")["Profit"].sum().reset_index()
            gp["Profit_Display"] = gp["Profit"].apply(format_currency)

            fig = px.bar(gp, x="Region", y="Profit", color="Region", title="Profit by Region")

            fig.update_traces(
                customdata=gp[["Profit_Display"]],
                hovertemplate="<b>Region: %{x}</b><br>Profit: %{customdata[0]}<extra></extra>",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Region", yaxis_title="Profit ($)",
                yaxis=dict(tickformat="-$"), height=650,
                margin=dict(t=90, b=80, l=80, r=80)
            )
            chart_html = "<div class='scroll-x'>" + fig_to_html(fig) + "</div>"

            table_html = gp[["Region", "Profit_Display"]].rename(
                columns={"Profit_Display": "Profit"}
            ).to_html(
                classes="table table-striped table-hover",
                index=False,
                escape=False
            )

        # ====================================================
        # QUERY: Products with Negative Profit (AGGREGATED)
        # ====================================================
        elif selected_query == "Products with Negative Profit":
            neg = filtered[filtered["Profit"] < 0].copy()

            if neg.empty:
                table_html = "<p>No products with negative profit.</p>"
                fig = go.Figure()
                fig.add_annotation(
                    text="No products with negative profit.",
                    x=0.5, y=0.5, showarrow=False
                )
                chart_html = fig_to_html(fig)

            else:
                agg = (
                    neg.groupby(["Product ID", "Product Name"], as_index=False)
                    .agg({
                        "Profit": "sum",
                        "Sales": "sum",
                        "Discount": "mean"
                    })
                )

                agg["Profit"] = agg["Profit"].astype(float).round(2)
                agg["Sales"] = agg["Sales"].astype(float).round(2)
                agg["Discount"] = agg["Discount"].astype(float)

                agg["Profit_Display"] = agg["Profit"].apply(format_currency)
                agg["Sales_Display"] = agg["Sales"].apply(format_currency)
                agg["Discount_Display"] = agg["Discount"].apply(format_percentage)

                agg = agg.sort_values("Profit", ascending=True)
                top_neg = agg.head(10)

                table_html = agg[[
                    "Product ID", "Product Name", "Profit_Display",
                    "Sales_Display", "Discount_Display"
                ]].rename(columns={
                    "Profit_Display": "Profit",
                    "Sales_Display": "Sales",
                    "Discount_Display": "Discount"
                }).to_html(
                    classes="table table-striped table-hover",
                    index=False,
                    escape=False
                )

                fig = px.bar(
                    top_neg,
                    x="Profit",
                    y="Product Name",
                    orientation="h",
                    text="Profit_Display",
                    color_discrete_sequence=["#fca5a5"],
                    title="Most Unprofitable Products (Aggregated)"
                )

                # Advanced multi-line tooltip using customdata
                fig.update_traces(
                    customdata=top_neg[["Profit_Display", "Sales_Display", "Discount_Display"]],
                    hovertemplate="""<b>%{y}</b><br>
                    Total Profit: %{customdata[0]}<br>
                    Total Sales: %{customdata[1]}<br>
                    Avg Discount: %{customdata[2]}<extra></extra>""",
                    textposition="inside"
                )

                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    xaxis_title="Total Profit ($)",
                    yaxis_title="Product Name",
                    xaxis=dict(tickformat="-$")
                )

                chart_html = "<div class='scroll-x'>" + fig_to_html(fig) + "</div>"

        # ====================================================
        # AI RECOMMENDATIONS
        # ====================================================
        summary = build_llm_summary(
            filtered, selected_query, category, subcategory, region, segment
        )
        raw_llm = call_gemini(summary)
        if raw_llm.startswith("Gemini API"):
            recommendations_html = f"<p class='text-danger fw-bold'>{raw_llm}</p>"
        else:
            recommendations_html = format_recommendations(raw_llm)


    return render_template(
        "index.html",
        categories=categories,
        subcategories=subcategories,
        regions=regions,
        segments=segments,
        query_name=selected_query,
        chart_html=chart_html,
        table_html=table_html,
        recommendations_html=recommendations_html,
        selected_category=category,
        selected_subcategory=subcategory,
        selected_region=region,
        selected_segment=segment,
        selected_query=selected_query,
    )


if __name__ == "__main__":
    app.run(debug=False)
