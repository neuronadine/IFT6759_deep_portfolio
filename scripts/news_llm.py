from typing import Literal, List
from pydantic import BaseModel, Field

# --- Schema Definitions ---

class Sector(BaseModel):
    sector_name: (
        Literal[
            "communication",
            "consumer_discretionary",
            "consumer_staples",
            "financials",
            "health_care",
            "industrials",
            "information_technology",
        ]
        | None
    ) = Field(
        description="This is the sector name, representing a specific category of the market. Each sector includes companies with similar business activities or market focus: "
        "'communication' includes companies like DIS (The Walt Disney Company) and VZ (Verizon Communications); "
        "'consumer_discretionary' includes companies like HD (Home Depot), MCD (McDonald’s), and NKE (Nike); "
        "'consumer_staples' includes companies like KO (Coca-Cola), PG (Procter & Gamble), WBA (Walgreens Boots Alliance), and WMT (Walmart); "
        "'energy' includes companies like CVX (Chevron); "
        "'financials' includes companies like AXP (American Express), GS (Goldman Sachs), JPM (JPMorgan Chase), TRV (Travelers Companies), and V (Visa); "
        "'health_care' includes companies like AMGN (Amgen), JNJ (Johnson & Johnson), MRK (Merck), and UNH (UnitedHealth Group); "
        "'industrials' includes companies like BA (Boeing), CAT (Caterpillar), HON (Honeywell), and MMM (3M); "
        "'information_technology' includes companies like AAPL (Apple), CRM (Salesforce), CSCO (Cisco), IBM (IBM), INTC (Intel), and MSFT (Microsoft)."
    )
    sentiment_score: Literal[-1, 0, 1] | None = Field(
        description="This score represents the sentiment of investors or the market toward a sector. A score of -1 indicates negative sentiment (bearish outlook), 1 indicates positive sentiment (bullish outlook), and 0 indicates neutral sentiment (indifference or uncertainty)."
    )


class Stock(BaseModel):
    stock_id: (
        Literal[
            "DIS",
            "VZ",
            "HD",
            "MCD",
            "NKE",
            "KO",
            "PG",
            "WBA",
            "WMT",
            "CVX",
            "AXP",
            "GS",
            "JPM",
            "TRV",
            "V",
            "AMGN",
            "JNJ",
            "MRK",
            "UNH",
            "BA",
            "CAT",
            "HON",
            "MMM",
            "AAPL",
            "CRM",
            "CSCO",
            "IBM",
            "INTC",
            "MSFT",
        ]
        | None
    ) = Field(
        description="""This is the stock ticker symbol, a unique identifier for publicly traded companies. Each ticker corresponds to a specific company:
    - DIS: The Walt Disney Company
    - VZ: Verizon Communications
    - HD: Home Depot
    - MCD: McDonald’s
    - NKE: Nike
    - KO: Coca-Cola
    - PG: Procter & Gamble
    - WBA: Walgreens Boots Alliance
    - WMT: Walmart
    - CVX: Chevron
    - AXP: American Express
    - GS: Goldman Sachs
    - JPM: JPMorgan Chase
    - TRV: Travelers Companies
    - V: Visa
    - AMGN: Amgen
    - JNJ: Johnson & Johnson
    - MRK: Merck
    - UNH: UnitedHealth Group
    - BA: Boeing
    - CAT: Caterpillar
    - HON: Honeywell
    - MMM: 3M
    - AAPL: Apple
    - CRM: Salesforce
    - CSCO: Cisco
    - IBM: IBM
    - INTC: Intel
    - MSFT: Microsoft

    Return None if the stock ticker is not found in the document."""
    )

    sentiment_score: Literal[-1, 0, 1] | None = Field(
        description="This score represents the sentiment of investors or the market toward a specific stock or sector. A score of -1 indicates negative sentiment (bearish outlook), 1 indicates positive sentiment (bullish outlook), and 0 indicates neutral sentiment (indifference or uncertainty)."
    )


class Newspaper(BaseModel):
    sectors: Sector = Field(
        description="A list of sectors mentioned in the news article. Perform sentiment analysis for each sector to determine the market's attitude. For example, 'communication' includes companies like DIS (The Walt Disney Company) and VZ (Verizon Communications), while 'financials' includes companies like JPM (JPMorgan Chase) and GS (Goldman Sachs)."
    )
    stocks: Stock = Field(
        description="A list of stocks mentioned in the news article, identified by their ticker symbols. Perform sentiment analysis for each stock to determine the market's attitude. For example, DIS represents The Walt Disney Company, and AAPL represents Apple."
    )

# --- Field Warehouse and DocumentProcessor ---

class Warehouse:
    def __init__(self):
        # Use only the Newspaper model fields.
        self.fields = Newspaper.model_fields

    def get_field_schema(self, field_name: str) -> Field:
        """Retrieve a field definition by its key, supporting dot-notation for nested fields."""
        if '.' in field_name:
            main_field, nested_field = field_name.split('.', 1)
            main_def = self.fields.get(main_field)
            if main_def and hasattr(main_def.annotation, "model_fields"):
                return main_def.annotation.model_fields.get(nested_field)
            return None
        return self.fields.get(field_name)

class Factory:
    """Document processor that assembles prompts retaining nested model structures."""
    def __init__(self):
        self.field_warehouse = Warehouse()

    def assemble_prompt(self, keys: List[str]) -> str:
        """Create system prompt for field extraction with nested model structures where available."""
        fields_desc = []
        model_structures = []

        for key in keys:
            field = self.field_warehouse.get_field_schema(key)
            if field:
                if field.description:
                    fields_desc.append(f"- {key}: {field.description}")
                # If the field's annotation has nested model definitions, add their structure.
                if hasattr(field.annotation, "model_fields"):
                    nested_fields = field.annotation.model_fields
                    model_name = field.annotation.__name__
                    model_structures.append(f"\n{model_name} structure:")
                    for nested_name, nested_field in nested_fields.items():
                        if nested_field.description:
                            model_structures.append(f"- {nested_name}: {nested_field.description}")

        prompt = (
            "You are a document analysis expert. I will provide you with a news article.\n"
            "Your task is to extract specific information from the article and return it in JSON format.\n"
            "Please extract the following information:\n"
            f"{chr(10).join(fields_desc)}\n"
        )

        if model_structures:
            prompt += (
                "\nImportant: The response must follow these model structures:\n"
                f"{chr(10).join(model_structures)}\n"
            )

        prompt += (
            "\nReturn the results in valid JSON format with exactly these field names and structures.\n"
            "If a field is not found, use null as the value.\n"
            "Only return the JSON object, no other text."
        )

        return prompt
