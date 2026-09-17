"""Central configuration: dataset profiles, treatments, economic assumptions.

Every business assumption lives here and nowhere else. If a judge asks "where
did that number come from", the answer is this file.

Two dataset profiles ship. `cell2cell` is the default because it contains an
OBSERVED retention intervention (`RetentionOffersAccepted`) -- no other public
churn dataset hands you the treatment variable directly. `telco` is kept
working so the project runs on the familiar benchmark too.

Switch with the CHURNSHIELD_DATASET environment variable:

    $env:CHURNSHIELD_DATASET = "telco"
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"

for _d in (DATA_DIR, ARTIFACT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Causal forests are fitted on at most this many rows, then used to predict an
# effect for every customer. Fitting on 50k+ rows costs minutes per treatment
# and buys almost nothing: the estimator is recovering subgroup structure, and
# 20k rows already pins that down. Raise it for a final run if you like.
CAUSAL_FIT_MAX_ROWS = 20_000

DATASET = os.environ.get("CHURNSHIELD_DATASET", "cell2cell").strip().lower()


# ===========================================================================
# Cell2Cell -- Duke / Teradata Center for CRM telecom churn
# 71,047 customers, 58 features, ~50% churn by design (oversampled)
# ===========================================================================
CELL2CELL = {
    "name": "cell2cell",
    "display": "Cell2Cell (Duke/Teradata CRM)",
    "filenames": [
        "cell2celltrain.csv", "cell2cell.csv", "Cell2Cell.csv",
        "cell2cell_train.csv", "telecom_churn_cell2cell.csv",
    ],
    "id_col": "CustomerID",
    "target": "Churn",
    "revenue_col": "MonthlyRevenue",
    "tenure_col": "MonthsInService",

    "numeric": [
        "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
        "OverageMinutes", "RoamingCalls", "PercChangeMinutes",
        "PercChangeRevenues", "DroppedCalls", "BlockedCalls",
        "CustomerCareCalls", "ReceivedCalls", "OutboundCalls", "InboundCalls",
        "PeakCallsInOut", "OffPeakCallsInOut", "MonthsInService", "UniqueSubs",
        "ActiveSubs", "Handsets", "HandsetModels", "CurrentEquipmentDays",
        "AgeHH1", "IncomeGroup", "AdjustmentsToCreditRating",
        "ReferralsMadeBySubscriber", "RetentionCalls",
    ],
    "categorical": [
        "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable",
        "Homeownership", "BuysViaMailOrder", "RespondsToMailOffers",
        "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard",
        "NewCellphoneUser", "CreditRating", "PrizmCode", "Occupation",
        "MaritalStatus", "MadeCallToRetentionTeam",
    ],

    # Columns computed during load, so a treatment can be an expression.
    "derived": {
        # Did this customer accept a retention offer?
        "AcceptedRetentionOffer":
            lambda df: (df["RetentionOffersAccepted"].fillna(0) >= 1)
                       .map({True: "Yes", False: "No"}),
        # Age band, used as a fairness dimension (Cell2Cell has no gender).
        "AgeBand":
            lambda df: df["AgeHH1"].fillna(0).map(
                lambda a: "Unknown" if a <= 0 else
                          "Under 35" if a < 35 else
                          "35-54" if a < 55 else "55+"),
    },

    "sensitive": ["MaritalStatus", "ChildrenInHH", "AgeBand"],

    "profile_fields": [
        "MonthsInService", "MonthlyRevenue", "TotalRecurringCharge",
        "MonthlyMinutes", "OverageMinutes", "DroppedCalls",
        "CustomerCareCalls", "RetentionCalls", "AcceptedRetentionOffer",
        "CurrentEquipmentDays", "HandsetWebCapable", "HandsetRefurbished",
        "HasCreditCard", "CreditRating", "MaritalStatus", "AgeBand",
    ],

    # (label, function) pairs fed to the error-analysis tree.
    "error_features": [
        ("Months in service", lambda d: d["MonthsInService"]),
        ("Monthly revenue", lambda d: d["MonthlyRevenue"]),
        ("Dropped calls", lambda d: d["DroppedCalls"]),
        ("Care calls", lambda d: d["CustomerCareCalls"]),
        ("Equipment age (days)", lambda d: d["CurrentEquipmentDays"]),
        ("Called retention", lambda d: (d["RetentionCalls"] > 0).astype(int)),
        ("Web-capable handset", lambda d: (d["HandsetWebCapable"] == "Yes").astype(int)),
    ],

    "treatments": {
        "retention_offer": {
            "label": "Retention offer",
            "short": "Retention offer",
            "detail": "The concession the retention desk is authorised to make",
            "column": "AcceptedRetentionOffer",
            "treated_values": ["Yes"],
            "control_values": ["No"],
            # Only customers who actually reached the retention desk are
            # comparable. Everyone else was never in a position to accept.
            "eligible": lambda df: df["RetentionCalls"].fillna(0) >= 1,
            "caveat": "Treated = accepted an offer. Acceptance is a choice, so this "
                      "compares acceptors against decliners who both reached the desk. "
                      "Cleanest available, still not randomised.",
            "cost": lambda row, a: 0.15 * float(row["MonthlyRevenue"]) * a["horizon_months"],
        },
        "handset_upgrade": {
            "label": "Handset upgrade",
            "short": "Handset upgrade",
            "detail": "Subsidised move to a web-capable handset",
            "column": "HandsetWebCapable",
            "treated_values": ["Yes"],
            "control_values": ["No"],
            "cost": lambda row, a: a["handset_subsidy"],
        },
        "new_device": {
            "label": "New (non-refurbished) device",
            "short": "New device",
            "detail": "Replace a refurbished handset with new stock",
            "column": "HandsetRefurbished",
            "treated_values": ["No"],
            "control_values": ["Yes"],
            "cost": lambda row, a: a["new_device_cost"],
        },
        "auto_billing": {
            "label": "Card auto-billing",
            "short": "Auto-billing",
            "detail": "One-time credit to move billing onto a card",
            "column": "HasCreditCard",
            "treated_values": ["Yes"],
            "control_values": ["No"],
            "cost": lambda row, a: a["autopay_incentive"],
        },
    },

    "assumptions": {
        "budget": 250000.0,
        "gross_margin": 0.45,
        "horizon_months": 12,
        "handset_subsidy": 120.0,
        "new_device_cost": 180.0,
        "autopay_incentive": 15.0,
        "currency": "$",
    },
}


# ===========================================================================
# IBM Telco -- the familiar benchmark, kept working
# ===========================================================================
TELCO = {
    "name": "telco",
    "display": "IBM Telco Customer Churn",
    "filenames": [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv", "telco_customer_churn.csv",
        "Telco-Customer-Churn.csv",
    ],
    "id_col": "customerID",
    "target": "Churn",
    "revenue_col": "MonthlyCharges",
    "tenure_col": "tenure",

    "numeric": ["tenure", "MonthlyCharges", "TotalCharges"],
    "categorical": [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ],
    "derived": {},
    "sensitive": ["gender", "SeniorCitizen", "Partner"],
    "profile_fields": [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod",
        "gender", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling",
    ],
    "error_features": [
        ("Tenure (months)", lambda d: d["tenure"]),
        ("Monthly charge", lambda d: d["MonthlyCharges"]),
        ("Fibre optic", lambda d: (d["InternetService"] == "Fiber optic").astype(int)),
        ("Month-to-month", lambda d: (d["Contract"] == "Month-to-month").astype(int)),
        ("Electronic cheque", lambda d: (d["PaymentMethod"] == "Electronic check").astype(int)),
        ("Has tech support", lambda d: (d["TechSupport"] == "Yes").astype(int)),
        ("Senior citizen", lambda d: d["SeniorCitizen"].astype(int)),
    ],
    "treatments": {
        "contract": {
            "label": "12-month contract", "short": "Contract lock-in",
            "detail": "15% discount for a 12-month commitment",
            "column": "Contract",
            "treated_values": ["One year", "Two year"],
            "control_values": ["Month-to-month"],
            "cost": lambda row, a: 0.15 * float(row["MonthlyCharges"]) * a["horizon_months"],
        },
        "support": {
            "label": "Tech support add-on", "short": "Tech support",
            "detail": "Priority technical support bundled at no charge",
            "column": "TechSupport",
            "treated_values": ["Yes"], "control_values": ["No"],
            "cost": lambda row, a: a["support_cost_per_month"] * a["horizon_months"],
        },
        "security": {
            "label": "Security bundle", "short": "Security bundle",
            "detail": "Online security and backup, waived for the term",
            "column": "OnlineSecurity",
            "treated_values": ["Yes"], "control_values": ["No"],
            "cost": lambda row, a: a["security_cost_per_month"] * a["horizon_months"],
        },
        "autopay": {
            "label": "Auto-pay migration", "short": "Auto-pay",
            "detail": "One-time credit to switch to auto-debit",
            "column": "PaymentMethod",
            "treated_values": ["Bank transfer (automatic)", "Credit card (automatic)"],
            "control_values": ["Electronic check", "Mailed check"],
            "cost": lambda row, a: a["autopay_incentive"],
        },
    },
    "assumptions": {
        "budget": 40000.0, "gross_margin": 0.45, "horizon_months": 12,
        "support_cost_per_month": 6.0, "security_cost_per_month": 4.5,
        "autopay_incentive": 15.0, "currency": "$",
    },
}


PROFILES = {"cell2cell": CELL2CELL, "telco": TELCO}
PROFILE = PROFILES.get(DATASET, CELL2CELL)

# --- names the rest of the codebase imports -------------------------------
DATASET_DISPLAY = PROFILE["display"]
FILENAMES = PROFILE["filenames"]
ID_COL = PROFILE["id_col"]
TARGET = PROFILE["target"]
REVENUE_COL = PROFILE["revenue_col"]
TENURE_COL = PROFILE["tenure_col"]
NUMERIC_FEATURES = PROFILE["numeric"]
CATEGORICAL_FEATURES = PROFILE["categorical"]
DERIVED = PROFILE["derived"]
SENSITIVE_FEATURES = PROFILE["sensitive"]
PROFILE_FIELDS = PROFILE["profile_fields"]
ERROR_FEATURES = PROFILE["error_features"]
TREATMENTS = PROFILE["treatments"]
DEFAULT_ASSUMPTIONS = dict(PROFILE["assumptions"])

# Quadrant boundaries. A customer is a sleeping dog when the best available
# treatment effect is meaningfully negative: contacting them makes it worse.
SLEEPING_DOG_THRESHOLD = -0.012
PERSUADABLE_UPLIFT = 0.045
PERSUADABLE_RISK = 0.40
LOST_CAUSE_RISK = 0.55
