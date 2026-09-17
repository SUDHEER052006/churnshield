"""Synthetic stand-ins, one per dataset profile.

These exist so the repository runs end to end for anyone who clones it before
downloading the real CSV. They are NOT a substitute for real data and the UI
says so wherever they are used.

Both generators deliberately build in treatment-effect heterogeneity,
including subgroups where an offer BACKFIRES. Without that, every customer
responds to every offer identically, all four quadrants collapse into two,
and the entire premise of the project has nothing to find. Each subgroup is
defined by a single numeric threshold, because a causal forest recovers a
clean split on one continuous feature, while a three-way categorical
interaction at this sample size gets shrunk back toward the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ===========================================================================
# Cell2Cell
# ===========================================================================
def synth_cell2cell(n: int = 51_047, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    months = np.clip(np.abs(rng.normal(19, 14, n)).round(), 1, 61).astype(int)
    revenue = np.clip(rng.gamma(6.0, 9.5, n), 10, 380).round(2)
    recurring = np.clip(revenue * rng.uniform(0.45, 0.85, n), 5, 250).round(2)
    minutes = np.clip(rng.gamma(3.5, 150, n), 0, 4000).round()
    overage = np.clip(rng.gamma(1.2, 45, n) - 20, 0, 900).round()
    equip_days = np.clip(rng.normal(380, 240, n), 10, 1800).round().astype(int)
    care_calls = rng.poisson(1.3, n)
    dropped = np.clip(rng.gamma(1.4, 4.5, n), 0, 60).round(1)
    blocked = np.clip(rng.gamma(1.1, 3.0, n), 0, 50).round(1)

    # Reaching the retention desk is itself driven by dissatisfaction.
    retention_propensity = _sigmoid(
        -2.3 + dropped / 14 + care_calls / 3.2 + (revenue - 58) / 95)
    retention_calls = rng.binomial(2, np.clip(retention_propensity, 0, 0.75), n)
    reached_desk = retention_calls >= 1
    # Among those who reached it, roughly half accept whatever was offered.
    accepted = np.where(reached_desk, rng.binomial(1, 0.48, n), 0)

    web_capable = rng.choice(["Yes", "No"], n, p=[0.58, 0.42])
    refurbished = rng.choice(["Yes", "No"], n, p=[0.24, 0.76])
    has_card = rng.choice(["Yes", "No"], n, p=[0.32, 0.68])

    age = np.where(rng.uniform(size=n) < 0.18, 0,
                   np.clip(rng.normal(44, 15, n), 18, 92).round()).astype(int)

    df = pd.DataFrame({
        "CustomerID": [f"{3000000 + i}" for i in range(n)],
        "MonthlyRevenue": revenue,
        "MonthlyMinutes": minutes,
        "TotalRecurringCharge": recurring,
        "OverageMinutes": overage,
        "RoamingCalls": np.clip(rng.gamma(0.8, 2.5, n), 0, 60).round(1),
        "PercChangeMinutes": rng.normal(-11, 260, n).round(1),
        "PercChangeRevenues": rng.normal(-1.2, 32, n).round(1),
        "DroppedCalls": dropped,
        "BlockedCalls": blocked,
        "CustomerCareCalls": care_calls,
        "ReceivedCalls": np.clip(rng.gamma(2.2, 55, n), 0, 1600).round(),
        "OutboundCalls": np.clip(rng.gamma(2.0, 22, n), 0, 800).round(),
        "InboundCalls": np.clip(rng.gamma(1.8, 12, n), 0, 500).round(),
        "PeakCallsInOut": np.clip(rng.gamma(2.4, 45, n), 0, 1500).round(),
        "OffPeakCallsInOut": np.clip(rng.gamma(1.9, 30, n), 0, 1200).round(),
        "MonthsInService": months,
        "UniqueSubs": rng.choice([1, 2, 3, 4], n, p=[0.72, 0.19, 0.06, 0.03]),
        "ActiveSubs": rng.choice([1, 2, 3], n, p=[0.80, 0.16, 0.04]),
        "Handsets": rng.choice([1, 2, 3, 4], n, p=[0.62, 0.25, 0.09, 0.04]),
        "HandsetModels": rng.choice([1, 2, 3], n, p=[0.70, 0.23, 0.07]),
        "CurrentEquipmentDays": equip_days,
        "AgeHH1": age,
        "IncomeGroup": rng.integers(0, 10, n),
        "AdjustmentsToCreditRating": rng.poisson(0.35, n),
        "ReferralsMadeBySubscriber": rng.poisson(0.18, n),
        "RetentionCalls": retention_calls,
        "RetentionOffersAccepted": accepted,
        "ChildrenInHH": rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "HandsetRefurbished": refurbished,
        "HandsetWebCapable": web_capable,
        "Homeownership": rng.choice(["Known", "Unknown"], n, p=[0.61, 0.39]),
        "BuysViaMailOrder": rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
        "RespondsToMailOffers": rng.choice(["Yes", "No"], n, p=[0.40, 0.60]),
        "OptOutMailings": rng.choice(["Yes", "No"], n, p=[0.06, 0.94]),
        "NonUSTravel": rng.choice(["Yes", "No"], n, p=[0.06, 0.94]),
        "OwnsComputer": rng.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "HasCreditCard": has_card,
        "NewCellphoneUser": rng.choice(["Yes", "No"], n, p=[0.24, 0.76]),
        "CreditRating": rng.choice(
            ["1-Highest", "2-High", "3-Good", "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"],
            n, p=[0.13, 0.17, 0.20, 0.19, 0.15, 0.09, 0.07]),
        "PrizmCode": rng.choice(["Suburban", "Town", "Other", "Rural"], n,
                                p=[0.36, 0.25, 0.28, 0.11]),
        "Occupation": rng.choice(
            ["Professional", "Crafts", "Clerical", "Student", "Homemaker",
             "Retired", "Self", "Other"],
            n, p=[0.36, 0.09, 0.07, 0.04, 0.04, 0.05, 0.08, 0.27]),
        "MaritalStatus": rng.choice(["Yes", "No", "Unknown"], n, p=[0.36, 0.34, 0.30]),
        "MadeCallToRetentionTeam": np.where(reached_desk, "Yes", "No"),
    })

    # ---- churn generating process ----------------------------------------
    z = -0.10                                  # Cell2Cell is ~50% churn by design
    z = z + (revenue - 58) / 120
    z = z + dropped / 22 + blocked / 40 + care_calls / 5.5
    z = z - months / 34
    z = z + equip_days / 900
    z = z + np.where(refurbished == "Yes", 0.14, 0.0)
    z = z + np.where(web_capable == "Yes", -0.20, 0.0)
    z = z + np.where(has_card == "Yes", -0.16, 0.0)
    z = z + np.where(df.NewCellphoneUser.values == "Yes", 0.18, 0.0)
    z = z + df.AdjustmentsToCreditRating.values * 0.12
    z = z - df.ReferralsMadeBySubscriber.values * 0.16
    z = z + reached_desk * 0.55                # asking about leaving predicts leaving

    treated_offer = accepted == 1
    # Deliberately modest. An offer that moves everyone by 20 points makes
    # every customer persuadable and the quadrant split degenerate -- real
    # retention effects are single-digit percentage points for most people.
    z = z + np.where(treated_offer, -0.42, 0.0)

    # ---- heterogeneity, including two genuine reversals ------------------
    # Lost causes: at the top of the price book people leave on price alone,
    # and every concession is cancelled out.
    price_driven = revenue > 105
    z = z + np.where(price_driven, 1.25, 0.0)
    z = z + np.where(price_driven & treated_offer, 0.95, 0.0)
    z = z + np.where(price_driven & (web_capable == "Yes"), 0.30, 0.0)
    z = z + np.where(price_driven & (has_card == "Yes"), 0.26, 0.0)
    z = z + np.where(price_driven & (refurbished == "No"), 0.22, 0.0)

    # Sleeping dogs: long-tenure customers are not thinking about leaving.
    # Any intervention reminds them to shop around.
    dormant = months > 30
    z = z + np.where(dormant & treated_offer, 1.45, 0.0)
    z = z + np.where(dormant & (web_capable == "Yes"), 0.75, 0.0)
    z = z + np.where(dormant & (has_card == "Yes"), 0.70, 0.0)
    z = z + np.where(dormant & (refurbished == "No"), 0.70, 0.0)

    # A handset upgrade earns its keep mainly for heavy data-era users.
    z = z + np.where((web_capable == "Yes") & (minutes > 900), -0.40, 0.0)

    z = z + rng.normal(0, 0.42, n)
    df["Churn"] = np.where(rng.uniform(size=n) < _sigmoid(z), "Yes", "No")
    return df


# ===========================================================================
# IBM Telco
# ===========================================================================
def synth_telco(n: int = 7043, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = np.clip(np.abs(rng.normal(26, 22, n)).round(), 1, 72).astype(int)
    contract = np.where(
        tenure > 40,
        rng.choice(["Two year", "One year", "Month-to-month"], n, p=[0.45, 0.30, 0.25]),
        rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.70, 0.20, 0.10]))
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    has_net = internet != "No"

    base = np.where(internet == "Fiber optic", 88.0, np.where(internet == "DSL", 58.0, 21.0))
    monthly = np.clip(rng.normal(base, 14.0), 18.0, 125.0).round(2)
    total = (monthly * tenure * rng.uniform(0.93, 1.05, n)).round(2)

    def addon(p_yes):
        v = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        return np.where(has_net, v, "No internet service")

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n, p=[0.48, 0.52]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": addon(0.33), "OnlineBackup": addon(0.35),
        "DeviceProtection": addon(0.34), "TechSupport": addon(0.33),
        "StreamingTV": addon(0.39), "StreamingMovies": addon(0.39),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.23, 0.22, 0.21]),
        "MonthlyCharges": monthly, "TotalCharges": total,
    })

    locked_in = (df.Contract != "Month-to-month").values
    has_support = (df.TechSupport.values == "Yes")
    has_security = (df.OnlineSecurity.values == "Yes")
    on_autopay = df.PaymentMethod.str.contains("automatic").values

    z = -1.30
    z = z + np.where(df.Contract == "Month-to-month", 1.15,
            np.where(df.Contract == "One year", 0.0, -0.65))
    z = z + np.where(internet == "Fiber optic", 0.72, np.where(internet == "DSL", 0.05, -0.55))
    z = z + np.where(df.PaymentMethod == "Electronic check", 0.60,
            np.where(on_autopay, -0.36, 0.04))
    z = z + np.where(has_support, -0.36, np.where(df.TechSupport == "No", 0.42, 0.0))
    z = z + np.where(has_security, -0.30, np.where(df.OnlineSecurity == "No", 0.30, 0.0))
    z = z - tenure / 26.0 + (monthly - 65.0) / 42.0
    z = z + np.where(df.SeniorCitizen == 1, 0.28, 0.0)
    z = z + np.where(df.Partner == "Yes", -0.22, 0.06)
    z = z + np.where(df.Dependents == "Yes", -0.18, 0.0)

    resents_lock_in = (internet == "Fiber optic") & (monthly > 85) & (df.Dependents.values == "No")
    z = z + np.where(resents_lock_in & locked_in, 1.9, 0.0)

    price_driven = monthly > 100
    z = z + np.where(price_driven, 1.15, 0.0)
    z = z + np.where(price_driven & locked_in, 0.85, 0.0)
    z = z + np.where(price_driven & has_support, 0.80, 0.0)
    z = z + np.where(price_driven & has_security, 0.70, 0.0)
    z = z + np.where(price_driven & on_autopay, 0.80, 0.0)

    dormant = tenure > 48
    z = z + np.where(dormant & locked_in, 1.60, 0.0)
    z = z + np.where(dormant & has_support, 1.30, 0.0)
    z = z + np.where(dormant & has_security, 1.30, 0.0)
    z = z + np.where(dormant & on_autopay, 1.30, 0.0)

    z = z + np.where(has_support & (monthly > 75), -0.40, 0.0)
    z = z + rng.normal(0, 0.45, n)

    df["Churn"] = np.where(rng.uniform(size=n) < _sigmoid(z), "Yes", "No")
    return df
