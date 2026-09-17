# ChurnShield — Plain English Explainer

Everything in this project, in the simplest words possible. Written so any team
member can explain any screen, any number, and any word a judge asks about.

**Read this before the demo. It is the script.**

---

## Part 0 — The whole idea in 30 seconds

A phone company loses customers. That's called **churn**.

Normal churn projects build a model that says *"this customer will probably
leave."* Then the company sends those customers a discount.

**That wastes most of the money.** Because "likely to leave" and "can be saved"
are not the same thing. Some people are leaving no matter what you offer. Some
were staying anyway and you just gave them a discount for free. And some people
only start thinking about leaving *because* you called them.

ChurnShield asks the harder question: **"If I make this offer to this person,
does their decision actually change?"**

Then it spends the budget only on the people whose answer is yes.

> **One line:** Don't predict who leaves. Predict who you can save.

---

## Part 1 — Words you need to know

### The basics

| Word | Simplest meaning |
|---|---|
| **Churn** | A customer leaves / cancels. |
| **Churn probability** | The model's guess, from 0% to 100%, that this person will leave. |
| **Base rate** | How often churn happens across everyone. If 26% of customers leave, the base rate is 26%. |
| **Feature** | One column of information about a customer (their bill, how long they've been with you, how many dropped calls they had). |
| **Model** | A program that learned patterns from past customers and uses them to guess about new ones. |
| **Training** | Showing the model old customers where we already know who left, so it can learn. |
| **Holdout / test set** | Customers we hide from the model during training, so we can check if it actually learned anything or just memorised. |

### The important distinction (this is the whole project)

| Word | Simplest meaning |
|---|---|
| **Prediction** | "What will happen?" — Who is likely to leave. |
| **Causation** | "What happens if I *do* something?" — Whether an offer changes their mind. |
| **Treatment** | Something you *do* to a customer. Here: a retention offer, a handset upgrade, a new device, moving them to card billing. We also just call these **offers**. |
| **Uplift** | How much an offer improves the chance that a customer stays. Measured in **percentage points**. |
| **CATE** | The technical name for uplift. Stands for *Conditional Average Treatment Effect*. "Conditional" just means "for this particular kind of customer". When a judge says CATE, they mean uplift. |

**The classic example of why this matters:** hospitals. People who go to
hospitals die more often than people who don't. That's a true pattern
(prediction). But hospitals don't cause death — sick people go to hospitals. If
you acted on the pattern, you'd close hospitals. Churn models have the same trap.

### Percentage points vs percent

If churn goes from **40% down to 32%**, that is a drop of **8 percentage points**
— not 8 percent. We always write "pp" to avoid confusion. This matters because
judges notice when you get it wrong.

### The four kinds of customer

This is the most important table in the project. Everything on screen is
colour-coded by it.

| Group | What they do | What to do about them | Colour |
|---|---|---|---|
| **Persuadable** | Leaves if you do nothing. Stays if you make an offer. | **Spend here. This is the only group that makes money.** | Green |
| **Sure thing** | Stays either way. | Don't spend. A discount here is money given away for nothing. | Blue |
| **Lost cause** | Leaves either way. | Don't spend. Nothing you offer will change it. | Grey |
| **Sleeping dog** | Was staying quietly — but leaves *because* you contacted them. | **Never contact. Your call is what reminds them to shop around.** | Red |

A normal churn model ranks by risk. Ranking by risk sends your budget straight
at Lost causes (they have the highest risk scores) and Sleeping dogs (contacting
them makes it worse). That's the money ChurnShield saves.

### Explaining the model

| Word | Simplest meaning |
|---|---|
| **SHAP** | A method that says *why* the model made one particular guess. It splits the prediction into pieces: "the short tenure pushed risk up by 8 points, the low bill pulled it down by 3." |
| **Global importance** | Which features matter most across *everyone*. |
| **Local explanation** | Which features mattered for *this one customer*. |

Keep those two separate. Mixing them up is the most common way churn dashboards
mislead people.

### Scoring the model

| Word | Simplest meaning |
|---|---|
| **Accuracy** | What fraction of guesses were right. **Misleading here** — see the accuracy trap below. |
| **Precision** | Of the people we flagged as churners, how many really churned. |
| **Recall** | Of the people who really churned, how many we caught. |
| **F1** | One number that balances precision and recall. |
| **ROC-AUC** | How good the model is at *ranking* — putting churners above non-churners. 0.5 = coin flip, 1.0 = perfect. |
| **PR-AUC** | Like ROC-AUC, but fairer when churn is rare. |
| **Brier score** | How honest the probabilities are. **Lower is better.** |
| **Calibration** | Whether "80%" really means 80%. If you take everyone the model said 80% about, did 80% of them actually leave? |
| **Confusion matrix** | A 2×2 box counting right and wrong guesses of each type. |
| **False positive** | We said they'd leave, they stayed. Cost: a wasted offer. |
| **False negative** | We said they'd stay, they left. Cost: a customer nobody tried to save. **The expensive one.** |
| **Threshold** | The cut-off. Above 42% probability we call it "at risk". Lower threshold = catch more churners but more false alarms. |

**The accuracy trap — memorise this one.** A model that predicts "nobody ever
leaves" gets high accuracy when most people stay. It is completely useless. That
is exactly why we put a **Dummy** model in our comparison table: to show on
screen that accuracy alone would have picked a useless model. Judges love this.

### Scoring the uplift model

There's no "correct answer" to check uplift against, because you never see both
branches for the same person — you either made the offer or you didn't. So we
score it differently:

| Word | Simplest meaning |
|---|---|
| **Qini curve** | A graph showing how many extra customers you keep as you work down your list. Higher line = better targeting. |
| **Qini coefficient** | One number summarising that curve. Higher is better. |
| **AUUC** | Another version of the same idea (*Area Under the Uplift Curve*). |
| **Uplift @ 20%** | How well you do if you only contact your top 20% of customers. |
| **Confidence interval (CI)** | A range instead of a single guess: "somewhere between +2 and +9 points." **If the range crosses zero, we don't know if the offer helps at all, so we don't spend money on it.** |

### Trust and honesty

| Word | Simplest meaning |
|---|---|
| **Observational data** | Nobody ran an experiment. Customers chose things themselves. This is what we have. |
| **Randomised experiment** | Someone flipped a coin to decide who got the offer. This is the gold standard. We don't have it. |
| **Confounding** | A hidden cause that fools you. Example: loyal people *choose* long contracts. So long contracts *look* like they cause loyalty, when really loyalty caused the contract. Backwards. |
| **Refutation test** | We attack our own result and see if it survives. |
| **Placebo test** | Replace the real offer with a fake random one. If we still "find" an effect, our method is broken. **A good result here is *no* effect.** |
| **Random common cause** | Add a meaningless extra column. The answer shouldn't move. |
| **Data subset test** | Re-run on a random 80% of the data. The answer should stay about the same. |
| **Fairness** | Checking the model doesn't fail one group of people more than others. |
| **Demographic parity difference** | The gap in how often different groups get flagged. Smaller is better. Our gate is 0.10. |
| **Equalised odds difference** | The gap in *error rates* between groups. Smaller is better. |
| **Error analysis** | Finding the specific pockets of customers where the model is worst. |
| **Drift** | Customer behaviour changes over time, so the model slowly goes stale and needs retraining. |

### Money words

| Word | Simplest meaning |
|---|---|
| **CLV** | Customer Lifetime Value. What this customer is worth to us over the chosen time window. |
| **Gross margin** | The share of revenue that's actual profit. We assume 45%. |
| **Value horizon** | How far ahead we count the money. We use 12 months. |
| **Cost of offer** | What the offer costs us (a discount, a subsidised handset, a billing credit). |
| **Expected net value** | `uplift × customer value − cost of offer`. **This is what the worklist is ranked by.** |
| **Knapsack** | The budget-packing method: keep buying the best value-per-dollar until the money runs out. |
| **ROI** | Return on investment. Profit recovered per dollar spent. |

### The libraries we use (and why judges care)

| Tool | What it does | Who made it |
|---|---|---|
| **scikit-learn / XGBoost** | The churn prediction models | Open source |
| **SHAP** | Explains individual predictions | Open source |
| **EconML** | Estimates uplift / CATE | **Microsoft Research** |
| **DoWhy** | Refutation tests | **Microsoft** |
| **Fairlearn** | Fairness audit | **Microsoft** |
| **FastAPI** | The backend server | Open source |
| **React + Recharts** | The dashboard you see | Open source |

Three of those are Microsoft's own open-source projects. They are **free** — no
Azure account needed. Azure is only where we'd deploy it.

---

## Part 2 — The dataset

**Cell2Cell** (Duke University / Teradata Center for CRM). About 51,000–71,000
mobile customers, 58 columns.

**Why this one and not the usual "Telco" dataset:** Cell2Cell contains
`RetentionCalls` and `RetentionOffersAccepted` — it actually records *retention
offers being made and accepted*. No other public churn dataset gives you the
intervention as a column. For a project about interventions, that's the whole
ballgame. The IBM Telco dataset everyone uses has no offer column at all.

**The four offers we model** (each is a real column, not something we invented):

| Offer | Column in the data |
|---|---|
| Retention offer | `RetentionOffersAccepted` |
| Handset upgrade | `HandsetWebCapable` |
| New (non-refurbished) device | `HandsetRefurbished` |
| Card auto-billing | `HasCreditCard` |

**If a judge asks "did you make the data up?":** the app shows a red *Synthetic
data* badge in the sidebar whenever it is running on the built-in stand-in
generator. Drop the real CSV into `backend/data/` and the badge disappears. We
never present generated numbers as real ones.

---

## Part 3 — The screens, one by one

There are seven, grouped into three sections in the left sidebar:
**Operate** (do the work), **Verify** (prove it's trustworthy), **Ask**.

---

### 1. Command Center — *"Here's the problem, in money"*

**What it's for:** the opening slide. It sets up the entire argument in one look.

**The four tiles across the top:**

| Tile | What it means |
|---|---|
| **Revenue at risk / yr** | Yearly money tied to everyone likely to leave. This is the number every other churn dashboard shows. |
| **Savable revenue** | The money we can *actually* recover. Much smaller. |
| **Persuadables found** | How many customers an offer would genuinely sway. |
| **Would be harmed by an offer** | How many customers an offer would make *worse*. |

> **The line to say out loud:** *"Look at the first two tiles. Every other churn
> dashboard shows you the first number. The gap between them is money that
> cannot be recovered no matter what you spend — and no risk model can see that
> gap, because it's a causal question, not a prediction."*

**The scatter plot — "Risk is not savability".** Every dot is a customer.

- **Left to right** = how likely they are to leave (churn probability)
- **Bottom to top** = how much an offer would help (uplift)
- The dashed line across the middle is **zero effect**. Below it, offers backfire.

Normal churn models only look at the horizontal axis and target the right-hand
edge. **The money is at the top, not the right.** Dots are coloured by quadrant,
and clicking any dot opens that customer.

**Portfolio by quadrant.** Bars showing what share of the customer base falls
into each of the four groups, and how much annual revenue each carries.

**Churn drivers across the base.** The SHAP global importance chart — which
features matter most overall. Gender (or any sensitive attribute) is shown in
grey to make a point: it contributes nearly nothing, and the fairness audit
confirms it.

**Last pipeline run.** Every stage and score from the most recent run: dataset,
rows, model chosen, ROC-AUC, Brier, causal estimator, Qini, refutation results,
fairness gate, runtime. This is your "we're not hiding anything" panel.

---

### 2. Action List — *"Here's what to do today"*

**What it's for:** proving this is a product, not a science project.

**Say this:** *"This is not a list of who is at risk. It's a list of who to call,
which offer to make, and what we expect to get back. The output is a worklist, not
a chart."*

**The columns:**

| Column | Meaning |
|---|---|
| **#** | Rank. Best expected return first. |
| **Customer** | Their ID. Click any row to open them. |
| **Quadrant** | Which of the four groups they're in. |
| **Churn P** | Probability they leave if we do nothing. |
| **Uplift** | How many percentage points the offer improves their chance of staying. |
| **Recommended offer** | Which of the four offers is best *for this person*. |
| **Cost** | What that offer costs us. |
| **Expected net** | `uplift × their value − cost`. **The list is sorted by this.** |
| **Profile** | Quick facts — tenure, monthly revenue. |

**The filters:** switch between *All actionable*, *Persuadables*, *Lost causes*
and *Sleeping dogs*. Filter by offer type. Search a customer ID.

**Important point to make:** Sleeping dogs are **excluded by default**, and not
just sorted to the bottom — they're removed structurally, so increasing the
budget can never reach them. That's a deliberate safety design.

---

### 3. Customer 360 — *"Here's the proof, on one human being"*

**What it's for:** this is where you win the demo.

**Top four tiles:** who they are, their churn probability, the best uplift
available, and their quadrant.

**"Why the model says this"** — the SHAP chart for this one person.
- **Red bars pointing right** = pushed their risk *up*
- **Green bars pointing left** = pulled their risk *down*
- Bar length = how much

**"Would an offer change the outcome?"** — the most important chart in the app.
Each of the four offers gets a dot and a horizontal line:
- The **dot** is our best estimate of the effect
- The **line through it** is the confidence interval — our uncertainty
- The **dashed vertical line** is zero: no effect

**If the line crosses zero, we don't know if that offer works, so we don't spend
on it.** Green = confidently helps. Red = confidently *hurts*. Grey = unclear.

**The button that wins the round: "Jump to a sleeping dog."**

It takes you to a customer with high churn risk whose uplift is **negative**.

> **Say this:** *"Every risk-ranked model in this room would put this customer
> near the top of the call list. Our model says contacting them makes it worse —
> they were staying quietly, and our phone call is what reminds them to shop
> around. That's not a smaller number. That's the opposite sign."*

**What-if simulator.** Change a customer's details and watch the prediction move.

> **Be precise here, because a sharp judge will test you:** *"Every control here
> moves a real column the model was trained on. We deliberately do not have a
> 'discount' slider, because there is no discount column in the data — a slider
> for a feature the model never saw would produce a number that means nothing.
> And this shows how the **model's prediction** changes. It is not a promise
> about what the customer would really do."*

---

### 4. Budget Optimizer — *"Here's the money, proven"*

**What it's for:** turning the idea into a number a business person cares about.

**Three sliders:** retention budget, gross margin, value horizon. Drag any one
and the entire allocation re-solves live.

**The four result tiles:**

| Tile | Meaning |
|---|---|
| **Customers targeted** | How many people we'd contact. |
| **Expected retained margin** | Profit we expect to keep. |
| **Return on spend** | Profit per dollar spent. |
| **Lift over risk targeting** | **The headline.** How much better we do than ranking by churn risk with the *same budget*. |

**The Qini chart — "Uplift targeting vs risk targeting".**

Three lines, all on the same axis:
- **Blue** — our way, ranked by uplift
- **Orange** — the normal way, ranked by churn risk
- **Grey dashed** — random targeting, the floor

Left to right = contacting more and more people. Up = more customers kept.

> **Say this:** *"The shaded gap between the blue and orange lines is this entire
> project, measured in customers. Same budget, same offers, same data. The only
> difference is who we chose to call."*

**Where the budget goes.** Which offers the money is being spent on and how many
customers each reaches.

**Stated assumptions.** Every assumption written out in the open — margin,
horizon, each offer's cost, and the big one: *we assume 100% of people accept the
offer*, which real campaigns never achieve.

> **Say this before they ask:** *"These assumptions are inputs, not findings. The
> acceptance assumption makes every return figure an upper bound — scale it by
> your real conversion rate. We put that on the screen rather than in a footnote."*

---

### 5. Model Lab — *"Here's every model we tried, including the ones we threw away"*

**What it's for:** showing rigour. Most teams show one model and its best score.

**Stage 1 — churn classifiers.** A table of five models: Dummy, Logistic
Regression, Random Forest, XGBoost, and XGBoost + isotonic calibration. The one
we ship is highlighted green; rejected ones say so.

> **Point at the Dummy row:** *"That model never predicts churn at all, and its
> accuracy still looks respectable. That's why we don't select on accuracy. We
> select on PR-AUC and Brier score."*

**Calibration chart.** Predicted probability vs what actually happened. The
dashed diagonal is perfect. Our calibrated model hugs it; the raw one doesn't.
This matters because every money figure multiplies by that probability.

**Confusion matrix.** The four outcome counts. Call out the **false negatives** —
customers who left while the system said they were safe. That's the ethical cost
centre, not just a statistic.

**Stage 2 — causal uplift learners.** Five estimators: S-Learner, T-Learner,
X-Learner, DR-Learner, CausalForestDML.

> **A detail worth volunteering:** *"We ship CausalForestDML even when another
> learner scores slightly higher on Qini. It's the only one that gives a
> confidence interval per customer, and our allocator refuses to spend money
> against an estimate whose interval crosses zero. We'd rather have honest
> uncertainty than a better leaderboard score."*

---

### 6. Responsible AI — *"Here's us trying to break our own work"*

**What it's for:** Microsoft publishes Responsible AI as a company pillar. A
judge is very likely to ask. Answer before they do.

**Three tiles:** fairness gate (pass/fail), refutation tests (how many survived),
worst cohort error.

**Fairlearn — performance by group.** Two bars per group: how often we flag them,
and **false negative rate** — how often we miss a real churner.

> **Say this:** *"The false negative rate is the one that matters ethically. A
> missed churner is a customer nobody tried to save. If that rate were much
> higher for one group, we'd be quietly abandoning them."*

**Refutation tests.** The three attacks on our own causal claim, with expected vs
observed values and pass/fail.

> **Say this:** *"The placebo test is the important one. We swap the real offer
> for a random fake one. If we still found an effect, our method would be finding
> patterns in noise. A good result here is finding nothing."*

**Error analysis.** The customer groups where the model is worst. These are
**discovered automatically** by fitting a small decision tree to the model's own
mistakes — the same idea as Microsoft's Error Analysis dashboard. We didn't pick
them by hand. Flagged cohorts are sent to humans instead of automated action.

**Operating limits.** What we do *not* claim. Read this one aloud if you have
time; it's unusual and it lands well.

---

### 7. Churn Copilot — *"Ask it a question"*

**What it's for:** an approachable finish.

Type a question, get an answer from the real scored data.

**The thing to point out:** under every answer there's a line showing the
**intent** and **parameters** that ran.

> **Say this:** *"We deliberately do not let a language model write queries
> against the customer database. The question is matched to a fixed set of
> approved queries with bound parameters, and every answer shows you exactly
> which one ran. It's auditable, and it can't be talked into doing something it
> shouldn't."*

**Good questions to demo:**
- "Who should I spend my next 20000 on?"
- "Which customers should we NOT contact?" ← ties back to sleeping dogs
- "What are the main churn factors?"
- "Is the model fair?"

---

## Part 4 — The 90-second demo script

| Time | Screen | What you say |
|---|---|---|
| 0:00–0:20 | **Command Center** | "Revenue at risk is the number every churn dashboard shows. Savable revenue is what we can actually recover. The gap is invisible to a prediction model." |
| 0:20–0:35 | Scatter plot | "Everyone targets the right-hand edge — highest risk. The money is at the top. Watch this red dot at the bottom right." *(click it)* |
| 0:35–1:00 | **Customer 360** | "90% likely to leave. Every risk model would call this person first. Our estimate of the effect is **negative** — calling them makes it worse. They're a sleeping dog." |
| 1:00–1:20 | **Budget Optimizer** | *(drag the budget slider)* "Same budget, same offers. Ranked by uplift instead of risk, we keep this many more customers. That gap is the project." |
| 1:20–1:30 | **Responsible AI** | "Fairness gate passes, and all three refutation tests survived — including the placebo test, where finding nothing is the correct answer." |

---

## Part 5 — Questions judges ask, and how to answer

**"Isn't this just a churn model with extra steps?"**
> No. A churn model answers *who will leave*. This answers *whether an offer
> changes that*, which is a different mathematical object — a treatment effect,
> not a prediction. They give different rankings, and we show the gap on the
> Qini chart.

**"How do you know the causal estimates are real?"**
> We don't claim certainty. The data is observational, so we assume no hidden
> confounders, and we say that on screen. Then we attack the assumption three
> ways with refutation tests and report the results. That's a stronger claim than
> a good accuracy score.

**"Why not just use accuracy?"**
> Because a model that never predicts churn scores well on accuracy and is
> useless. That's why our comparison table includes a dummy baseline — to show it
> on screen. We select on PR-AUC and Brier.

**"Where did the cost and margin numbers come from?"**
> They're assumptions, not findings. They're editable sliders on the Budget page
> and they're listed in full underneath. Change them and every number updates.

**"Is the data real?"**
> Cell2Cell from Duke/Teradata. If we're running on the built-in stand-in
> generator, the sidebar shows a red *Synthetic data* badge — we never pass
> generated numbers off as real ones.

**"What would you do next?"**
> Validate the uplift engine on the Hillstrom dataset — a real randomised
> experiment — to show it recovers a known experimental effect. That closes the
> gap between "assumed causal" and "measured causal".

---

## Part 6 — Running it

```powershell
# backend (port 8000)
cd backend
.venv\Scripts\python -m src.build_artifacts      # trains everything, ~8 min
.venv\Scripts\python -m uvicorn app.main:app --port 8000

# frontend (port 5173)
cd frontend
npm run dev
```

Then open **http://localhost:5173**.

The **Theme** button top-right switches light/dark. Both are designed; the chart
colours were checked for colour-blind readability in each.

Switch datasets with `$env:CHURNSHIELD_DATASET = "telco"` before rebuilding.

**Rehearse the demo at least five times.** The idea is unusual, and unusual ideas
need a confident delivery to land.
