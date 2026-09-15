from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


INTENTS = [
    "account_access",
    "account_security",
    "payments_billing",
    "refunds",
    "ps_plus_subscription",
    "game_purchase_store",
    "download_installation",
    "console_hardware",
    "network_psn",
    "codes_dlc_content",
    "content_availability",
    "other",
]

ESCALATION_INTENTS = {
    "account_security",
    "refunds",
    "payments_billing",
}


class PlaystationAgent:

    def __init__(self, model_dir="artifacts"):
        self.model_dir = Path(model_dir)

        self.model = joblib.load(
            self.model_dir / "intent_model.joblib"
        )

        self.vectorizer = joblib.load(
            self.model_dir / "retrieval_vectorizer.joblib"
        )

        self.retrieval_matrix = joblib.load(
            self.model_dir / "retrieval_matrix.joblib"
        )

        self.responses = pd.read_pickle(
            self.model_dir / "retrieval_rows.pkl"
        )

    def classify(self, text):
        """
        Hybrid intent classification.

        High-signal domain rules handle explicit phrases such as
        PSN error codes, refunds, hacked accounts and DLC.
        Logistic Regression is used as the fallback for ambiguous
        messages.
        """

        t = str(text).lower()

        rules = {
            "account_security": [
                "hacked", "hack", "account compromised",
                "unauthorized", "someone accessed",
                "someone changed", "password changed",
                "stolen account", "fraud", "2fa",
                "two factor"
            ],

            "network_psn": [
                "nw-", "ce-", "ws-", "dns error",
                "network error", "connection error",
                "cannot connect", "can't connect",
                "unable to connect", "connecting",
                "connection", "wifi", "wi-fi", "psn"
            ],

            "refunds": [
                "refund", "refunds", "money back",
                "return purchase", "cancel purchase"
            ],

            "payments_billing": [
                "charged twice", "charged me",
                "payment failed", "payment declined",
                "billing", "credit card", "debit card",
                "transaction", "add funds", "adding funds"
            ],

            "download_installation": [
                "download", "downloading", "install",
                "installation", "installing",
                "corrupt download", "corrupted download"
            ],

            "codes_dlc_content": [
                "redeem code", "voucher", "voucher code",
                "dlc", "add-on", "addon", "entitlement",
                "redeem"
            ],

            "ps_plus_subscription": [
                "ps plus", "playstation plus", "ps+",
                "subscription", "renewal",
                "renew subscription"
            ],

            "game_purchase_store": [
                "buy the game", "buy a game",
                "bought the game", "purchase",
                "purchased", "playstation store",
                "ps store", "game price"
            ],

            "console_hardware": [
                "controller", "disc eject", "eject disc",
                "console turns on", "console won't turn on",
                "ps4 turns on", "ps5 turns on",
                "overheat", "overheating", "safe mode"
            ],

            "content_availability": [
                "when will", "when is", "available",
                "availability", "release date",
                "added to", "add to the", "catalog",
                "catalogue", "offerings", "streaming",
                "my shows", "watch"
            ],

            "account_access": [
                "can't log in", "cannot log in",
                "can't login", "cannot login",
                "can't sign in", "cannot sign in",
                "login problem", "sign in problem",
                "locked out", "forgot password",
                "forgot my password"
            ],
        }

        scores = {}

        for intent, keywords in rules.items():
            score = 0

            for keyword in keywords:
                if keyword in t:
                    # Multi-word phrases are stronger evidence.
                    score += 2 if (" " in keyword or "-" in keyword) else 1

            if score:
                scores[intent] = score

        if scores:
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]

            if best_score >= 2:
                return best_intent, 0.99

        intent = self.model.predict([text])[0]

        confidence = float(
            np.max(
                self.model.predict_proba([text])[0]
            )
        )

        return intent, confidence

    def retrieve(self, text, k=3):
        query_vector = self.vectorizer.transform([text])

        scores = cosine_similarity(
            query_vector,
            self.retrieval_matrix
        ).ravel()

        indices = np.argsort(-scores)[:k]

        return [
            (
                self.responses.iloc[i].to_dict(),
                float(scores[i])
            )
            for i in indices
        ]

    def decide(self, text, intent, confidence):
        """
        Conservative risk policy.

        Security, refund and payment issues are escalated.
        Low ML confidence alone does not trigger escalation.
        """

        t = str(text).lower()
        reasons = []

        security_terms = [
            "hacked", "hack", "stolen",
            "unauthorized", "account compromised",
            "someone accessed", "someone changed",
            "password changed", "fraud",
            "do not recognize"
        ]

        payment_terms = [
            "refund", "charged twice",
            "charged me", "payment dispute",
            "money taken", "payment reversed",
            "purchase i did not make"
        ]

        if intent == "account_security":
            reasons.append(
                "account security issue requiring verification"
            )

        elif intent == "refunds":
            reasons.append(
                "refund issue requiring review"
            )

        elif intent == "payments_billing":
            reasons.append(
                "payment issue requiring review"
            )

        if any(term in t for term in security_terms):
            reasons.append(
                "account security issue requiring verification"
            )

        if any(term in t for term in payment_terms):
            reasons.append(
                "refund or payment issue requiring review"
            )

        if reasons:
            return (
                "escalate",
                "; ".join(dict.fromkeys(reasons))
            )

        return (
            "auto_handle",
            "safe informational, troubleshooting, "
            "follow-up, or clarification request"
        )

    @staticmethod
    def clean_response(text):
        text = re.sub(r"@\w+", "", str(text))
        text = re.sub(r"https?://t\.co/\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def reply(self, text, intent, action, retrieved):
        """
        Grounded response drafting.

        The response is based on the highest-similarity historical
        PlayStation support response rather than inventing a solution.
        """

        if not retrieved:
            return (
                "Thanks for reaching out. Please share a little "
                "more detail about the issue so we can help."
            )

        best, score = retrieved[0]

        historical = self.clean_response(
            best.get("support_response", "")
        )

        # Avoid copying an extremely weak retrieval.
        if score < 0.10 or not historical:
            if action == "escalate":
                return (
                    "Thanks for reaching out. This issue may require "
                    "account or purchase-specific review. Please "
                    "provide the requested details through PlayStation "
                    "Support so the team can assist securely."
                )

            return (
                "Thanks for reaching out. Please share a little "
                "more detail about the issue so we can help."
            )

        if action == "escalate":
            return (
                "Thanks for reaching out. This issue may require "
                "account or purchase-specific review. Please provide "
                "the requested details through PlayStation Support "
                "so the team can assist securely."
            )

        return historical

    def run(self, text):
        intent, confidence = self.classify(text)

        action, reason = self.decide(
            text,
            intent,
            confidence
        )

        retrieved = self.retrieve(text)

        reply = self.reply(
            text,
            intent,
            action,
            retrieved
        )

        return {
            "message": text,
            "intent": intent,
            "intent_confidence": round(confidence, 3),
            "action": action,
            "reason": reason,
            "reply": reply,
            "evidence": [
                {
                    "customer_message": r["customer_message"],
                    "historical_response": r["support_response"],
                    "similarity": round(score, 3)
                }
                for r, score in retrieved
            ]
        }
