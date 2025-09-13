import numpy as np
from typing import Dict, Optional

from production_lstm.config import settings


class BusinessMetricsCalculator:
    def __init__(self):
        self.config = settings

    def calculate_business_impact(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        if volumes is None:
            volumes = np.ones_like(predictions)

        errors = np.abs(predictions - actuals)
        accurate_predictions_mask = errors < np.std(errors)

        total_error_cost = np.sum(errors * volumes * self.config.cost_per_unit_error)
        accurate_revenue = np.sum(
            accurate_predictions_mask
            * volumes
            * self.config.revenue_per_accurate_prediction
        )

        accuracy_rate = np.mean(accurate_predictions_mask)
        inventory_savings = (
            np.sum(volumes)
            * accuracy_rate
            * self.config.inventory_cost_reduction_factor
        )

        total_benefit = accurate_revenue + inventory_savings
        net_benefit = total_benefit - total_error_cost
        roi = (net_benefit / max(total_error_cost, 1e-6)) * 100

        return {
            "total_error_cost": total_error_cost,
            "accurate_revenue": accurate_revenue,
            "inventory_savings": inventory_savings,
            "total_benefit": total_benefit,
            "net_benefit": net_benefit,
            "roi_percentage": roi,
            "accuracy_rate": accuracy_rate,
            "total_volume": np.sum(volumes),
        }

    def generate_business_report(
        self, metrics: Dict[str, float], period: str = "período recente"
    ) -> str:

        status_message = (
            "✅ MODELO GERANDO VALOR POSITIVO"
            if metrics.get("net_benefit", 0) > 0
            else "⚠️ MODELO PRECISA DE AJUSTES (CUSTO SUPERA BENEFÍCIO)"
        )

        return f"""
📊 RELATÓRIO DE IMPACTO NO NEGÓCIO - {period.upper()}
----------------------------------------------------
💰 MÉTRICAS FINANCEIRAS:
• Receita de Predições Precisas: R$ {metrics.get('accurate_revenue', 0):,.2f}
• Economia em Inventário:      R$ {metrics.get('inventory_savings', 0):,.2f}
• Custo Total de Erros:        R$ {metrics.get('total_error_cost', 0):,.2f}
----------------------------------------------------
• Benefício Líquido:           R$ {metrics.get('net_benefit', 0):,.2f}
• ROI do Modelo:               {metrics.get('roi_percentage', 0):.1f}%

🎯 PERFORMANCE OPERACIONAL:
• Taxa de Precisão (vs. std dev): {metrics.get('accuracy_rate', 0):.1%}
• Volume Total Processado:        {metrics.get('total_volume', 0):,.0f} unidades
• Benefício por Unidade:          R$ {metrics.get('net_benefit', 0) / max(metrics.get('total_volume', 1), 1):.2f}

STATUS: {status_message}
        """.strip()
