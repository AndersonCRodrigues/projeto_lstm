import numpy as np
from typing import Dict, Optional
import warnings

from production_lstm.config import settings


class BusinessMetricsCalculator:
    def __init__(self, accuracy_threshold_percent: float = 10.0):
        """
        Inicializa o calculador de métricas de negócio.

        Args:
            accuracy_threshold_percent: Percentual de erro aceitável para considerar
                                      uma predição como "precisa" (default: 10%)
        """
        self.config = settings
        self.accuracy_threshold_percent = accuracy_threshold_percent / 100.0

    def _validate_inputs(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Valida e normaliza os inputs."""
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        if predictions.shape != actuals.shape:
            raise ValueError(
                f"Predictions e actuals devem ter o mesmo shape. "
                f"Got {predictions.shape} vs {actuals.shape}"
            )

        if predictions.size == 0:
            raise ValueError("Arrays não podem estar vazios")

        if volumes is None:
            volumes = np.ones_like(predictions)
        else:
            volumes = np.asarray(volumes)
            if volumes.shape != predictions.shape:
                raise ValueError(
                    f"Volumes deve ter o mesmo shape que predictions. "
                    f"Got {volumes.shape} vs {predictions.shape}"
                )
            if np.any(volumes < 0):
                raise ValueError("Volumes não podem ser negativos")

        if np.any(np.isnan(predictions)) or np.any(np.isnan(actuals)):
            warnings.warn("Encontrados valores NaN nos dados. Eles serão removidos.")
            valid_mask = ~(
                np.isnan(predictions) | np.isnan(actuals) | np.isnan(volumes)
            )
            predictions = predictions[valid_mask]
            actuals = actuals[valid_mask]
            volumes = volumes[valid_mask]

        return predictions, actuals, volumes

    def calculate_business_impact(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calcula o impacto do modelo nas métricas de negócio.

        Args:
            predictions: Array de predições do modelo
            actuals: Array de valores reais
            volumes: Array de volumes/quantidades (opcional, default=1 para cada predição)

        Returns:
            Dict com métricas de negócio calculadas
        """
        predictions, actuals, volumes = self._validate_inputs(
            predictions, actuals, volumes
        )

        errors = np.abs(predictions - actuals)
        relative_errors = errors / np.maximum(
            np.abs(actuals), 1e-6
        )  # Evita divisão por zero

        accurate_predictions_mask = relative_errors < self.accuracy_threshold_percent

        # Métricas financeiras
        total_error_cost = np.sum(errors * volumes * self.config.cost_per_unit_error)
        accurate_revenue = np.sum(
            accurate_predictions_mask.astype(float)
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

        if total_error_cost > 0:
            roi = (net_benefit / total_error_cost) * 100
        elif net_benefit > 0:
            roi = float("inf")
        else:
            roi = 0.0

        avg_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)

        return {
            "total_error_cost": float(total_error_cost),
            "accurate_revenue": float(accurate_revenue),
            "inventory_savings": float(inventory_savings),
            "total_benefit": float(total_benefit),
            "net_benefit": float(net_benefit),
            "roi_percentage": float(roi),
            "accuracy_rate": float(accuracy_rate),
            "total_volume": float(np.sum(volumes)),
            "avg_error": float(avg_error),
            "median_error": float(median_error),
            "max_error": float(max_error),
            "num_predictions": len(predictions),
            "accuracy_threshold_used": float(self.accuracy_threshold_percent * 100),
        }

    def generate_business_report(
        self, metrics: Dict[str, float], period: str = "período recente"
    ) -> str:
        """
        Gera relatório formatado das métricas de negócio.

        Args:
            metrics: Dict com métricas calculadas
            period: Descrição do período analisado

        Returns:
            String com relatório formatado
        """
        # Validação dos inputs
        if not metrics:
            return "ERRO: Métricas não fornecidas para o relatório"

        def safe_get(key: str, default: float = 0.0) -> float:
            return metrics.get(key, default)

        net_benefit = safe_get("net_benefit")
        roi = safe_get("roi_percentage")

        # Status baseado no benefício líquido
        if net_benefit > 0:
            status_message = "✅ MODELO GERANDO VALOR POSITIVO"
            status_color = "VERDE"
        elif net_benefit == 0:
            status_message = "⚠️  MODELO EM PONTO DE EQUILÍBRIO"
            status_color = "AMARELO"
        else:
            status_message = "❌ MODELO PRECISA DE AJUSTES (CUSTO SUPERA BENEFÍCIO)"
            status_color = "VERMELHO"

        # Formatação especial para ROI infinito
        roi_display = f"{roi:.1f}%" if roi != float("inf") else "∞% (sem custos)"

        # Cálculo do benefício por unidade
        total_volume = safe_get("total_volume", 1)
        benefit_per_unit = net_benefit / max(total_volume, 1)

        report = f"""
RELATÓRIO DE IMPACTO NO NEGÓCIO - {period.upper()}
{'=' * 60}

📊 MÉTRICAS FINANCEIRAS:
   • Receita de Predições Precisas: R$ {safe_get('accurate_revenue'):>12,.2f}
   • Economia em Inventário:        R$ {safe_get('inventory_savings'):>12,.2f}
   • Custo Total de Erros:          R$ {safe_get('total_error_cost'):>12,.2f}
   {'-' * 60}
   • Benefício Líquido:             R$ {net_benefit:>12,.2f}
   • ROI do Modelo:                 {roi_display:>15s}

🔍 PERFORMANCE OPERACIONAL:
   • Taxa de Precisão:              {safe_get('accuracy_rate'):.1%}
   • Threshold de Precisão Usado:   {safe_get('accuracy_threshold_used'):.1f}%
   • Volume Total Processado:       {safe_get('total_volume'):>12,.0f} unidades
   • Número de Predições:           {safe_get('num_predictions'):>12,.0f}
   • Benefício por Unidade:         R$ {benefit_per_unit:>12,.2f}

📈 ANÁLISE DE ERROS:
   • Erro Médio:                    {safe_get('avg_error'):>12,.2f}
   • Erro Mediano:                  {safe_get('median_error'):>12,.2f}
   • Erro Máximo:                   {safe_get('max_error'):>12,.2f}

🎯 STATUS: {status_message}
   Classificação: {status_color}
        """
        return report.strip()

    def get_performance_grade(self, metrics: Dict[str, float]) -> str:
        """
        Atribui uma nota de performance baseada nas métricas.

        Returns:
            Nota de A+ a F
        """
        roi = metrics.get("roi_percentage", 0)
        accuracy = metrics.get("accuracy_rate", 0)
        net_benefit = metrics.get("net_benefit", 0)

        score = 0

        if roi >= 50:
            score += 40
        elif roi >= 20:
            score += 30
        elif roi >= 10:
            score += 20
        elif roi > 0:
            score += 10

        if accuracy >= 0.9:
            score += 35
        elif accuracy >= 0.8:
            score += 28
        elif accuracy >= 0.7:
            score += 21
        elif accuracy >= 0.6:
            score += 14
        elif accuracy >= 0.5:
            score += 7

        if net_benefit > 0:
            score += 25

        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"


def calculate_model_business_impact(
    predictions: np.ndarray,
    actuals: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    accuracy_threshold: float = 10.0,
    generate_report: bool = True,
) -> Dict:
    """
    Função de conveniência para calcular impacto de negócio.

    Returns:
        Dict contendo 'metrics', 'report' (se solicitado) e 'grade'
    """
    calculator = BusinessMetricsCalculator(accuracy_threshold)
    metrics = calculator.calculate_business_impact(predictions, actuals, volumes)

    result = {"metrics": metrics, "grade": calculator.get_performance_grade(metrics)}

    if generate_report:
        result["report"] = calculator.generate_business_report(metrics)

    return result
