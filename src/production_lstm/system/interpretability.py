import numpy as np
import shap
import torch
import logging
from typing import List, Dict, Any, Optional
from production_lstm.config import settings
from production_lstm.models.predictor import LSTMPredictor

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Classe para interpretação de modelos LSTM usando SHAP.

    Esta classe fornece funcionalidades para explicar predições de modelos LSTM
    através de valores SHAP, permitindo entender quais features são mais importantes
    para as predições do modelo.
    """

    def __init__(self, device: str = "cpu"):
        """
        Inicializa o interpretador de modelos.

        Args:
            device: Dispositivo para executar os cálculos ('cpu' ou 'cuda')
        """
        self.config = settings
        self.device = torch.device(device)
        self.explainer: Optional[shap.DeepExplainer] = None
        self.feature_names: Optional[List[str]] = None

    def initialize_explainer(
        self,
        model: LSTMPredictor,
        background_data: np.ndarray,
        feature_names: List[str],
    ) -> bool:
        """
        Inicializa o explicador SHAP.

        Args:
            model: Modelo LSTM treinado
            background_data: Dados de referência para o SHAP (formato 3D)
            feature_names: Lista com nomes das features

        Returns:
            bool: True se inicializado com sucesso, False caso contrário
        """
        if not self.config.enable_shap_explanations:
            logger.info("Explicações SHAP estão desabilitadas na configuração")
            return False

        try:
            return self._extracted_from_initialize_explainer_23(
                background_data, feature_names, model
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar SHAP: {e}")
            self.explainer = None
            return False

    def _extracted_from_initialize_explainer_23(self, background_data, feature_names, model):
        self._validate_inputs(background_data, feature_names)

        self.feature_names = feature_names

        # Validar formato dos dados de fundo
        if len(background_data.shape) != 3:
            raise ValueError(
                f"Dados de fundo devem ter formato 3D (samples, sequence, features). "
                f"Recebido: {background_data.shape}"
            )

        # Selecionar amostra dos dados de fundo
        sample_size = min(self.config.shap_sample_size, background_data.shape[0])
        background_sample = background_data[:sample_size]

        # Converter para tensor
        background_tensor = (
            torch.from_numpy(background_sample).float().to(self.device)
        )

        self.explainer = shap.DeepExplainer(model, background_tensor)

        logger.info(
            f"Explicador SHAP inicializado com sucesso. "
            f"Amostra de fundo: {background_sample.shape}, "
            f"Features: {len(feature_names)}"
        )
        return True

    def _validate_inputs(
        self, background_data: np.ndarray, feature_names: List[str]
    ) -> None:
        """
        Valida os dados de entrada.

        Args:
            background_data: Dados de referência
            feature_names: Nomes das features

        Raises:
            ValueError: Se os dados não são válidos
        """
        if background_data is None or len(background_data) == 0:
            raise ValueError("Dados de fundo não podem estar vazios")

        if not feature_names:
            raise ValueError("Lista de nomes de features não pode estar vazia")

        if len(background_data.shape) == 3:
            n_features = background_data.shape[-1]
            if n_features != len(feature_names):
                raise ValueError(
                    f"Número de features nos dados ({n_features}) não corresponde "
                    f"ao número de nomes de features ({len(feature_names)})"
                )

    def explain_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Gera explicação SHAP para uma predição.

        Args:
            input_data: Dados de entrada para explicar (formato 2D ou 3D)

        Returns:
            Dict contendo a explicação ou mensagem de erro
        """
        if not self.explainer:
            return {
                "error": "Explicador SHAP não foi inicializado",
                "explanation": None,
            }

        try:
            return self._generate_shap_explanation(input_data)

        except Exception as e:
            logger.error(f"Erro na geração da explicação SHAP: {str(e)}")
            return {
                "error": f"Erro na geração da explicação SHAP: {str(e)}",
                "explanation": None,
            }

    def _generate_shap_explanation(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Gera a explicação SHAP para os dados de entrada.

        Args:
            input_data: Dados para explicar

        Returns:
            Dict com a importância das features
        """
        input_tensor = self._prepare_input_tensor(input_data)

        # Calcular valores SHAP
        shap_values = self.explainer.shap_values(input_tensor)

        feature_importance = self._process_shap_values(shap_values)

        return {
            "error": None,
            "explanation": {
                "feature_importance": feature_importance,
                "total_features": len(feature_importance),
                "input_shape": input_data.shape,
            },
        }

    def _prepare_input_tensor(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Prepara o tensor de entrada para o SHAP.

        Args:
            input_data: Dados de entrada

        Returns:
            Tensor preparado
        """
        # Se os dados são 2D, adicionar dimensão de batch
        if len(input_data.shape) == 2:
            input_data = input_data[np.newaxis, :]
        elif len(input_data.shape) != 3:
            raise ValueError(
                f"Dados de entrada devem ter formato 2D ou 3D. "
                f"Recebido: {input_data.shape}"
            )

        return torch.from_numpy(input_data).float().to(self.device)

    def _process_shap_values(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processa os valores SHAP para criar ranking de importância.

        Args:
            shap_values: Valores SHAP calculados

        Returns:
            Lista ordenada com importância das features
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if len(shap_values.shape) == 3:
            shap_values = shap_values[0]

        # Calcular importância agregada por feature
        agg_shap_values = np.abs(shap_values).sum(axis=0)

        # Criar lista de importância
        feature_importance = [
            {
                "feature": name,
                "shap_importance": float(importance),
                "normalized_importance": 0.0,  # Será calculado abaixo
            }
            for name, importance in zip(self.feature_names, agg_shap_values)
        ]

        feature_importance.sort(key=lambda x: x["shap_importance"], reverse=True)

        total_importance = sum(item["shap_importance"] for item in feature_importance)

        for i, item in enumerate(feature_importance):
            item["rank"] = i + 1
            if total_importance > 0:
                item["normalized_importance"] = (
                    item["shap_importance"] / total_importance
                )

        return feature_importance

    def get_top_features(
        self, input_data: np.ndarray, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retorna as top-k features mais importantes.

        Args:
            input_data: Dados de entrada
            top_k: Número de features top para retornar

        Returns:
            Dict com as features mais importantes
        """
        explanation = self.explain_prediction(input_data)

        if explanation.get("error"):
            return explanation

        feature_importance = explanation["explanation"]["feature_importance"]
        top_features = feature_importance[:top_k]

        return {
            "error": None,
            "top_features": top_features,
            "total_features_analyzed": len(feature_importance),
        }

    def is_initialized(self) -> bool:
        """
        Verifica se o explicador foi inicializado.

        Returns:
            bool: True se inicializado, False caso contrário
        """
        return self.explainer is not None
