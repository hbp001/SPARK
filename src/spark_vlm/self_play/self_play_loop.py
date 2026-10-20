import logging
from .reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class SelfPlayLoop:
    @staticmethod
    def _load_reward_calculator(cfg: dict) -> "RewardCalculator":
        module_path = cfg.get("reward_module_path")
        if not module_path:
            return RewardCalculator(cfg)
        import importlib.util

        spec = importlib.util.spec_from_file_location("reward_variant", module_path)
        if spec is None or spec.loader is None:
            logger.warning(
                f"[SelfPlayLoop] reward_module_path 로드 실패, 기본 RewardCalculator 사용: {module_path}"
            )
            return RewardCalculator(cfg)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(
                f"[SelfPlayLoop] reward_module_path 실행 실패({e}), 기본 RewardCalculator 사용: {module_path}"
            )
            return RewardCalculator(cfg)
        variant_cls = getattr(module, "RewardCalculator", None)
        if variant_cls is None:
            logger.warning(
                f"[SelfPlayLoop] {module_path}에 RewardCalculator 클래스가 없음, 기본값 사용"
            )
            return RewardCalculator(cfg)
        logger.info(f"[SelfPlayLoop] Reward variant 로드: {module_path} ({variant_cls.__name__})")
        return variant_cls(cfg)
