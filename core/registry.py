from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type
import difflib

class Registry:
    """
    이름(문자열) -> 클래스(혹은 팩토리 함수) 매핑을 보관하는 작은 컨테이너.
    - register(name)(cls): 데코레이터로 등록
    - get(name): 등록된 것을 꺼내기
    - create(name, **kwargs): 바로 인스턴스 만들기(선택)
    - list(): 현재 등록된 키 확인
    """
    def __init__(self, name: str):
        self._name = name           # 디버깅용, 예: "SAMPLERS"
        self._map: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None, *, overwrite: bool = False) -> Callable:
        """
        @REG.register("gaussian") 처럼 쓰는 데코레이터.
        name=None이면 클래스 이름을 소문자로 자동 사용.
        overwrite=True면 같은 이름 덮어쓰기 허용.
        """
        def deco(obj: Any):
            key = name or obj.__name__.lower()
            if (not overwrite) and (key in self._map):
                raise KeyError(f"[{self._name}] '{key}'는 이미 등록되어 있습니다.")
            self._map[key] = obj
            return obj
        return deco

    def get(self, name: str) -> Any:
        """이름으로 등록된 것을 가져오고, 없으면 유사 후보를 제안."""
        if name in self._map:
            return self._map[name]
        # 못 찾을 때 비슷한 이름 추천
        candidates = difflib.get_close_matches(name, self._map.keys(), n=3, cutoff=0.5)
        hint = f" 가까운 후보: {candidates}" if candidates else ""
        raise KeyError(f"[{self._name}] '{name}'를 찾지 못했습니다.{hint}")

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        등록된 '클래스 또는 팩토리'를 즉시 호출해 인스턴스를 생성.
        (선택 기능) 보통 빌더 코드에서 편리함.
        """
        cls_or_factory = self.get(name)
        return cls_or_factory(*args, **kwargs)

    def list(self) -> List[str]:
        """현재 등록된 키 목록."""
        return sorted(self._map.keys())

# 카테고리별 전역 레지스트리 생성
ALGOS     = Registry("ALGOS")
DYNAMICS  = Registry("DYNAMICS")
COSTS     = Registry("COSTS")
SAMPLERS  = Registry("SAMPLERS")
CONSTRS   = Registry("CONSTRS")
